"""Modal H200 global-VRAM SCReLU v4 (Asymmetric 16-King-Bucket + 2024 T80 + S1 UHO + S2) NNUE trainer.

Pipeline:
1. Parallel CPU workers download & convert `linrock/test80-2024` (Stockfish/Lc0 2024 T80 binpacks)
   via `PGG106/Primer` into 32-byte Bullet records (`--max-score 3500 --min-ply 16`, filtered for
   no in-check and no capture bestmove), cached on `/vol/data/t80_2024_*.bin`.
2. Parallel CPU workers stream `linrock/bullet-training-data/S1` (UHO Stockfish-scored Bullet shards)
   onto `/vol/data/s1_uho_iter*.bin`.
3. Single H200 (141 GB HBM3e VRAM) loads up to 2.4B unique positions (76.8 GB) simultaneously into
   VRAM and trains Asymmetric 16-King-Bucket Factorized SCReLU (CHNN v4) with Power-2.5 loss,
   coprime global shuffling, WSD + Cosine LR decay, and 10-SB SWA averaging.
"""

import os
import modal

app = modal.App("chatarunga-v3-screlu-h200")
vol = modal.Volume.from_name("chess-nnue-vol", create_if_missing=True)

primer_image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.11")
    .apt_install("git", "g++", "make", "zstd", "curl", "ca-certificates")
    .pip_install("zstandard")
    .run_commands(
        "git clone --depth 1 https://github.com/PGG106/Primer.git /opt/Primer",
        "cd /opt/Primer/src && make -j$(nproc) build ARCH=x86-64-avx2",
    )
)

cuda_image = modal.Image.from_registry(
    "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
).pip_install("zstandard")

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

struct alignas(32) BulletRec {
    uint64_t occ;
    uint8_t pcs[16];
    int16_t score;
    uint8_t result;
    uint8_t ksq;
    uint8_t opp_ksq;
    uint8_t pad[3];
};
static_assert(sizeof(BulletRec) == 32, "BulletRec must be 32 bytes");

__constant__ uint8_t d_bucket4[32] = {
    0,0,1,1, 0,0,1,1, 2,2,2,2, 2,2,2,2,
    3,3,3,3, 3,3,3,3, 3,3,3,3, 3,3,3,3,
};
__constant__ uint8_t d_bucket8[32] = {
    0,0,1,1, 2,2,3,3, 4,4,5,5, 4,4,5,5,
    6,6,6,6, 6,6,6,6, 7,7,7,7, 7,7,7,7,
};
// Asymmetric 16-bucket layout (CHNN v4): exact match to `BUCKET_16_ASYM` in chess/src/nnue.rs
__constant__ uint8_t d_bucket16_asym[32] = {
     0,  1,  2,  3,
     4,  5,  6,  7,
     8,  8,  9,  9,
    10, 10, 11, 11,
    12, 12, 13, 13,
    12, 12, 13, 13,
    14, 14, 15, 15,
    14, 14, 15, 15,
};

__device__ __forceinline__ int king_bucket(int kb, int asym, int km) {
    if (kb <= 1) return 0;
    int idx32 = (km >> 3) * 4 + (km & 7);
    if (kb == 4) return d_bucket4[idx32];
    if (kb == 8) return d_bucket8[idx32];
    if (kb == 16) return asym ? d_bucket16_asym[idx32] : ((km >> 4) * 4 + (km & 7));
    return idx32 % kb;
}

__device__ __forceinline__ float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void fwd_bwd_screlu_v4(
    const BulletRec* __restrict__ recs,
    uint64_t total_n,
    uint64_t base_step,
    uint64_t perm_offset,
    uint64_t perm_stride,
    int batch_size,
    int kb,
    int asym,
    int hl,
    int ob,
    int factorized,
    float wdl_w,
    float scale_f,
    const float* __restrict__ ft_w,
    const float* __restrict__ ft_b,
    const float* __restrict__ out_w,
    const float* __restrict__ out_b,
    float* __restrict__ g_ft_w,
    float* __restrict__ g_ft_b,
    float* __restrict__ g_out_w,
    float* __restrict__ g_out_b,
    double* __restrict__ mse_sum,
    int do_bwd
) {
    int bid = blockIdx.x;
    if (bid >= batch_size) return;
    int t = threadIdx.x;
    int half = hl >> 1;

    __shared__ int s_cnt;
    __shared__ int s_bkt;
    __shared__ float s_target;
    __shared__ float s_weight;
    __shared__ uint16_t s_f_stm[64];
    __shared__ uint16_t s_f_opp[64];
    __shared__ float s_warp[32];
    __shared__ float s_d_out;

    if (t == 0) {
        uint64_t pos_idx = (perm_offset + (base_step + (uint64_t)bid) * perm_stride) % total_n;
        BulletRec r = recs[pos_idx];
        uint64_t occ = r.occ;
        int cnt = __popcll(occ);
        int div = (ob > 1) ? ((32 + ob - 1) / ob) : 32;
        int bkt = (ob > 1) ? min(max(cnt - 2, 0) / div, ob - 1) : 0;
        s_bkt = bkt;

        s_weight = (abs((int)r.score) <= 3500) ? 1.0f : 0.0f;
        float sc_clamped = fminf(fmaxf((float)r.score, -3500.0f), 3500.0f);
        float sc_sig = sigmoidf(sc_clamped / scale_f);
        float wdl = (float)r.result * 0.5f;
        s_target = wdl_w * wdl + (1.0f - wdl_w) * sc_sig;

        int mirror_s = (kb > 1) && ((r.ksq & 7) >= 4);
        int km_s = mirror_s ? (r.ksq ^ 7) : r.ksq;
        int kb_s = king_bucket(kb, asym, km_s);

        int mirror_o = (kb > 1) && ((r.opp_ksq & 7) >= 4);
        int km_o = mirror_o ? (r.opp_ksq ^ 7) : r.opp_ksq;
        int kb_o = king_bucket(kb, asym, km_o);

        int base_off = kb * 768;
        int nfeat = 0;
        int idx = 0;
        while (occ && idx < 32) {
            int sq = __ffsll(occ) - 1;
            occ &= occ - 1;
            uint8_t byte = r.pcs[idx >> 1];
            uint8_t nib = (idx & 1) == 0 ? (byte & 0xF) : (byte >> 4);
            int piece = nib & 7;
            int col = nib >> 3;

            int sq_s = mirror_s ? (sq ^ 7) : sq;
            int fs_base = col * 384 + piece * 64 + sq_s;
            s_f_stm[nfeat] = (uint16_t)(kb_s * 768 + fs_base);

            int sq_o = sq ^ 56;
            if (mirror_o) sq_o ^= 7;
            int fo_base = (col ^ 1) * 384 + piece * 64 + sq_o;
            s_f_opp[nfeat] = (uint16_t)(kb_o * 768 + fo_base);
            nfeat++;

            if (factorized) {
                s_f_stm[nfeat] = (uint16_t)(base_off + fs_base);
                s_f_opp[nfeat] = (uint16_t)(base_off + fo_base);
                nfeat++;
            }
            idx++;
        }
        s_cnt = nfeat;
    }
    __syncthreads();

    int nf = s_cnt;
    int bkt = s_bkt;

    float s0 = ft_b[t];
    float s1 = ft_b[t + half];
    float o0 = s0;
    float o1 = s1;

    for (int k = 0; k < nf; k++) {
        size_t off_s = (size_t)s_f_stm[k] * hl;
        s0 += ft_w[off_s + t];
        s1 += ft_w[off_s + t + half];
        size_t off_o = (size_t)s_f_opp[k] * hl;
        o0 += ft_w[off_o + t];
        o1 += ft_w[off_o + t + half];
    }

    float cs0 = fminf(fmaxf(s0, 0.0f), 1.0f);
    float cs1 = fminf(fmaxf(s1, 0.0f), 1.0f);
    float co0 = fminf(fmaxf(o0, 0.0f), 1.0f);
    float co1 = fminf(fmaxf(o1, 0.0f), 1.0f);

    int ow_stm = bkt * 2 * hl;
    int ow_opp = ow_stm + hl;
    float ws0 = out_w[ow_stm + t];
    float ws1 = out_w[ow_stm + t + half];
    float wo0 = out_w[ow_opp + t];
    float wo1 = out_w[ow_opp + t + half];

    float partial = (cs0 * cs0) * ws0 + (cs1 * cs1) * ws1 + (co0 * co0) * wo0 + (co1 * co1) * wo1;

    for (int mask = 16; mask > 0; mask >>= 1) {
        partial += __shfl_down_sync(0xffffffff, partial, mask);
    }
    int lane = t & 31;
    int wid = t >> 5;
    if (lane == 0) s_warp[wid] = partial;
    __syncthreads();

    int nwarps = (half + 31) >> 5;
    if (wid == 0) {
        float v = (lane < nwarps) ? s_warp[lane] : 0.0f;
        for (int mask = 16; mask > 0; mask >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, mask);
        }
        if (lane == 0) {
            float out = v + out_b[bkt];
            float pred = sigmoidf(out);
            float diff = pred - s_target;
            float w = s_weight;
            atomicAdd(mse_sum, (double)(w * diff * diff));
            float grad_p = 2.5f * diff * sqrtf(fabsf(diff) + 1e-12f);
            s_d_out = w * grad_p * pred * (1.0f - pred);
            if (do_bwd && w > 0.0f) atomicAdd(&g_out_b[bkt], s_d_out);
        }
    }
    __syncthreads();

    if (!do_bwd) return;

    float d_out = s_d_out;
    atomicAdd(&g_out_w[ow_stm + t], d_out * (cs0 * cs0));
    atomicAdd(&g_out_w[ow_stm + t + half], d_out * (cs1 * cs1));
    atomicAdd(&g_out_w[ow_opp + t], d_out * (co0 * co0));
    atomicAdd(&g_out_w[ow_opp + t + half], d_out * (co1 * co1));

    float ds0 = (s0 > 0.0f && s0 < 1.0f) ? (2.0f * d_out * ws0 * cs0) : 0.0f;
    float ds1 = (s1 > 0.0f && s1 < 1.0f) ? (2.0f * d_out * ws1 * cs1) : 0.0f;
    float do0 = (o0 > 0.0f && o0 < 1.0f) ? (2.0f * d_out * wo0 * co0) : 0.0f;
    float do1 = (o1 > 0.0f && o1 < 1.0f) ? (2.0f * d_out * wo1 * co1) : 0.0f;

    atomicAdd(&g_ft_b[t], ds0 + do0);
    atomicAdd(&g_ft_b[t + half], ds1 + do1);

    for (int k = 0; k < nf; k++) {
        size_t off_s = (size_t)s_f_stm[k] * hl;
        if (ds0 != 0.0f) atomicAdd(&g_ft_w[off_s + t], ds0);
        if (ds1 != 0.0f) atomicAdd(&g_ft_w[off_s + t + half], ds1);
        size_t off_o = (size_t)s_f_opp[k] * hl;
        if (do0 != 0.0f) atomicAdd(&g_ft_w[off_o + t], do0);
        if (do1 != 0.0f) atomicAdd(&g_ft_w[off_o + t + half], do1);
    }
}

__global__ void adamw_kernel(
    float* __restrict__ w,
    float* __restrict__ m,
    float* __restrict__ v,
    float* __restrict__ g,
    size_t n,
    float inv_bs,
    float lr,
    float wd,
    float bc1,
    float bc2,
    float max_w
) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i] * inv_bs;
    g[i] = 0.0f;
    float mi = 0.9f * m[i] + 0.1f * gi;
    float vi = 0.999f * v[i] + 0.001f * gi * gi;
    m[i] = mi;
    v[i] = vi;
    float m_hat = mi * bc1;
    float v_hat = vi * bc2;
    float wi = w[i] * (1.0f - lr * wd) - lr * m_hat / (sqrtf(v_hat) + 1e-8f);
    if (max_w > 0.0f) wi = fminf(fmaxf(wi, -max_w), max_w);
    w[i] = wi;
}

static uint64_t rng_state = 20261127ULL;
static inline uint64_t xorshift64() {
    uint64_t x = rng_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    rng_state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

void save_chnn_screlu(
    const char* path, uint32_t ver, int kb, int hl, int ob, int factorized, int scale,
    const std::vector<float>& ft_w, const std::vector<float>& ft_b,
    const std::vector<float>& out_w, const std::vector<float>& out_b
) {
    const int QA = 255, QB = 64;
    FILE* f = fopen(path, "wb");
    if (!f) { perror("fopen out"); exit(1); }
    uint32_t ukb = kb, uhl = hl, uob = ob;
    fwrite("CHNN", 1, 4, f);
    fwrite(&ver, 4, 1, f);
    fwrite(&ukb, 4, 1, f);
    fwrite(&uhl, 4, 1, f);
    fwrite(&uob, 4, 1, f);
    fwrite(&QA, 4, 1, f);
    fwrite(&QB, 4, 1, f);
    fwrite(&scale, 4, 1, f);

    std::vector<int16_t> q_ft_w((size_t)kb * 768 * hl);
    size_t base_start = (size_t)kb * 768 * hl;
    for (int b = 0; b < kb; b++) {
        for (int feat = 0; feat < 768; feat++) {
            size_t row = ((size_t)b * 768 + feat) * hl;
            size_t brow = base_start + (size_t)feat * hl;
            for (int h = 0; h < hl; h++) {
                float val = ft_w[row + h] + (factorized ? ft_w[brow + h] : 0.0f);
                int q = (int)llroundf(val * QA);
                q_ft_w[row + h] = (int16_t)std::max(-32768, std::min(32767, q));
            }
        }
    }
    fwrite(q_ft_w.data(), 2, q_ft_w.size(), f);

    std::vector<int16_t> q_ft_b(hl);
    for (int h = 0; h < hl; h++) {
        int q = (int)llroundf(ft_b[h] * QA);
        q_ft_b[h] = (int16_t)std::max(-32768, std::min(32767, q));
    }
    fwrite(q_ft_b.data(), 2, q_ft_b.size(), f);

    std::vector<int16_t> q_out_w(out_w.size());
    for (size_t i = 0; i < out_w.size(); i++) {
        float c = std::max(-1.98f, std::min(1.98f, out_w[i]));
        int q = (int)llroundf(c * QB);
        q_out_w[i] = (int16_t)std::max(-127, std::min(127, q));
    }
    fwrite(q_out_w.data(), 2, q_out_w.size(), f);

    std::vector<int32_t> q_out_b(ob);
    for (int i = 0; i < ob; i++) q_out_b[i] = (int32_t)llroundf(out_b[i] * (QA * QB));
    fwrite(q_out_b.data(), 4, q_out_b.size(), f);
    fclose(f);
}

int main(int argc, char** argv) {
    std::vector<std::string> data_paths;
    const char* out_path = nullptr;
    int kb = 16, asym = 1, hl = 1024, ob = 8, superbatches = 100, batch_size = 16384, scale_out = 160;
    uint64_t sb_pos = 100000000ULL;
    float lr_start = 0.0012f, lr_end = 0.00001f, wdl_start = 0.35f, wdl_end = 0.35f, scale_f = 400.0f;
    int swa_last = 10;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--data") && i+1 < argc) data_paths.push_back(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i+1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--kb") && i+1 < argc) kb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--asym") && i+1 < argc) asym = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--hl") && i+1 < argc) hl = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ob") && i+1 < argc) ob = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--superbatches") && i+1 < argc) superbatches = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sb-pos") && i+1 < argc) sb_pos = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--batch") && i+1 < argc) batch_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr") && i+1 < argc) lr_start = atof(argv[++i]);
        else if (!strcmp(argv[i], "--lr-end") && i+1 < argc) lr_end = atof(argv[++i]);
        else if (!strcmp(argv[i], "--wdl-start") && i+1 < argc) wdl_start = atof(argv[++i]);
        else if (!strcmp(argv[i], "--wdl-end") && i+1 < argc) wdl_end = atof(argv[++i]);
        else if (!strcmp(argv[i], "--scale-out") && i+1 < argc) scale_out = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--swa") && i+1 < argc) swa_last = atoi(argv[++i]);
    }
    if (data_paths.empty() || !out_path) { fprintf(stderr, "Need --data and --out\n"); return 1; }

    std::vector<uint64_t> shard_counts;
    uint64_t total_raw = 0;
    for (const auto& p : data_paths) {
        FILE* f = fopen(p.c_str(), "rb");
        if (!f) { perror(p.c_str()); return 1; }
        fseek(f, 0, SEEK_END);
        uint64_t n = (uint64_t)ftell(f) / 32;
        fclose(f);
        shard_counts.push_back(n);
        total_raw += n;
    }

    // Hold out 50k positions per shard (up to 400k total) for balanced multi-source validation!
    uint64_t val_per_shard = std::min<uint64_t>(50000, 400000 / std::max<size_t>(1, data_paths.size()));
    uint64_t val_n = val_per_shard * data_paths.size();
    uint64_t train_n = total_raw - val_n;

    auto t_load0 = std::chrono::steady_clock::now();
    BulletRec* d_recs = nullptr;
    BulletRec* d_val_recs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_recs, train_n * 32));
    CUDA_CHECK(cudaMalloc(&d_val_recs, val_n * 32));

    const uint64_t chunk_n = 8000000;
    BulletRec* h_buf = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_buf, chunk_n * 32));
    uint64_t dst_off = 0, val_off = 0;
    for (size_t si = 0; si < data_paths.size(); si++) {
        FILE* f = fopen(data_paths[si].c_str(), "rb");
        uint64_t n = shard_counts[si];
        uint64_t take_train = n - val_per_shard;
        uint64_t rem = take_train;
        while (rem > 0) {
            uint64_t want = std::min(rem, chunk_n);
            size_t got = fread(h_buf, 32, want, f);
            CUDA_CHECK(cudaMemcpy(d_recs + dst_off, h_buf, got * 32, cudaMemcpyHostToDevice));
            dst_off += got;
            rem -= got;
        }
        size_t got = fread(h_buf, 32, val_per_shard, f);
        CUDA_CHECK(cudaMemcpy(d_val_recs + val_off, h_buf, got * 32, cudaMemcpyHostToDevice));
        val_off += got;
        fclose(f);
    }
    CUDA_CHECK(cudaFreeHost(h_buf));
    double t_load = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load0).count();
    printf("Loaded %.3fB train + %llu val positions across %zu shards into VRAM in %.1fs\n",
           (double)train_n / 1e9, (unsigned long long)val_n, data_paths.size(), t_load);
    fflush(stdout);

    int factorized = (kb > 1) ? 1 : 0;
    size_t n_feats = (size_t)kb * 768 + (factorized ? 768 : 0);
    size_t n_ft_w = n_feats * hl;
    size_t n_out_w = (size_t)ob * 2 * hl;

    std::vector<float> h_ft_w(n_ft_w, 0.0f), h_ft_b(hl, 0.0f), h_out_w(n_out_w, 0.0f), h_out_b(ob, 0.0f);
    float ft_sc = 0.04f;
    float out_sc = 1.0f / sqrtf((float)(2 * hl));
    size_t init_start = factorized ? (size_t)kb * 768 * hl : 0;
    for (size_t i = init_start; i < n_ft_w; i++)
        h_ft_w[i] = ((float)(xorshift64() & 0xFFFFFF) / 8388608.0f - 1.0f) * ft_sc;
    for (size_t i = 0; i < n_out_w; i++)
        h_out_w[i] = ((float)(xorshift64() & 0xFFFFFF) / 8388608.0f - 1.0f) * out_sc;

    float *d_ft_w, *d_ft_b, *d_out_w, *d_out_b;
    float *g_ft_w, *g_ft_b, *g_out_w, *g_out_b;
    float *m_ft_w, *v_ft_w, *m_ft_b, *v_ft_b, *m_out_w, *v_out_w, *m_out_b, *v_out_b;
    double *d_loss;
    auto alloc_zero = [](float** p, size_t n) { CUDA_CHECK(cudaMalloc(p, n * sizeof(float))); CUDA_CHECK(cudaMemset(*p, 0, n * sizeof(float))); };
    alloc_zero(&d_ft_w, n_ft_w); alloc_zero(&g_ft_w, n_ft_w); alloc_zero(&m_ft_w, n_ft_w); alloc_zero(&v_ft_w, n_ft_w);
    alloc_zero(&d_ft_b, hl);     alloc_zero(&g_ft_b, hl);     alloc_zero(&m_ft_b, hl);     alloc_zero(&v_ft_b, hl);
    alloc_zero(&d_out_w, n_out_w); alloc_zero(&g_out_w, n_out_w); alloc_zero(&m_out_w, n_out_w); alloc_zero(&v_out_w, n_out_w);
    alloc_zero(&d_out_b, ob);    alloc_zero(&g_out_b, ob);    alloc_zero(&m_out_b, ob);    alloc_zero(&v_out_b, ob);
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_ft_w, h_ft_w.data(), n_ft_w * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ft_b, h_ft_b.data(), hl * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_w, h_out_w.data(), n_out_w * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_b, h_out_b.data(), ob * 4, cudaMemcpyHostToDevice));

    uint64_t batches_per_sb = sb_pos / batch_size;
    uint64_t total_steps = (uint64_t)superbatches * batches_per_sb;
    uint64_t step = 0;
    double best_val = 1e9;
    std::vector<float> best_ft_w(n_ft_w), best_ft_b(hl), best_out_w(n_out_w), best_out_b(ob);
    std::vector<float> swa_ft_w(n_ft_w, 0.0f), swa_ft_b(hl, 0.0f), swa_out_w(n_out_w, 0.0f), swa_out_b(ob, 0.0f);
    std::vector<float> tmp_ft_w(n_ft_w), tmp_ft_b(hl), tmp_out_w(n_out_w), tmp_out_b(ob);
    int swa_count = 0;
    int best_sb = 0;

    uint32_t out_ver = (kb == 16 && asym) ? 4 : 1;
    printf("Chatarunga SCReLU v%u H200: (768*%d%s -> %d)x2 -> %d | %d SBs x %lluM = %.1fB steps | Power-2.5 | WDL %.2f->%.2f | LR %.5f->%.5f | SWA %d\n",
           out_ver, kb, (kb == 16 && asym) ? "a" : "", hl, ob, superbatches,
           (unsigned long long)(sb_pos / 1000000ULL),
           (double)(superbatches * sb_pos) / 1e9, wdl_start, wdl_end, lr_start, lr_end, swa_last);
    fflush(stdout);

    auto eval_val = [&]() -> double {
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(double)));
        for (uint64_t off = 0; off < val_n; off += batch_size) {
            int bs = (int)std::min<uint64_t>(batch_size, val_n - off);
            fwd_bwd_screlu_v4<<<bs, hl / 2>>>(
                d_val_recs, val_n, off, 0, 1, bs,
                kb, asym, hl, ob, factorized, 0.35f, scale_f,
                d_ft_w, d_ft_b, d_out_w, d_out_b,
                g_ft_w, g_ft_b, g_out_w, g_out_b, d_loss, 0
            );
        }
        double vloss_sum = 0.0;
        CUDA_CHECK(cudaMemcpy(&vloss_sum, d_loss, sizeof(double), cudaMemcpyDeviceToHost));
        return vloss_sum / (double)val_n;
    };

    auto gcd = [](uint64_t a, uint64_t b) { while (b) { uint64_t t = a % b; a = b; b = t; } return a; };
    auto t_start = std::chrono::steady_clock::now();

    for (int sb = 1; sb <= superbatches; sb++) {
        auto t0 = std::chrono::steady_clock::now();
        uint64_t stride = ((xorshift64() % 50000000ULL) + 1000003ULL) | 1ULL;
        while (gcd(stride, train_n) != 1) stride += 2;
        uint64_t offset = xorshift64() % train_n;

        float sb_prog = (float)(sb - 1) / (float)std::max(1, superbatches - 1);
        float wdl_w = wdl_start + (wdl_end - wdl_start) * sb_prog;
        double sb_mse = 0.0;

        for (uint64_t b = 0; b < batches_per_sb; b++) {
            step++;
            float prog = fminf(1.0f, (float)step / (float)total_steps);
            float lr;
            if (prog < 0.02f) lr = lr_start * (prog / 0.02f);
            else if (prog < 0.60f) lr = lr_start;
            else {
                float dprog = (prog - 0.60f) / 0.40f;
                lr = lr_end + (lr_start - lr_end) * 0.5f * (1.0f + cosf(3.14159265f * dprog));
            }

            CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(double)));
            fwd_bwd_screlu_v4<<<batch_size, hl / 2>>>(
                d_recs, train_n, b * batch_size, offset, stride, batch_size,
                kb, asym, hl, ob, factorized, wdl_w, scale_f,
                d_ft_w, d_ft_b, d_out_w, d_out_b,
                g_ft_w, g_ft_b, g_out_w, g_out_b, d_loss, 1
            );

            float inv_bs = 1.0f / (float)batch_size;
            float bc1 = 1.0f / (1.0f - powf(0.9f, (float)step));
            float bc2 = 1.0f / (1.0f - powf(0.999f, (float)step));
            adamw_kernel<<<(n_ft_w + 255) / 256, 256>>>(d_ft_w, m_ft_w, v_ft_w, g_ft_w, n_ft_w, inv_bs, lr, 0.0f, bc1, bc2, 8.0f);
            adamw_kernel<<<(hl + 255) / 256, 256>>>(d_ft_b, m_ft_b, v_ft_b, g_ft_b, hl, inv_bs, lr, 0.0f, bc1, bc2, 8.0f);
            adamw_kernel<<<(n_out_w + 255) / 256, 256>>>(d_out_w, m_out_w, v_out_w, g_out_w, n_out_w, inv_bs, lr, 0.0f, bc1, bc2, 1.98f);
            adamw_kernel<<<(ob + 255) / 256, 256>>>(d_out_b, m_out_b, v_out_b, g_out_b, ob, inv_bs, lr, 0.0f, bc1, bc2, 0.0f);

            if ((b & 63) == 63 || b + 1 == batches_per_sb) {
                double bl = 0.0;
                CUDA_CHECK(cudaMemcpy(&bl, d_loss, sizeof(double), cudaMemcpyDeviceToHost));
                sb_mse += bl * ((b + 1 == batches_per_sb && (b & 63) != 63) ? (double)((b & 63) + 1) : 64.0);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        double val_mse = eval_val();
        double tr_mse = sb_mse / (double)(batches_per_sb * batch_size);

        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const char* mark = "";
        bool copied = false;
        if (val_mse < best_val) {
            best_val = val_mse; best_sb = sb; mark = " *";
            CUDA_CHECK(cudaMemcpy(best_ft_w.data(), d_ft_w, n_ft_w * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(best_ft_b.data(), d_ft_b, hl * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(best_out_w.data(), d_out_w, n_out_w * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(best_out_b.data(), d_out_b, ob * 4, cudaMemcpyDeviceToHost));
            copied = true;
        }
        if (swa_last > 0 && sb > superbatches - swa_last) {
            const auto& fw = copied ? best_ft_w : tmp_ft_w;
            const auto& fb = copied ? best_ft_b : tmp_ft_b;
            const auto& ow = copied ? best_out_w : tmp_out_w;
            const auto& ob_v = copied ? best_out_b : tmp_out_b;
            if (!copied) {
                CUDA_CHECK(cudaMemcpy(tmp_ft_w.data(), d_ft_w, n_ft_w * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(tmp_ft_b.data(), d_ft_b, hl * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(tmp_out_w.data(), d_out_w, n_out_w * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(tmp_out_b.data(), d_out_b, ob * 4, cudaMemcpyDeviceToHost));
            }
            for (size_t i = 0; i < n_ft_w; i++) swa_ft_w[i] += fw[i];
            for (size_t i = 0; i < (size_t)hl; i++) swa_ft_b[i] += fb[i];
            for (size_t i = 0; i < n_out_w; i++) swa_out_w[i] += ow[i];
            for (size_t i = 0; i < (size_t)ob; i++) swa_out_b[i] += ob_v[i];
            swa_count++;
        }
        printf("SB %3d/%d | train~%.6f val(0.35)=%.6f%s | wdl=%.2f | %.2fs (%.2fM pos/s)\n",
               sb, superbatches, tr_mse, val_mse, mark, wdl_w, dt, (double)(batches_per_sb * batch_size) / dt / 1e6);
        fflush(stdout);
    }

    if (swa_count > 1) {
        float inv = 1.0f / (float)swa_count;
        for (float& x : swa_ft_w) x *= inv;
        for (float& x : swa_ft_b) x *= inv;
        for (float& x : swa_out_w) x *= inv;
        for (float& x : swa_out_b) x *= inv;
        CUDA_CHECK(cudaMemcpy(d_ft_w, swa_ft_w.data(), n_ft_w * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ft_b, swa_ft_b.data(), hl * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_out_w, swa_out_w.data(), n_out_w * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_out_b, swa_out_b.data(), ob * 4, cudaMemcpyHostToDevice));
        double swa_val = eval_val();
        printf("SWA (%d SBs) val(0.35)=%.6f vs best single SB %d val=%.6f\n", swa_count, swa_val, best_sb, best_val);
        if (swa_val <= best_val) {
            best_val = swa_val;
            best_ft_w.swap(swa_ft_w);
            best_ft_b.swap(swa_ft_b);
            best_out_w.swap(swa_out_w);
            best_out_b.swap(swa_out_b);
            printf("Using SWA weights!\n");
        }
    }

    save_chnn_screlu(out_path, out_ver, kb, hl, ob, factorized, scale_out, best_ft_w, best_ft_b, best_out_w, best_out_b);
    double tot_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    printf("Saved CHNN v%u SCReLU NNUE (val %.6f) -> %s in %.1fs\n", out_ver, best_val, out_path, tot_s);
    return 0;
}
"""


@app.function(image=primer_image, volumes={"/vol": vol}, cpu=8, memory=16384, timeout=3600)
def fetch_t80_month(month_file: str, out_tag: str, limit_pos: int = 200_000_000) -> str:
    """Download a `linrock/test80-2024` .binpack.zst month and convert `limit_pos` filtered positions via Primer."""
    import subprocess
    import time

    os.makedirs("/vol/data", exist_ok=True)
    out_path = f"/vol/data/{out_tag}"
    min_bytes = int(limit_pos * 32 * 0.90)
    if os.path.exists(out_path) and os.path.getsize(out_path) >= min_bytes:
        print(f"Using cached {out_path} ({os.path.getsize(out_path) / 1e9:.2f} GB)", flush=True)
        return out_path

    url = f"https://huggingface.co/datasets/linrock/test80-2024/resolve/main/{month_file}"
    tmp_zst = "/tmp/in.binpack.zst"
    tmp_bp = "/tmp/in.binpack"
    tmp_out = "/tmp/out.bin"
    t0 = time.time()
    print(f"Downloading {url} ...", flush=True)
    subprocess.run(["curl", "-L", "-f", "--retry", "3", "-o", tmp_zst, url], check=True)
    print(f"Downloaded {os.path.getsize(tmp_zst) / 1e9:.2f} GB in {time.time() - t0:.1f}s; decompressing...", flush=True)
    t1 = time.time()
    subprocess.run(["zstd", "-d", "-T0", "-f", tmp_zst, "-o", tmp_bp], check=True)
    os.remove(tmp_zst)
    print(f"Decompressed {os.path.getsize(tmp_bp) / 1e9:.2f} GB in {time.time() - t1:.1f}s; running Primer...", flush=True)
    t2 = time.time()
    subprocess.run(
        [
            "/opt/Primer/src/primer",
            "convert",
            tmp_bp,
            tmp_out,
            "--max-score", "3500",
            "--min-ply", "16",
            "--limit-positions", str(limit_pos),
        ],
        check=True,
    )
    os.remove(tmp_bp)
    sz = os.path.getsize(tmp_out)
    # Trim to exact 32-byte multiple and copy to Modal Volume
    rem = sz % 32
    if rem:
        with open(tmp_out, "r+b") as f:
            f.truncate(sz - rem)
    import shutil
    shutil.move(tmp_out, out_path)
    vol.commit()
    final_sz = os.path.getsize(out_path)
    print(
        f"Converted {final_sz // 32:,} positions ({final_sz / 1e9:.2f} GB) -> {out_path} "
        f"(primer {time.time() - t2:.1f}s, total {time.time() - t0:.1f}s)",
        flush=True,
    )
    return out_path


@app.function(image=primer_image, volumes={"/vol": vol}, cpu=4, memory=8192, timeout=1800)
def fetch_s1_uho(shard: int, positions: int = 100_000_000) -> str:
    """Stream `positions` 32-byte Bullet records from linrock/bullet-training-data S1 (UHO Stockfish)."""
    import time
    import urllib.request
    import zstandard

    os.makedirs("/vol/data", exist_ok=True)
    out_path = f"/vol/data/s1_uho_iter{shard}_{positions // 1_000_000}m.bin"
    target_bytes = positions * 32
    if os.path.exists(out_path) and os.path.getsize(out_path) >= target_bytes:
        print(f"Using cached {out_path} ({os.path.getsize(out_path) / 1e9:.2f} GB)", flush=True)
        return out_path

    url = f"https://huggingface.co/datasets/linrock/bullet-training-data/resolve/main/S1/UHO.pdist.iter-{shard}.bullet.bin.zst"
    print(f"Streaming {positions:,} UHO positions from {url} -> {out_path}", flush=True)
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "games-ai-modal"})
    written = 0
    dctx = zstandard.ZstdDecompressor()
    with urllib.request.urlopen(req, timeout=60) as resp:
        with dctx.stream_reader(resp) as reader, open(out_path, "wb") as out:
            while written < target_bytes:
                want = min(16 * 1024 * 1024, target_bytes - written)
                chunk = reader.read(want)
                if not chunk:
                    break
                if written + len(chunk) > target_bytes:
                    chunk = chunk[: target_bytes - written]
                rem = len(chunk) % 32
                if rem:
                    chunk = chunk[:-rem]
                out.write(chunk)
                written += len(chunk)
    vol.commit()
    print(f"Done streaming {written // 32:,} UHO positions to {out_path} in {time.time() - t0:.1f}s", flush=True)
    return out_path


@app.function(image=cuda_image, volumes={"/vol": vol}, gpu=["H200", "H100"], cpu=8, memory=65536, timeout=3600)
def train_screlu_v4_gpu(
    data_paths: list[str],
    out_name: str,
    kb: int = 16,
    asym: int = 1,
    hl: int = 1024,
    ob: int = 8,
    superbatches: int = 100,
    sb_pos: int = 100_000_000,
    batch_size: int = 16384,
    lr: float = 0.0012,
    lr_end: float = 0.00001,
    wdl_start: float = 0.35,
    wdl_end: float = 0.35,
    scale_out: int = 160,
    swa: int = 10,
) -> bytes:
    import subprocess

    os.makedirs("/vol/nets", exist_ok=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False)
    cu_path = "/tmp/train_screlu_v4.cu"
    bin_path = "/tmp/train_screlu_v4"
    with open(cu_path, "w") as f:
        f.write(CUDA_SRC)

    subprocess.run(["nvcc", "-O3", "-arch=native", "-use_fast_math", cu_path, "-o", bin_path], check=True)
    vol_out = f"/vol/nets/{out_name}"
    cmd = [
        bin_path,
        "--out", vol_out,
        "--kb", str(kb),
        "--asym", str(asym),
        "--hl", str(hl),
        "--ob", str(ob),
        "--superbatches", str(superbatches),
        "--sb-pos", str(sb_pos),
        "--batch", str(batch_size),
        "--lr", str(lr),
        "--lr-end", str(lr_end),
        "--wdl-start", str(wdl_start),
        "--wdl-end", str(wdl_end),
        "--scale-out", str(scale_out),
        "--swa", str(swa),
    ]
    for dp in data_paths:
        cmd += ["--data", dp]

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    vol.commit()
    with open(vol_out, "rb") as f:
        return f.read()


T80_MONTHS = [
    ("test80-2024-01-jan-2tb7p.min-v2.v6.binpack.zst", "t80_2024_01_200m.bin"),
    ("test80-2024-02-feb-2tb7p.min-v2.v6.binpack.zst", "t80_2024_02_200m.bin"),
    ("test80-2024-03-mar-2tb7p.min-v2.v6.binpack.zst", "t80_2024_03_200m.bin"),
    ("test80-2024-04-apr-2tb7p.min-v2.v6.binpack.zst", "t80_2024_04_200m.bin"),
]


@app.local_entrypoint()
def main(
    dataset: str = "t80+s1+s2",
    t80_months: int = 4,
    t80_pos: int = 200_000_000,
    s1_shards: str = "1,2,3,4",
    s2_shards: str = "1,2,3,4,5,6,7,8,9,10,11,12",
    kb: int = 16,
    asym: int = 1,
    hl: int = 1024,
    ob: int = 8,
    superbatches: int = 100,
    sb_pos: int = 100_000_000,
    batch: int = 16384,
    lr: float = 0.0012,
    lr_end: float = 0.00001,
    wdl_start: float = 0.35,
    wdl_end: float = 0.35,
    scale_out: int = 160,
    swa: int = 10,
    out: str = "chess/nets/h200_v8_asym16_1024_10b.nnue",
):
    data_paths: list[str] = []
    parts = [p.strip() for p in dataset.split("+") if p.strip()]

    t80_handles = []
    s1_handles = []
    if "t80" in parts:
        for mf, tag in T80_MONTHS[:t80_months]:
            t80_handles.append(fetch_t80_month.spawn(mf, tag, t80_pos))
    if "s1" in parts:
        for s in [int(x.strip()) for x in s1_shards.split(",") if x.strip()]:
            s1_handles.append(fetch_s1_uho.spawn(s, 100_000_000))

    for h in t80_handles:
        data_paths.append(h.get())
    for h in s1_handles:
        data_paths.append(h.get())
    if "s2" in parts:
        for s in [int(x.strip()) for x in s2_shards.split(",") if x.strip()]:
            data_paths.append(f"/vol/data/s2_iter{s}_100m.bin")

    print(f"Training on {len(data_paths)} shards: {data_paths}", flush=True)
    out_name = os.path.basename(out)
    weights = train_screlu_v4_gpu.remote(
        data_paths=data_paths,
        out_name=out_name,
        kb=kb,
        asym=asym,
        hl=hl,
        ob=ob,
        superbatches=superbatches,
        sb_pos=sb_pos,
        batch_size=batch,
        lr=lr,
        lr_end=lr_end,
        wdl_start=wdl_start,
        wdl_end=wdl_end,
        scale_out=scale_out,
        swa=swa,
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "wb") as f:
        f.write(weights)
    print(f"Downloaded trained CHNN v{4 if (kb == 16 and asym) else 1} NNUE ({len(weights) / 1e6:.2f} MB) -> {out}")
