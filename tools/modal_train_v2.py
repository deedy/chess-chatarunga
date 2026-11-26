"""Modal H100/H200 global-VRAM NNUE trainer (CHNN v2 Pairwise-CReLU, Power-2.5 loss, WDL annealing).

Loads up to 1.2B+ unique Bullet S2 positions (38.4 GB) simultaneously into H100/H200 HBM3 VRAM
and trains with zero-overhead coprime modular global shuffling across all shards, Power-2.5 loss
(|pred - target|^2.5), WDL annealing (0.25 -> 0.50), and WSD + Cosine LR decay.
"""

import os
import modal

app = modal.App("chatarunga-v2-h100")
vol = modal.Volume.from_name("chess-nnue-vol", create_if_missing=True)

image = modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11").pip_install("zstandard")

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

__device__ __forceinline__ int king_bucket(int kb, int km) {
    if (kb <= 1) return 0;
    int idx32 = (km >> 3) * 4 + (km & 7);
    if (kb == 4) return d_bucket4[idx32];
    if (kb == 8) return d_bucket8[idx32];
    if (kb == 16) return ((km >> 4) * 4 + (km & 7));
    return idx32 % kb;
}

__device__ __forceinline__ float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// One block per position in the batch; blockDim.x = HALF = HL / 2 (up to 1024).
// Uses zero-VRAM coprime modular permutation across the ENTIRE multi-shard dataset in VRAM!
__global__ void fwd_bwd_pairwise_v2(
    const BulletRec* __restrict__ recs,
    uint64_t total_n,
    uint64_t base_step,
    uint64_t perm_offset,
    uint64_t perm_stride,
    int batch_size,
    int kb,
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
        int kb_s = king_bucket(kb, km_s);

        int mirror_o = (kb > 1) && ((r.opp_ksq & 7) >= 4);
        int km_o = mirror_o ? (r.opp_ksq ^ 7) : r.opp_ksq;
        int kb_o = king_bucket(kb, km_o);

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

    float a_s = fminf(fmaxf(s0, 0.0f), 1.0f);
    float b_s = fminf(fmaxf(s1, 0.0f), 1.0f);
    float a_o = fminf(fmaxf(o0, 0.0f), 1.0f);
    float b_o = fminf(fmaxf(o1, 0.0f), 1.0f);

    int ow_base = bkt * hl;
    float ws = out_w[ow_base + t];
    float wo = out_w[ow_base + half + t];
    float partial = (a_s * b_s) * ws + (a_o * b_o) * wo;

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
            // Report clean MSE for direct comparison, optimize Power-2.5 loss |diff|^2.5!
            atomicAdd(mse_sum, (double)(w * diff * diff));
            float grad_p = 2.5f * diff * sqrtf(fabsf(diff) + 1e-12f);
            s_d_out = w * grad_p * pred * (1.0f - pred);
            if (do_bwd && w > 0.0f) atomicAdd(&g_out_b[bkt], s_d_out);
        }
    }
    __syncthreads();

    if (!do_bwd) return;

    float d_out = s_d_out;
    atomicAdd(&g_out_w[ow_base + t], d_out * (a_s * b_s));
    atomicAdd(&g_out_w[ow_base + half + t], d_out * (a_o * b_o));

    float ds0 = (s0 > 0.0f && s0 < 1.0f) ? (d_out * ws * b_s) : 0.0f;
    float ds1 = (s1 > 0.0f && s1 < 1.0f) ? (d_out * ws * a_s) : 0.0f;
    float do0 = (o0 > 0.0f && o0 < 1.0f) ? (d_out * wo * b_o) : 0.0f;
    float do1 = (o1 > 0.0f && o1 < 1.0f) ? (d_out * wo * a_o) : 0.0f;

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

static uint64_t rng_state = 20261126ULL;
static inline uint64_t xorshift64() {
    uint64_t x = rng_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    rng_state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

void save_chnn_v2(
    const char* path, int kb, int hl, int ob, int factorized, int scale,
    const std::vector<float>& ft_w, const std::vector<float>& ft_b,
    const std::vector<float>& out_w, const std::vector<float>& out_b
) {
    const int QA = 255, QB = 64;
    FILE* f = fopen(path, "wb");
    if (!f) { perror("fopen out"); exit(1); }
    uint32_t ver = 2, ukb = kb, uhl = hl, uob = ob;
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
    int kb = 16, hl = 1024, ob = 8, superbatches = 60, batch_size = 16384, scale_out = 160;
    uint64_t sb_pos = 100000000ULL;
    float lr_start = 0.0012f, lr_end = 0.00001f, wdl_start = 0.25f, wdl_end = 0.50f, scale_f = 400.0f;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--data") && i+1 < argc) data_paths.push_back(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i+1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--kb") && i+1 < argc) kb = atoi(argv[++i]);
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
    }
    if (data_paths.empty() || !out_path) { fprintf(stderr, "Need --data and --out\n"); return 1; }

    // Count total positions across all shards
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

    // Reserve last 250,000 positions of shard 0 for apples-to-apples validation comparison!
    const uint64_t val_n = 250000;
    uint64_t train_n = total_raw - val_n;
    printf("Loading ALL %llu training + %llu val positions (%.2f GB) into HBM3 VRAM...\n",
           (unsigned long long)train_n, (unsigned long long)val_n, (double)(total_raw * 32) / 1e9);
    fflush(stdout);

    auto t_load0 = std::chrono::steady_clock::now();
    BulletRec* d_recs = nullptr;
    BulletRec* d_val_recs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_recs, train_n * 32));
    CUDA_CHECK(cudaMalloc(&d_val_recs, val_n * 32));

    const uint64_t chunk_n = 8000000;
    BulletRec* h_buf = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_buf, chunk_n * 32));
    uint64_t dst_off = 0;
    for (size_t si = 0; si < data_paths.size(); si++) {
        FILE* f = fopen(data_paths[si].c_str(), "rb");
        uint64_t n = shard_counts[si];
        uint64_t take_train = (si == 0) ? (n - val_n) : n;
        uint64_t rem = take_train;
        while (rem > 0) {
            uint64_t want = std::min(rem, chunk_n);
            size_t got = fread(h_buf, 32, want, f);
            CUDA_CHECK(cudaMemcpy(d_recs + dst_off, h_buf, got * 32, cudaMemcpyHostToDevice));
            dst_off += got;
            rem -= got;
        }
        if (si == 0) {
            size_t got = fread(h_buf, 32, val_n, f);
            CUDA_CHECK(cudaMemcpy(d_val_recs, h_buf, got * 32, cudaMemcpyHostToDevice));
        }
        fclose(f);
    }
    CUDA_CHECK(cudaFreeHost(h_buf));
    double t_load = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load0).count();
    printf("Loaded %.2fB positions into VRAM in %.1fs (%.2f GB/s)\n",
           (double)train_n / 1e9, t_load, (double)(total_raw * 32) / t_load / 1e9);
    fflush(stdout);

    int factorized = (kb > 1) ? 1 : 0;
    size_t n_feats = (size_t)kb * 768 + (factorized ? 768 : 0);
    size_t n_ft_w = n_feats * hl;
    size_t n_out_w = (size_t)ob * hl;

    std::vector<float> h_ft_w(n_ft_w, 0.0f), h_ft_b(hl, 0.0f), h_out_w(n_out_w, 0.0f), h_out_b(ob, 0.0f);
    float ft_sc = 0.04f;
    float out_sc = 1.0f / sqrtf((float)hl);
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
    int best_sb = 0;

    printf("Chatarunga v2 H100 Trainer: (768*%d -> %d)x2 -> %d | %d SBs x %lluM = %.1fB pos | Power-2.5 | WDL %.2f->%.2f | LR %.5f->%.5f\n",
           kb, hl, ob, superbatches, (unsigned long long)(sb_pos / 1000000ULL),
           (double)(superbatches * sb_pos) / 1e9, wdl_start, wdl_end, lr_start, lr_end);
    fflush(stdout);

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
            if (prog < 0.03f) lr = lr_start * (prog / 0.03f);
            else if (prog < 0.65f) lr = lr_start;
            else {
                float dprog = (prog - 0.65f) / 0.35f;
                lr = lr_end + (lr_start - lr_end) * 0.5f * (1.0f + cosf(3.14159265f * dprog));
            }

            CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(double)));
            fwd_bwd_pairwise_v2<<<batch_size, hl / 2>>>(
                d_recs, train_n, b * batch_size, offset, stride, batch_size,
                kb, hl, ob, factorized, wdl_w, scale_f,
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

        // Validate at fixed wdl=0.35 so val_mse is directly comparable across superbatches and vs v1!
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(double)));
        for (uint64_t off = 0; off < val_n; off += batch_size) {
            int bs = (int)std::min<uint64_t>(batch_size, val_n - off);
            fwd_bwd_pairwise_v2<<<bs, hl / 2>>>(
                d_val_recs, val_n, off, 0, 1, bs,
                kb, hl, ob, factorized, 0.35f, scale_f,
                d_ft_w, d_ft_b, d_out_w, d_out_b,
                g_ft_w, g_ft_b, g_out_w, g_out_b, d_loss, 0
            );
        }
        double vloss_sum = 0.0;
        CUDA_CHECK(cudaMemcpy(&vloss_sum, d_loss, sizeof(double), cudaMemcpyDeviceToHost));
        double val_mse = vloss_sum / (double)val_n;
        double tr_mse = sb_mse / (double)(batches_per_sb * batch_size);

        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const char* mark = "";
        if (val_mse < best_val) {
            best_val = val_mse; best_sb = sb; mark = " *";
            CUDA_CHECK(cudaMemcpy(best_ft_w.data(), d_ft_w, n_ft_w * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(best_ft_b.data(), d_ft_b, hl * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(best_out_w.data(), d_out_w, n_out_w * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(best_out_b.data(), d_out_b, ob * 4, cudaMemcpyDeviceToHost));
        }
        printf("SB %2d/%d | train~%.6f val(0.35)=%.6f%s | wdl=%.2f | %.2fs (%.2fM pos/s)\n",
               sb, superbatches, tr_mse, val_mse, mark, wdl_w, dt, (double)(batches_per_sb * batch_size) / dt / 1e6);
        fflush(stdout);
    }

    // Save final converged weights (also best-val if better; during WDL annealing final weights are often best in play!)
    std::vector<float> fin_ft_w(n_ft_w), fin_ft_b(hl), fin_out_w(n_out_w), fin_out_b(ob);
    CUDA_CHECK(cudaMemcpy(fin_ft_w.data(), d_ft_w, n_ft_w * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(fin_ft_b.data(), d_ft_b, hl * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(fin_out_w.data(), d_out_w, n_out_w * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(fin_out_b.data(), d_out_b, ob * 4, cudaMemcpyDeviceToHost));
    save_chnn_v2(out_path, kb, hl, ob, factorized, scale_out, fin_ft_w, fin_ft_b, fin_out_w, fin_out_b);

    double tot_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    printf("Saved final CHNN v2 Pairwise NNUE (best val %.6f @ SB %d) -> %s in %.1fs\n",
           best_val, best_sb, out_path, tot_s);
    return 0;
}
"""


@app.function(image=image, volumes={"/vol": vol}, cpu=4, memory=8192, timeout=1800)
def fetch_bullet_shard(shard: int = 1, positions: int = 100_000_000) -> str:
    import time
    import urllib.request
    import zstandard

    os.makedirs("/vol/data", exist_ok=True)
    tag = f"s2_iter{shard}_{positions // 1_000_000}m.bin"
    out_path = f"/vol/data/{tag}"
    target_bytes = positions * 32

    if os.path.exists(out_path) and os.path.getsize(out_path) >= target_bytes:
        print(f"Using cached {out_path} ({os.path.getsize(out_path) / 1e9:.2f} GB)")
        return out_path

    url = (
        f"https://huggingface.co/datasets/linrock/bullet-training-data/resolve/main/"
        f"S2/test77nov-unfilt-test79-maraprmay-v6-dd.skip-see-ge0.wdl-pdist.iter-{shard}.bullet.bin.zst"
    )
    print(f"Streaming {positions:,} positions from {url} -> {out_path}")
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "chatarunga-modal"})
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
    print(f"Done streaming {written // 32:,} positions to {out_path} in {time.time() - t0:.1f}s")
    return out_path


@app.function(image=image, volumes={"/vol": vol}, gpu=["H200", "H100", "A100-80GB"], cpu=8, memory=65536, timeout=3600)
def train_v2_gpu(
    data_paths: list[str],
    out_name: str,
    kb: int = 16,
    hl: int = 1024,
    ob: int = 8,
    superbatches: int = 60,
    sb_pos: int = 100_000_000,
    batch_size: int = 16384,
    lr: float = 0.0012,
    lr_end: float = 0.00001,
    wdl_start: float = 0.25,
    wdl_end: float = 0.50,
    scale_out: int = 160,
) -> bytes:
    import subprocess

    os.makedirs("/vol/nets", exist_ok=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False)
    cu_path = "/tmp/train_v2.cu"
    bin_path = "/tmp/train_v2"
    with open(cu_path, "w") as f:
        f.write(CUDA_SRC)

    print("Compiling fused CUDA trainer with nvcc -O3 -arch=native...", flush=True)
    subprocess.run(["nvcc", "-O3", "-arch=native", "-use_fast_math", cu_path, "-o", bin_path], check=True)

    vol_out = f"/vol/nets/{out_name}"
    cmd = [
        bin_path,
        "--out", vol_out,
        "--kb", str(kb),
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
    ]
    for dp in data_paths:
        cmd += ["--data", dp]

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    vol.commit()
    with open(vol_out, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(
    shards: str = "1,2,3,4,5,6,7,8,9,10,11,12",
    positions: int = 100_000_000,
    kb: int = 16,
    hl: int = 1024,
    ob: int = 8,
    superbatches: int = 60,
    sb_pos: int = 100_000_000,
    batch: int = 16384,
    lr: float = 0.0012,
    lr_end: float = 0.00001,
    wdl_start: float = 0.25,
    wdl_end: float = 0.50,
    scale_out: int = 160,
    out: str = "chess/nets/h100_v7_1024_6b.nnue",
):
    shard_list = [int(s.strip()) for s in shards.split(",") if s.strip()]
    data_paths = list(fetch_bullet_shard.map(shard_list, [positions] * len(shard_list)))
    out_name = os.path.basename(out)
    weights = train_v2_gpu.remote(
        data_paths=data_paths,
        out_name=out_name,
        kb=kb,
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
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "wb") as f:
        f.write(weights)
    print(f"Downloaded trained NNUE ({len(weights) / 1e6:.2f} MB) -> {out}")
