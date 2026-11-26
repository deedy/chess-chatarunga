"""Modal H100 multi-shard NNUE v3 trainer (CHNN v3: 16 King Buckets + Threat Inputs + 2-layer head L1->16->32->1).

Streams up to 10 shards (1 BILLION positions) of `linrock/bullet-training-data` S2 onto the
persistent Modal Volume `/vol`, then trains a factorized CHNN v3 network end-to-end on NVIDIA H100
using a single fused CUDA C++ forward+backward kernel with WSD (Warmup-Stable-Decay) LR schedule
and continuous AdamW state across all shards.

Usage:
  # 1. Pre-cache shards 1..10 (1B positions) onto Modal Volume in parallel:
  modal run chess/tools/modal_train_v3.py --fetch-only --shards 1,2,3,4,5,6,7,8,9,10

  # 2. Train CHNN v3 incrementally across 1B positions on H100:
  modal run chess/tools/modal_train_v3.py --shards 1,2,3,4,5,6,7,8,9,10 --epochs-per-shard 2 --out chess/nets/h100_v4.nnue
"""

import os
import modal

app = modal.App("chess-nnue-v3-h100")
vol = modal.Volume.from_name("chess-nnue-vol", create_if_missing=True)

image = modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11").pip_install("zstandard")

CUDA_V3_SRC = r"""
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

constexpr int L1_OUT = 16;
constexpr int L2_IN  = 32; // 16 CReLU + 16 SCReLU
constexpr int L2_OUT = 32;

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

constexpr uint64_t NOT_FILE_A = 0xFEFEFEFEFEFEFEFEULL;
constexpr uint64_t NOT_FILE_H = 0x7F7F7F7F7F7F7F7FULL;
constexpr uint64_t NOT_FILE_AB = 0xFCFCFCFCFCFCFCFCULL;
constexpr uint64_t NOT_FILE_GH = 0x3F3F3F3F3F3F3F3FULL;

__device__ __forceinline__ uint64_t knight_att_bb(int sq) {
    uint64_t b = 1ULL << sq;
    uint64_t l1 = (b >> 1) & NOT_FILE_H;
    uint64_t l2 = (b >> 2) & NOT_FILE_GH;
    uint64_t r1 = (b << 1) & NOT_FILE_A;
    uint64_t r2 = (b << 2) & NOT_FILE_AB;
    uint64_t h1 = l1 | r1;
    uint64_t h2 = l2 | r2;
    return (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8);
}

__device__ __forceinline__ uint64_t king_att_bb(int sq) {
    uint64_t b = 1ULL << sq;
    uint64_t lr = ((b & NOT_FILE_A) >> 1) | ((b & NOT_FILE_H) << 1);
    uint64_t c = b | lr;
    return lr | (c << 8) | (c >> 8);
}

__device__ __forceinline__ uint64_t bishop_att_bb(int sq, uint64_t occ) {
    uint64_t att = 0;
    int r0 = sq >> 3, f0 = sq & 7;
    for (int r = r0 + 1, f = f0 + 1; r < 8 && f < 8; r++, f++) {
        uint64_t b = 1ULL << (r * 8 + f); att |= b; if (occ & b) break;
    }
    for (int r = r0 + 1, f = f0 - 1; r < 8 && f >= 0; r++, f--) {
        uint64_t b = 1ULL << (r * 8 + f); att |= b; if (occ & b) break;
    }
    for (int r = r0 - 1, f = f0 + 1; r >= 0 && f < 8; r--, f++) {
        uint64_t b = 1ULL << (r * 8 + f); att |= b; if (occ & b) break;
    }
    for (int r = r0 - 1, f = f0 - 1; r >= 0 && f >= 0; r--, f--) {
        uint64_t b = 1ULL << (r * 8 + f); att |= b; if (occ & b) break;
    }
    return att;
}

__device__ __forceinline__ uint64_t rook_att_bb(int sq, uint64_t occ) {
    uint64_t att = 0;
    int r0 = sq >> 3, f0 = sq & 7;
    for (int r = r0 + 1; r < 8; r++) {
        uint64_t b = 1ULL << (r * 8 + f0); att |= b; if (occ & b) break;
    }
    for (int r = r0 - 1; r >= 0; r--) {
        uint64_t b = 1ULL << (r * 8 + f0); att |= b; if (occ & b) break;
    }
    for (int f = f0 + 1; f < 8; f++) {
        uint64_t b = 1ULL << (r0 * 8 + f); att |= b; if (occ & b) break;
    }
    for (int f = f0 - 1; f >= 0; f--) {
        uint64_t b = 1ULL << (r0 * 8 + f); att |= b; if (occ & b) break;
    }
    return att;
}

// One block per position; blockDim.x = HALF = HL / 2 (512 for HL = 1024; must be multiple of 32 and >= 512).
__global__ void fwd_bwd_v3_kernel(
    const BulletRec* __restrict__ recs,
    const uint32_t* __restrict__ indices,
    int batch_size,
    int kb,
    int hl,
    int ob,
    int factorized,
    int use_threats,
    float wdl_w,
    float scale_f,
    const float* __restrict__ ft_w,
    const float* __restrict__ ft_b,
    const float* __restrict__ l1_w,   // [ob][16][hl]
    const float* __restrict__ l1_b,   // [ob][16]
    const float* __restrict__ l2_w,   // [ob][32][32]
    const float* __restrict__ l2_b,   // [ob][32]
    const float* __restrict__ out_w,  // [ob][32]
    const float* __restrict__ out_b,  // [ob]
    float* __restrict__ g_ft_w,
    float* __restrict__ g_ft_b,
    float* __restrict__ g_l1_w,
    float* __restrict__ g_l1_b,
    float* __restrict__ g_l2_w,
    float* __restrict__ g_l2_b,
    float* __restrict__ g_out_w,
    float* __restrict__ g_out_b,
    double* __restrict__ loss_sum,
    int do_bwd
) {
    int bid = blockIdx.x;
    if (bid >= batch_size) return;
    int t = threadIdx.x;
    int half = hl >> 1;
    int wid = t >> 5;
    int lane = t & 31;

    __shared__ int s_cnt;
    __shared__ int s_bkt;
    __shared__ float s_target;
    __shared__ float s_weight;
    __shared__ uint16_t s_f_stm[128];
    __shared__ uint16_t s_f_opp[128];
    __shared__ float s_ft[1024];       // pairwise activations [hl]
    __shared__ float s_l1_pre[L1_OUT];
    __shared__ float s_l1_act[L2_IN];
    __shared__ float s_l2_pre[L2_OUT];
    __shared__ float s_l2_act[L2_OUT];
    __shared__ float s_d_out;
    __shared__ float s_d_l2_pre[L2_OUT];
    __shared__ float s_d_l1_act[L2_IN];
    __shared__ float s_d_l1_pre[L1_OUT];

    if (t == 0) {
        BulletRec r = recs[indices[bid]];
        uint64_t occ_all = r.occ;
        int cnt = __popcll(occ_all);
        int div = (ob > 1) ? ((32 + ob - 1) / ob) : 32;
        s_bkt = (ob > 1) ? min(max(cnt - 2, 0) / div, ob - 1) : 0;

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

        int threat_off = kb * 768;
        int factor_off = (kb + (use_threats ? 3 : 0)) * 768;

        // First pass: unpack pieces & compute attack bitboards for threat inputs
        uint8_t p_sq[32], p_type[32], p_col[32];
        uint64_t att_pawn[2] = {0, 0};
        uint64_t att_minor[2] = {0, 0};
        uint64_t att_rook[2] = {0, 0};
        uint64_t att_all[2] = {0, 0};

        uint64_t occ = occ_all;
        int np = 0;
        int nfeat = 0;
        while (occ && np < 32) {
            int sq = __ffsll(occ) - 1;
            occ &= occ - 1;
            uint8_t byte = r.pcs[np >> 1];
            uint8_t nib = (np & 1) == 0 ? (byte & 0xF) : (byte >> 4);
            int piece = nib & 7;
            int col = nib >> 3; // 0 = STM (pawns go UP), 1 = OPP (pawns go DOWN)
            p_sq[np] = (uint8_t)sq;
            p_type[np] = (uint8_t)piece;
            p_col[np] = (uint8_t)col;

            int sq_s = mirror_s ? (sq ^ 7) : sq;
            int fs_base = col * 384 + piece * 64 + sq_s;
            s_f_stm[nfeat] = (uint16_t)(kb_s * 768 + fs_base);

            int sq_o = sq ^ 56;
            if (mirror_o) sq_o ^= 7;
            int fo_base = (col ^ 1) * 384 + piece * 64 + sq_o;
            s_f_opp[nfeat] = (uint16_t)(kb_o * 768 + fo_base);
            nfeat++;

            if (factorized) {
                s_f_stm[nfeat] = (uint16_t)(factor_off + fs_base);
                s_f_opp[nfeat] = (uint16_t)(factor_off + fo_base);
                nfeat++;
            }

            if (use_threats) {
                uint64_t b = 1ULL << sq;
                if (piece == 0) {
                    uint64_t pa = (col == 0)
                        ? (((b & NOT_FILE_A) << 7) | ((b & NOT_FILE_H) << 9))
                        : (((b & NOT_FILE_A) >> 9) | ((b & NOT_FILE_H) >> 7));
                    att_pawn[col] |= pa;
                } else if (piece == 1) {
                    att_minor[col] |= knight_att_bb(sq);
                } else if (piece == 2) {
                    att_minor[col] |= bishop_att_bb(sq, occ_all);
                } else if (piece == 3) {
                    att_rook[col] |= rook_att_bb(sq, occ_all);
                } else if (piece == 4) {
                    att_all[col] |= bishop_att_bb(sq, occ_all) | rook_att_bb(sq, occ_all);
                } else if (piece == 5) {
                    att_all[col] |= king_att_bb(sq);
                }
            }
            np++;
        }

        if (use_threats) {
            for (int c = 0; c < 2; c++) {
                att_minor[c] |= att_pawn[c];
                att_rook[c]  |= att_minor[c];
                att_all[c]   |= att_rook[c];
            }
            for (int i = 0; i < np && nfeat + 2 <= 128; i++) {
                int sq = p_sq[i];
                int piece = p_type[i];
                int col = p_col[i];
                uint64_t b = 1ULL << sq;
                int enemy = col ^ 1;
                if (att_all[enemy] & b) {
                    int defended = (att_all[col] & b) ? 1 : 0;
                    int sq_s = mirror_s ? (sq ^ 7) : sq;
                    int fs_base = col * 384 + piece * 64 + sq_s;
                    int sq_o = sq ^ 56;
                    if (mirror_o) sq_o ^= 7;
                    int fo_base = enemy * 384 + piece * 64 + sq_o;

                    int tbase = threat_off + defended * 768;
                    s_f_stm[nfeat] = (uint16_t)(tbase + fs_base);
                    s_f_opp[nfeat] = (uint16_t)(tbase + fo_base);
                    nfeat++;

                    bool lesser = false;
                    if (piece == 1 || piece == 2) lesser = (att_pawn[enemy] & b) != 0;
                    else if (piece == 3) lesser = (att_minor[enemy] & b) != 0;
                    else if (piece == 4) lesser = (att_rook[enemy] & b) != 0;
                    if (lesser && nfeat < 128) {
                        int lbase = threat_off + 2 * 768;
                        s_f_stm[nfeat] = (uint16_t)(lbase + fs_base);
                        s_f_opp[nfeat] = (uint16_t)(lbase + fo_base);
                        nfeat++;
                    }
                }
            }
        }
        s_cnt = nfeat;
    }
    __syncthreads();

    int nf = s_cnt;
    int bkt = s_bkt;

    // 1. Feature Transformer Forward
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
    s_ft[t]        = a_s * b_s;
    s_ft[half + t] = a_o * b_o;
    __syncthreads();

    // 2. Layer 1 Forward: nwarps warps share L1_OUT=16 neurons (works for hl=512, 768, 1024, 1536, ...)
    int nwarps = half >> 5;
    const float* l1_w_b = l1_w + (size_t)bkt * L1_OUT * hl;
    for (int j = wid; j < L1_OUT; j += nwarps) {
        const float* w_row = l1_w_b + (size_t)j * hl;
        float acc = 0.0f;
        for (int i = lane; i < hl; i += 32) {
            acc += s_ft[i] * w_row[i];
        }
        for (int mask = 16; mask > 0; mask >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, mask);
        }
        if (lane == 0) {
            float pre = acc + l1_b[bkt * L1_OUT + j];
            s_l1_pre[j] = pre;
            float c = fminf(fmaxf(pre, 0.0f), 1.0f);
            s_l1_act[j] = c;
            s_l1_act[L1_OUT + j] = c * c;
        }
    }
    __syncthreads();

    // 3. Layer 2 Forward (32->32) + Output (32->1 + skip l1_pre[0]) in Warp 0
    const float* l2_w_b = l2_w + (size_t)bkt * L2_OUT * L2_IN;
    const float* out_w_b = out_w + (size_t)bkt * L2_OUT;
    if (wid == 0) {
        const float* w_row = l2_w_b + (size_t)lane * L2_IN;
        float acc = l2_b[bkt * L2_OUT + lane];
        #pragma unroll
        for (int j = 0; j < L2_IN; j++) {
            acc += s_l1_act[j] * w_row[j];
        }
        s_l2_pre[lane] = acc;
        float act = fminf(fmaxf(acc, 0.0f), 1.0f);
        s_l2_act[lane] = act;

        float out_p = act * out_w_b[lane];
        for (int mask = 16; mask > 0; mask >>= 1) {
            out_p += __shfl_down_sync(0xffffffff, out_p, mask);
        }
        if (lane == 0) {
            float out = out_p + out_b[bkt] + s_l1_pre[0];
            float pred = sigmoidf(out);
            float diff = pred - s_target;
            float w = s_weight;
            atomicAdd(loss_sum, (double)(w * diff * diff));
            s_d_out = w * 2.0f * diff * pred * (1.0f - pred);
            if (do_bwd && w > 0.0f) {
                atomicAdd(&g_out_b[bkt], s_d_out);
            }
        }
    }
    __syncthreads();

    if (!do_bwd || s_weight == 0.0f) return;

    // 4. Output & Layer 2 Backward
    float d_out = s_d_out;
    if (wid == 0) {
        atomicAdd(&g_out_w[bkt * L2_OUT + lane], d_out * s_l2_act[lane]);
        float d_l2_act = d_out * out_w_b[lane];
        float pre = s_l2_pre[lane];
        float d_l2_pre = (pre > 0.0f && pre < 1.0f) ? d_l2_act : 0.0f;
        s_d_l2_pre[lane] = d_l2_pre;
        if (d_l2_pre != 0.0f) atomicAdd(&g_l2_b[bkt * L2_OUT + lane], d_l2_pre);
    }
    __syncthreads();

    // Update g_l2_w across all nwarps warps (works for any hl >= 256)
    {
        float* gl2_b = g_l2_w + (size_t)bkt * L2_OUT * L2_IN;
        for (int k = wid; k < L2_OUT; k += nwarps) {
            float d = s_d_l2_pre[k];
            if (d != 0.0f) {
                atomicAdd(&gl2_b[k * L2_IN + lane], d * s_l1_act[lane]);
            }
        }
    }
    // Compute s_d_l1_act[j] (j = lane in 0..32) in Warp 0
    if (wid == 0) {
        float d_acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < L2_OUT; k++) {
            d_acc += s_d_l2_pre[k] * l2_w_b[k * L2_IN + lane];
        }
        s_d_l1_act[lane] = d_acc;
    }
    __syncthreads();

    // Compute s_d_l1_pre[j] (j = t in 0..16) including skip connection for j == 0
    if (t < L1_OUT) {
        float v = s_l1_pre[t];
        float dl = (v > 0.0f && v < 1.0f) ? (s_d_l1_act[t] + 2.0f * v * s_d_l1_act[L1_OUT + t]) : 0.0f;
        if (t == 0) dl += d_out; // direct skip connection!
        s_d_l1_pre[t] = dl;
        if (dl != 0.0f) atomicAdd(&g_l1_b[bkt * L1_OUT + t], dl);
    }
    __syncthreads();

    // 5. Layer 1 Backward (512 threads handle t and half+t across all 16 neurons)
    float* gl1_b = g_l1_w + (size_t)bkt * L1_OUT * hl;
    float d_ft_s = 0.0f;
    float d_ft_o = 0.0f;
    float ft_s = s_ft[t];
    float ft_o = s_ft[half + t];
    #pragma unroll
    for (int j = 0; j < L1_OUT; j++) {
        float dl = s_d_l1_pre[j];
        if (dl != 0.0f) {
            size_t row = (size_t)j * hl;
            d_ft_s += dl * l1_w_b[row + t];
            d_ft_o += dl * l1_w_b[row + half + t];
            if (ft_s != 0.0f) atomicAdd(&gl1_b[row + t], dl * ft_s);
            if (ft_o != 0.0f) atomicAdd(&gl1_b[row + half + t], dl * ft_o);
        }
    }

    // 6. Feature Transformer Backward
    float ds0 = (s0 > 0.0f && s0 < 1.0f) ? (d_ft_s * b_s) : 0.0f;
    float ds1 = (s1 > 0.0f && s1 < 1.0f) ? (d_ft_s * a_s) : 0.0f;
    float do0 = (o0 > 0.0f && o0 < 1.0f) ? (d_ft_o * b_o) : 0.0f;
    float do1 = (o1 > 0.0f && o1 < 1.0f) ? (d_ft_o * a_o) : 0.0f;

    if (ds0 + do0 != 0.0f) atomicAdd(&g_ft_b[t], ds0 + do0);
    if (ds1 + do1 != 0.0f) atomicAdd(&g_ft_b[t + half], ds1 + do1);

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

void save_chnn_v3(
    const char* path, int kb, int hl, int ob, int factorized, int use_threats, int scale,
    const std::vector<float>& ft_w, const std::vector<float>& ft_b,
    const std::vector<float>& l1_w, const std::vector<float>& l1_b,
    const std::vector<float>& l2_w, const std::vector<float>& l2_b,
    const std::vector<float>& out_w, const std::vector<float>& out_b
) {
    const int QA = 255, QB = 64, QBIAS = 127 * 64;
    FILE* f = fopen(path, "wb");
    if (!f) { perror("fopen out"); exit(1); }
    uint32_t ver = 3, ukb = kb, uhl = hl, uob = ob;
    uint32_t flags = (use_threats ? 1u : 0u) | ((uint32_t)L1_OUT << 8) | ((uint32_t)L2_OUT << 16);
    fwrite("CHNN", 1, 4, f);
    fwrite(&ver, 4, 1, f);
    fwrite(&ukb, 4, 1, f);
    fwrite(&uhl, 4, 1, f);
    fwrite(&uob, 4, 1, f);
    fwrite(&QA, 4, 1, f);
    fwrite(&QB, 4, 1, f);
    fwrite(&scale, 4, 1, f);
    fwrite(&flags, 4, 1, f);

    // Fold factorized base features into the kb king buckets; append threat features unchanged
    size_t stored_rows = (size_t)(kb + (use_threats ? 3 : 0)) * 768;
    size_t factor_start = stored_rows * hl;
    std::vector<int16_t> q_ft_w(stored_rows * hl);
    for (int b = 0; b < kb; b++) {
        for (int feat = 0; feat < 768; feat++) {
            size_t row = ((size_t)b * 768 + feat) * hl;
            size_t brow = factor_start + (size_t)feat * hl;
            for (int h = 0; h < hl; h++) {
                float val = ft_w[row + h] + (factorized ? ft_w[brow + h] : 0.0f);
                int q = (int)llroundf(val * QA);
                q_ft_w[row + h] = (int16_t)std::max(-32768, std::min(32767, q));
            }
        }
    }
    if (use_threats) {
        size_t t_start = (size_t)kb * 768 * hl;
        size_t t_len = (size_t)3 * 768 * hl;
        for (size_t i = 0; i < t_len; i++) {
            int q = (int)llroundf(ft_w[t_start + i] * QA);
            q_ft_w[t_start + i] = (int16_t)std::max(-32768, std::min(32767, q));
        }
    }
    fwrite(q_ft_w.data(), 2, q_ft_w.size(), f);

    std::vector<int16_t> q_ft_b(hl);
    for (int h = 0; h < hl; h++) {
        int q = (int)llroundf(ft_b[h] * QA);
        q_ft_b[h] = (int16_t)std::max(-32768, std::min(32767, q));
    }
    fwrite(q_ft_b.data(), 2, q_ft_b.size(), f);

    auto write_i8 = [&](const std::vector<float>& src) {
        std::vector<int8_t> q(src.size());
        for (size_t i = 0; i < src.size(); i++) {
            int v = (int)llroundf(std::max(-1.98f, std::min(1.98f, src[i])) * QB);
            q[i] = (int8_t)std::max(-127, std::min(127, v));
        }
        fwrite(q.data(), 1, q.size(), f);
    };
    auto write_i32_bias = [&](const std::vector<float>& src) {
        std::vector<int32_t> q(src.size());
        for (size_t i = 0; i < src.size(); i++) q[i] = (int32_t)llroundf(src[i] * QBIAS);
        fwrite(q.data(), 4, q.size(), f);
    };

    write_i8(l1_w);
    write_i32_bias(l1_b);
    write_i8(l2_w);
    write_i32_bias(l2_b);
    write_i8(out_w);
    write_i32_bias(out_b);
    fclose(f);
}

int main(int argc, char** argv) {
    std::vector<std::string> data_paths;
    const char* out_path = nullptr;
    int kb = 16, hl = 1024, ob = 8, epochs_per_shard = 2, batch_size = 16384, scale_out = 160;
    int use_threats = 1;
    float lr_max = 0.001f, lr_min = 0.00002f, wdl_w = 0.35f, scale_f = 400.0f, wd = 0.0f;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--data") && i+1 < argc) data_paths.push_back(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i+1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--kb") && i+1 < argc) kb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--hl") && i+1 < argc) hl = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ob") && i+1 < argc) ob = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--epochs-per-shard") && i+1 < argc) epochs_per_shard = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i+1 < argc) batch_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr") && i+1 < argc) lr_max = atof(argv[++i]);
        else if (!strcmp(argv[i], "--lr-end") && i+1 < argc) lr_min = atof(argv[++i]);
        else if (!strcmp(argv[i], "--wdl") && i+1 < argc) wdl_w = atof(argv[++i]);
        else if (!strcmp(argv[i], "--scale-out") && i+1 < argc) scale_out = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threats") && i+1 < argc) use_threats = atoi(argv[++i]);
    }
    if (data_paths.empty() || !out_path) { fprintf(stderr, "Need --data and --out\n"); return 1; }

    int factorized = (kb > 1) ? 1 : 0;
    size_t n_feats = (size_t)(kb + (use_threats ? 3 : 0) + (factorized ? 1 : 0)) * 768;
    size_t n_ft_w = n_feats * hl;
    size_t n_l1_w = (size_t)ob * L1_OUT * hl;
    size_t n_l1_b = (size_t)ob * L1_OUT;
    size_t n_l2_w = (size_t)ob * L2_OUT * L2_IN;
    size_t n_l2_b = (size_t)ob * L2_OUT;
    size_t n_out_w = (size_t)ob * L2_OUT;
    size_t n_out_b = (size_t)ob;

    std::vector<float> h_ft_w(n_ft_w, 0.0f), h_ft_b(hl, 0.0f);
    std::vector<float> h_l1_w(n_l1_w, 0.0f), h_l1_b(n_l1_b, 0.0f);
    std::vector<float> h_l2_w(n_l2_w, 0.0f), h_l2_b(n_l2_b, 0.0f);
    std::vector<float> h_out_w(n_out_w, 0.0f), h_out_b(n_out_b, 0.0f);

    auto randu = []() { return ((float)(xorshift64() & 0xFFFFFF) / 8388608.0f - 1.0f); };
    // Initialize factorized base + threat features with small uniform weights
    size_t init_start = factorized ? (size_t)kb * 768 * hl : 0;
    for (size_t i = init_start; i < n_ft_w; i++) h_ft_w[i] = randu() * 0.04f;
    float sc1 = 1.0f / sqrtf((float)hl);
    for (size_t i = 0; i < n_l1_w; i++) h_l1_w[i] = randu() * sc1;
    float sc2 = 1.0f / sqrtf((float)L2_IN);
    for (size_t i = 0; i < n_l2_w; i++) h_l2_w[i] = randu() * sc2;
    float sc3 = 1.0f / sqrtf((float)L2_OUT);
    for (size_t i = 0; i < n_out_w; i++) h_out_w[i] = randu() * sc3;

    auto alloc_zero = [](float** p, size_t n) { CUDA_CHECK(cudaMalloc(p, n * sizeof(float))); CUDA_CHECK(cudaMemset(*p, 0, n * sizeof(float))); };
    float *d_ft_w, *g_ft_w, *m_ft_w, *v_ft_w;
    float *d_ft_b, *g_ft_b, *m_ft_b, *v_ft_b;
    float *d_l1_w, *g_l1_w, *m_l1_w, *v_l1_w;
    float *d_l1_b, *g_l1_b, *m_l1_b, *v_l1_b;
    float *d_l2_w, *g_l2_w, *m_l2_w, *v_l2_w;
    float *d_l2_b, *g_l2_b, *m_l2_b, *v_l2_b;
    float *d_out_w, *g_out_w, *m_out_w, *v_out_w;
    float *d_out_b, *g_out_b, *m_out_b, *v_out_b;
    double *d_loss;

    alloc_zero(&d_ft_w, n_ft_w); alloc_zero(&g_ft_w, n_ft_w); alloc_zero(&m_ft_w, n_ft_w); alloc_zero(&v_ft_w, n_ft_w);
    alloc_zero(&d_ft_b, hl);     alloc_zero(&g_ft_b, hl);     alloc_zero(&m_ft_b, hl);     alloc_zero(&v_ft_b, hl);
    alloc_zero(&d_l1_w, n_l1_w); alloc_zero(&g_l1_w, n_l1_w); alloc_zero(&m_l1_w, n_l1_w); alloc_zero(&v_l1_w, n_l1_w);
    alloc_zero(&d_l1_b, n_l1_b); alloc_zero(&g_l1_b, n_l1_b); alloc_zero(&m_l1_b, n_l1_b); alloc_zero(&v_l1_b, n_l1_b);
    alloc_zero(&d_l2_w, n_l2_w); alloc_zero(&g_l2_w, n_l2_w); alloc_zero(&m_l2_w, n_l2_w); alloc_zero(&v_l2_w, n_l2_w);
    alloc_zero(&d_l2_b, n_l2_b); alloc_zero(&g_l2_b, n_l2_b); alloc_zero(&m_l2_b, n_l2_b); alloc_zero(&v_l2_b, n_l2_b);
    alloc_zero(&d_out_w, n_out_w); alloc_zero(&g_out_w, n_out_w); alloc_zero(&m_out_w, n_out_w); alloc_zero(&v_out_w, n_out_w);
    alloc_zero(&d_out_b, n_out_b); alloc_zero(&g_out_b, n_out_b); alloc_zero(&m_out_b, n_out_b); alloc_zero(&v_out_b, n_out_b);
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_ft_w, h_ft_w.data(), n_ft_w * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ft_b, h_ft_b.data(), hl * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_l1_w, h_l1_w.data(), n_l1_w * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_l1_b, h_l1_b.data(), n_l1_b * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_l2_w, h_l2_w.data(), n_l2_w * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_l2_b, h_l2_b.data(), n_l2_b * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_w, h_out_w.data(), n_out_w * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_b, h_out_b.data(), n_out_b * 4, cudaMemcpyHostToDevice));

    // Hold a fixed 250k validation set from the tail of shard 0 so val_loss is apples-to-apples across all shards
    BulletRec* d_val_recs = nullptr;
    uint32_t* d_val_idx = nullptr;
    uint64_t val_n = 250000;

    // Reusable shard VRAM buffer (up to 100M records = 3.2 GB)
    uint64_t max_shard_n = 100000000ULL;
    BulletRec* d_recs = nullptr;
    uint32_t* d_idx = nullptr;
    CUDA_CHECK(cudaMalloc(&d_recs, max_shard_n * 32));
    CUDA_CHECK(cudaMalloc(&d_idx, max_shard_n * 4));
    std::vector<uint32_t> h_idx(max_shard_n);

    int total_epochs = (int)data_paths.size() * epochs_per_shard;
    uint64_t approx_steps_per_ep = (max_shard_n - val_n) / batch_size;
    uint64_t total_steps = (uint64_t)total_epochs * approx_steps_per_ep;
    uint64_t step = 0;

    double best_val = 1e9;
    int best_ep = 0;
    std::vector<float> best_ft_w(n_ft_w), best_ft_b(hl);
    std::vector<float> best_l1_w(n_l1_w), best_l1_b(n_l1_b);
    std::vector<float> best_l2_w(n_l2_w), best_l2_b(n_l2_b);
    std::vector<float> best_out_w(n_out_w), best_out_b(n_out_b);

    printf("CHNN v3 H100 Trainer | (768*%d%s -> %d)x2 -> 16(dual+skip) -> 32 -> 1 (ob=%d) | %zu shards x %d ep = %d total epochs | wdl=%.2f lr=%.5f->%.5f\n",
           kb, use_threats ? "+threats(3*768)" : "", hl, ob, data_paths.size(), epochs_per_shard, total_epochs, wdl_w, lr_max, lr_min);
    fflush(stdout);

    auto t_start = std::chrono::steady_clock::now();
    int global_ep = 0;

    for (size_t si = 0; si < data_paths.size(); si++) {
        FILE* f = fopen(data_paths[si].c_str(), "rb");
        if (!f) { perror(data_paths[si].c_str()); return 1; }
        fseek(f, 0, SEEK_END);
        uint64_t n = std::min<uint64_t>((uint64_t)ftell(f) / 32, max_shard_n);
        fseek(f, 0, SEEK_SET);
        BulletRec* h_recs = nullptr;
        CUDA_CHECK(cudaMallocHost(&h_recs, n * 32));
        n = fread(h_recs, 32, n, f);
        fclose(f);

        uint64_t train_n = n;
        if (si == 0) {
            val_n = std::min<uint64_t>(250000, n / 40);
            train_n = n - val_n;
            CUDA_CHECK(cudaMalloc(&d_val_recs, val_n * 32));
            CUDA_CHECK(cudaMalloc(&d_val_idx, val_n * 4));
            CUDA_CHECK(cudaMemcpy(d_val_recs, h_recs + train_n, val_n * 32, cudaMemcpyHostToDevice));
            std::vector<uint32_t> h_val(val_n);
            for (uint64_t i = 0; i < val_n; i++) h_val[i] = (uint32_t)i;
            CUDA_CHECK(cudaMemcpy(d_val_idx, h_val.data(), val_n * 4, cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemcpy(d_recs, h_recs, train_n * 32, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaFreeHost(h_recs));
        printf("--- Loaded shard %zu/%zu (%s): %llu train pos ---\n", si + 1, data_paths.size(), data_paths[si].c_str(), (unsigned long long)train_n);
        fflush(stdout);

        uint64_t batches_per_ep = train_n / batch_size;

        for (int rep = 0; rep < epochs_per_shard; rep++) {
            global_ep++;
            auto t0 = std::chrono::steady_clock::now();

            // Coprime stride permutation over train_n
            uint64_t stride = (xorshift64() | 1ULL) % train_n;
            auto gcd = [](uint64_t a, uint64_t b) { while (b) { uint64_t t = a % b; a = b; b = t; } return a; };
            while (stride < 1000 || gcd(stride, train_n) != 1) stride = (stride + 2) % train_n;
            uint64_t offset = xorshift64() % train_n;
            for (uint64_t i = 0; i < train_n; i++) {
                offset += stride;
                if (offset >= train_n) offset -= train_n;
                h_idx[i] = (uint32_t)offset;
            }
            CUDA_CHECK(cudaMemcpy(d_idx, h_idx.data(), train_n * 4, cudaMemcpyHostToDevice));

            double ep_loss = 0.0;
            for (uint64_t b = 0; b < batches_per_ep; b++) {
                step++;
                float prog = fminf(1.0f, (float)step / (float)total_steps);
                // WSD schedule: 3% linear warmup, 67% stable at lr_max, 30% cosine decay to lr_min
                float lr;
                if (prog < 0.03f) {
                    lr = lr_max * (prog / 0.03f);
                } else if (prog < 0.70f) {
                    lr = lr_max;
                } else {
                    float dprog = (prog - 0.70f) / 0.30f;
                    lr = lr_min + (lr_max - lr_min) * 0.5f * (1.0f + cosf(3.14159265f * dprog));
                }

                CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(double)));
                fwd_bwd_v3_kernel<<<batch_size, hl / 2>>>(
                    d_recs, d_idx + b * batch_size, batch_size,
                    kb, hl, ob, factorized, use_threats, wdl_w, scale_f,
                    d_ft_w, d_ft_b, d_l1_w, d_l1_b, d_l2_w, d_l2_b, d_out_w, d_out_b,
                    g_ft_w, g_ft_b, g_l1_w, g_l1_b, g_l2_w, g_l2_b, g_out_w, g_out_b,
                    d_loss, 1
                );

                float inv_bs = 1.0f / (float)batch_size;
                float bc1 = 1.0f / (1.0f - powf(0.9f, (float)step));
                float bc2 = 1.0f / (1.0f - powf(0.999f, (float)step));
                adamw_kernel<<<(n_ft_w + 255) / 256, 256>>>(d_ft_w, m_ft_w, v_ft_w, g_ft_w, n_ft_w, inv_bs, lr, wd, bc1, bc2, 1.98f);
                adamw_kernel<<<(hl + 255) / 256, 256>>>(d_ft_b, m_ft_b, v_ft_b, g_ft_b, hl, inv_bs, lr, 0.0f, bc1, bc2, 1.98f);
                adamw_kernel<<<(n_l1_w + 255) / 256, 256>>>(d_l1_w, m_l1_w, v_l1_w, g_l1_w, n_l1_w, inv_bs, lr, wd, bc1, bc2, 1.98f);
                adamw_kernel<<<(n_l1_b + 255) / 256, 256>>>(d_l1_b, m_l1_b, v_l1_b, g_l1_b, n_l1_b, inv_bs, lr, 0.0f, bc1, bc2, 0.0f);
                adamw_kernel<<<(n_l2_w + 255) / 256, 256>>>(d_l2_w, m_l2_w, v_l2_w, g_l2_w, n_l2_w, inv_bs, lr, wd, bc1, bc2, 1.98f);
                adamw_kernel<<<(n_l2_b + 255) / 256, 256>>>(d_l2_b, m_l2_b, v_l2_b, g_l2_b, n_l2_b, inv_bs, lr, 0.0f, bc1, bc2, 0.0f);
                adamw_kernel<<<(n_out_w + 255) / 256, 256>>>(d_out_w, m_out_w, v_out_w, g_out_w, n_out_w, inv_bs, lr, wd, bc1, bc2, 1.98f);
                adamw_kernel<<<(n_out_b + 255) / 256, 256>>>(d_out_b, m_out_b, v_out_b, g_out_b, n_out_b, inv_bs, lr, 0.0f, bc1, bc2, 0.0f);

                if ((b & 63) == 63 || b + 1 == batches_per_ep) {
                    double bl = 0.0;
                    CUDA_CHECK(cudaMemcpy(&bl, d_loss, sizeof(double), cudaMemcpyDeviceToHost));
                    ep_loss += bl * ((b + 1 == batches_per_ep && (b & 63) != 63) ? (double)((b & 63) + 1) : 64.0);
                }
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            // Held-out validation loss (always same 250k positions from Shard 1!)
            CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(double)));
            for (uint64_t off = 0; off < val_n; off += batch_size) {
                int bs = (int)std::min<uint64_t>(batch_size, val_n - off);
                fwd_bwd_v3_kernel<<<bs, hl / 2>>>(
                    d_val_recs, d_val_idx + off, bs,
                    kb, hl, ob, factorized, use_threats, wdl_w, scale_f,
                    d_ft_w, d_ft_b, d_l1_w, d_l1_b, d_l2_w, d_l2_b, d_out_w, d_out_b,
                    g_ft_w, g_ft_b, g_l1_w, g_l1_b, g_l2_w, g_l2_b, g_out_w, g_out_b,
                    d_loss, 0
                );
            }
            double vloss_sum = 0.0;
            CUDA_CHECK(cudaMemcpy(&vloss_sum, d_loss, sizeof(double), cudaMemcpyDeviceToHost));
            double val_loss = vloss_sum / (double)val_n;
            double tr_loss = ep_loss / (double)(batches_per_ep * batch_size);

            auto t1 = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(t1 - t0).count();
            const char* mark = "";
            if (val_loss < best_val) {
                best_val = val_loss; best_ep = global_ep; mark = " *";
                CUDA_CHECK(cudaMemcpy(best_ft_w.data(), d_ft_w, n_ft_w * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(best_ft_b.data(), d_ft_b, hl * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(best_l1_w.data(), d_l1_w, n_l1_w * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(best_l1_b.data(), d_l1_b, n_l1_b * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(best_l2_w.data(), d_l2_w, n_l2_w * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(best_l2_b.data(), d_l2_b, n_l2_b * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(best_out_w.data(), d_out_w, n_out_w * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(best_out_b.data(), d_out_b, n_out_b * 4, cudaMemcpyDeviceToHost));
            }
            printf("Ep %2d/%d (shard %zu) | train~%.6f val=%.6f%s | %.2fs (%.2fM pos/s)\n",
                   global_ep, total_epochs, si + 1, tr_loss, val_loss, mark, dt,
                   (double)(batches_per_ep * batch_size) / dt / 1e6);
            fflush(stdout);
        }
    }

    save_chnn_v3(out_path, kb, hl, ob, factorized, use_threats, scale_out,
                 best_ft_w, best_ft_b, best_l1_w, best_l1_b, best_l2_w, best_l2_b, best_out_w, best_out_b);
    auto t_end = std::chrono::steady_clock::now();
    printf("Saved CHNN v3 NNUE (best ep %d, val=%.6f) -> %s in %.1fs\n",
           best_ep, best_val, out_path, std::chrono::duration<double>(t_end - t_start).count());
    return 0;
}
"""


@app.function(image=image, volumes={"/vol": vol}, cpu=4, memory=8192, timeout=1800)
def fetch_bullet_shard(shard: int = 1, positions: int = 100_000_000) -> str:
    """Stream-decompress `positions` 32-byte Bullet ChessBoard records from linrock/bullet-training-data S2."""
    import time
    import urllib.request
    import zstandard

    os.makedirs("/vol/data", exist_ok=True)
    tag = f"s2_iter{shard}_{positions // 1_000_000}m.bin"
    out_path = f"/vol/data/{tag}"
    target_bytes = positions * 32

    if os.path.exists(out_path) and os.path.getsize(out_path) >= target_bytes:
        print(f"[shard {shard}] Using cached {out_path} ({os.path.getsize(out_path) / 1e9:.2f} GB)")
        return out_path

    url = (
        f"https://huggingface.co/datasets/linrock/bullet-training-data/resolve/main/"
        f"S2/test77nov-unfilt-test79-maraprmay-v6-dd.skip-see-ge0.wdl-pdist.iter-{shard}.bullet.bin.zst"
    )
    print(f"[shard {shard}] Streaming {positions:,} positions -> {out_path}", flush=True)
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "games-ai-modal"})
    written = 0
    dctx = zstandard.ZstdDecompressor()
    with urllib.request.urlopen(req, timeout=90) as resp:
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
    print(f"[shard {shard}] Done {written // 32:,} pos in {time.time() - t0:.1f}s", flush=True)
    return out_path


@app.function(image=image, volumes={"/vol": vol}, gpu=["H100", "H200", "A100-80GB"], cpu=8, memory=32768, timeout=7200)
def train_v3_h100(
    data_paths: list[str],
    out_name: str,
    kb: int = 16,
    hl: int = 1024,
    ob: int = 8,
    epochs_per_shard: int = 2,
    batch_size: int = 16384,
    lr: float = 0.001,
    lr_end: float = 0.00002,
    wdl: float = 0.35,
    scale_out: int = 160,
    threats: int = 1,
) -> bytes:
    import subprocess

    os.makedirs("/vol/nets", exist_ok=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False)
    cu_path = "/tmp/train_v3_h100.cu"
    bin_path = "/tmp/train_v3_h100"
    with open(cu_path, "w") as f:
        f.write(CUDA_V3_SRC)

    print("Compiling CHNN v3 fused CUDA trainer...", flush=True)
    subprocess.run(["nvcc", "-O3", "-arch=native", "-use_fast_math", cu_path, "-o", bin_path], check=True)

    vol_out = f"/vol/nets/{out_name}"
    cmd = [
        bin_path,
        "--out", vol_out,
        "--kb", str(kb),
        "--hl", str(hl),
        "--ob", str(ob),
        "--epochs-per-shard", str(epochs_per_shard),
        "--batch", str(batch_size),
        "--lr", str(lr),
        "--lr-end", str(lr_end),
        "--wdl", str(wdl),
        "--scale-out", str(scale_out),
        "--threats", str(threats),
    ]
    for p in data_paths:
        cmd += ["--data", p]

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    vol.commit()
    with open(vol_out, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(
    shards: str = "1,2,3,4,5,6,7,8,9,10",
    positions: int = 100_000_000,
    kb: int = 16,
    hl: int = 1024,
    ob: int = 8,
    epochs_per_shard: int = 2,
    batch: int = 16384,
    lr: float = 0.001,
    lr_end: float = 0.00002,
    wdl: float = 0.35,
    scale_out: int = 160,
    threats: int = 1,
    fetch_only: bool = False,
    out: str = "chess/nets/h100_v4.nnue",
):
    shard_list = [int(s.strip()) for s in shards.split(",") if s.strip()]
    print(f"Fetching/verifying {len(shard_list)} shards ({len(shard_list) * positions / 1e9:.2f}B positions) in parallel...")
    data_paths = list(fetch_bullet_shard.map(shard_list, [positions] * len(shard_list)))
    print("Cached shards:", data_paths)
    if fetch_only:
        return

    out_name = os.path.basename(out)
    weights = train_v3_h100.remote(
        data_paths=data_paths,
        out_name=out_name,
        kb=kb,
        hl=hl,
        ob=ob,
        epochs_per_shard=epochs_per_shard,
        batch_size=batch,
        lr=lr,
        lr_end=lr_end,
        wdl=wdl,
        scale_out=scale_out,
        threats=threats,
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "wb") as f:
        f.write(weights)
    print(f"Downloaded CHNN v3 NNUE ({len(weights) / 1e6:.2f} MB) -> {out}")
