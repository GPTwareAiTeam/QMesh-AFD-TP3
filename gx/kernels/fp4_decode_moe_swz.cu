// SPDX-FileCopyrightText: 2026 QMesh contributors
// SPDX-License-Identifier: MIT
//
// QMesh NVFP4 decode-MoE implementation, distributed under the root MIT
// license. TensorFold's `zig/kernels/cuda/nvfp4_experts.cu` and `mma4.cuh`
// (Copyright 2026 TensorFold contributors, Apache-2.0) were consulted as a
// design starting reference. The current implementation is structurally
// different from the reviewed TensorFold kernel, and no preserved TensorFold
// source copy has been identified in this file. The provenance of any earlier
// prototype is not established. The block-scale layout follows vLLM's
// `nvfp4_utils.swizzle_blockscale` (Apache-2.0, Copyright contributors to the
// vLLM project); no vLLM source code is copied into this file. See CREDITS.md.
// Decode-row NVFP4 MoE body, v7: identical math and weight streaming to v6,
// but the block scales are read DIRECTLY from the vLLM/CUTLASS-swizzled arena
// buffers. No row-major copy exists anywhere: the 40-layer duplicate that v6
// needs is 13.05 GB (384 experts x 849,920 B x 40) and does not fit beside the
// 110.7 GiB code arena.
//
// Scale layout (vllm nvfp4_utils.swizzle_blockscale, per expert):
//   byte (row, kb) of a kbp-wide logical row lives at
//     (row/128)*(kbp*128) + (kb/4)*512 + (row%32)*16 + ((row%128)/32)*4 + kb%4
// so four consecutive kb of one row are four contiguous bytes. The kernel
// consumes two consecutive kb (one 16-B code vector = two blocks of 16), i.e.
// an aligned uint16 at (kb/4)*512 + 2*(kb%4)/2.
// kbp is the PADDED width: 320 for w13 (K/16), 56 for w2 (CP/16, only 54 used).
//
// Why this is not an 8x DRAM blowup on scales (the usual failure mode): a
// 32-B sector of the swizzled scale buffer holds one row pair (r, r+1) x one
// 4-kb group, four quarters each = the 4-byte slices of rows
// {r, r+1, r+32, r+33, r+64, r+65, r+96, r+97}. Each block therefore maps its
// 8 warps onto exactly those 8 rows (row8 below), so all 32 B of every sector
// it touches are consumed by the block. DRAM traffic equals the logical scale
// bytes, same as the row-major copy.
//
// Cooperative staging (ss, ss2 below). Those 8 warps used to issue the same
// sector reads 8 times over, so 11% of the bytes (the scales) generated ~73%
// of the L1 wavefronts: one warp instruction reading 32 uint16 at the 512-B
// group stride touches 16 lines for 64 useful bytes. Each block now stages the
// sectors it shares once -- one task per warp, 512 B per task -- and reads the
// scales back as 64-B smem broadcasts. Only the access path changes: the
// staged uint16 are the same bytes at the same addresses, so e4m3 decode, dot
// order and every output bit are unchanged (proof_scale_map.cu proves the byte
// selection on the CPU from this file's own addressing).
//
// Interface is byte-identical to v6's run(): pass the ARENA tensors
//   s13: uint8 [E, 2*CP, K/16] swizzled, s2: uint8 [E, K, CP/16] swizzled
// instead of row-major copies. Per-expert strides are derived from K, CP.
//
// Measured on GX1 (v6, row-major scales): rows=1 199-206 us vs CUTLASS
// 269.3, rows=6 748 us at 98% of the 237.6 GB/s in-situ ceiling. This variant
// keeps the same bytes and arithmetic, minus 13.05 GB of scale storage.
//
// Memory-level parallelism: every lane issues ALL of its weight loads for a
// unit of work (16-B code vectors + scale bytes) before any math.
//
//   Stage 1 (w13 + SwiGLU): row = 2560 B = 160 vectors -> lane owns i = lane+32j.
//   Stage 2 (w2 + top-k sum): row = 432 valid B = 27 vectors -> lane u<27 owns u.
// Nibble order: low nibble = even k element. value = e2m1 * e4m3(scale) * gscale.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int TOPK = 6;
constexpr int W1 = 8;  // stage-1 warps per block; row8 needs exactly 8
constexpr int W2 = 8;  // stage-2 warps per block
constexpr int KR = 1;  // stage-2 rows per warp; row8 needs exactly 1

// Stage-1 staging width: (band, j) scale tasks staged per block, UPL..2*UPL.
// 8 tasks keep stage 1 at 4,096 B static + 20,480 B dynamic = 24,576 B/block,
// i.e. 4 blocks/SM under the 1 KB/block reservation (4 x 25,600 = 102,400 =
// the cc-12.x per-SM budget). 10 stages every task but needs 25,600 B/block,
// which drops to 3 blocks/SM.
#ifndef GX_STG
#define GX_STG 8
#endif

// Hardware e2m1 -> f16x2 (two nibbles of one byte) and e4m3 -> f16 (sm_121a).
// Low nibble = even k element, so the f16x2 result is (even, odd).
__device__ __forceinline__ void e2m1x2_to_f32(uint32_t byte, float& lo, float& hi) {
  uint32_t r;
  asm("{ .reg .b8 b; cvt.u8.u32 b, %1; cvt.rn.f16x2.e2m1x2 %0, b; }"
      : "=r"(r) : "r"(byte & 0xffu));
  __half2 h = *reinterpret_cast<__half2*>(&r);
  lo = __low2float(h);
  hi = __high2float(h);
}

__device__ __forceinline__ float e4m3_to_f32(uint32_t b) {
  uint32_t r;
  asm("{ .reg .b16 b; cvt.u16.u32 b, %1; cvt.rn.f16x2.e4m3x2 %0, b; }"
      : "=r"(r) : "r"(b & 0xffu));
  return __low2float(*reinterpret_cast<__half2*>(&r));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Shared-activation layout: flat activation chunk i=k/32 stores its float4 q at
// 32*i + 4*(q ^ (i & 7)). Without the XOR every lane of a warp reads the same
// bank group (lane stride is 128 B) => 32-way conflict on every vec4 load; with
// it the conflict is 4-way, the floor for a 128 B/warp vec4 read.
__device__ __forceinline__ int swz(int k) {
  return (k & ~31) + 4 * (((k >> 2) & 7) ^ ((k >> 5) & 7)) + (k & 3);
}

// 16 code bytes (32 elements, activations xv[0..31]) -> two block dot products.
__device__ __forceinline__ void dot_vec(uint4 v, const float* xv, float& b0, float& b1) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int q = 0; q < 4; q++) {
#pragma unroll
    for (int b = 0; b < 4; b++) {
      float lo, hi;
      e2m1x2_to_f32(w[q] >> (8 * b), lo, hi);
      const int k = 8 * q + 2 * b;  // element offset in the 32-element vector
      const float d = lo * xv[k] + hi * xv[k + 1];
      if (q < 2) b0 += d; else b1 += d;
    }
  }
}

// Block's 8 warps -> one 32-B sector worth of rows: {b, b+1} x 4 quarters of
// the 128-row tile. y is the block index within a 128-row tile group (16 per
// tile); returns the row index inside the expert.
__host__ __device__ __forceinline__ int row8(int y, int warp) {
  return ((y >> 4) << 7) + ((y & 15) << 1) + ((warp >> 1) << 5) + (warp & 1);
}

// First scale byte of `row` in one expert's swizzled buffer; kbp = padded
// k-blocks per row (320 for w13, 56 for w2), expert stride rows*kbp bytes.
__host__ __device__ __forceinline__ size_t srow_off(int kbp, int row) {
  return (size_t)(row >> 7) * (size_t)(kbp * 128) + 16 * (row & 31)
         + 4 * ((row >> 5) & 3);
}

// The two adjacent scale bytes of blocks kb = 2i, 2i+1 (one 16-B code vector).
__host__ __device__ __forceinline__ size_t s16_off(int i) {
  return (size_t)(i >> 1) * 512 + 2 * (i & 1);
}

// Byte offset of one 16-B half of a row-pair sector: tile = 128-row tile,
// a = row pair inside the tile, g = scale group (kb/4), h = row half. Half h
// holds the four quarters x four bytes of row 2a+h, quarter q at byte 4q.
// Stage 1: tile = band*(CP/128) + y>>4 (exact because CP is a whole number of
// 128-row tiles and CP/32 is a multiple of 4, so the up band's quarter term
// matches), g = 16*j + (lane>>1). Stage 2: tile = y>>4, g = lane>>1.
__host__ __device__ __forceinline__ size_t sector_off(int kbp, int rows, int e,
                                                      int tile, int a, int g,
                                                      int h) {
  return (size_t)e * (size_t)(rows * kbp) + (size_t)tile * (size_t)(kbp * 128)
         + 32 * a + 512 * g + 16 * h;
}

// Staged word index (within one task's 128 words) that receives quarter q of
// half h, group g; and the uint16 index a warp reads back for (task, warp,
// lane). The store of lane (g,h) lands on bank 16h+g, so one warp covers the
// 32 banks once; the read is 16 consecutive words = 64 B on 16 banks, two
// lanes per word (broadcast). Both directions are conflict-free.
__host__ __device__ __forceinline__ int stage_word(int q, int h, int g) {
  return (2 * q + h) * 16 + g;
}
__host__ __device__ __forceinline__ int stage_read16(int task, int warp,
                                                     int lane) {
  return task * 256 + warp * 32 + lane;
}

template <int ROWS, int KBP>
__device__ __forceinline__ const uint8_t* srow(const uint8_t* s, int e,
                                               int row) {
  return s + (size_t)e * (ROWS * KBP) + srow_off(KBP, row);
}

}  // namespace

// ---------- stage 1 ----------------------------------------------------------
// grid: (TOPK * R, CP / W1), block: 32 * W1. One warp per (pair, n). The y
// range must cover CP gate columns, not C: blocks y=108..111 (C/W1..CP/W1)
// carry real columns 792..863, so a C/W1 grid silently drops 24 channels.
// x_bf16 [R, K]; ids [R, TOPK]; act [R*TOPK, C] fp32 out. s13 is swizzled
// [E, 2*CP, K/16]; gate rows [0,C), up rows [CP, CP+C).
template <int K, int CP, int C>
__global__ void __launch_bounds__(32 * W1, 4)
stage1_kernel(const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids,
              const uint8_t* __restrict__ w13, const uint8_t* __restrict__ s13,
              const float* __restrict__ g13, float* __restrict__ act,
              float swiglu_limit, int has_limit) {
  constexpr int KH = K / 2, KBP = K / 16, SR = 2 * CP;
  constexpr int UPL = K / 1024;  // 16-B vectors per lane: KH/16 / 32
  constexpr int NSTG = GX_STG;   // (band, j) scale tasks staged per block
  static_assert(K % 1024 == 0 && CP % (16 * W1) == 0, "shape");
  static_assert(NSTG >= UPL && NSTG <= 2 * UPL && NSTG <= W1 * UPL, "staging");
  extern __shared__ float xs[];  // K floats: this row's activation
  __shared__ uint32_t ss[NSTG * 128];  // staged scale words: 512 B per task
  const int pair = blockIdx.x;
  const int row = pair / TOPK;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = row8(blockIdx.y, warp);
  const int e = ids[pair];  // the staging load below needs it: 4 B, uniform

  // Fill: one thread per 32-float chunk (64 B of x, 4 x uint4) so the copy is
  // vectorized end to end and normally takes a single iteration.
  const __nv_bfloat16* xr = x + (size_t)row * K;
  for (int c = threadIdx.x; c < K / 32; c += blockDim.x) {
    const uint4* src = reinterpret_cast<const uint4*>(xr + 32 * c);
    float* dst = xs + 32 * c;
    const int sw = c & 7;
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const uint4 u = src[q];
      const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int b = 0; b < 4; b++) {
        const __nv_bfloat162 h = *reinterpret_cast<const __nv_bfloat162*>(&w[b]);
        const int k = 8 * q + 2 * b;
        float* g = dst + 4 * ((k >> 2) ^ sw);
        g[k & 3] = __low2float(h);
        g[(k + 1) & 3] = __high2float(h);
      }
    }
  }
  // Stage the block's scale sectors once, before the barrier. Warp t owns
  // task t = band*UPL + j and loads the 16 sectors (16 groups x 2 row halves
  // = 512 B) that all 8 warps read; every lane's 16-B chunk is one half of one
  // sector. Nothing here depends on `n`, so it is uniform even for the
  // padding blocks below.
#pragma unroll
  for (int t = warp; t < NSTG; t += W1) {
    const int band = t / UPL, j = t - band * UPL;
    const int gg = lane >> 1, hh = lane & 1;
    const size_t off = sector_off(KBP, SR, e, band * (CP / 128)
                                                 + (blockIdx.y >> 4),
                                  blockIdx.y & 15, 16 * j + gg, hh);
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(s13 + off));
    uint32_t* d = ss + t * 128;
    d[stage_word(0, hh, gg)] = u.x;
    d[stage_word(1, hh, gg)] = u.y;
    d[stage_word(2, hh, gg)] = u.z;
    d[stage_word(3, hh, gg)] = u.w;
  }
  __syncthreads();
  if (n >= C) return;  // zero padding rows C..CP-1; must follow the barrier

  const uint8_t* gr = w13 + ((size_t)e * SR + n) * KH;
  const uint8_t* ur = gr + (size_t)CP * KH;
  const int s_lo = (lane >> 1) * 512 + 2 * (lane & 1);
  // Up-band tasks this block does not stage (none when GX_STG == 2*UPL).
  const uint8_t* us = srow<SR, KBP>(s13, e, CP + n) + s_lo;

  // Phase 1: every load of this lane, before any math. Codes stream as before;
  // the scales are block-staged, so a warp instruction is one 64-B smem read.
  uint4 vg[UPL], vu[UPL];
  uint32_t sgv[UPL], suv[UPL];
  const uint16_t* sw = reinterpret_cast<const uint16_t*>(ss);
#pragma unroll
  for (int j = 0; j < UPL; j++) {
    const int i = lane + 32 * j;
    vg[j] = __ldg(reinterpret_cast<const uint4*>(gr) + i);
    vu[j] = __ldg(reinterpret_cast<const uint4*>(ur) + i);
    sgv[j] = sw[stage_read16(j, warp, lane)];
    suv[j] = (j < NSTG - UPL)
                 ? sw[stage_read16(UPL + j, warp, lane)]
                 : __ldg(reinterpret_cast<const uint16_t*>(us + j * 8192));
  }

  // Phase 2: math.
  float ag = 0.f, au = 0.f;
#pragma unroll
  for (int j = 0; j < UPL; j++) {
    const int i = lane + 32 * j;
    float xv[32];
    const float* xb = xs + 32 * i;
    const int sw = i & 7;
#pragma unroll
    for (int q = 0; q < 8; q++) {
      const float4 t = *reinterpret_cast<const float4*>(xb + 4 * (q ^ sw));
      xv[4 * q] = t.x; xv[4 * q + 1] = t.y; xv[4 * q + 2] = t.z; xv[4 * q + 3] = t.w;
    }
    float g0 = 0.f, g1 = 0.f, u0 = 0.f, u1 = 0.f;
    dot_vec(vg[j], xv, g0, g1);
    dot_vec(vu[j], xv, u0, u1);
    ag += g0 * e4m3_to_f32(sgv[j]) + g1 * e4m3_to_f32(sgv[j] >> 8);
    au += u0 * e4m3_to_f32(suv[j]) + u1 * e4m3_to_f32(suv[j] >> 8);
  }
  ag = warp_sum(ag);
  au = warp_sum(au);
  if (lane == 0) {
    const float gs_e = g13[e];
    float g = ag * gs_e, u = au * gs_e;
    if (has_limit) {
      g = fminf(g, swiglu_limit);
      u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
    }
    act[(size_t)pair * C + n] = (g / (1.f + expf(-g))) * u;
  }
}

// ---------- stage 2 ----------------------------------------------------------
// grid: (R, K / (W2 * KR)), block: 32 * W2. A warp computes one output row k
// over all TOPK experts; lane u<NV owns code vector u of each row. K/(W2*KR)
// must be a whole number of 16-block row tiles (K % 128 == 0 for KR=1) or the
// trailing row8 warps are never launched. s2 is swizzled [E, K, CP/16]; only
// the first C/16 scale bytes of each row are consumed.
template <int K, int CP, int C>
__global__ void __launch_bounds__(32 * W2)
stage2_kernel(const float* __restrict__ act, const int* __restrict__ ids,
              const float* __restrict__ tw, const uint8_t* __restrict__ w2,
              const uint8_t* __restrict__ s2, const float* __restrict__ g2,
              __nv_bfloat16* __restrict__ out) {
  constexpr int NV = C / 32, KBV = CP / 16;  // 27 vectors per row, padded 56
  constexpr int NGS = (NV + 1) / 2;  // scale groups a row actually consumes
  static_assert(C % 32 == 0 && NV <= 32 && (TOPK * C) % 4 == 0, "shape");
  static_assert(KBV % 4 == 0 && K % (16 * W2 * KR) == 0, "shape");
  static_assert(NGS <= KBV / 4 && TOPK <= W2, "staging");
  extern __shared__ float as[];  // TOPK * C floats: this row's activations
  __shared__ uint32_t ss2[TOPK * 128];  // staged scale words: 512 B per expert
  const int row = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int k0 = row8(blockIdx.y, warp);
  {
    const float* src = act + (size_t)row * TOPK * C;
    for (int k = threadIdx.x; k < TOPK * C; k += blockDim.x) as[swz(k)] = src[k];
  }
  // Stage the block's scale sectors once: one warp per expert loads the 14
  // groups (28 row halves, 448 B) that this block's row pair consumes, instead
  // of the 8 warps re-issuing 14 sectors each. Lane (g,h) reads the 16-B half
  // of sector (g, row pair); g < NGS keeps the chunk inside the expert's
  // 56-group row tile (groups 14,15 belong to the next tile). The lanes cut by
  // the guard only skip their own load and store, with no barrier between.
  if (warp < TOPK && (lane >> 1) < NGS) {
    const size_t off = sector_off(KBV, K, ids[row * TOPK + warp],
                                  blockIdx.y >> 4, blockIdx.y & 15,
                                  lane >> 1, lane & 1);
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(s2 + off));
    uint32_t* d = ss2 + warp * 128;
    d[stage_word(0, lane & 1, lane >> 1)] = u.x;
    d[stage_word(1, lane & 1, lane >> 1)] = u.y;
    d[stage_word(2, lane & 1, lane >> 1)] = u.z;
    d[stage_word(3, lane & 1, lane >> 1)] = u.w;
  }
  __syncthreads();  // the staged sectors are block-wide; the act fill is too
  const bool on = lane < NV;
  const int u = on ? lane : 0;

  // Phase 1: all loads of this lane, before any math.
  int eid[TOPK];
  float gw[TOPK];
  uint4 v[TOPK];
  uint32_t sc[TOPK];
#pragma unroll
  for (int j = 0; j < TOPK; j++) {
    eid[j] = ids[row * TOPK + j];
    gw[j] = g2[eid[j]] * tw[row * TOPK + j];
  }
#pragma unroll
  for (int j = 0; j < TOPK; j++) {
    const size_t rw = (size_t)eid[j] * K + k0;
    v[j] = __ldg(reinterpret_cast<const uint4*>(w2 + rw * (CP / 2)) + u);
    sc[j] = reinterpret_cast<const uint16_t*>(ss2)[stage_read16(j, warp, u)];
  }

  // Phase 2: math.
  float tot = 0.f;
#pragma unroll
  for (int j = 0; j < TOPK; j++) {
    float xv[32];
    const int base = j * C + 32 * u;
    const int sw = (base >> 5) & 7;
    const float* xb = as + base;
#pragma unroll
    for (int q = 0; q < 8; q++) {
      const float4 t = *reinterpret_cast<const float4*>(xb + 4 * (q ^ sw));
      xv[4 * q] = t.x; xv[4 * q + 1] = t.y; xv[4 * q + 2] = t.z; xv[4 * q + 3] = t.w;
    }
    float b0 = 0.f, b1 = 0.f;
    dot_vec(v[j], xv, b0, b1);
    if (on) tot += (b0 * e4m3_to_f32(sc[j]) + b1 * e4m3_to_f32(sc[j] >> 8)) * gw[j];
  }
  const float t = warp_sum(tot);
  if (lane == 0) out[(size_t)row * K + k0] = __float2bfloat16(t);
}

// Host wrapper: two launches, no activation quant, no shuffle, no allocations.
// act: fp32 scratch [R*TOPK, C]. Launch: stage1 then stage2, same stream.
template <int K, int CP, int C>
void decode_moe(const __nv_bfloat16* x, const int* ids, const float* tw,
                const uint8_t* w13, const uint8_t* s13, const float* g13,
                const uint8_t* w2, const uint8_t* s2, const float* g2,
                float* act, __nv_bfloat16* out, int R, float swiglu_limit,
                int has_limit, cudaStream_t st) {
  dim3 g1(R * TOPK, CP / W1);
  stage1_kernel<K, CP, C><<<g1, 32 * W1, K * sizeof(float), st>>>(
      x, ids, w13, s13, g13, act, swiglu_limit, has_limit);
  dim3 grid2(R, K / (W2 * KR));
  stage2_kernel<K, CP, C><<<grid2, 32 * W2, TOPK * C * sizeof(float), st>>>(
      act, ids, tw, w2, s2, g2, out);
}
