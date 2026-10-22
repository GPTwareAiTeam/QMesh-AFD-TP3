/* qmesh_afd_seam: device-driven, CUDA-graph-capturable AFD seam.
 *
 * Included at the end of qmesh_afd_peer.cpp, so it shares the wire layout, the
 * Peer struct and the wire helpers. Chunk 1 is the device side: the amax
 * reduction, the scale finish, the e4m3 pack into every peer's mapped in-slot,
 * the doorbell publish, the rx spin and the combine. Host-side proxies and the
 * Python binding follow in this file.
 *
 * Numerics replicate afd_session.PeerSession.post exactly:
 *   m     = amax of |x| (bf16, exact in fp32)
 *   mc    = max(m, bf16(1e-6))                      (clamp_min on a bf16 tensor)
 *   s     = (float)(448.0 / mc)                     (x * scale, scalar as fp32)
 *   inv   = (float)(1.0 / (448.0 / mc))             (header, 1/s in double)
 *   y     = bf16_rn(float(x) * s)                   (torch bf16 product rounding)
 *   q     = e4m3_rne_satfinite(clamp(y, +-448))
 */

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <emmintrin.h>
#include <thread>
#include <vector>
#include <cstdlib>
#include <cstring>

static constexpr int SEAM_MAXP = 4;
static constexpr uint32_t F32_INF_BITS = 0x7f800000u;

struct SeamStatus {
  unsigned long long nonfinite;   /* posts whose amax was nan/inf (payload skipped) */
  unsigned long long timeouts;    /* rounds that missed their budget: device collect spin or peer wait */
  unsigned long long posts;       /* device-side publishes */
  /* Sticky terminal failure (0 = healthy). Set by a peer wait that misses its
   * deadline or by a collect spin that exhausts its cycle budget. Once set no
   * post publishes, no collect combines, and the slab is never handed to the
   * model as if the round had completed. */
  unsigned long long failed;
};

/* One doorbell per peer, in mapped host memory. The device writes layer and n,
 * fences, then writes ticket; the proxy acquire-loads ticket before reading
 * the rest. */
struct alignas(64) SeamDoorbell {
  uint64_t ticket;
  int32_t  layer;
  int32_t  n;
};

/* Everything a kernel needs, passed by value (captured by value in graphs).
 * All pointers are device-visible addresses; the pinned slabs are mapped. */
struct SeamDev {
  int np;
  uint8_t*              in[SEAM_MAXP];    /* in-slot payload, n*K e4m3 */
  char*                 rout[SEAM_MAXP];  /* routing slot: hdr, ids, tw */
  SeamDoorbell*         db[SEAM_MAXP];
  const __nv_bfloat16*  out[SEAM_MAXP];   /* out-slot partial, n*K bf16 */
  uint64_t*             rx[SEAM_MAXP];    /* per-peer completion word */
  uint32_t*             amax;             /* device: max |x| bits */
  uint32_t*             fin;              /* device: amax block-completion counter (0 between posts) */
  float*                scale;            /* device: [0]=s, [1]=inv */
  uint64_t*             ticket;           /* device: monotonic post counter */
  SeamStatus*           st;               /* mapped host status */
  long long             timeout_cyc;     /* collect budget in SM cycles */
};

/* ---- finish: one thread turns amax into s, inv and the routing header ----
 * A nonfinite amax counts in status, keeps s=0/inv=1 and leaves the payload
 * skipped in the pack kernel. The header is still written so the worker's round
 * advances and nothing hangs.
 *
 * Called by the last amax block in the post path. That caller must have ordered
 * every atomicMax before this read: each amax block release-fences before its
 * fin increment, and the last block acquire-fences after observing the final
 * increment (see seam_amax_kernel). */
__device__ void seam_finish(SeamDev d, int n, float bf16_floor) {
  /* Volatile load: in the fused path this block reads the final atomicMax
   * result from inside the same kernel, so it must not hit a stale L1 line. */
  const uint32_t bits = *(volatile uint32_t*)d.amax;
  float s = 0.f, inv = 1.f;
  if (bits >= F32_INF_BITS) {
    atomicAdd(&d.st->nonfinite, 1ull);
  } else {
    const float mc = fmaxf(__uint_as_float(bits), bf16_floor);
    const double sd = 448.0 / (double)mc;
    s = (float)sd;
    inv = (float)(1.0 / sd);
  }
  d.scale[0] = s;
  d.scale[1] = inv;
  const int32_t invb = __float_as_int(inv);
  if (d.rout[0]) for (int p = 0; p < d.np; p++) {
    int32_t* h = (int32_t*)d.rout[p];
    h[1] = n;
    h[2] = invb;
    h[3] = 1;   /* FMT_FP8 */
  }
}

/* ---- amax: grid-stride over n*K bf16, integer max of |x| bits ----
 * Non-negative fp32 bit patterns order like their values, and NaN/inf sort above
 * every finite value, so one integer atomicMax yields the nonfinite test too.
 *
 * The last block to arrive runs the finish inline, so the post path has no
 * separate finish node. This is the standard threadfence-reduction handshake:
 * each block publish-fences its atomicMax before incrementing d.fin, and the
 * block that observes the final increment acquire-fences before reading amax.
 * The same block resets d.fin for the next post. d.fin == nullptr keeps the
 * plain reduction without the finish (test/quant-only use). */
/* Single-post contract: the caller must stream-order each post after the
 * previous post's publish (same stream or explicit GPU ordering). d.amax/d.fin/
 * d.scale are single-instance; overlapping posts or unsynchronized streams are
 * unsupported and are not repaired by the resets below. */
__global__ void seam_amax_kernel(const __nv_bfloat16* __restrict__ x, size_t xs, int n,
                                 SeamDev d, float bf16_floor) {
  uint32_t m = 0;
  const size_t total = (size_t)n * K_HIDDEN;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += stride) {
    const size_t r = i / K_HIDDEN, c = i - r * K_HIDDEN;
    const uint32_t b = __float_as_uint(fabsf(__bfloat162float(x[r * xs + c])));
    m = max(m, b);
  }
  for (int o = 16; o; o >>= 1) m = max(m, __shfl_xor_sync(0xffffffffu, m, o));
  __shared__ uint32_t sm[32];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  if (lane == 0) sm[w] = m;
  __syncthreads();
  if (w == 0) {
    m = lane < (int)(blockDim.x >> 5) ? sm[lane] : 0u;
    for (int o = 16; o; o >>= 1) m = max(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) {
      atomicMax(d.amax, m);
      if (d.fin) {
        __threadfence();                          /* release this block's max */
        if (atomicAdd(d.fin, 1u) == gridDim.x - 1) {
          __threadfence();                        /* acquire every other max */
          *d.fin = 0;                             /* re-arm; single post in flight */
          seam_finish(d, n, bf16_floor);
        }
      }
    }
  }
}

/* ---- pack: e4m3 rows into every peer's in-slot, ids/tw into routing ----
 * Grid-stride over n*K payload bytes and n*6 ids/tw entries. ids is int32 or
 * int64 (flattened row-major [n,6]); tw is fp32 [n,6]. */
__global__ void seam_pack_kernel(SeamDev d, const __nv_bfloat16* __restrict__ x, size_t xs,
                                 const void* ids, int ids64, const float* __restrict__ tw,
                                 int n) {
  const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  if (*d.amax < F32_INF_BITS) {
    const float s = d.scale[0];
    const size_t total = (size_t)n * K_HIDDEN;
    for (size_t i = tid; i < total; i += stride) {
      const size_t r = i / K_HIDDEN, c = i - r * K_HIDDEN;
      const __nv_bfloat16 y = __float2bfloat16_rn(__bfloat162float(x[r * xs + c]) * s);
      const float yf = fminf(fmaxf(__bfloat162float(y), -448.f), 448.f);
      const uint8_t q = __nv_cvt_float_to_fp8(yf, __NV_SATFINITE, __NV_E4M3);
      for (int p = 0; p < d.np; p++) d.in[p][i] = q;
    }
  }
  const size_t ne = (size_t)n * 6;
  for (size_t e = tid; ids && e < ne; e += stride) {
    const int32_t id = ids64 ? (int32_t)((const int64_t*)ids)[e] : ((const int32_t*)ids)[e];
    const uint32_t wb = __float_as_uint(tw[e]);
    for (int p = 0; p < d.np; p++) {
      ((int32_t*)(d.rout[p] + ROUT_HDR))[e] = id;
      ((uint32_t*)(d.rout[p] + ROUT_HDR + 24 * (size_t)n))[e] = wb;
    }
  }
}

/* ---- publish body: ticket++ in device memory, then the doorbells ----
 * The caller must have ordered every in-slot and routing write ahead of this
 * call (system fence or kernel boundary); the fence below orders layer/n ahead
 * of the ticket word the proxies poll. */
__device__ __forceinline__ void seam_publish_body(SeamDev d, int layer, int n) {
  if (*(volatile unsigned long long*)&d.st->failed) return;
  /* Re-arm amax and fin for the next post. pack (earlier in this stream) has
   * already read amax and the last amax block already cleared fin, so both
   * stores are idempotent here. This is a single-post re-arm, ordered by the
   * caller's stream (see Seam.post); it does NOT recover concurrent
   * unsynchronized streams, overlapping graph replays, or an abandoned round,
   * which remain unsupported and can still corrupt amax/fin/scale. A sticky
   * failure returned above and the host gate then suppresses every later post,
   * so dirty state on the failed path never reaches a published round. */
  *d.amax = 0;
  if (d.fin) *d.fin = 0;
  const uint64_t t = *d.ticket + 1;
  *d.ticket = t;
  atomicAdd(&d.st->posts, 1ull);
  for (int p = 0; p < d.np; p++) {
    d.db[p]->layer = layer;
    d.db[p]->n = n;
  }
  __threadfence_system();
  for (int p = 0; p < d.np; p++) {
    volatile uint64_t* tp = &d.db[p]->ticket;
    *tp = t;
  }
}

/* ---- publish: its own launch after the pack ----
 * Kept for the scalar pack path: the kernel boundary already orders every pack
 * write ahead of the body. The vector pack folds this in (publish != 0), which
 * removes one launch from the post path the peers wait on. */
__global__ void seam_publish_kernel(SeamDev d, int layer, int n) {
  if (threadIdx.x | blockIdx.x) return;
  seam_publish_body(d, layer, n);
}

/* ---- pack (vectorised): the same e4m3 quantisation, 8 payload bytes per
 * thread per peer ----
 * Reads one 16B chunk of 8 bf16 inputs and writes one 8B chunk per peer instead
 * of eight 1B stores, so a warp covers 256B of each in-slot contiguously. The
 * per-element chain (bf16 round, clamp, e4m3 satfinite) is identical to
 * seam_pack_kernel and no element depends on its neighbours, so the in-slot
 * bytes match the scalar pack exactly. Host dispatch takes this kernel only when
 * x is 16B aligned, xs and the in-slot bases are 8B aligned, and K_HIDDEN % 8.
 *
 * The doorbell is folded in: every block system-fences its in-slot and routing
 * stores and increments d.fin, and the block observing the last increment runs
 * the publish body (d.fin is required; the caller checks it). */
__global__ void seam_pack_vec_kernel(SeamDev d, const __nv_bfloat16* __restrict__ x, size_t xs,
                                     const void* ids, int ids64, const float* __restrict__ tw,
                                     int n, int layer) {
  const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  if (*d.amax < F32_INF_BITS) {
    const float s = d.scale[0];
    const size_t nchunk = (size_t)n * (K_HIDDEN / 8);
    for (size_t j = tid; j < nchunk; j += stride) {
      const size_t i = j * 8;
      const size_t r = i / K_HIDDEN, c = i - r * K_HIDDEN;
      const uint4 u = *reinterpret_cast<const uint4*>(x + r * xs + c);
      const uint32_t w[4] = {u.x, u.y, u.z, u.w};
      uint32_t lo = 0, hi = 0;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const __nv_bfloat16 xb =
              __ushort_as_bfloat16((uint16_t)(h ? (w[k] >> 16) : (w[k] & 0xffffu)));
          const __nv_bfloat16 y = __float2bfloat16_rn(__bfloat162float(xb) * s);
          const float yf = fminf(fmaxf(__bfloat162float(y), -448.f), 448.f);
          const uint32_t q = __nv_cvt_float_to_fp8(yf, __NV_SATFINITE, __NV_E4M3);
          const int e = 2 * k + h;
          if (e < 4) lo |= q << (8 * e);
          else       hi |= q << (8 * (e - 4));
        }
      }
      const uint2 q2 = make_uint2(lo, hi);
      for (int p = 0; p < d.np; p++) *reinterpret_cast<uint2*>(d.in[p] + i) = q2;
    }
  }
  const size_t ne = (size_t)n * 6;
  for (size_t e = tid; ids && e < ne; e += stride) {
    const int32_t id = ids64 ? (int32_t)((const int64_t*)ids)[e] : ((const int32_t*)ids)[e];
    const uint32_t wb = __float_as_uint(tw[e]);
    for (int p = 0; p < d.np; p++) {
      ((int32_t*)(d.rout[p] + ROUT_HDR))[e] = id;
      ((uint32_t*)(d.rout[p] + ROUT_HDR + 24 * (size_t)n))[e] = wb;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();                   /* every block's in-slot stores */
    if (atomicAdd(d.fin, 1u) == gridDim.x - 1) {
      __threadfence_system();                 /* acquire every other block */
      seam_publish_body(d, layer, n);
    }
  }
}

/* ---- collect wait: 1 CTA, spin until every rx word equals the current ticket ----
 * Reads the ticket from device memory, never from a header. The proxy's 2 s peer
 * budget normally detects a dead worker first and sets the sticky failure; this
 * kernel's cycle budget (3 s) is only the backstop for a proxy that is itself
 * stuck. Either way the failure is sticky and the combine is suppressed: a
 * missed round must never be handed to the model as a completed one. */
__global__ void seam_collect_wait_kernel(SeamDev d) {
  if (threadIdx.x | blockIdx.x) return;
  if (*(volatile unsigned long long*)&d.st->failed) return;
  const long long t0 = clock64();
  const uint64_t want = *(volatile uint64_t*)d.ticket;
  unsigned int it = 0;
  for (int p = 0; p < d.np; p++) {
    volatile uint64_t* rx = d.rx[p];
    while (*rx != want) {
      __threadfence_system();
      /* Poll the host-mapped flag sparsely: each test is a PCIe read. */
      if (((++it) & 255u) == 0 && *(volatile unsigned long long*)&d.st->failed) return;
      if (clock64() - t0 > d.timeout_cyc) {
        atomicAdd(&d.st->timeouts, 1ull);
        __threadfence_system();
        atomicExch(&d.st->failed, 1ull);   /* sticky: the combine must not run */
        return;
      }
    }
  }
  __threadfence_system();
}

/* The preload wave below names pv0..pv3 explicitly. */
static_assert(SEAM_MAXP == 4, "combine preload wave is written for SEAM_MAXP == 4");
/* ---- combine: out[:n] = cast(fp32(out) + fp32(p0) + fp32(p1) ...) in peer order ---- */
template <bool OUT_F32>
__global__ void seam_combine_kernel(SeamDev d, void* out, size_t out_stride, int n) {
  /* Fail-stop: no combine on a missed round, and the gate still precedes every
   * store. One host-mapped read per block (each is a PCIe read), broadcast
   * through shared memory.
   *
   * Wave 0 issues every independent load this thread owns before the gate
   * barrier, so the local value and the peer partials overlap instead of
   * forming a dependent chain of host reads. The ordered fp32 sum (local, then
   * peers ascending) and the single bf16 round are unchanged. At the 1024-block
   * cap the wave covers rows <= 51 exactly; larger row counts continue with the
   * unchanged grid-stride loop for the remaining rounds. */
  __shared__ int bad;
  const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  const size_t total = (size_t)n * K_HIDDEN;
  const bool have0 = tid < total;
  float acc0 = 0.f, pv0 = 0.f, pv1 = 0.f, pv2 = 0.f, pv3 = 0.f;
  size_t oi0 = 0;
  if (have0) {
    const size_t r = tid / K_HIDDEN, c = tid - r * K_HIDDEN;
    oi0 = r * out_stride + c;
    if (OUT_F32) acc0 = ((const float*)out)[oi0];
    else         acc0 = __bfloat162float(((const __nv_bfloat16*)out)[oi0]);
    if (d.np > 0) pv0 = __bfloat162float(d.out[0][tid]);
    if (d.np > 1) pv1 = __bfloat162float(d.out[1][tid]);
    if (d.np > 2) pv2 = __bfloat162float(d.out[2][tid]);
    if (d.np > 3) pv3 = __bfloat162float(d.out[3][tid]);
  }
  if (threadIdx.x == 0) bad = (*(volatile unsigned long long*)&d.st->failed != 0ull);
  __syncthreads();
  if (bad) return;
  if (have0) {
    float acc = acc0;
    if (d.np > 0) acc += pv0;
    if (d.np > 1) acc += pv1;
    if (d.np > 2) acc += pv2;
    if (d.np > 3) acc += pv3;
    if (OUT_F32) ((float*)out)[oi0] = acc;
    else         ((__nv_bfloat16*)out)[oi0] = __float2bfloat16_rn(acc);
  }
  for (size_t i = tid + stride; i < total; i += stride) {
    const size_t r = i / K_HIDDEN, c = i - r * K_HIDDEN;
    const size_t oi = r * out_stride + c;
    float acc;
    if (OUT_F32) acc = ((const float*)out)[oi];
    else         acc = __bfloat162float(((const __nv_bfloat16*)out)[oi]);
    for (int p = 0; p < d.np; p++) acc += __bfloat162float(d.out[p][i]);
    if (OUT_F32) ((float*)out)[oi] = acc;
    else         ((__nv_bfloat16*)out)[oi] = __float2bfloat16_rn(acc);
  }
}

/* ---- vector combine: the same ordered sum, one 16B chunk per thread ----
 * Peer partials are a contiguous [n, K_HIDDEN] bf16 block in mapped host
 * memory, so eight elements are one uint4 read per peer: 16B transactions
 * instead of 2B ones. The local row carries out_stride and is read and written
 * with the same 16B granularity, so seam_launch_collect selects this kernel
 * only when the out base and stride keep every chunk aligned and otherwise
 * falls back to seam_combine_kernel. K_HIDDEN is a multiple of 8, so no chunk
 * and no row tail is ever partial here.
 *
 * The arithmetic is the scalar kernel's, element for element: fp32 local, then
 * peers ascending, then exactly one bf16 round for the store. Wave 0 still
 * issues all of its loads before the fail gate, the gate is still one
 * host-mapped read per block by thread 0, and every store follows it. */

/* 16B of eight bf16 -> eight fp32 (exact). */
__device__ __forceinline__ void seam_unpack8(const uint4& u, float (&dst)[8]) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float2 f = __bfloat1622float2(h[k]);
    dst[2 * k] = f.x;
    dst[2 * k + 1] = f.y;
  }
}

/* Eight fp32 -> 16B of eight bf16 with one round-to-nearest-even each. */
__device__ __forceinline__ uint4 seam_pack8(const float (&src)[8]) {
  union { uint4 u; __nv_bfloat162 h[4]; } v;
#pragma unroll
  for (int k = 0; k < 4; ++k) v.h[k] = __floats2bfloat162_rn(src[2 * k], src[2 * k + 1]);
  return v.u;
}

template <bool OUT_F32>
__global__ void seam_combine_vec_kernel(SeamDev d, void* out, size_t out_stride, int n) {
  static_assert(K_HIDDEN % 8 == 0, "vector combine chunks each row in 8s");
  constexpr size_t RV = K_HIDDEN / 8;   /* 8-element chunks per row */
  __shared__ int bad;
  const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  const size_t total = (size_t)n * RV;
  const bool have0 = tid < total;
  uint4 loc_u = make_uint4(0, 0, 0, 0);
  float4 loc_f0 = make_float4(0.f, 0.f, 0.f, 0.f), loc_f1 = loc_f0;
  uint4 pv[SEAM_MAXP];
#pragma unroll
  for (int p = 0; p < SEAM_MAXP; ++p) pv[p] = loc_u;
  size_t oi0 = 0;
  if (have0) {
    const size_t r = tid / RV, c = (tid - r * RV) * 8;
    oi0 = r * out_stride + c;
    if (OUT_F32) {
      const float4* s = reinterpret_cast<const float4*>((const float*)out + oi0);
      loc_f0 = s[0];
      loc_f1 = s[1];
    } else {
      loc_u = *reinterpret_cast<const uint4*>((const __nv_bfloat16*)out + oi0);
    }
#pragma unroll
    for (int p = 0; p < SEAM_MAXP; ++p)
      if (p < d.np) pv[p] = *reinterpret_cast<const uint4*>(d.out[p] + tid * 8);
  }
  if (threadIdx.x == 0) bad = (*(volatile unsigned long long*)&d.st->failed != 0ull);
  __syncthreads();
  if (bad) return;
  if (have0) {
    /* Ordered adds with constant peer indices, so the peer chunks stay in
     * registers: local, then p0..p3 ascending, exactly the scalar kernel. */
    float acc[8], q[8];
    if (OUT_F32) {
      acc[0] = loc_f0.x; acc[1] = loc_f0.y; acc[2] = loc_f0.z; acc[3] = loc_f0.w;
      acc[4] = loc_f1.x; acc[5] = loc_f1.y; acc[6] = loc_f1.z; acc[7] = loc_f1.w;
    } else {
      seam_unpack8(loc_u, acc);
    }
#pragma unroll
    for (int p = 0; p < SEAM_MAXP; ++p) {
      if (p < d.np) {
        seam_unpack8(pv[p], q);
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] += q[e];
      }
    }
    if (OUT_F32) {
      float4* s = reinterpret_cast<float4*>((float*)out + oi0);
      s[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
      s[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
    } else {
      *reinterpret_cast<uint4*>((__nv_bfloat16*)out + oi0) = seam_pack8(acc);
    }
  }
  for (size_t j = tid + stride; j < total; j += stride) {
    const size_t r = j / RV, c = (j - r * RV) * 8;
    const size_t oi = r * out_stride + c;
    float acc[8], q[8];
    if (OUT_F32) {
      const float4* s = reinterpret_cast<const float4*>((const float*)out + oi);
      const float4 a = s[0], b = s[1];
      acc[0] = a.x; acc[1] = a.y; acc[2] = a.z; acc[3] = a.w;
      acc[4] = b.x; acc[5] = b.y; acc[6] = b.z; acc[7] = b.w;
    } else {
      const uint4 u = *reinterpret_cast<const uint4*>((const __nv_bfloat16*)out + oi);
      seam_unpack8(u, acc);
    }
#pragma unroll
    for (int p = 0; p < SEAM_MAXP; ++p) {
      if (p < d.np) {
        const uint4 u = *reinterpret_cast<const uint4*>(d.out[p] + j * 8);
        seam_unpack8(u, q);
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] += q[e];
      }
    }
    if (OUT_F32) {
      float4* s = reinterpret_cast<float4*>((float*)out + oi);
      s[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
      s[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
    } else {
      *reinterpret_cast<uint4*>((__nv_bfloat16*)out + oi) = seam_pack8(acc);
    }
  }
}



/* The peer slabs always qualify (K_HIDDEN % 8 == 0, slab and OUT_OFF are 16B
 * aligned); the local row qualifies when its base is 16B aligned and its
 * stride keeps every row start aligned to the 16B chunk it is read with. */
/* Rows from which the vector pack wins (measured: 4.64 vs 5.64 us at 6 rows,
 * 43.5 vs 29.5 us at 128). Below it the byte-at-a-time pack plus its separate
 * publish launch is the faster and equally exact path. */
static constexpr int SEAM_PACK_VEC_MIN_ROWS = 128;

static bool seam_combine_vec_ok(const void* out, size_t out_stride, bool out_f32) {
  if (K_HIDDEN % 8 != 0) return false;
  if ((reinterpret_cast<uintptr_t>(out) & 15u) != 0) return false;
  return (out_stride & (out_f32 ? 3u : 7u)) == 0;
}


/* The vector pack reads 16B chunks of x and writes 8B chunks per in-slot, so it
 * needs x 16B aligned, a row stride that keeps every chunk start aligned, and
 * 8B-aligned in-slot bases. Its folded publish needs the d.fin counter. */
static bool seam_pack_vec_ok(const void* x, size_t xs, const SeamDev& d) {
  if (K_HIDDEN % 8 != 0) return false;
  if ((reinterpret_cast<uintptr_t>(x) & 15u) != 0) return false;
  if ((xs & 7u) != 0) return false;
  if (d.fin == nullptr) return false;
  for (int p = 0; p < d.np; ++p) {
    if ((reinterpret_cast<uintptr_t>(d.in[p]) & 7u) != 0) return false;
  }
  return true;
}
