// M5 fused causal-mask + softmax kernel (winning variant rv=1, ev=1 — verbatim
// softmax_looped<bfloat16_t, float, 4> with the causal select injected at both
// in[] load sites). Bitwise-identical to the 3-op fallback chain
// (greater_equal -> where(bf16 finfo.min) -> softmax precise) on all tested
// seeds / S / offsets. Kernel name: m5_fused_rv1_ev1.
// Generated signature (MLXFast wraps this):
//   inputs:  scores (device bfloat16_t*, [B,H,qL,S] contiguous),
//            params (constant int*, [axis_size=S, cofs=kL-qL, qL])
//   output:  y (device bfloat16_t*, same shape)
//   grid (threads): (B*H*qL * 1024, 1, 1), threadgroup: (1024, 1, 1)
//   NOTE: threadgroup size MUST equal the production softmax dispatch
//   (maxTotalThreadsPerThreadgroup, 1024 here) — the f32 accumulation order
//   depends on lsize; a different lsize changes the bits.
// ---- header (prepended by MLXFast after metal::utils()) ----
constant constexpr float BF16_MINF = -0x1.FEp+127f;  // bf16 0xFF7F as f32 (exact)
template <typename T>
inline T softmax_exp(T x) {
  return fast::exp(x);
}
// ---- body ----
const int axis_size = params[0];
const int cofs = params[1];
const int qL = params[2];

uint gid = threadgroup_position_in_grid.x;
uint lid = thread_position_in_threadgroup.x;
uint lsize = threads_per_threadgroup.x;
uint simd_lane_id = thread_index_in_simdgroup;
uint simd_group_id = simdgroup_index_in_threadgroup;

const device bfloat16_t* in = scores;
device bfloat16_t* out = y;
in += gid * size_t(axis_size);
const int rowlim = cofs + int(gid) % qL;

constexpr int SIMD_SIZE = 32;
constexpr int N_READS = 4;

threadgroup float local_max[SIMD_SIZE];
threadgroup float local_normalizer[SIMD_SIZE];

// Get the max and the normalizer in one go
float prevmax;
float maxval = Limits<float>::finite_min;
float normalizer = 0;
for (int r = 0; r < static_cast<int>(ceildiv(axis_size, N_READS * lsize));
     r++) {
  int offset = r * lsize * N_READS + lid * N_READS;
  float vals[N_READS];
  if (offset + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      vals[i] = (offset + i <= rowlim)
          ? float(in[offset + i]) : BF16_MINF;
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      vals[i] =
          (offset + i < axis_size)
          ? ((offset + i <= rowlim) ? float(in[offset + i]) : BF16_MINF)
          : Limits<float>::min;
    }
  }
  prevmax = maxval;
  for (int i = 0; i < N_READS; i++) {
    maxval = (maxval < vals[i]) ? vals[i] : maxval;
  }
  normalizer *= softmax_exp(prevmax - maxval);
  for (int i = 0; i < N_READS; i++) {
    normalizer += softmax_exp(vals[i] - maxval);
  }
}
// Now we got partial normalizer of N_READS * ceildiv(axis_size, N_READS *
// lsize) parts. We need to combine them.
//    1. We start by finding the max across simd groups
//    2. We then change the partial normalizers to account for a possible
//       change in max
//    3. We sum all normalizers
prevmax = maxval;
maxval = simd_max(maxval);
normalizer *= softmax_exp(prevmax - maxval);
normalizer = simd_sum(normalizer);

// Now the normalizer and max value is correct for each simdgroup. We write
// them shared memory and combine them.
prevmax = maxval;
if (simd_lane_id == 0) {
  local_max[simd_group_id] = maxval;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
maxval = simd_max(local_max[simd_lane_id]);
normalizer *= softmax_exp(prevmax - maxval);
if (simd_lane_id == 0) {
  local_normalizer[simd_group_id] = normalizer;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
normalizer = simd_sum(local_normalizer[simd_lane_id]);
normalizer = 1 / normalizer;

// Finally given the normalizer and max value we can directly write the
// softmax output
out += gid * size_t(axis_size);
for (int r = 0; r < static_cast<int>(ceildiv(axis_size, N_READS * lsize));
     r++) {
  int offset = r * lsize * N_READS + lid * N_READS;
  if (offset + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      out[offset + i] = bfloat16_t(softmax_exp(
          ((offset + i <= rowlim) ? float(in[offset + i]) : BF16_MINF)
          - maxval) * normalizer);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if (offset + i < axis_size) {
        out[offset + i] = bfloat16_t(softmax_exp(
            ((offset + i <= rowlim) ? float(in[offset + i]) : BF16_MINF)
            - maxval) * normalizer);
      }
    }
  }
}