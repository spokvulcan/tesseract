// M4 fused rotate+dequant+dot kernel (winning variant p1v=1, p2v=1 — verbatim
// arithmetic on both phases; bitwise-identical to the two-kernel pipeline on
// all tested seeds/shapes). Kernel name: m4_K2048_p1v1_p2v1_m0_rps4.
// Generated signature (MLXFast wraps this):
//   inputs:  x (device bfloat16_t*), packed_pairs (device int*),
//            cos_theta/sin_theta (device bfloat16_t*),
//            channel_scales (device bfloat16_t*),
//            w (device uint*), scales/biases (device bfloat16_t*)
//   output:  y (device bfloat16_t*, [1, 2048])
//   grid (threads): (32, N/4, 1), threadgroup: (32, 2, 1)
//   attributes: threadgroup_position_in_grid, thread_index_in_simdgroup,
//               simdgroup_index_in_threadgroup
// ---- header (helpers, prepended by MLXFast after metal::utils()) ----
// load_vector<T=bfloat16_t, U=float, values_per_thread=16, bits=4>
// (quantized.h) with x in threadgroup address space.
inline float load_vector_tg(const threadgroup bfloat16_t* x, thread float* x_thread) {
  float sum = 0;
  for (int i = 0; i < 16; i += 4) {
    sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 16.0f;
    x_thread[i + 2] = x[i + 2] / 256.0f;
    x_thread[i + 3] = x[i + 3] / 4096.0f;
  }
  return sum;
}
// Same body, original device address space (clone/isolation mode).
inline float load_vector_dev(const device bfloat16_t* x, thread float* x_thread) {
  float sum = 0;
  for (int i = 0; i < 16; i += 4) {
    sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 16.0f;
    x_thread[i + 2] = x[i + 2] / 256.0f;
    x_thread[i + 3] = x[i + 3] / 4096.0f;
  }
  return sum;
}
// qdot<U=float, values_per_thread=16, bits=4> (quantized.h)
inline float qdot4(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
  float accum = 0;
  const device uint16_t* ws = (const device uint16_t*)w;
  for (int i = 0; i < 4; i++) {
    accum +=
        (x_thread[4 * i] * (ws[i] & 0x000f) +
         x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
         x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
         x_thread[4 * i + 3] * (ws[i] & 0xf000));
  }
  return scale * accum + sum * bias;
}
// ---- body ----
// ---- phase 1: pairwise rotation, arithmetic identical to
// paro_rotate_r1_k8_bfloat16; simdgroup s handles groups
// [s*G, (s+1)*G). Same per-lane coefficient caching, same f32
// expressions, same simdgroup barriers, same bf16 rounding.
constexpr int KHID = 2048;
constexpr int KROTV = 8;
constexpr int GSV = 128;
constexpr int HALF_HIDDENV = KHID / 2;
constexpr int GROUPS_PER_SIMD = 8;

const uint lane_u = thread_index_in_simdgroup;
const uint sgid = simdgroup_index_in_threadgroup;
const int lane = int(lane_u);

threadgroup float stage[2 * 128];
threadgroup bfloat16_t xrot[KHID];

for (int g = 0; g < GROUPS_PER_SIMD; g++) {
    const int group_idx = int(sgid) * GROUPS_PER_SIMD + g;
    const int gbase = group_idx * GSV;

    float cos_vals[KROTV][2], sin_vals[KROTV][2];
    int   pair_vals[KROTV][2];

    for (int k = 0; k < KROTV; k++) {
        for (int u = 0; u < 2; u++) {
            int idx = k * HALF_HIDDENV + group_idx * (GSV / 2) + lane + u * 32;
            cos_vals[k][u]  = float(cos_theta[idx]);
            sin_vals[k][u]  = float(sin_theta[idx]);
            pair_vals[k][u] = int(packed_pairs[idx]);
        }
    }

    threadgroup float* tile = stage + sgid * 128;

    float sc0 = float(channel_scales[gbase + lane * 4 + 0]);
    float sc1 = float(channel_scales[gbase + lane * 4 + 1]);
    float sc2 = float(channel_scales[gbase + lane * 4 + 2]);
    float sc3 = float(channel_scales[gbase + lane * 4 + 3]);
    {
        bfloat4 xh = ((const device bfloat4*)(x + gbase))[lane];
        float4 tv;
        tv[0] = float(xh[0]) * sc0;
        tv[1] = float(xh[1]) * sc1;
        tv[2] = float(xh[2]) * sc2;
        tv[3] = float(xh[3]) * sc3;
        *(threadgroup float4*)(tile + lane * 4) = tv;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 0; k < KROTV; k++) {
        for (int u = 0; u < 2; u++) {
            int i_local = pair_vals[k][u] & 0xFFFF;
            int j_local = pair_vals[k][u] >> 16;
            float c = cos_vals[k][u], s = sin_vals[k][u];
            float a = tile[i_local];
            float b = tile[j_local];
            tile[i_local] = a * c + b * s;
            tile[j_local] = b * c - a * s;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    {
        float4 tv = *(threadgroup float4*)(tile + lane * 4);
        bfloat4 ov;
        ov[0] = bfloat16_t(tv[0]);
        ov[1] = bfloat16_t(tv[1]);
        ov[2] = bfloat16_t(tv[2]);
        ov[3] = bfloat16_t(tv[3]);
        *(threadgroup bfloat4*)(xrot + gbase + lane * 4) = ov;
    }
}
// ---- phase 2: qmv_fast_impl<bfloat16_t, 128, 4> verbatim body,
// x from the threadgroup tile.
// in_vec_size=KHID, out_vec_size=N (grid y covers N/(2*RPS)), tid.x=0.
// RPS=results_per_simdgroup: per-row arithmetic is RPS-independent
// (same lane->data mapping, same simd_sum tree), so RPS != 4 stays
// bitwise-identical to the production rps=4 kernel while amortising
// the redundant phase-1 rotation over 2*RPS rows per threadgroup.
{
    const uint lane_u2 = lane_u;
    const uint sgid2 = sgid;
    const int lane2 = int(lane_u2);
    constexpr int RPS = 4;
    constexpr int in_vec_size_w = 2048 * 4 / 8;   // K * bytes_per_pack / pack_factor
    constexpr int in_vec_size_g = 2048 / 128;   // K / group_size

    const int out_row = int(threadgroup_position_in_grid.y) * (2 * RPS) + int(sgid2) * RPS;

    const device uint8_t* ws = (const device uint8_t*)w;
    const device bfloat16_t* scales_p = scales;
    const device bfloat16_t* biases_p = biases;
    device bfloat16_t* yp = y;

    thread float x_thread[16];
    thread float result[RPS] = {0};

    ws += out_row * in_vec_size_w + lane2 * 2 * 4;
    scales_p += out_row * in_vec_size_g + lane2 / 8;
    biases_p += out_row * in_vec_size_g + lane2 / 8;
    threadgroup_barrier(mem_flags::mem_threadgroup);
const threadgroup bfloat16_t* xp = xrot + lane2 * 16;
    yp += out_row;

    for (int k = 0; k < 2048; k += 512) {
        float sum = load_vector_tg(xp, x_thread);

        for (int row = 0; row < RPS; row++) {
            const device uint8_t* wl = ws + row * in_vec_size_w;
            const device bfloat16_t* sl = scales_p + row * in_vec_size_g;
            const device bfloat16_t* bl = biases_p + row * in_vec_size_g;

            float s = sl[0];
            float b = bl[0];
            result[row] += qdot4(wl, x_thread, s, b, sum);
        }

        ws += 512 * 4 / 8;
        scales_p += 512 / 128;
        biases_p += 512 / 128;
        xp += 512;
    }

    for (int row = 0; row < RPS; row++) {
        result[row] = simd_sum(result[row]);
        if (lane2 == 0) {
            yp[row] = static_cast<bfloat16_t>(result[row]);
        }
    }
}