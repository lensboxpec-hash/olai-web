/**
 * OLAI WGSL kernels v0.1.
 * Wired & tested in-page:  GEMV_F32, RMSNORM.
 * Defined interface, stubbed body (marked): SILU_MUL, ARGMAX — the full
 * LFM3 op set (short-conv k=7 ring state, GQA+SWA attention, RoPE) lands
 * once CI fills the packets branch; interfaces are frozen now.
 */

export const GEMV_F32 = /* wgsl */ `
struct Params { N: u32, K: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> W : array<f32>;
@group(0) @binding(1) var<storage, read> x : array<f32>;
@group(0) @binding(2) var<storage, read_write> y : array<f32>;
@group(0) @binding(3) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let row = gid.x;
  if (row >= params.N) { return; }
  var acc : f32 = 0.0;
  let K = params.K;
  var k : u32 = 0u;
  while (k + 4u <= K) {
    acc = acc + W[row*K+k]   * x[k]
              + W[row*K+k+1u] * x[k+1u]
              + W[row*K+k+2u] * x[k+2u]
              + W[row*K+k+3u] * x[k+3u];
    k = k + 4u;
  }
  while (k < K) { acc = acc + W[row*K+k] * x[k]; k = k + 1u; }
  y[row] = acc;
}
`;

export const RMSNORM = /* wgsl */ `
struct Params { n: u32, eps_bits: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> x : array<f32>;
@group(0) @binding(1) var<storage, read> w : array<f32>;
@group(0) @binding(2) var<storage, read_write> y : array<f32>;
@group(0) @binding(3) var<uniform> params : Params;

@compute @workgroup_size(1)
fn main() {
  let n = params.n;
  var ss : f32 = 0.0;
  for (var i : u32 = 0u; i < n; i = i + 1u) {
    ss = ss + x[i] * x[i];
  }
  let eps = f32(params.eps_bits) * 1e-8; // eps encoded as integer micro-units
  let inv = 1.0 / sqrt(ss / f32(n) + eps);
  for (var i : u32 = 0u; i < n; i = i + 1u) {
    y[i] = x[i] * inv * w[i];
  }
}
`;

// ---- STUBS (interfaces frozen, bodies land with the packets branch) ----

export const SILU_MUL = /* wgsl */ `
// y = silu(a) * b ; a,b: [n] — LFM3 SwiGLU FFN tail
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> y : array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  let v = a[i];
  y[i] = v / (1.0 + exp(-v)) * b[i]; // TODO(lfm3-kernels): numerics review
}
`;

export const ARGMAX = /* wgsl */ `
// single-workgroup argmax over vocab — TODO(lfm3-kernels): parallel reduction
@group(0) @binding(0) var<storage, read> x : array<f32>;
@group(0) @binding(1) var<storage, read_write> out : array<u32>;
@compute @workgroup_size(1)
fn main() {
  var best : f32 = x[0]; var bi : u32 = 0u;
  for (var i : u32 = 1u; i < arrayLength(&x); i = i + 1u) {
    if (x[i] > best) { best = x[i]; bi = i; }
  }
  out[0] = bi;
}
`;
