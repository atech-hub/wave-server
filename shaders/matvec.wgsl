// Matrix-vector multiply: output[row] = sum(weight[row][j] * input[j]) + bias[row]
// One thread per output row. Suitable for 768-dim+ inference projections.

struct Dims {
    out_dim: u32,
    in_dim: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> input: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= dims.out_dim) { return; }

    let in_dim = dims.in_dim;
    var sum: f32 = bias[row];
    let base = row * in_dim;

    for (var j: u32 = 0u; j < in_dim; j = j + 1u) {
        sum = sum + weight[base + j] * input[j];
    }

    output[row] = sum;
}
