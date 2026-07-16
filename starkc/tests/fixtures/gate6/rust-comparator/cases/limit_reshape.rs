// The wall — NOT a defect case.
// This is a *valid* program (a flatten/reshape whose output dim is the product
// of input dims) that the strongest stable-Rust host CANNOT EXPRESS, because a
// computed const in return position requires `feature(generic_const_exprs)`
// (unstable). STARK types this directly. Expected on stable rustc: COMPILE
// ERROR ("generic parameters may not be used in const operations"), which is
// the host failing to express the program, not catching a bug.
//
// This measures the boundary of the comparator: pure permutations + elementwise
// ops (all this CV preprocessing needs) type fine on stable Rust, but the moment
// a pipeline needs shape *arithmetic* (flatten, matmul contraction, conv stride,
// broadcast) the stable-Rust host must drop those dims to runtime.
include!("../lib.rs");

// flatten [N,C,H,W] -> [N, C*H*W]
fn flatten<T, const N: usize, const C: usize, const H: usize, const W: usize, Dev>(
    _t: Tensor4<T, N, C, H, W, Dev>,
) -> Tensor2<T, N, { C * H * W }, Dev> {
    unimplemented!()
}

fn main() {
    let t: Tensor4<f32, 1, 3, 224, 224, Cpu> = Tensor4::zeros();
    let _flat = flatten(t);
}
