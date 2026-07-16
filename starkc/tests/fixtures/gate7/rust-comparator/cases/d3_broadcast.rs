include!("../src/typed.rs");
fn f(x: Tensor4<f32, 1, 1, 13, 13, Cpu, Unspecified>) {
    // broadcast_channels input [N,1,H,W] -> [N,C,H,W] shares H,W; declaring a
    // result with a different spatial dim (14 vs 13) is a compile-time error.
    let _bad: Tensor4<f32, 1, 5, 14, 13, Cpu, Unspecified> = broadcast_channels(x);
}
fn main() {}
