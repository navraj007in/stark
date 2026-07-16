include!("../src/typed.rs");
fn expects_unspecified(_x: Tensor4<f32, 1, 3, 4, 4, Cpu, Unspecified>) {}
fn f(x: Tensor4<f32, 1, 3, 4, 4, Cpu, ByteRange>) {
    // Passing a ByteRange tensor where Unspecified is expected -> type error
    // (the marker never widens to Unspecified).
    expects_unspecified(x);
}
fn main() {}
