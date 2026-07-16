include!("../src/typed.rs");
fn f(a: Tensor4<f32, 1, 3, 4, 4, Cpu, ByteRange>, b: Tensor4<f32, 1, 3, 4, 4, Cpu, UnitRange>) {
    // add requires both operands share range R; ByteRange vs UnitRange -> error.
    let _ = add(a, b);
}
fn main() {}
