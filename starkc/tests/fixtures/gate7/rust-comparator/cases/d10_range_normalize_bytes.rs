include!("../src/typed.rs");
fn f(x: Tensor4<f32, 1, 3, 4, 4, Cpu, ByteRange>) {
    // normalize is defined only on UnitRange -> no method for ByteRange.
    let _ = x.normalize();
}
fn main() {}
