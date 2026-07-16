include!("../src/typed.rs");
fn f(x: Tensor4<f32, 1, 3, 4, 4, Cpu, ByteRange>) {
    let unit = x.scale_255();   // UnitRange
    let _ = unit.scale_255();   // scale_255 not defined on UnitRange -> error
}
fn main() {}
