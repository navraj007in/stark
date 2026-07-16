include!("../src/typed.rs");
fn f(unit: Tensor4<f32, 1, 3, 4, 4, Cpu, UnitRange>, out: Vec<f32>) {
    let _ = NormNet::predict_ty::<1>(&unit, out);
}
fn main() {}
