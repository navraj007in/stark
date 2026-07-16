include!("../src/typed.rs");
fn f(image: Tensor4<f32, 1, 3, 416, 416, Cuda<0>, Unspecified>, out: Vec<f32>) {
    // predict input port is on Cpu; Cuda<0> -> type error.
    let _ = TinyYoloV2::predict_ty::<1>(&image, out);
}
fn main() {}
