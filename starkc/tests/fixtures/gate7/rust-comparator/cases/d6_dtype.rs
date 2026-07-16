include!("../src/typed.rs");
fn f(image: Tensor4<u8, 1, 3, 416, 416, Cpu, Unspecified>, out: Vec<f32>) {
    // predict expects f32 input; u8 -> type error.
    let _ = TinyYoloV2::predict_ty::<1>(&image, out);
}
fn main() {}
