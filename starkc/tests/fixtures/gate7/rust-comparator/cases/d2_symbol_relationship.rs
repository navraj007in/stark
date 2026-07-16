include!("../src/typed.rs");
fn f(image: Tensor4<f32, 1, 3, 416, 416, Cpu, Unspecified>, out: Vec<f32>) {
    // predict_ty::<2> expects &Tensor4<..,2,..>; the input is N=1 -> type error.
    let _ = TinyYoloV2::predict_ty::<2>(&image, out);
}
fn main() {}
