// Defect 2 — incorrect element type.
// preprocess omits the cast and returns UInt8 NCHW (like STARK's bad_dtype.stark);
// the model expects f32, so the predict call is a type error.
// Expected: rustc COMPILE ERROR (defect caught at compile time).
include!("../lib.rs");

fn preprocess(nhwc: Tensor4<u8, 1, 224, 224, 3, Cpu>) -> Tensor4<u8, 1, 3, 224, 224, Cpu> {
    nhwc_to_nchw(nhwc) // no .cast::<f32>()
}

fn main() {
    let model = Resnet50V17;
    let nchw = preprocess(Tensor4::zeros());
    let _ = model.predict(&nchw); // expects &Tensor4<f32,..>, found <u8,..>
}
