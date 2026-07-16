// Defect 1 — incompatible tensor dimensions.
// The preprocess input is declared [1,224,100,3] (100 instead of 224), exactly
// like STARK's bad_shape.stark. The wrong dim propagates through the typed
// permutation, so the produced NCHW shape is [1,3,224,100], which cannot match
// the declared f32 [1,3,224,224] return type.
// Expected: rustc COMPILE ERROR (defect caught at compile time).
include!("../lib.rs");

fn preprocess(nhwc: Tensor4<u8, 1, 224, 100, 3, Cpu>) -> Tensor4<f32, 1, 3, 224, 224, Cpu> {
    nhwc_to_nchw(nhwc).cast::<f32>()
}

fn main() {
    let model = Resnet50V17;
    let nchw = preprocess(Tensor4::zeros());
    let _ = model.predict(&nchw);
}
