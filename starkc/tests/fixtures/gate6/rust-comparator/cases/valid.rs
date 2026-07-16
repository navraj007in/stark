// Baseline: the correct pipeline. Expected: COMPILES (and would run).
include!("../lib.rs");

// preprocess: NHWC u8 [1,224,224,3] -> NCHW f32 [1,3,224,224]
fn preprocess(nhwc: Tensor4<u8, 1, 224, 224, 3, Cpu>) -> Tensor4<f32, 1, 3, 224, 224, Cpu> {
    nhwc_to_nchw(nhwc).cast::<f32>()
}

fn infer(model: &Resnet50V17, nhwc: Tensor4<u8, 1, 224, 224, 3, Cpu>) -> Tensor2<f32, 1, 1000, Cpu> {
    let nchw = preprocess(nhwc);
    model.predict(&nchw)
}

fn main() {
    let _ = infer(&Resnet50V17, Tensor4::zeros());
}
