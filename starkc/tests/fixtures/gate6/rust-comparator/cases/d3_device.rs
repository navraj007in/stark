// Defect 3 — incompatible device placement.
// The preprocessed tensor is moved to Cuda<0> and then fed to a model whose
// input port lives on Cpu (like STARK's bad_device.stark).
// Expected with this MAXIMAL host (phantom device tag): rustc COMPILE ERROR.
// NOTE: this is the non-idiomatic case — an ordinary `ort` host does NOT encode
// device in the tensor type, so it would catch this at runtime or silently fall
// back (see the Python baseline). Compile-time catching here requires the
// generator to emit a phantom-device API mirroring STARK.
include!("../lib.rs");

fn preprocess(nhwc: Tensor4<u8, 1, 224, 224, 3, Cpu>) -> Tensor4<f32, 1, 3, 224, 224, Cpu> {
    nhwc_to_nchw(nhwc).cast::<f32>()
}

fn main() {
    let model = Resnet50V17;
    let nchw = preprocess(Tensor4::zeros());
    let on_gpu = nchw.to_device::<Cuda<0>>();
    let _ = model.predict(&on_gpu); // expects Cpu, found Cuda<0>
}
