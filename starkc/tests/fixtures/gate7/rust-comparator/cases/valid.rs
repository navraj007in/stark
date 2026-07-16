// Valid Tiny YOLOv2 typed pipeline (const batch B=1). Expected: COMPILES.
include!("../src/typed.rs");
fn pipeline(shape: &[usize], input: Vec<f32>, model_out: Vec<f32>)
    -> Result<Tensor5<f32, 1, 5, 13, 13, 25, Cpu, Unspecified>, String> {
    let image = Tensor4::<f32, 1, 3, 416, 416, Cpu, Unspecified>::refine(shape, input)?;
    let grid = TinyYoloV2::predict_ty::<1>(&image, model_out)?;
    let split = split_channels(grid);
    Ok(permute_grid(split))
}
fn main() { let _ = pipeline; }
