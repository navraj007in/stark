include!("../src/typed.rs");
fn f(grid: Tensor4<f32, 1, 125, 13, 13, Cpu, Unspecified>) {
    // split_channels yields [1,5,25,13,13]; declaring [1,5,26,13,13] is a type error.
    let _bad: Tensor5<f32, 1, 5, 26, 13, 13, Cpu, Unspecified> = split_channels(grid);
}
fn main() {}
