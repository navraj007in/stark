use std::env;
use std::path::PathBuf;
use std::process::Command;
use common::TempDir;

mod common;

fn find_workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.pop();
    dir
}

#[test]
fn test_real_inference_agrees_with_reference() {
    let ws_root = find_workspace_root();
    
    let model_env = env::var("STARK_GATE5_MODEL").ok();
    let model_path = model_env.map(PathBuf::from).unwrap_or_else(|| {
        ws_root.join("starkc/tmp/resnet50-v1-7.onnx")
    });
    
    let image_env = env::var("STARK_GATE5_IMAGE").ok();
    let image_path = image_env.map(PathBuf::from).unwrap_or_else(|| {
        ws_root.join("starkc/tmp/dog.jpg")
    });
    
    if !model_path.exists() {
        println!("Skipping real-backend test: model file not found at {:?}", model_path);
        return;
    }
    if !image_path.exists() {
        println!("Skipping real-backend test: image file not found at {:?}", image_path);
        return;
    }
    
    let temp = TempDir::new();
    let out_dir = temp.path().join("host");
    
    let mut deploy_cmd = Command::new("cargo");
    deploy_cmd
        .arg("run")
        .arg("--")
        .arg("deploy")
        .arg(ws_root.join("starkc/examples/gate5/valid_pipeline.stark"))
        .arg("--model")
        .arg(&model_path)
        .arg("--entry")
        .arg("infer")
        .arg("--out")
        .arg(&out_dir)
        .arg("--force")
        .current_dir(ws_root.join("starkc"));
        
    let deploy_status = deploy_cmd.status().expect("failed to execute starkc deploy");
    assert!(deploy_status.success(), "starkc deploy failed");
    
    let mut build_cmd = Command::new("cargo");
    build_cmd
        .arg("build")
        .arg("--release")
        .arg("--locked")
        .arg("--manifest-path")
        .arg(out_dir.join("Cargo.toml"));
        
    let build_status = build_cmd.status().expect("failed to build generated host");
    assert!(build_status.success(), "failed to build generated host");
    
    let binary = out_dir.join("target/release/stark-resnet50");
    let mut run_cmd = Command::new(&binary);
    run_cmd
        .arg("--model")
        .arg(&model_path)
        .arg("--image")
        .arg(&image_path);
        
    let run_output = run_cmd.output().expect("failed to run compiled binary");
    assert!(run_output.status.success(), "compiled binary exited with failure");
    let host_stdout = String::from_utf8(run_output.stdout).unwrap();
    
    let mut host_class = None;
    let mut host_prob = None;
    for line in host_stdout.lines() {
        if line.starts_with("top-1 class :") {
            let class_str = line.split(':').nth(1).unwrap().trim();
            host_class = Some(class_str.parse::<i32>().unwrap());
        } else if line.starts_with("probability :") {
            let prob_str = line.split(':').nth(1).unwrap().trim();
            host_prob = Some(prob_str.parse::<f32>().unwrap());
        }
    }
    
    let host_class = host_class.expect("missing top-1 class in host output");
    let host_prob = host_prob.expect("missing probability in host output");
    
    let python_bin = ws_root.join("starkc/tmp/venv/bin/python3");
    let reference_script = ws_root.join("starkc/tests/fixtures/gate5/reference.py");
    
    let mut py_cmd = Command::new(&python_bin);
    py_cmd
        .arg(&reference_script)
        .arg(&model_path)
        .arg(&image_path);
        
    let py_output = py_cmd.output().expect("failed to run Python reference");
    assert!(py_output.status.success(), "Python reference script failed");
    let py_stdout = String::from_utf8(py_output.stdout).unwrap();
    
    let mut py_class = None;
    let mut py_prob = None;
    for line in py_stdout.lines() {
        if line.starts_with("top-1 class :") {
            let class_str = line.split(':').nth(1).unwrap().trim();
            py_class = Some(class_str.parse::<i32>().unwrap());
        } else if line.starts_with("probability :") {
            let prob_str = line.split(':').nth(1).unwrap().trim();
            py_prob = Some(prob_str.parse::<f32>().unwrap());
        }
    }
    
    let py_class = py_class.expect("missing top-1 class in Python output");
    let py_prob = py_prob.expect("missing probability in Python output");
    
    println!("Host: class={}, prob={}", host_class, host_prob);
    println!("Python: class={}, prob={}", py_class, py_prob);
    
    assert_eq!(host_class, py_class, "top-1 class mismatch");
    let diff = (host_prob - py_prob).abs();
    assert!(diff <= 1e-3, "probability mismatch: host={} vs python={}, diff={}", host_prob, py_prob, diff);
}
