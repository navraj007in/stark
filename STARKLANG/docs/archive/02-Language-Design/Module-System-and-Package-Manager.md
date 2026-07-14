# Module System & Package Manager Specification

The STARK module system provides a hierarchical namespace organization with explicit import/export declarations, semantic versioning, and a powerful package manager for dependency resolution. It emphasizes reproducible builds, security, and seamless integration with AI/ML workflows.

## Module System Design

### Module Definition and Structure

```stark
// Module declaration (optional for file-level modules)
module ml::vision::detection;

// Imports from standard library
use std::tensor::{Tensor, Device};
use std::dataset::{Dataset, DataLoader};
use std::model::{Model, Loss, Optimizer};

// Imports from external packages
use opencv::{Mat, imread, imwrite};
use torchvision::transforms::{Resize, Normalize};

// Imports from local modules
use super::backbone::{ResNet, Backbone};
use crate::utils::{Timer, Logger};

// Module contents
pub struct YOLODetector {
    backbone: Box<dyn Backbone>,
    head: DetectionHead,
    anchors: Tensor<f32, [?, 2]>
}

impl YOLODetector {
    pub fn new(num_classes: i32) -> Self { ... }
    
    pub fn detect(image: Tensor<f32, [3, 640, 640]>) -> [Detection] { ... }
    
    // Private function (not exported)
    fn nms(boxes: [BoundingBox], threshold: f32) -> [BoundingBox] { ... }
}

// Export specific items
pub use backbone::ResNet50;
pub use utils::Timer as PerfTimer;

// Re-export with renaming
pub use external_crate::Model as BaseModel;
```

### Module Hierarchy and File Organization

```
src/
├── lib.stark              // Root module (defines crate public API)
├── ml/
│   ├── mod.stark          // ml module declaration
│   ├── vision/
│   │   ├── mod.stark      // ml::vision module
│   │   ├── detection.stark // ml::vision::detection
│   │   ├── classification.stark
│   │   └── segmentation.stark
│   ├── nlp/
│   │   ├── mod.stark
│   │   ├── tokenizers.stark
│   │   └── transformers.stark
│   └── audio/
│       ├── mod.stark
│       └── preprocessing.stark
├── utils/
│   ├── mod.stark
│   ├── logging.stark
│   └── metrics.stark
└── examples/
    ├── yolo_detection.stark
    └── sentiment_analysis.stark
```

### Import and Export Syntax

#### Import Declarations

```stark
// Basic imports
use std::tensor::Tensor;                    // Import single item
use std::dataset::{Dataset, DataLoader};   // Import multiple items
use std::model::*;                          // Import all public items (discouraged)

// Qualified imports
use std::tensor;                            // Import module, use as tensor::Tensor
use std::dataset as data;                   // Import with alias

// Relative imports
use super::backbone::ResNet;                // Parent module
use crate::utils::Logger;                   // Crate root
use self::submodule::Helper;                // Current module's submodule

// Conditional imports (platform/feature specific)
#[cfg(feature = "gpu")]
use cuda::kernels::*;

#[cfg(target_os = "linux")]
use linux_specific::optimizations;

// External package imports
use numpy as np;                            // Python interop
use opencv::{Mat, Size};                    // C++ library binding
use @huggingface/transformers::{AutoModel, AutoTokenizer};  // External registry

// Version-specific imports
use torch@">=1.13,<2.0"::{nn, optim};     // Version constraints
use tensorflow@"2.12.0"::{keras};          // Exact version

// Optional imports with fallbacks
use gpu_accelerated::fast_conv2d catch std::tensor::conv2d;
```

#### Export Declarations

```stark
// Public exports (available to external crates)
pub fn train_model() -> Model { ... }
pub struct NeuralNetwork { ... }
pub enum ActivationFunction { ... }
pub const DEFAULT_LEARNING_RATE: f32 = 0.001;

// Conditional exports
#[cfg(feature = "experimental")]
pub fn experimental_feature() { ... }

// Re-exports from submodules
pub use vision::{Detection, Classification, Segmentation};
pub use nlp::transformers::{BERT, GPT, T5};

// Renamed exports
pub use internal::ComplexName as SimpleName;

// Macro exports
#[macro_export]
macro_rules! define_layer {
    // Macro definition
}

// Trait exports with implementations
pub trait Trainable {
    fn train(&mut self, data: Dataset) -> Result<Metrics, TrainingError>;
}

// Module-level visibility modifiers
pub(crate) fn internal_api() { ... }        // Visible within crate
pub(super) fn parent_accessible() { ... }   // Visible to parent module
pub(in crate::ml) fn ml_only() { ... }     // Visible within ml module tree
```

#### Module Resolution Rules

```stark
// Resolution order for `use ml::vision::detection::YOLOv5`:

1. Current crate modules:
   src/ml/vision/detection.stark → YOLOv5

2. Standard library:
   std::ml::vision::detection::YOLOv5

3. External dependencies (Package.stark):
   ml-toolkit::vision::detection::YOLOv5

4. Global package registry:
   @pytorch/vision::detection::YOLOv5

// Disambiguation with explicit paths
use crate::ml::vision::detection::YOLOv5;     // Local module
use std::ml::vision::detection::YOLOv5;       // Standard library
use pytorch_vision::detection::YOLOv5;        // External dependency
use @pytorch/vision::detection::YOLOv5;       // Registry package
```

## Package Manager Design

### Package.stark Manifest Format

```stark
// Package.stark - Project manifest file
[package]
name = "stark-cv"
version = "0.3.1"
description = "Computer Vision library for STARK"
authors = ["AI Research Team <ai@company.com>"]
license = "MIT OR Apache-2.0"
homepage = "https://github.com/company/stark-cv"
repository = "https://github.com/company/stark-cv.git"
documentation = "https://docs.company.com/stark-cv"
readme = "README.md"
keywords = ["computer-vision", "ai", "ml", "deep-learning"]
categories = ["machine-learning", "computer-vision"]
edition = "2024"
rust-version = "1.70"  // Minimum STARK version

// Package metadata
[package.metadata]
maintainers = ["John Doe <john@company.com>"]
funding = "https://github.com/sponsors/company"
ci = "https://github.com/company/stark-cv/actions"
msrv-policy = "latest-stable"

// Binary targets
[[bin]]
name = "stark-cv-cli"
path = "src/bin/cli.stark"
required-features = ["cli"]

[[bin]]
name = "inference-server"
path = "src/bin/server.stark"

// Library configuration
[lib]
name = "stark_cv"
path = "src/lib.stark"
crate-type = ["cdylib", "rlib"]  // Both dynamic and static linking

// Dependencies
[dependencies]
# Standard dependencies
stark-std = "1.0"
tensor-lib = "2.1.0"
dataset-lib = "1.8"

# External dependencies with version constraints
opencv = { version = "4.8", features = ["contrib", "opencv_dnn"] }
numpy = "1.24"
torch = { version = ">=2.0,<3.0", optional = true }

# Git dependencies
ml-commons = { git = "https://github.com/ml-commons/stark", branch = "main" }
internal-tools = { git = "ssh://git@internal.com/ml/tools.git", tag = "v1.2.0" }

# Path dependencies (local development)
shared-models = { path = "../shared-models" }

# Registry dependencies
transformers = { registry = "huggingface", version = "4.30" }
vision-models = { registry = "pytorch", version = "0.15.0" }

# Platform-specific dependencies
[target.'cfg(target_os = "linux")'.dependencies]
cuda-toolkit = "11.8"
tensorrt = { version = "8.6", optional = true }

[target.'cfg(target_os = "macos")'.dependencies]
metal-performance-shaders = "14.0"

[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = "0.2"
web-sys = "0.3"

# Development dependencies (for tests, examples, benchmarks)
[dev-dependencies]
stark-test = "1.0"
criterion = "0.5"  // Benchmarking
proptest = "1.2"   // Property-based testing
tempfile = "3.0"

# Build dependencies (for build scripts)
[build-dependencies]
cmake = "0.1"
pkg-config = "0.3"
cc = "1.0"

// Feature flags
[features]
default = ["std", "tensor-ops"]

# Core features
std = ["stark-std/std"]
no-std = []
tensor-ops = ["tensor-lib/ops"]

# Optional ML frameworks
pytorch = ["torch", "torchvision"]
tensorflow = ["tensorflow-rs"]
onnx = ["onnxruntime"]

# Hardware acceleration
gpu = ["cuda", "opencl"]
cuda = ["cuda-toolkit", "cuDNN"]
opencl = ["opencl-sys"]
metal = ["metal-performance-shaders"]
tensorrt = ["tensorrt-sys"]

# Optimization levels
fast = ["tensor-ops/simd", "gpu"]
experimental = ["unstable-features"]

# Python interoperability
python = ["pyo3", "numpy"]

# Web assembly support
wasm = ["wasm-bindgen", "web-sys"]

# CLI tools
cli = ["clap", "colored", "indicatif"]

// Workspace configuration (for multi-package projects)
[workspace]
members = [
    "packages/stark-cv-core",
    "packages/stark-cv-models", 
    "packages/stark-cv-datasets",
    "packages/stark-cv-cli"
]

exclude = [
    "examples/*",
    "benchmarks/*"
]

resolver = "2"

// Workspace dependencies (shared across workspace members)
[workspace.dependencies]
stark-std = "1.0"
tensor-lib = "2.1.0"
serde = "1.0"

// Workspace metadata
[workspace.metadata.release]
tag-prefix = "v"
sign-tag = true
pre-release-commit-message = "Release {{version}}"
post-release-commit-message = "Bump version to {{next_version}}"

// Profile configurations
[profile.dev]
opt-level = 0
debug = 2
debug-assertions = true
overflow-checks = true
lto = false
panic = "unwind"
incremental = true

[profile.release]
opt-level = 3
debug = 0
debug-assertions = false
overflow-checks = false
lto = "fat"
panic = "abort"
codegen-units = 1
strip = "symbols"

[profile.test]
opt-level = 1
debug = 2
debug-assertions = true
overflow-checks = true

[profile.bench]
opt-level = 3
debug = 0
debug-assertions = false
overflow-checks = false
lto = true

// Custom profiles for specific use cases
[profile.gpu-optimized]
inherits = "release"
opt-level = 3
lto = "fat"
codegen-units = 1
target-cpu = "native"

[profile.memory-optimized]
inherits = "release"
opt-level = "s"  // Optimize for size
lto = "thin"
panic = "abort"

// Linting and code quality
[lints.stark]
unused-imports = "warn"
missing-docs = "warn"
unsafe-code = "forbid"

[lints.clippy]
all = "warn"
pedantic = "warn"
nursery = "warn"
cargo = "warn"

// Documentation configuration
[package.metadata.docs.rs]
features = ["std", "pytorch", "tensorflow"]
rustdoc-args = ["--cfg", "docsrs"]
targets = ["x86_64-unknown-linux-gnu", "x86_64-apple-darwin"]

// Benchmark configuration
[[bench]]
name = "tensor_operations"
harness = false
path = "benches/tensor_ops.stark"

[[bench]]
name = "model_inference"
harness = false
required-features = ["pytorch"]

// Example configurations
[[example]]
name = "image_classification"
path = "examples/classification.stark"
required-features = ["pytorch", "cli"]

[[example]]
name = "object_detection"
path = "examples/detection.stark"
required-features = ["opencv"]

// Integration test configurations  
[[test]]
name = "integration_tests"
path = "tests/integration_tests.stark"
required-features = ["test-utils"]
```

### Dependency Management and Versioning

#### Semantic Versioning (SemVer) Support

```stark
// Version specification formats in Package.stark

// Exact version
opencv = "4.8.0"

// SemVer ranges
tensor-lib = "^2.1.0"    // >=2.1.0, <3.0.0 (compatible)
dataset = "~1.8.0"       // >=1.8.0, <1.9.0 (bugfix updates)
ml-utils = ">=1.5,<2.0"  // Range specification

// Pre-release versions
experimental-ai = "3.0.0-alpha.1"
beta-features = "2.5.0-beta"
nightly = "1.0.0-nightly.20241126"

// Version with build metadata
custom-build = "1.2.3+build.456"

// Wildcard versions (discouraged in production)
dev-tools = "*"
test-utils = "1.*"

// Git-based versioning
ml-research = { git = "https://github.com/research/stark-ml", rev = "abc123" }
internal = { git = "...", tag = "v2.1.0" }
latest = { git = "...", branch = "main" }
```

#### Lock File Format (Package.lock)

```stark
// Package.lock - Exact dependency tree for reproducible builds
# This file is automatically generated by stark package manager.
# Do not edit manually.

version = 3

[[package]]
name = "stark-cv"
version = "0.3.1"
dependencies = [
    "opencv",
    "tensor-lib", 
    "dataset-lib"
]

[[package]]
name = "opencv"
version = "4.8.1"
source = "registry+https://packages.stark-lang.org/"
checksum = "sha256:1234567890abcdef..."
dependencies = [
    "opencv-sys"
]

[[package]]
name = "opencv-sys"
version = "0.88.0"
source = "registry+https://packages.stark-lang.org/"
checksum = "sha256:fedcba0987654321..."
dependencies = []

[[package]]
name = "tensor-lib"
version = "2.1.3"
source = "registry+https://packages.stark-lang.org/"
checksum = "sha256:abcdef1234567890..."
dependencies = [
    "blas-sys",
    "cuda-sys"
]

[[package]]
name = "ml-research"
version = "0.1.0"
source = "git+https://github.com/research/stark-ml#abc123def456"
dependencies = [
    "experimental-features"
]

[[package]]
name = "local-tools"
version = "0.2.0"
source = "path+file:///Users/dev/stark-cv/../local-tools"
dependencies = []

# Package metadata and resolution information
[metadata]
resolver = "stark-resolver-v2"
generated-at = "2024-11-26T10:30:00Z"
stark-version = "1.0.0"

# Checksums for security verification
[metadata.checksums]
"opencv 4.8.1" = "sha256:1234567890abcdef..."
"tensor-lib 2.1.3" = "sha256:abcdef1234567890..."
```

#### Package Registry Configuration

```stark
// .stark/config.toml - Global package manager configuration

[registries]
# Primary registry (default)
stark-lang = { index = "https://packages.stark-lang.org/", priority = 100 }

# Secondary registries
pytorch = { index = "https://packages.pytorch.org/stark/", priority = 90 }
huggingface = { index = "https://packages.huggingface.co/stark/", priority = 80 }
nvidia = { index = "https://packages.nvidia.com/stark/", priority = 70 }

# Private/corporate registries
internal = { 
    index = "https://packages.internal.company.com/",
    token-file = "~/.stark/tokens/internal",
    priority = 95
}

# Local registries for development
local-dev = { index = "file:///opt/stark-packages/", priority = 60 }

[registry.stark-lang]
token = "sk_1234567890abcdef"
default = true

[registry.internal]
username = "dev@company.com"
token-file = "~/.stark/tokens/internal"
ca-cert = "/etc/ssl/certs/company-ca.pem"

# Package sources configuration
[source]
# Replace specific packages with local versions
[source.local-opencv]
replace-with = "local-registry"
[source.local-registry]
local-registry = "/opt/local-packages"

# Patch dependencies from git
[patch.stark-lang]
tensor-lib = { git = "https://github.com/company/tensor-lib", branch = "optimization" }

[patch."https://github.com/ml-commons/stark"]
ml-commons = { path = "../local-ml-commons" }

# Build configuration
[build]
jobs = 8                      # Parallel build jobs
target-dir = "target"         # Build output directory
rustflags = ["-C", "target-cpu=native"]

[net]
retry = 3                     # Network request retries
git-fetch-with-cli = true     # Use git CLI for fetching
offline = false               # Offline mode

[profile]
# Override profile settings globally
[profile.dev.build-override]
opt-level = 1

# Cache configuration
[cache]
registry-index = "~/.stark/registry/index"
git-repos = "~/.stark/git"
packages = "~/.stark/packages"
max-size = "10GB"
cleanup-policy = "lru"        # Least recently used
```

#### Dependency Resolution Algorithm

```stark
// Dependency resolution follows these principles:

1. **Semantic Version Resolution**
   - Use highest compatible version within constraints
   - Prefer stable versions over pre-release
   - Respect minimum version requirements

2. **Conflict Resolution Strategy**
   - Multiple versions of same package are not allowed
   - Use "minimal version selection" algorithm
   - Fail fast on incompatible constraints

3. **Feature Unification**
   - Features are additive across the dependency graph
   - If any dependency enables a feature, it's enabled everywhere
   - Optional dependencies are only included if required

4. **Platform-Specific Resolution**
   - Resolve platform-specific dependencies based on target
   - Support cross-compilation scenarios
   - Handle conditional dependencies correctly

// Example resolution process:
Package A requires: opencv ^4.8.0
Package B requires: opencv ~4.8.1  
Package C requires: opencv >=4.8.0,<4.9.0

Resolution: opencv 4.8.1 (highest compatible version)

// Conflict example:
Package A requires: tensor-lib ^2.0.0
Package B requires: tensor-lib ^3.0.0

Resolution: ERROR - No compatible version found
```

## Package Manager Commands

### Core Commands

```bash
# Project initialization
stark new my-project                    # Create new project
stark new --lib my-library             # Create library project
stark new --bin my-binary              # Create binary project
stark init                             # Initialize in existing directory

# Dependency management
stark add opencv@4.8                   # Add dependency
stark add --dev criterion              # Add dev dependency
stark add --optional pytorch           # Add optional dependency
stark add --features "gpu,cuda" tensor-lib  # Add with specific features
stark remove opencv                    # Remove dependency
stark update                           # Update all dependencies
stark update opencv                    # Update specific dependency

# Building and testing
stark build                            # Build project
stark build --release                  # Release build
stark build --target wasm32-wasi       # Cross-compilation
stark test                             # Run tests
stark test --integration              # Run integration tests
stark bench                           # Run benchmarks
stark check                           # Check without building

# Documentation
stark doc                             # Generate documentation  
stark doc --open                      # Generate and open docs
stark doc --no-deps                   # Skip dependency docs

# Package publishing
stark login                           # Login to registry
stark logout                          # Logout from registry
stark publish                         # Publish to registry
stark publish --registry internal     # Publish to specific registry
stark yank 1.0.0                     # Yank published version
stark owner --add user@email.com      # Add package owner

# Package information
stark search "computer vision"         # Search packages
stark info opencv                     # Package information
stark deps                           # Show dependency tree
stark deps --duplicates              # Show duplicate dependencies
stark licenses                       # Show dependency licenses
stark audit                          # Security vulnerability audit

# Registry management
stark registry add pytorch https://packages.pytorch.org/
stark registry remove pytorch
stark registry list

# Maintenance commands
stark clean                           # Clean build artifacts
stark cache --clear                   # Clear package cache
stark fix                            # Auto-fix issues
stark fmt                            # Format code
stark lint                           # Lint code
stark outdated                       # Check for outdated dependencies
```

### Advanced Package Manager Features

#### Feature Flags and Conditional Compilation

```stark
// Package.stark feature configuration
[features]
default = ["std"]

# Feature groups
std = ["tensor-lib/std", "dataset-lib/std"]
gpu = ["cuda", "opencl"]
ml-frameworks = ["pytorch", "tensorflow"]

# Hardware-specific features
cuda = ["cuda-toolkit", "cuDNN"]
opencl = ["opencl-sys"]
metal = ["metal-performance-shaders"]

# Optional dependencies
pytorch = ["torch", "torchvision"]
tensorflow = ["tensorflow-rs"]
onnx = ["onnxruntime"]

# Mutually exclusive features  
cpu-only = []
gpu-only = ["gpu"]

[dependencies.torch]
version = "2.0"
optional = true

[dependencies.tensorflow-rs]
version = "0.20"
optional = true

// Usage in code
#[cfg(feature = "pytorch")]
use torch::{nn, Tensor as TorchTensor};

#[cfg(feature = "tensorflow")]
use tensorflow::Graph;

#[cfg(all(feature = "gpu", target_os = "linux"))]
fn gpu_accelerated_training() { ... }

#[cfg(not(feature = "gpu"))]
fn cpu_fallback_training() { ... }
```

#### Workspace Management

```stark
// Multi-package workspace structure
workspace/
├── Package.stark          # Workspace manifest
├── packages/
│   ├── stark-cv-core/
│   │   └── Package.stark
│   ├── stark-cv-models/
│   │   └── Package.stark
│   ├── stark-cv-datasets/
│   │   └── Package.stark
│   └── stark-cv-cli/
│       └── Package.stark
├── examples/
├── benchmarks/
└── docs/

// Workspace Package.stark
[workspace]
members = [
    "packages/stark-cv-core",
    "packages/stark-cv-models",
    "packages/stark-cv-datasets", 
    "packages/stark-cv-cli"
]

resolver = "2"

# Shared dependencies across workspace
[workspace.dependencies]
stark-std = "1.0"
tensor-lib = "2.1.0"
serde = { version = "1.0", features = ["derive"] }

# Workspace metadata
[workspace.metadata]
authors = ["AI Team <ai@company.com>"]
license = "MIT"
repository = "https://github.com/company/stark-cv"

// Package-specific Package.stark inherits from workspace
[package]
name = "stark-cv-core"
version = "0.3.1"
workspace = true  # Inherit workspace metadata

[dependencies]
stark-std = { workspace = true }
tensor-lib = { workspace = true, features = ["gpu"] }
opencv = "4.8"

// Workspace commands
stark build                           # Build all packages
stark build -p stark-cv-core         # Build specific package
stark test --workspace              # Test all packages
stark publish --workspace           # Publish all packages
```

#### Package Templates and Scaffolding

```bash
# Project templates
stark new --template ml-project my-ai-app     # ML project template
stark new --template wasm-lib my-wasm-lib     # WebAssembly template
stark new --template cli-tool my-cli          # CLI application template

# Custom templates from git
stark new --template https://github.com/company/stark-template my-project

# Template with parameters
stark new --template ml-project \
  --param framework=pytorch \
  --param license=MIT \
  my-pytorch-project
```

## Security and Trust

### Package Verification

```stark
// Package signing and verification
[package.metadata.security]
signing-key = "ed25519:1234567890abcdef..."
require-signatures = true
trusted-publishers = [
    "stark-lang-team",
    "pytorch-team", 
    "nvidia"
]

// Security policies in .stark/config.toml
[security]
require-signatures = true
allow-git-dependencies = false
audit-on-install = true
max-package-size = "100MB"

blocked-packages = [
    "malicious-package@*",
    "vulnerable-lib@<1.2.0"
]

trusted-keys = [
    "ed25519:abcdef1234567890...",
    "rsa:fedcba0987654321..."
]

// Audit database integration
[audit]
database = "https://security.stark-lang.org/advisory-db"
ignore = [
    "STARK-2024-001",  # Ignore specific advisory
]
```

### Supply Chain Security

```bash
# Security commands
stark audit                           # Check for vulnerabilities
stark audit --fix                     # Auto-fix vulnerable dependencies
stark verify                         # Verify package signatures
stark sbom                           # Generate Software Bill of Materials
stark license-check                   # Verify license compatibility

# Example output
$ stark audit
    Checking 145 dependencies for security advisories
    Found 2 vulnerabilities:
    
    ┌─────────────────────────────────────────────────────────────┐
    │                        Advisory                              │
    │                        ━━━━━━━━                              │
    │ Package      ┆ tensor-lib                                    │
    │ Version      ┆ 2.0.1                                        │
    │ Advisory     ┆ STARK-2024-0123                              │
    │ Severity     ┆ High                                         │
    │ Description  ┆ Buffer overflow in tensor operations         │
    │ Solution     ┆ Upgrade to >= 2.0.2                         │
    └─────────────────────────────────────────────────────────────┘
    
    Run `stark audit --fix` to upgrade vulnerable dependencies.
```

This comprehensive Module System & Package Manager specification provides:

1. **Hierarchical Module Organization** - Clear namespace management with explicit imports/exports
2. **Rich Package Manifest** - Comprehensive Package.stark format supporting all modern package management features
3. **Semantic Versioning** - Full SemVer support with sophisticated dependency resolution
4. **Security-First Design** - Package signing, vulnerability scanning, and supply chain security
5. **Workspace Support** - Multi-package projects with shared dependencies and metadata
6. **Cross-Platform Support** - Platform-specific dependencies and conditional compilation
7. **Registry Ecosystem** - Multiple registry support with private/corporate registry integration
8. **Developer Experience** - Rich CLI commands and helpful tooling for package management

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Create Module System & Package Manager specification", "status": "completed", "priority": "high", "id": "1"}, {"content": "Define import/export syntax and resolution rules", "status": "completed", "priority": "high", "id": "2"}, {"content": "Design Package.stark manifest format", "status": "completed", "priority": "high", "id": "3"}, {"content": "Specify dependency management and versioning", "status": "completed", "priority": "high", "id": "4"}]