# TensorLib API Specification

The TensorLib is STARK's core library for tensor operations, providing high-performance, hardware-accelerated tensor computation with automatic differentiation support.

## Core Tensor Type

```stark
// Core tensor type with compile-time shape checking
type Tensor<T, Shape> where T: Numeric, Shape: TensorShape {
    data: *T
    shape: Shape
    strides: [i64]
    device: Device
    requires_grad: bool
    grad: ?Tensor<T, Shape>
}

// Device enumeration
enum Device {
    CPU,
    GPU(i32),      // GPU device ID
    TPU(i32),      // TPU device ID
    Custom(str)    // Custom device string
}

// Shape type for compile-time shape checking
trait TensorShape {
    fn rank() -> i32
    fn size() -> i64
    fn dims() -> [i32]
}
```

## Creation Functions

### Basic Creation

```stark
module tensor {
    // Create tensor filled with zeros
    fn zeros<T, S>(shape: S) -> Tensor<T, S> where T: Numeric, S: TensorShape
    fn zeros(shape: [i32], dtype: Type) -> Tensor<?, ?>
    
    // Create tensor filled with ones
    fn ones<T, S>(shape: S) -> Tensor<T, S> where T: Numeric, S: TensorShape
    fn ones(shape: [i32], dtype: Type) -> Tensor<?, ?>
    
    // Create tensor filled with specific value
    fn full<T, S>(shape: S, value: T) -> Tensor<T, S> where T: Numeric, S: TensorShape
    fn full(shape: [i32], value: f64, dtype: Type) -> Tensor<?, ?>
    
    // Create uninitialized tensor (faster but unsafe)
    fn empty<T, S>(shape: S) -> Tensor<T, S> where T: Numeric, S: TensorShape
    fn empty(shape: [i32], dtype: Type) -> Tensor<?, ?>
    
    // Create identity matrix
    fn eye<T>(n: i32) -> Tensor<T, [n, n]> where T: Numeric
    fn eye(n: i32, dtype: Type) -> Tensor<?, ?>
    
    // Create from array/list
    fn from_array<T>(data: [T]) -> Tensor<T, [?]> where T: Numeric
    fn from_nested<T>(data: [[T]]) -> Tensor<T, [?, ?]> where T: Numeric
    
    // Load from memory pointer (unsafe)
    unsafe fn from_ptr<T, S>(ptr: *T, shape: S) -> Tensor<T, S>
}
```

### Random Tensor Creation

```stark
module tensor::random {
    // Uniform random in [0, 1)
    fn rand<T, S>(shape: S) -> Tensor<T, S> where T: Float, S: TensorShape
    fn rand(shape: [i32], dtype: Type) -> Tensor<?, ?>
    
    // Uniform random in [low, high)
    fn uniform<T, S>(shape: S, low: T, high: T) -> Tensor<T, S> where T: Float, S: TensorShape
    
    // Normal distribution (mean=0, std=1)
    fn randn<T, S>(shape: S) -> Tensor<T, S> where T: Float, S: TensorShape
    
    // Normal distribution with specified mean and std
    fn normal<T, S>(shape: S, mean: T, std: T) -> Tensor<T, S> where T: Float, S: TensorShape
    
    // Random integers in [low, high)
    fn randint<T, S>(shape: S, low: T, high: T) -> Tensor<T, S> where T: Integer, S: TensorShape
    
    // Random permutation
    fn permutation<T>(n: i32) -> Tensor<T, [n]> where T: Integer
    
    // Random choice from array
    fn choice<T>(array: Tensor<T, [?]>, size: i32, replace: bool = true) -> Tensor<T, [?]>
}
```

### Sequence Creation

```stark
module tensor::sequence {
    // Linear sequence [start, stop) with step
    fn arange<T>(start: T, stop: T, step: T = 1) -> Tensor<T, [?]> where T: Numeric
    
    // Linear sequence with num points
    fn linspace<T>(start: T, stop: T, num: i32) -> Tensor<T, [num]> where T: Float
    
    // Logarithmic sequence
    fn logspace<T>(start: T, stop: T, num: i32, base: T = 10.0) -> Tensor<T, [num]> where T: Float
    
    // Geometric sequence
    fn geomspace<T>(start: T, stop: T, num: i32) -> Tensor<T, [num]> where T: Float
}
```

## Shape Operations

```stark
impl<T, S> Tensor<T, S> where T: Numeric, S: TensorShape {
    // Get tensor properties
    fn shape() -> S
    fn size() -> i64
    fn rank() -> i32
    fn dtype() -> Type
    fn device() -> Device
    fn requires_grad() -> bool
    
    // Reshape tensor (compile-time shape checking when possible)
    fn reshape<NewS>(new_shape: NewS) -> Tensor<T, NewS> where NewS: TensorShape
    fn reshape(new_shape: [i32]) -> Tensor<T, ?>
    
    // View with different shape (no copy)
    fn view<NewS>(new_shape: NewS) -> Tensor<T, NewS> where NewS: TensorShape
    fn view(new_shape: [i32]) -> Tensor<T, ?>
    
    // Flatten to 1D
    fn flatten() -> Tensor<T, [?]>
    fn flatten(start_dim: i32, end_dim: i32) -> Tensor<T, ?>
    
    // Squeeze dimensions of size 1
    fn squeeze() -> Tensor<T, ?>
    fn squeeze(dim: i32) -> Tensor<T, ?>
    
    // Add dimension of size 1
    fn unsqueeze(dim: i32) -> Tensor<T, ?>
    
    // Transpose operations
    fn transpose(dim0: i32, dim1: i32) -> Tensor<T, ?>
    fn T() -> Tensor<T, ?> // 2D transpose shorthand
    fn permute(dims: [i32]) -> Tensor<T, ?>
    
    // Expand dimensions (broadcasting)
    fn expand<NewS>(new_shape: NewS) -> Tensor<T, NewS> where NewS: TensorShape
    fn expand_as<OtherS>(other: Tensor<?, OtherS>) -> Tensor<T, OtherS>
    
    // Repeat tensor along dimensions
    fn repeat(repeats: [i32]) -> Tensor<T, ?>
    fn tile(reps: [i32]) -> Tensor<T, ?>
}
```

## Indexing and Slicing

```stark
impl<T, S> Tensor<T, S> where T: Numeric, S: TensorShape {
    // Element access
    fn get(indices: [i32]) -> T
    fn set(indices: [i32], value: T)
    
    // Slice operations
    fn slice(dim: i32, start: i32, end: i32, step: i32 = 1) -> Tensor<T, ?>
    fn slice_all(ranges: [Range]) -> Tensor<T, ?>
    
    // Index with tensor
    fn index(indices: Tensor<i64, ?>) -> Tensor<T, ?>
    fn index_select(dim: i32, indices: Tensor<i64, [?]>) -> Tensor<T, ?>
    
    // Boolean indexing
    fn masked_select(mask: Tensor<bool, S>) -> Tensor<T, [?]>
    fn masked_fill(mask: Tensor<bool, S>, value: T) -> Tensor<T, S>
    
    // Gather and scatter operations
    fn gather(dim: i32, index: Tensor<i64, ?>) -> Tensor<T, ?>
    fn scatter(dim: i32, index: Tensor<i64, ?>, src: Tensor<T, ?>) -> Tensor<T, S>
    fn scatter_add(dim: i32, index: Tensor<i64, ?>, src: Tensor<T, ?>) -> Tensor<T, S>
    
    // Advanced indexing
    fn take(indices: Tensor<i64, [?]>) -> Tensor<T, [?]>
    fn put(indices: Tensor<i64, [?]>, values: Tensor<T, [?]>) -> Tensor<T, S>
}

// Range type for slicing
struct Range {
    start: ?i32,
    end: ?i32,
    step: i32 = 1
}

// Slice syntax sugar
fn range(start: ?i32, end: ?i32, step: i32 = 1) -> Range
```

## Arithmetic Operations

```stark
impl<T, S> Tensor<T, S> where T: Numeric, S: TensorShape {
    // Element-wise arithmetic
    fn add<OtherS>(other: Tensor<T, OtherS>) -> Tensor<T, ?> where OtherS: TensorShape
    fn sub<OtherS>(other: Tensor<T, OtherS>) -> Tensor<T, ?> where OtherS: TensorShape
    fn mul<OtherS>(other: Tensor<T, OtherS>) -> Tensor<T, ?> where OtherS: TensorShape
    fn div<OtherS>(other: Tensor<T, OtherS>) -> Tensor<T, ?> where OtherS: TensorShape
    fn mod<OtherS>(other: Tensor<T, OtherS>) -> Tensor<T, ?> where OtherS: TensorShape
    fn pow<OtherS>(other: Tensor<T, OtherS>) -> Tensor<T, ?> where OtherS: TensorShape
    
    // Scalar operations
    fn add_scalar(scalar: T) -> Tensor<T, S>
    fn sub_scalar(scalar: T) -> Tensor<T, S>
    fn mul_scalar(scalar: T) -> Tensor<T, S>
    fn div_scalar(scalar: T) -> Tensor<T, S>
    fn pow_scalar(scalar: T) -> Tensor<T, S>
    
    // In-place operations (mutating)
    fn add_<OtherS>(other: Tensor<T, OtherS>) -> Tensor<T, S>
    fn sub_<OtherS>(other: Tensor<T, OtherS>) -> Tensor<T, S>
    fn mul_<OtherS>(other: Tensor<T, OtherS>) -> Tensor<T, S>
    fn div_<OtherS>(other: Tensor<T, OtherS>) -> Tensor<T, S>
    
    // Unary operations
    fn neg() -> Tensor<T, S>
    fn abs() -> Tensor<T, S>
    fn sign() -> Tensor<T, S>
    fn sqrt() -> Tensor<T, S> where T: Float
    fn square() -> Tensor<T, S>
    fn reciprocal() -> Tensor<T, S> where T: Float
    
    // Clamping
    fn clamp(min: T, max: T) -> Tensor<T, S>
    fn clamp_min(min: T) -> Tensor<T, S>
    fn clamp_max(max: T) -> Tensor<T, S>
}

// Operator overloading
impl<T, S1, S2> Add<Tensor<T, S2>> for Tensor<T, S1> where T: Numeric {
    type Output = Tensor<T, ?>
    fn add(self, other: Tensor<T, S2>) -> Self::Output
}

// Similar for -, *, /, %, **
```

## Linear Algebra

```stark
module tensor::linalg {
    // Matrix multiplication
    fn matmul<T>(a: Tensor<T, [?, ?]>, b: Tensor<T, [?, ?]>) -> Tensor<T, [?, ?]> where T: Numeric
    fn bmm<T>(a: Tensor<T, [?, ?, ?]>, b: Tensor<T, [?, ?, ?]>) -> Tensor<T, [?, ?, ?]> // Batch matmul
    
    // Dot products
    fn dot<T>(a: Tensor<T, [?]>, b: Tensor<T, [?]>) -> T where T: Numeric
    fn vdot<T>(a: Tensor<T, [?]>, b: Tensor<T, [?]>) -> T where T: Float // Complex conjugate dot
    
    // Matrix operations
    fn det<T>(matrix: Tensor<T, [?, ?]>) -> T where T: Float
    fn inv<T>(matrix: Tensor<T, [?, ?]>) -> Tensor<T, [?, ?]> where T: Float
    fn pinv<T>(matrix: Tensor<T, [?, ?]>) -> Tensor<T, [?, ?]> where T: Float // Pseudo-inverse
    fn trace<T>(matrix: Tensor<T, [?, ?]>) -> T where T: Numeric
    
    // Decompositions
    fn svd<T>(matrix: Tensor<T, [?, ?]>) -> (Tensor<T, [?, ?]>, Tensor<T, [?]>, Tensor<T, [?, ?]>) where T: Float
    fn eig<T>(matrix: Tensor<T, [?, ?]>) -> (Tensor<T, [?]>, Tensor<T, [?, ?]>) where T: Float
    fn qr<T>(matrix: Tensor<T, [?, ?]>) -> (Tensor<T, [?, ?]>, Tensor<T, [?, ?]>) where T: Float
    fn cholesky<T>(matrix: Tensor<T, [?, ?]>) -> Tensor<T, [?, ?]> where T: Float
    fn lu<T>(matrix: Tensor<T, [?, ?]>) -> (Tensor<T, [?, ?]>, Tensor<T, [?, ?]>) where T: Float
    
    // Norms
    fn norm<T>(tensor: Tensor<T, ?>, p: f64 = 2.0, dim: ?i32 = null, keepdim: bool = false) -> Tensor<T, ?>
    fn frobenius_norm<T>(matrix: Tensor<T, [?, ?]>) -> T where T: Float
    
    // Solving linear systems
    fn solve<T>(A: Tensor<T, [?, ?]>, b: Tensor<T, [?]>) -> Tensor<T, [?]> where T: Float
    fn lstsq<T>(A: Tensor<T, [?, ?]>, b: Tensor<T, [?]>) -> Tensor<T, [?]> where T: Float
}

// @ operator for matrix multiplication
impl<T> MatMul<Tensor<T, [?, ?]>> for Tensor<T, [?, ?]> where T: Numeric {
    type Output = Tensor<T, [?, ?]>
    fn matmul(self, other: Tensor<T, [?, ?]>) -> Self::Output
}
```

## Mathematical Functions

```stark
module tensor::math {
    // Trigonometric functions
    fn sin<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
    fn cos<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
    fn tan<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
    fn asin<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
    fn acos<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
    fn atan<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
    fn atan2<T, S>(y: Tensor<T, S>, x: Tensor<T, S>) -> Tensor<T, S> where T: Float
    
    // Hyperbolic functions
    fn sinh<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
    fn cosh<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
    fn tanh<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
    fn asinh<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
    fn acosh<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
    fn atanh<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
    
    // Exponential and logarithmic
    fn exp<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
    fn exp2<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
    fn expm1<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float // exp(x) - 1
    fn log<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
    fn log2<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
    fn log10<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
    fn log1p<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float // log(1 + x)
    
    // Power and roots
    fn pow<T, S>(base: Tensor<T, S>, exp: Tensor<T, S>) -> Tensor<T, S> where T: Float
    fn sqrt<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
    fn rsqrt<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float // 1/sqrt(x)
    fn cbrt<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
    
    // Rounding
    fn floor<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
    fn ceil<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
    fn round<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
    fn trunc<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
    fn frac<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float // Fractional part
    
    // Special functions
    fn erf<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
    fn erfc<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
    fn gamma<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
    fn lgamma<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
    fn digamma<T, S>(tensor: Tensor<T, S>) -> Tensor<T, S> where T: Float
}
```

## Reduction Operations

```stark
impl<T, S> Tensor<T, S> where T: Numeric, S: TensorShape {
    // Sum operations
    fn sum() -> T
    fn sum(dim: i32, keepdim: bool = false) -> Tensor<T, ?>
    fn sum(dims: [i32], keepdim: bool = false) -> Tensor<T, ?>
    fn cumsum(dim: i32) -> Tensor<T, S>
    
    // Product operations
    fn prod() -> T
    fn prod(dim: i32, keepdim: bool = false) -> Tensor<T, ?>
    fn cumprod(dim: i32) -> Tensor<T, S>
    
    // Mean and statistics
    fn mean() -> T where T: Float
    fn mean(dim: i32, keepdim: bool = false) -> Tensor<T, ?> where T: Float
    fn var(unbiased: bool = true) -> T where T: Float
    fn var(dim: i32, unbiased: bool = true, keepdim: bool = false) -> Tensor<T, ?> where T: Float
    fn std(unbiased: bool = true) -> T where T: Float
    fn std(dim: i32, unbiased: bool = true, keepdim: bool = false) -> Tensor<T, ?> where T: Float
    
    // Min/max operations
    fn min() -> T
    fn min(dim: i32, keepdim: bool = false) -> (Tensor<T, ?>, Tensor<i64, ?>) // Values and indices
    fn max() -> T
    fn max(dim: i32, keepdim: bool = false) -> (Tensor<T, ?>, Tensor<i64, ?>) // Values and indices
    fn argmin() -> i64
    fn argmin(dim: i32, keepdim: bool = false) -> Tensor<i64, ?>
    fn argmax() -> i64
    fn argmax(dim: i32, keepdim: bool = false) -> Tensor<i64, ?>
    
    // Quantiles and percentiles
    fn median() -> T
    fn median(dim: i32, keepdim: bool = false) -> (Tensor<T, ?>, Tensor<i64, ?>)
    fn quantile(q: f64) -> T
    fn quantile(q: f64, dim: i32, keepdim: bool = false) -> Tensor<T, ?>
    
    // Logical reductions (for boolean tensors)
    fn all() -> bool where T: Bool
    fn all(dim: i32, keepdim: bool = false) -> Tensor<bool, ?> where T: Bool
    fn any() -> bool where T: Bool
    fn any(dim: i32, keepdim: bool = false) -> Tensor<bool, ?> where T: Bool
    
    // Unique values
    fn unique(sorted: bool = true, return_inverse: bool = false, return_counts: bool = false) 
        -> (Tensor<T, [?]>, ?Tensor<i64, [?]>, ?Tensor<i64, [?]>)
}
```

## Comparison Operations

```stark
impl<T, S> Tensor<T, S> where T: Numeric, S: TensorShape {
    // Element-wise comparisons
    fn eq<OtherS>(other: Tensor<T, OtherS>) -> Tensor<bool, ?> where OtherS: TensorShape
    fn ne<OtherS>(other: Tensor<T, OtherS>) -> Tensor<bool, ?> where OtherS: TensorShape
    fn lt<OtherS>(other: Tensor<T, OtherS>) -> Tensor<bool, ?> where OtherS: TensorShape
    fn le<OtherS>(other: Tensor<T, OtherS>) -> Tensor<bool, ?> where OtherS: TensorShape
    fn gt<OtherS>(other: Tensor<T, OtherS>) -> Tensor<bool, ?> where OtherS: TensorShape
    fn ge<OtherS>(other: Tensor<T, OtherS>) -> Tensor<bool, ?> where OtherS: TensorShape
    
    // Scalar comparisons
    fn eq_scalar(scalar: T) -> Tensor<bool, S>
    fn ne_scalar(scalar: T) -> Tensor<bool, S>
    fn lt_scalar(scalar: T) -> Tensor<bool, S>
    fn le_scalar(scalar: T) -> Tensor<bool, S>
    fn gt_scalar(scalar: T) -> Tensor<bool, S>
    fn ge_scalar(scalar: T) -> Tensor<bool, S>
    
    // Utility comparisons
    fn isnan() -> Tensor<bool, S> where T: Float
    fn isinf() -> Tensor<bool, S> where T: Float
    fn isfinite() -> Tensor<bool, S> where T: Float
    fn isclose<OtherS>(other: Tensor<T, OtherS>, rtol: T = 1e-5, atol: T = 1e-8) -> Tensor<bool, ?> where T: Float
    
    // Sorting
    fn sort(dim: i32 = -1, descending: bool = false) -> (Tensor<T, S>, Tensor<i64, S>)
    fn argsort(dim: i32 = -1, descending: bool = false) -> Tensor<i64, S>
    fn topk(k: i32, dim: i32 = -1, largest: bool = true, sorted: bool = true) 
        -> (Tensor<T, ?>, Tensor<i64, ?>)
}
```

## Automatic Differentiation

```stark
impl<T, S> Tensor<T, S> where T: Float, S: TensorShape {
    // Gradient computation
    fn requires_grad(requires_grad: bool = true) -> Tensor<T, S>
    fn detach() -> Tensor<T, S> // Remove from computation graph
    fn backward(gradient: ?Tensor<T, S> = null, retain_graph: bool = false, create_graph: bool = false)
    
    // Gradient properties
    fn grad() -> ?Tensor<T, S>
    fn grad_fn() -> ?GradFn
    fn is_leaf() -> bool
    
    // Context managers for gradient computation
    fn no_grad<R>(f: fn() -> R) -> R // Disable gradients
    fn enable_grad<R>(f: fn() -> R) -> R // Enable gradients
    
    // Higher-order gradients
    fn jacobian<InputS>(input: Tensor<T, InputS>) -> Tensor<T, ?> where InputS: TensorShape
    fn hessian<InputS>(input: Tensor<T, InputS>) -> Tensor<T, ?> where InputS: TensorShape
}

// Gradient function trait
trait GradFn {
    fn apply(grad_output: Tensor<?, ?>) -> [Tensor<?, ?>]
    fn next_functions() -> [?GradFn]
}
```

## Device Management

```stark
impl<T, S> Tensor<T, S> where T: Numeric, S: TensorShape {
    // Device operations
    fn to_device(device: Device) -> Tensor<T, S>
    fn cpu() -> Tensor<T, S>
    fn cuda(device: ?i32 = null) -> Tensor<T, S>
    fn device() -> Device
    fn is_cpu() -> bool
    fn is_cuda() -> bool
    
    // Data type conversion
    fn to_dtype<NewT>(dtype: Type) -> Tensor<NewT, S> where NewT: Numeric
    fn float() -> Tensor<f32, S>
    fn double() -> Tensor<f64, S>
    fn int() -> Tensor<i32, S>
    fn long() -> Tensor<i64, S>
    fn bool() -> Tensor<bool, S>
    
    // Memory operations
    fn pin_memory() -> Tensor<T, S> // Pin memory for faster CPU-GPU transfer
    fn is_pinned() -> bool
    fn share_memory() -> Tensor<T, S> // Enable sharing between processes
    fn is_shared() -> bool
}

module tensor::cuda {
    fn is_available() -> bool
    fn device_count() -> i32
    fn current_device() -> i32
    fn set_device(device: i32)
    fn get_device_properties(device: i32) -> DeviceProperties
    fn synchronize()
    fn empty_cache() // Clear GPU memory cache
}

struct DeviceProperties {
    name: str,
    major: i32,
    minor: i32,
    total_memory: i64,
    multi_processor_count: i32,
    max_threads_per_block: i32,
    max_block_dimensions: [i32; 3],
    max_grid_dimensions: [i32; 3]
}
```

## Serialization and I/O

```stark
impl<T, S> Tensor<T, S> where T: Numeric, S: TensorShape {
    // Save/load operations
    fn save(path: str) -> Result<(), IOError>
    fn load(path: str) -> Result<Tensor<?, ?>, IOError>
    
    // NumPy compatibility
    fn to_numpy() -> numpy.ndarray // Requires numpy interop
    fn from_numpy(array: numpy.ndarray) -> Tensor<?, ?>
    
    // PyTorch compatibility
    fn to_torch() -> torch.Tensor // Requires PyTorch interop
    fn from_torch(tensor: torch.Tensor) -> Tensor<?, ?>
    
    // Memory views
    fn as_bytes() -> [u8]
    fn from_bytes<NewT, NewS>(bytes: [u8], shape: NewS, dtype: Type) -> Tensor<NewT, NewS>
    
    // Cloning and copying
    fn clone() -> Tensor<T, S> // Deep copy
    fn copy_from<OtherS>(other: Tensor<T, OtherS>) -> Tensor<T, S> // Copy data
}

module tensor::io {
    // File format support
    fn save_npz(path: str, tensors: Map<str, Tensor<?, ?>>) -> Result<(), IOError>
    fn load_npz(path: str) -> Result<Map<str, Tensor<?, ?>>, IOError>
    
    fn save_hdf5(path: str, tensors: Map<str, Tensor<?, ?>>) -> Result<(), IOError>
    fn load_hdf5(path: str) -> Result<Map<str, Tensor<?, ?>>, IOError>
    
    // Binary formats
    fn save_safetensors(path: str, tensors: Map<str, Tensor<?, ?>>) -> Result<(), IOError>
    fn load_safetensors(path: str) -> Result<Map<str, Tensor<?, ?>>, IOError>
}
```

## Performance and Memory

```stark
module tensor::memory {
    // Memory statistics
    fn allocated_memory(device: ?Device = null) -> i64
    fn cached_memory(device: ?Device = null) -> i64
    fn max_memory_allocated(device: ?Device = null) -> i64
    fn memory_stats(device: ?Device = null) -> MemoryStats
    
    // Memory management
    fn empty_cache(device: ?Device = null)
    fn set_memory_fraction(fraction: f64, device: ?Device = null)
    fn memory_snapshot(device: ?Device = null) -> MemorySnapshot
}

struct MemoryStats {
    allocated_bytes: i64,
    active_bytes: i64,
    inactive_split_bytes: i64,
    reserved_bytes: i64,
    num_alloc_retries: i64,
    num_ooms: i64
}

module tensor::profiler {
    // Performance profiling
    fn profile<R>(enabled: bool, f: fn() -> R) -> (R, ProfileResult)
    fn benchmark<R>(warmup: i32, iterations: i32, f: fn() -> R) -> BenchmarkResult
    
    // Kernel timing
    fn time_kernel<R>(f: fn() -> R) -> (R, f64) // Returns result and time in seconds
}

struct BenchmarkResult {
    mean_time: f64,
    std_time: f64,
    min_time: f64,
    max_time: f64,
    iterations: i32
}
```

## Error Handling

```stark
// Tensor-specific errors
enum TensorError {
    ShapeMismatch { expected: [i32], actual: [i32] },
    DeviceMismatch { expected: Device, actual: Device },
    DTypeMismatch { expected: Type, actual: Type },
    IndexOutOfBounds { index: [i32], shape: [i32] },
    SingularMatrix,
    IncompatibleShapes { op: str, shapes: [[i32]] },
    OutOfMemory { requested: i64, available: i64 },
    CudaError { code: i32, message: str },
    InvalidOperation { op: str, reason: str }
}

// Result types for fallible operations
type TensorResult<T> = Result<T, TensorError>
```

## Examples

```stark
// Basic tensor operations
fn example_basic() {
    let a = tensor::rand([1000, 1000])@gpu  // Random tensor on GPU
    let b = tensor::ones([1000, 1000])@gpu   // Ones tensor on GPU
    
    let c = a @ b  // Matrix multiplication
    let d = (a + b).sum(dim: 0)  // Element-wise add and sum
    
    print(f"Result shape: {c.shape()}")
    print(f"Sum: {d}")
}

// Automatic differentiation
fn example_autograd() {
    let x = tensor::rand([10, 5]).requires_grad(true)
    let y = tensor::rand([5, 3]).requires_grad(true)
    
    let z = (x @ y).sum()
    z.backward()
    
    print(f"x.grad: {x.grad()}")
    print(f"y.grad: {y.grad()}")
}

// Advanced indexing
fn example_indexing() {
    let data = tensor::arange(0, 24).reshape([4, 6])
    
    // Slice operations
    let sub1 = data[1:3, 2:5]  // Slice rows 1-2, columns 2-4
    let sub2 = data[range(1, 3), range(2, 5)]  // Same as above
    
    // Boolean indexing
    let mask = data > 10
    let filtered = data.masked_select(mask)
    
    // Advanced indexing
    let indices = tensor::from_array([0, 2, 3])
    let selected = data.index_select(dim: 0, indices: indices)
}
```

This comprehensive TensorLib API provides:

1. **Type Safety**: Compile-time shape checking where possible
2. **Performance**: GPU acceleration and memory optimization
3. **Compatibility**: Interop with NumPy, PyTorch, and other frameworks
4. **Automatic Differentiation**: Full gradient computation support
5. **Broadcasting**: NumPy-compatible broadcasting semantics
6. **Error Handling**: Comprehensive error types and recovery
7. **Device Management**: Seamless CPU/GPU/TPU operations
8. **Memory Management**: Fine-grained control over memory allocation