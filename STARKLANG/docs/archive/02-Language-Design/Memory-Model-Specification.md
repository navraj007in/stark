# Memory Model Specification

STARK employs a hybrid memory management system that combines ownership-based memory safety with garbage collection for high-level objects, optimized specifically for AI/ML workloads. The memory model provides zero-cost abstractions for performance-critical tensor operations while ensuring memory safety and preventing common vulnerabilities.

## Overview

### Memory Management Strategy

STARK uses a **dual-tier memory management system**:

1. **Stack-Allocated & Owned Memory** - For performance-critical data (tensors, primitives)
2. **Garbage-Collected Memory** - For high-level objects and complex data structures

This hybrid approach optimizes for:
- **Zero-cost tensor operations** with predictable memory layout
- **Memory safety** without runtime overhead for critical paths  
- **Productivity** with automatic memory management for complex objects
- **Interoperability** with C/C++/Python through controlled unsafe interfaces

```stark
// Stack-allocated and owned (zero-cost)
let tensor = Tensor::<f32, [1024, 1024]>::zeros();  // Stack metadata, owned data
let array = [1, 2, 3, 4, 5];                       // Stack-allocated array

// Garbage-collected (managed)
let model = torch::load_model("resnet50.pt");       // GC-managed complex object
let dataset = Dataset::from_csv("data.csv");        // GC-managed with lazy loading
let cache: Map<str, Model> = Map::new();           // GC-managed collections
```

## Ownership and Borrowing System

### Ownership Rules

STARK follows ownership principles similar to Rust but with relaxed rules for AI/ML ergonomics:

```stark
// Ownership rules:
// 1. Each value has exactly one owner
// 2. When the owner goes out of scope, the value is dropped
// 3. There can be multiple immutable borrows OR one mutable borrow
// 4. Borrows must be valid for their entire lifetime

struct TensorOwnership {
    data: Box<[f32]>,        // Owned heap allocation
    shape: [i32],            // Owned stack allocation
    device: Device           // Owned enum value
}

fn ownership_example() {
    let tensor1 = Tensor::rand([1000, 1000]);
    let tensor2 = tensor1;               // Move (tensor1 is no longer valid)
    // print(tensor1.shape());          // ERROR: tensor1 moved
    print(tensor2.shape());             // OK: tensor2 owns the data
    
    let tensor3 = tensor2.clone();      // Explicit deep copy
    print(tensor2.shape());             // OK: tensor2 still owns its data
    print(tensor3.shape());             // OK: tensor3 owns separate data
}
```

### Borrowing and References

```stark
// Immutable borrowing
fn immutable_borrows() {
    let tensor = Tensor::ones([512, 512]);
    let ref1 = &tensor;                 // Immutable borrow
    let ref2 = &tensor;                 // Multiple immutable borrows OK
    
    print(f"Shape: {ref1.shape()}");    // OK
    print(f"Device: {ref2.device()}");  // OK
    // tensor.fill_(0.0);               // ERROR: Cannot mutate while borrowed
}

// Mutable borrowing
fn mutable_borrows() {
    let mut tensor = Tensor::zeros([256, 256]);
    let ref_mut = &mut tensor;          // Mutable borrow
    // let ref2 = &tensor;              // ERROR: Cannot borrow while mutably borrowed
    
    ref_mut.fill_(1.0);                 // OK: Mutate through mutable reference
    // print(tensor.sum());             // ERROR: Cannot use tensor while borrowed
}

// Lifetime annotations for complex scenarios
fn lifetime_example<'a>(x: &'a Tensor<f32, [?, ?]>, y: &'a Tensor<f32, [?, ?]>) 
    -> &'a Tensor<f32, [?, ?]> {
    if x.sum() > y.sum() { x } else { y }
}
```

### Move Semantics and Copy Types

```stark
// Copy types (implement Copy trait)
#[derive(Copy, Clone)]
struct Point {
    x: f32,
    y: f32
}

// Move-only types (large or resource-owning)
struct Tensor<T, S> {
    data: Box<[T]>,
    shape: S,
    device: Device
}

fn move_vs_copy() {
    // Copy types
    let point1 = Point { x: 1.0, y: 2.0 };
    let point2 = point1;               // Copy (both point1 and point2 valid)
    print(f"{point1.x}, {point2.x}"); // OK
    
    // Move types
    let tensor1 = Tensor::rand([1000, 1000]);
    let tensor2 = tensor1;            // Move (tensor1 invalid)
    // print(tensor1.shape());        // ERROR: tensor1 moved
    print(tensor2.shape());           // OK
}

// Copy trait for small, stack-allocated types
trait Copy: Clone {}

impl Copy for i32 {}
impl Copy for f64 {}
impl Copy for bool {}
impl Copy for Device {}
impl<T: Copy, const N: usize> Copy for [T; N] where [T; N]: Sized {}

// Clone trait for explicit deep copying
trait Clone {
    fn clone(&self) -> Self;
}

impl<T: Clone, S: Clone> Clone for Tensor<T, S> {
    fn clone(&self) -> Self {
        Tensor {
            data: self.data.clone(),    // Deep copy of data
            shape: self.shape.clone(),
            device: self.device
        }
    }
}
```

### Ownership Transfer Patterns

```stark
// Function parameter patterns
fn take_ownership(tensor: Tensor<f32, [?, ?]>) {
    // Function owns tensor, will drop when function ends
}

fn borrow_immutable(tensor: &Tensor<f32, [?, ?]>) {
    // Function borrows tensor, cannot modify
}

fn borrow_mutable(tensor: &mut Tensor<f32, [?, ?]>) {
    // Function can modify tensor through reference
}

fn return_ownership() -> Tensor<f32, [1024, 1024]> {
    Tensor::zeros()  // Return ownership to caller
}

// Method patterns
impl<T, S> Tensor<T, S> {
    // Consuming method (takes ownership)
    fn into_raw(self) -> Box<[T]> {
        self.data
    }
    
    // Borrowing method (immutable)
    fn shape(&self) -> &S {
        &self.shape
    }
    
    // Borrowing method (mutable)
    fn fill_(&mut self, value: T) where T: Copy {
        for element in &mut self.data {
            *element = value;
        }
    }
    
    // Builder pattern (move and return)
    fn to_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }
}

// Usage patterns
fn ownership_patterns() {
    let tensor = Tensor::rand([512, 512]);
    
    // Method chaining with moves
    let gpu_tensor = tensor
        .to_device(Device::CUDA(0))
        .normalize()
        .transpose();
    
    // Reference methods
    print(f"Shape: {gpu_tensor.shape()}");
    
    // Mutable reference methods
    let mut mutable_tensor = gpu_tensor;
    mutable_tensor.fill_(0.0);
    
    // Consuming methods
    let raw_data = mutable_tensor.into_raw();
}
```

## Garbage Collection System

### Hybrid GC Design

STARK uses a **generational, concurrent, low-latency garbage collector** for managed objects:

```stark
// GC-managed types (automatically allocated on GC heap)
struct Model {
    layers: Vec<Layer>,           // GC-managed vector
    optimizer: Box<Optimizer>,   // GC-managed box
    metadata: Map<str, Any>      // GC-managed map
}

struct Dataset {
    data_source: DataSource,     // GC-managed trait object
    transforms: Vec<Transform>,  // GC-managed vector of closures
    cache: LRUCache<str, Tensor> // GC-managed cache
}

// Mixed ownership (owned data + GC references)
struct TrainingLoop {
    model: Gc<Model>,            // GC reference to model
    dataset: Gc<Dataset>,        // GC reference to dataset
    batch_size: i32,             // Stack-allocated primitive
    learning_rate: f32,          // Stack-allocated primitive
    current_batch: Tensor<f32, [?, ?]> // Owned tensor data
}
```

### Garbage Collection Modes

```stark
// GC configuration and control
module std::gc {
    // GC modes for different workloads
    enum GCMode {
        Throughput,      // Optimize for throughput (larger heaps, less frequent GC)
        LowLatency,      // Optimize for low latency (concurrent GC, small pauses)
        Memory,          // Optimize for memory usage (frequent GC, compact heaps)
        Training,        // Optimized for ML training (batch-aware collection)
        Inference        // Optimized for inference (predictable, low-latency)
    }
    
    // GC configuration
    struct GCConfig {
        mode: GCMode,
        max_heap_size: ?usize,          // None = unlimited
        young_gen_size: usize,          // Young generation size
        concurrent_threads: usize,       // Concurrent GC threads
        pause_target: Duration,         // Target maximum pause time
        throughput_target: f64          // Target throughput percentage
    }
    
    // GC control functions
    fn configure(config: GCConfig);
    fn collect();                       // Force full collection
    fn collect_young();                 // Force young generation collection
    fn disable();                       // Disable automatic collection
    fn enable();                        // Re-enable automatic collection
    
    // GC statistics and monitoring
    fn stats() -> GCStats;
    fn memory_usage() -> MemoryUsage;
    fn set_pause_callback(callback: fn(Duration));
}

struct GCStats {
    total_collections: u64,
    young_collections: u64,
    old_collections: u64,
    total_pause_time: Duration,
    average_pause_time: Duration,
    max_pause_time: Duration,
    bytes_allocated: u64,
    bytes_freed: u64,
    heap_size: usize,
    fragmentation: f64
}

// Training-specific GC integration
fn training_loop_with_gc() {
    // Configure GC for training workload
    gc::configure(GCConfig {
        mode: GCMode::Training,
        max_heap_size: Some(8 * 1024 * 1024 * 1024), // 8GB
        young_gen_size: 512 * 1024 * 1024,           // 512MB
        concurrent_threads: 4,
        pause_target: Duration::milliseconds(10),
        throughput_target: 0.95
    });
    
    for epoch in 0..100 {
        for batch in dataset.batches(batch_size) {
            // Training step with owned tensors (no GC pressure)
            let predictions = model.forward(&batch.inputs);
            let loss = loss_fn(&predictions, &batch.targets);
            
            // GC cleanup between batches
            if batch.id % 10 == 0 {
                gc::collect_young(); // Quick young generation cleanup
            }
        }
        
        // Full GC between epochs
        gc::collect();
        
        let stats = gc::stats();
        println(f"Epoch {epoch}: GC pause time {stats.average_pause_time:?}");
    }
}
```

### Smart Pointers and Reference Types

```stark
// Smart pointer types for GC integration
use std::gc::{Gc, Weak, RefCell, Cell};

// Garbage-collected pointer (like Rc but GC-managed)
struct Gc<T> {
    // Points to GC-managed memory
}

impl<T> Gc<T> {
    fn new(value: T) -> Gc<T>;
    fn clone(&self) -> Gc<T>;           // Cheap reference copy
    fn ptr_eq(&self, other: &Gc<T>) -> bool;
    fn try_unwrap(self) -> Result<T, Gc<T>>;
}

// Weak reference (doesn't prevent collection)
struct Weak<T> {
    // Weak reference to GC object
}

impl<T> Weak<T> {
    fn upgrade(&self) -> Option<Gc<T>>;
    fn clone(&self) -> Weak<T>;
}

// Interior mutability for GC objects
struct RefCell<T> {
    // Runtime borrow checking
}

impl<T> RefCell<T> {
    fn new(value: T) -> RefCell<T>;
    fn borrow(&self) -> Ref<T>;
    fn borrow_mut(&self) -> RefMut<T>;
    fn try_borrow(&self) -> Result<Ref<T>, BorrowError>;
    fn try_borrow_mut(&self) -> Result<RefMut<T>, BorrowMutError>;
}

// Example: Complex object graph with GC
struct NeuralNetwork {
    layers: Vec<Gc<Layer>>,
    optimizer: Gc<RefCell<Optimizer>>,
    loss_history: RefCell<Vec<f64>>
}

struct Layer {
    weights: Tensor<f32, [?, ?]>,      // Owned tensor data
    bias: Tensor<f32, [?]>,            // Owned tensor data
    activation: Box<dyn ActivationFn>, // GC-managed trait object
    parent: Weak<Layer>,               // Weak reference to prevent cycles
    children: Vec<Gc<Layer>>           // Strong references to children
}

fn build_network() -> Gc<NeuralNetwork> {
    let layer1 = Gc::new(Layer::new(784, 128));
    let layer2 = Gc::new(Layer::new(128, 64));
    let layer3 = Gc::new(Layer::new(64, 10));
    
    // Set up parent-child relationships
    layer2.set_parent(Gc::downgrade(&layer1));
    layer3.set_parent(Gc::downgrade(&layer2));
    
    Gc::new(NeuralNetwork {
        layers: vec![layer1, layer2, layer3],
        optimizer: Gc::new(RefCell::new(Adam::new(0.001))),
        loss_history: RefCell::new(Vec::new())
    })
}
```

### Memory Regions and Allocation Strategies

```stark
// Memory region management
module std::alloc {
    // Custom allocators for different use cases
    trait Allocator {
        fn allocate(&self, layout: Layout) -> Result<*mut u8, AllocError>;
        fn deallocate(&self, ptr: *mut u8, layout: Layout);
        fn reallocate(&self, ptr: *mut u8, old_layout: Layout, new_layout: Layout) 
            -> Result<*mut u8, AllocError>;
    }
    
    // Built-in allocators
    struct SystemAllocator;     // System malloc/free
    struct PoolAllocator;       // Object pools for fixed-size allocations
    struct StackAllocator;      // Stack-based allocation
    struct GPUAllocator;        // GPU memory allocation
    struct PinnedAllocator;     // Pinned memory for CPU-GPU transfers
    
    // Region-based allocation for batch processing
    struct Region {
        allocator: Box<dyn Allocator>,
        size: usize,
        used: usize
    }
    
    impl Region {
        fn new(size: usize, allocator: Box<dyn Allocator>) -> Region;
        fn allocate<T>(&mut self) -> Result<*mut T, AllocError>;
        fn allocate_array<T>(&mut self, count: usize) -> Result<*mut T, AllocError>;
        fn reset(&mut self);  // Reset region (invalidates all pointers)
        fn clear(&mut self);  // Clear region and return memory
    }
    
    // Memory arena for batch operations
    fn with_arena<T>(size: usize, f: fn(&mut Region) -> T) -> T {
        let mut arena = Region::new(size, Box::new(SystemAllocator));
        let result = f(&mut arena);
        // Arena automatically cleaned up
        result
    }
}

// Usage example for batch processing
fn process_batch_with_arena(images: &[Image]) -> Vec<ProcessedImage> {
    alloc::with_arena(64 * 1024 * 1024, |arena| {  // 64MB arena
        let mut results = Vec::new();
        
        for image in images {
            // Temporary allocations in arena
            let temp_buffer = arena.allocate_array::<f32>(image.size())?;
            let processed = process_image_in_place(image, temp_buffer);
            results.push(processed);
        }
        
        results
        // Arena automatically freed here
    })
}
```

## Memory Safety Guarantees

### Compile-Time Safety Checks

```stark
// Borrow checker prevents common memory errors
fn memory_safety_examples() {
    // 1. Use after free prevention
    let tensor = Tensor::rand([100, 100]);
    drop(tensor);
    // print(tensor.shape());        // ERROR: tensor used after drop
    
    // 2. Double free prevention
    let tensor = Tensor::rand([100, 100]);
    drop(tensor);
    // drop(tensor);                 // ERROR: tensor already dropped
    
    // 3. Dangling pointer prevention  
    let reference: &Tensor<f32, [?, ?]>;
    {
        let tensor = Tensor::rand([100, 100]);
        reference = &tensor;
    }
    // print(reference.shape());     // ERROR: tensor dropped, reference invalid
    
    // 4. Data race prevention
    let mut tensor = Tensor::zeros([100, 100]);
    let ref1 = &tensor;
    // let ref2 = &mut tensor;       // ERROR: cannot borrow mutably while borrowed
    
    // 5. Iterator invalidation prevention
    let mut vec = vec![1, 2, 3, 4, 5];
    for item in &vec {
        // vec.push(6);               // ERROR: cannot modify while iterating
        print(*item);
    }
}
```

### Runtime Safety Checks

```stark
// Runtime checks for unsafe operations
module std::mem {
    // Safe memory operations
    fn size_of<T>() -> usize;
    fn align_of<T>() -> usize;
    fn size_of_val<T: ?Sized>(val: &T) -> usize;
    
    // Checked operations
    fn swap<T>(x: &mut T, y: &mut T);
    fn replace<T>(dest: &mut T, src: T) -> T;
    fn take<T: Default>(dest: &mut T) -> T;
    
    // Transmutation (unsafe, but checked)
    unsafe fn transmute<T, U>(e: T) -> U where T: TransmuteSafe<U>;
    
    // Memory validation
    fn is_aligned_to<T>(ptr: *const T, align: usize) -> bool;
    fn is_valid_range(ptr: *const u8, len: usize) -> bool;
}

// Safe array access with bounds checking
impl<T> [T] {
    fn get(&self, index: usize) -> Option<&T> {
        if index < self.len() {
            Some(unsafe { self.get_unchecked(index) })
        } else {
            None
        }
    }
    
    fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len() {
            Some(unsafe { self.get_unchecked_mut(index) })
        } else {
            None
        }
    }
}

// Panic on out-of-bounds access (debug mode)
impl<T> Index<usize> for [T] {
    type Output = T;
    
    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.len() {
            panic!("Index out of bounds: index={}, len={}", index, self.len());
        }
        unsafe { self.get_unchecked(index) }
    }
}
```

### Unsafe Code Guidelines

```stark
// Unsafe operations for performance-critical code
unsafe trait UnsafeOperations {
    // Raw pointer operations
    unsafe fn from_raw_parts<T>(ptr: *const T, len: usize) -> &[T];
    unsafe fn from_raw_parts_mut<T>(ptr: *mut T, len: usize) -> &mut [T];
    
    // Uninitialized memory
    unsafe fn uninitialized<T>() -> T;
    unsafe fn zeroed<T>() -> T;
    
    // Type punning
    unsafe fn cast<T, U>(value: T) -> U where T: TransmuteSafe<U>;
}

// Safe wrappers for unsafe operations
impl<T> Tensor<T, [?]> {
    // Safe constructor that validates inputs
    fn from_raw_parts(ptr: *mut T, len: usize, capacity: usize) -> Result<Self, InvalidPointer> {
        if ptr.is_null() {
            return Err(InvalidPointer::Null);
        }
        if len > capacity {
            return Err(InvalidPointer::InvalidLength);
        }
        if !mem::is_aligned_to(ptr, mem::align_of::<T>()) {
            return Err(InvalidPointer::Misaligned);
        }
        
        Ok(unsafe { Self::from_raw_parts_unchecked(ptr, len, capacity) })
    }
    
    // Unsafe version for when safety is already guaranteed
    unsafe fn from_raw_parts_unchecked(ptr: *mut T, len: usize, capacity: usize) -> Self {
        Tensor {
            data: Vec::from_raw_parts(ptr, len, capacity),
            shape: [len as i32],
            device: Device::CPU
        }
    }
}

// FFI safety guidelines
extern "C" {
    // All extern functions are unsafe
    fn cuda_malloc(size: usize) -> *mut c_void;
    fn cuda_free(ptr: *mut c_void);
    fn cublas_gemm(handle: *mut c_void, ...);
}

// Safe wrapper for FFI
struct CudaAllocator;

impl Allocator for CudaAllocator {
    fn allocate(&self, layout: Layout) -> Result<*mut u8, AllocError> {
        let ptr = unsafe { cuda_malloc(layout.size()) };
        if ptr.is_null() {
            Err(AllocError::OutOfMemory)
        } else {
            Ok(ptr as *mut u8)
        }
    }
    
    fn deallocate(&self, ptr: *mut u8, _layout: Layout) {
        unsafe { cuda_free(ptr as *mut c_void) };
    }
}
```

### Memory Layout Guarantees

```stark
// Guaranteed memory layouts for interop
#[repr(C)]
struct CCompatible {
    x: f32,
    y: f32,
    z: f32
}

#[repr(packed)]
struct PackedStruct {
    a: u8,
    b: u32,  // No padding
    c: u16
}

#[repr(transparent)]
struct NewType(u32);  // Same layout as u32

// Tensor memory layout guarantees
impl<T, S> Tensor<T, S> {
    // Guaranteed contiguous memory layout
    fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    
    fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }
    
    // Guaranteed stride information
    fn strides(&self) -> &[usize] {
        &self.strides
    }
    
    // Memory layout validation
    fn is_contiguous(&self) -> bool {
        // Check if tensor data is laid out contiguously
        let mut expected_stride = 1;
        for &dim_size in self.shape.dims().iter().rev() {
            if self.strides[self.strides.len() - 1] != expected_stride {
                return false;
            }
            expected_stride *= dim_size as usize;
        }
        true
    }
    
    // Force contiguous layout (may copy)
    fn contiguous(self) -> Self {
        if self.is_contiguous() {
            self
        } else {
            self.clone_contiguous()
        }
    }
}
```

### Thread Safety and Concurrency

```stark
// Thread safety markers
unsafe trait Send {}    // Can be sent between threads
unsafe trait Sync {}    // Can be shared between threads

// Automatic derivation for safe types
impl<T: Send> Send for Vec<T> {}
impl<T: Sync> Sync for Vec<T> {}

// Manual implementation for custom types
struct ThreadSafeTensor<T> {
    data: Arc<[T]>,           // Atomically reference counted
    shape: [i32],
    device: Device
}

unsafe impl<T: Send> Send for ThreadSafeTensor<T> {}
unsafe impl<T: Sync> Sync for ThreadSafeTensor<T> {}

// Atomic operations for lock-free programming
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};

struct AtomicCounter {
    count: AtomicUsize
}

impl AtomicCounter {
    fn new() -> Self {
        AtomicCounter { count: AtomicUsize::new(0) }
    }
    
    fn increment(&self) -> usize {
        self.count.fetch_add(1, Ordering::SeqCst)
    }
    
    fn get(&self) -> usize {
        self.count.load(Ordering::SeqCst)
    }
}

// Thread-safe reference counting
use std::sync::Arc;

fn parallel_tensor_processing() {
    let tensor = Arc::new(Tensor::rand([1000, 1000]));
    let mut handles = vec![];
    
    for i in 0..4 {
        let tensor_clone = Arc::clone(&tensor);
        let handle = thread::spawn(move || {
            // Each thread has access to the same tensor
            let slice = tensor_clone.slice(i * 250, (i + 1) * 250);
            process_slice(slice)
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
}
```

## Performance Characteristics

### Memory Allocation Performance

```stark
// Allocation strategies and their performance characteristics

// Stack allocation (fastest, limited size)
let small_array: [f32; 1024] = [0.0; 1024];    // ~4KB, microsecond allocation

// Owned heap allocation (fast, predictable)
let tensor = Tensor::<f32, [1024, 1024]>::zeros(); // ~4MB, millisecond allocation

// GC allocation (moderate overhead, unpredictable timing)
let model = Model::new();                       // Variable size, GC overhead

// Arena allocation (fastest for batch operations)
alloc::with_arena(1024 * 1024, |arena| {       // Pre-allocated region
    let temp1 = arena.allocate_array::<f32>(1000);  // Nanosecond allocation
    let temp2 = arena.allocate_array::<f32>(2000);  // Nanosecond allocation
    // Process with temp1 and temp2
});

// Memory pool allocation (consistent performance)
static TENSOR_POOL: Pool<Tensor<f32, [512, 512]>> = Pool::new();

fn get_pooled_tensor() -> PooledTensor<f32, [512, 512]> {
    TENSOR_POOL.get()  // Reuse pre-allocated tensor
}
```

### Garbage Collection Performance

```stark
// GC performance characteristics and tuning

// Generational GC assumptions:
// - Most objects die young (ML temporary tensors)
// - Few objects live long (models, datasets)
// - Collection frequency: young >> old

fn gc_performance_example() {
    // Young generation collection: 1-5ms typical
    // - Collects recent allocations
    // - Triggered frequently (every 10-50MB allocated)
    // - Concurrent with application
    
    // Old generation collection: 10-100ms typical  
    // - Collects long-lived objects
    // - Triggered infrequently (heap pressure)
    // - May pause application briefly
    
    // Configuration for inference workloads
    gc::configure(GCConfig {
        mode: GCMode::LowLatency,
        pause_target: Duration::milliseconds(5),   // Max 5ms pauses
        concurrent_threads: 2,                     // Concurrent collection
        throughput_target: 0.99                    // 99% app throughput
    });
    
    // Configuration for training workloads
    gc::configure(GCConfig {
        mode: GCMode::Training,
        pause_target: Duration::milliseconds(20),  // Can tolerate longer pauses
        concurrent_threads: 4,                     // More GC threads
        throughput_target: 0.95                    // 95% app throughput
    });
}
```

This comprehensive Memory Model Specification provides:

1. **Hybrid Memory Management** - Combines ownership for performance with GC for productivity
2. **Zero-Cost Abstractions** - Stack allocation and ownership for critical tensor operations
3. **Memory Safety** - Compile-time borrow checking prevents common memory errors
4. **GC Optimization** - Generational, concurrent GC optimized for AI/ML workloads
5. **Thread Safety** - Send/Sync traits and atomic operations for concurrent code
6. **Unsafe Interfaces** - Controlled unsafe code for FFI and performance-critical operations
7. **Memory Layout Guarantees** - Predictable layouts for interoperability
8. **Performance Characteristics** - Clear performance expectations for different allocation strategies

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Create Memory Model Specification", "status": "completed", "priority": "high", "id": "1"}, {"content": "Define ownership and borrowing semantics", "status": "completed", "priority": "high", "id": "2"}, {"content": "Specify garbage collection behavior", "status": "completed", "priority": "high", "id": "3"}, {"content": "Document memory safety guarantees", "status": "completed", "priority": "high", "id": "4"}]