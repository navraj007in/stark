# Error Handling System Specification

STARK employs a comprehensive error handling system that combines algebraic error types (Result/Option) with structured exception handling, optimized for AI/ML workflows. The system provides type-safe error propagation, detailed error context, and recovery mechanisms suitable for both development and production environments.

## Overview

### Error Handling Philosophy

STARK's error handling is designed around:

1. **Type Safety** - Errors are part of the type system, preventing unhandled errors
2. **Explicit Error Handling** - Errors must be explicitly handled or propagated
3. **Rich Error Context** - Detailed error information for debugging and monitoring
4. **Composable Error Types** - Easy composition and transformation of errors
5. **Performance** - Zero-cost error handling for the happy path
6. **ML-Specific Errors** - Domain-specific error types for AI/ML operations

```stark
// High-level error handling overview
async fn ml_pipeline() -> Result<TrainingMetrics, MLError> {
    // Type-safe error propagation with ?
    let dataset = Dataset::load("train.csv")?;
    let model = Model::from_config(&config)?;
    
    // Error context with custom error types
    let metrics = train_model(model, dataset)
        .await
        .context("Failed to train model")?;
    
    // Error recovery with fallback
    save_model(&model, "model.onnx")
        .or_else(|e| {
            warn!("Primary save failed: {e}, trying backup location");
            save_model(&model, "backup/model.onnx")
        })?;
    
    Ok(metrics)
}
```

## Result and Option Types

### Core Types Definition

```stark
// Result type for operations that may fail
enum Result<T, E> {
    Ok(T),
    Err(E)
}

impl<T, E> Result<T, E> {
    // Construction
    fn ok(value: T) -> Result<T, E> { Result::Ok(value) }
    fn err(error: E) -> Result<T, E> { Result::Err(error) }
    
    // Query methods
    fn is_ok(&self) -> bool;
    fn is_err(&self) -> bool;
    fn contains<U>(&self, x: &U) -> bool where T: PartialEq<U>;
    fn contains_err<F>(&self, f: &F) -> bool where E: PartialEq<F>;
    
    // Extract methods
    fn ok(self) -> Option<T>;
    fn err(self) -> Option<E>;
    fn as_ref(&self) -> Result<&T, &E>;
    fn as_mut(&mut self) -> Result<&mut T, &mut E>;
    
    // Transform methods
    fn map<U, F: FnOnce(T) -> U>(self, op: F) -> Result<U, E>;
    fn map_or<U, F: FnOnce(T) -> U>(self, default: U, f: F) -> U;
    fn map_or_else<U, D: FnOnce(E) -> U, F: FnOnce(T) -> U>(self, default: D, f: F) -> U;
    fn map_err<F, O: FnOnce(E) -> F>(self, op: O) -> Result<T, F>;
    
    // Boolean operations
    fn and<U>(self, res: Result<U, E>) -> Result<U, E>;
    fn and_then<U, F: FnOnce(T) -> Result<U, E>>(self, op: F) -> Result<U, E>;
    fn or<F>(self, res: Result<T, F>) -> Result<T, F>;
    fn or_else<F, O: FnOnce(E) -> Result<T, F>>(self, op: O) -> Result<T, F>;
    
    // Unwrap methods (panic on error)
    fn unwrap(self) -> T where E: Debug;
    fn expect(self, msg: &str) -> T where E: Debug;
    fn unwrap_err(self) -> E where T: Debug;
    fn expect_err(self, msg: &str) -> E where T: Debug;
    
    // Safe unwrap methods
    fn unwrap_or(self, default: T) -> T;
    fn unwrap_or_else<F: FnOnce(E) -> T>(self, op: F) -> T;
    fn unwrap_or_default(self) -> T where T: Default;
}

// Option type for nullable values
enum Option<T> {
    Some(T),
    None
}

impl<T> Option<T> {
    // Construction
    fn some(value: T) -> Option<T> { Option::Some(value) }
    fn none() -> Option<T> { Option::None }
    
    // Query methods
    fn is_some(&self) -> bool;
    fn is_none(&self) -> bool;
    fn contains<U>(&self, x: &U) -> bool where T: PartialEq<U>;
    
    // Extract methods
    fn as_ref(&self) -> Option<&T>;
    fn as_mut(&mut self) -> Option<&mut T>;
    fn as_deref(&self) -> Option<&T::Target> where T: Deref;
    fn as_deref_mut(&mut self) -> Option<&mut T::Target> where T: DerefMut;
    
    // Transform methods
    fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Option<U>;
    fn map_or<U, F: FnOnce(T) -> U>(self, default: U, f: F) -> U;
    fn map_or_else<U, D: FnOnce() -> U, F: FnOnce(T) -> U>(self, default: D, f: F) -> U;
    
    // Boolean operations
    fn and<U>(self, optb: Option<U>) -> Option<U>;
    fn and_then<U, F: FnOnce(T) -> Option<U>>(self, f: F) -> Option<U>;
    fn filter<P: FnOnce(&T) -> bool>(self, predicate: P) -> Option<T>;
    fn or(self, optb: Option<T>) -> Option<T>;
    fn or_else<F: FnOnce() -> Option<T>>(self, f: F) -> Option<T>;
    fn xor(self, optb: Option<T>) -> Option<T>;
    
    // Conversion methods
    fn ok_or<E>(self, err: E) -> Result<T, E>;
    fn ok_or_else<E, F: FnOnce() -> E>(self, err: F) -> Result<T, E>;
    
    // Unwrap methods
    fn unwrap(self) -> T;
    fn expect(self, msg: &str) -> T;
    fn unwrap_or(self, default: T) -> T;
    fn unwrap_or_else<F: FnOnce() -> T>(self, f: F) -> T;
    fn unwrap_or_default(self) -> T where T: Default;
}

// Try operator (?) for error propagation
trait Try {
    type Output;
    type Residual;
    
    fn from_output(output: Self::Output) -> Self;
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output>;
}

trait FromResidual<R> {
    fn from_residual(residual: R) -> Self;
}

// Implementation for Result
impl<T, E> Try for Result<T, E> {
    type Output = T;
    type Residual = Result<Infallible, E>;
    
    fn from_output(output: T) -> Self { Result::Ok(output) }
    
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        match self {
            Result::Ok(value) => ControlFlow::Continue(value),
            Result::Err(err) => ControlFlow::Break(Result::Err(err))
        }
    }
}

// Implementation for Option
impl<T> Try for Option<T> {
    type Output = T;
    type Residual = Option<Infallible>;
    
    fn from_output(output: T) -> Self { Option::Some(output) }
    
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        match self {
            Option::Some(value) => ControlFlow::Continue(value),
            Option::None => ControlFlow::Break(Option::None)
        }
    }
}
```

### Error Composition and Conversion

```stark
// Automatic error conversion with From trait
trait From<T> {
    fn from(value: T) -> Self;
}

// Automatic error conversion in Result
impl<T, E, F> FromResidual<Result<Infallible, E>> for Result<T, F>
where F: From<E> {
    fn from_residual(residual: Result<Infallible, E>) -> Self {
        match residual {
            Result::Err(e) => Result::Err(F::from(e)),
            Result::Ok(_) => unreachable!()
        }
    }
}

// Example: Automatic error conversion
fn process_data() -> Result<ProcessedData, ProcessingError> {
    let raw_data = read_file("data.txt")?;      // IOError -> ProcessingError
    let parsed = parse_csv(&raw_data)?;         // ParseError -> ProcessingError
    let validated = validate_data(parsed)?;     // ValidationError -> ProcessingError
    Ok(transform_data(validated))
}

// Error composition with multiple error types
enum CombinedError {
    Io(IOError),
    Parse(ParseError),
    Validation(ValidationError),
    Network(NetworkError)
}

impl From<IOError> for CombinedError {
    fn from(err: IOError) -> Self { CombinedError::Io(err) }
}

impl From<ParseError> for CombinedError {
    fn from(err: ParseError) -> Self { CombinedError::Parse(err) }
}

// Error boxing for dynamic error types
type BoxError = Box<dyn std::error::Error + Send + Sync>;

fn flexible_operation() -> Result<String, BoxError> {
    let data = risky_io_operation()?;           // Any error implementing Error
    let processed = complex_computation(data)?;  // Any error implementing Error
    Ok(processed)
}

// Anyhow-style error handling for applications
struct AnyhowError {
    error: Box<dyn std::error::Error + Send + Sync>,
    context: Vec<String>
}

impl AnyhowError {
    fn new<E: std::error::Error + Send + Sync + 'static>(error: E) -> Self {
        AnyhowError {
            error: Box::new(error),
            context: Vec::new()
        }
    }
    
    fn context<C: Display>(mut self, context: C) -> Self {
        self.context.push(context.to_string());
        self
    }
    
    fn with_context<C: Display, F: FnOnce() -> C>(mut self, f: F) -> Self {
        self.context.push(f().to_string());
        self
    }
}

// Context extension trait
trait Context<T> {
    fn context<C: Display>(self, context: C) -> Result<T, AnyhowError>;
    fn with_context<C: Display, F: FnOnce() -> C>(self, f: F) -> Result<T, AnyhowError>;
}

impl<T, E: std::error::Error + Send + Sync + 'static> Context<T> for Result<T, E> {
    fn context<C: Display>(self, context: C) -> Result<T, AnyhowError> {
        self.map_err(|e| AnyhowError::new(e).context(context))
    }
    
    fn with_context<C: Display, F: FnOnce() -> C>(self, f: F) -> Result<T, AnyhowError> {
        self.map_err(|e| AnyhowError::new(e).context(f()))
    }
}
```

## Exception Hierarchy

### Base Error Traits

```stark
// Core error trait (similar to std::error::Error)
trait Error: Debug + Display {
    // Optional source of this error
    fn source(&self) -> Option<&(dyn Error + 'static)> { None }
    
    // Optional backtrace
    fn backtrace(&self) -> Option<&Backtrace> { None }
    
    // Error description (deprecated, use Display)
    fn description(&self) -> &str { "an error occurred" }
    
    // Error cause (deprecated, use source)
    fn cause(&self) -> Option<&dyn Error> { self.source() }
    
    // Type ID for downcasting
    fn type_id(&self, _: private::Internal) -> TypeId { TypeId::of::<Self>() }
    
    // Provide method for additional context
    fn provide<'a>(&'a self, request: &mut Request<'a>) {}
}

// Automatic implementation for types with Debug + Display
impl<T: Debug + Display> Error for T {}

// Error downcasting
impl dyn Error + 'static {
    fn downcast_ref<T: Error + 'static>(&self) -> Option<&T> {
        if self.type_id(private::Internal) == TypeId::of::<T>() {
            unsafe { Some(&*(self as *const dyn Error as *const T)) }
        } else {
            None
        }
    }
    
    fn downcast_mut<T: Error + 'static>(&mut self) -> Option<&mut T> {
        if self.type_id(private::Internal) == TypeId::of::<T>() {
            unsafe { Some(&mut *(self as *mut dyn Error as *mut T)) }
        } else {
            None
        }
    }
    
    fn downcast<T: Error + 'static>(self: Box<Self>) -> Result<Box<T>, Box<dyn Error + 'static>> {
        if self.type_id(private::Internal) == TypeId::of::<T>() {
            unsafe {
                let raw: *mut dyn Error = Box::into_raw(self);
                Ok(Box::from_raw(raw as *mut T))
            }
        } else {
            Err(self)
        }
    }
}

// Chain of errors
trait ErrorChain {
    fn iter_chain(&self) -> ErrorChainIter<'_>;
    fn find_root_cause(&self) -> &dyn Error;
}

impl<T: Error> ErrorChain for T {
    fn iter_chain(&self) -> ErrorChainIter<'_> {
        ErrorChainIter { current: Some(self) }
    }
    
    fn find_root_cause(&self) -> &dyn Error {
        let mut current = self as &dyn Error;
        while let Some(source) = current.source() {
            current = source;
        }
        current
    }
}

struct ErrorChainIter<'a> {
    current: Option<&'a dyn Error>
}

impl<'a> Iterator for ErrorChainIter<'a> {
    type Item = &'a dyn Error;
    
    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current?;
        self.current = current.source();
        Some(current)
    }
}
```

### Standard Error Types

```stark
// System-level errors
#[derive(Debug)]
enum SystemError {
    OutOfMemory { requested: usize, available: usize },
    StackOverflow { current_depth: usize, max_depth: usize },
    ThreadPanic { thread_id: ThreadId, message: String },
    Timeout { operation: String, duration: Duration },
    ResourceExhausted { resource: String, limit: u64 }
}

impl Display for SystemError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            SystemError::OutOfMemory { requested, available } => 
                write!(f, "Out of memory: requested {requested} bytes, {available} available"),
            SystemError::StackOverflow { current_depth, max_depth } =>
                write!(f, "Stack overflow: depth {current_depth} exceeds maximum {max_depth}"),
            SystemError::ThreadPanic { thread_id, message } =>
                write!(f, "Thread {thread_id:?} panicked: {message}"),
            SystemError::Timeout { operation, duration } =>
                write!(f, "Operation '{operation}' timed out after {duration:?}"),
            SystemError::ResourceExhausted { resource, limit } =>
                write!(f, "Resource '{resource}' exhausted (limit: {limit})")
        }
    }
}

// I/O errors
#[derive(Debug)]
enum IOError {
    NotFound { path: String },
    PermissionDenied { path: String, operation: String },
    ConnectionRefused { address: String },
    ConnectionAborted { reason: String },
    ConnectionReset,
    TimedOut,
    WriteZero,
    Interrupted,
    UnexpectedEof,
    InvalidData { details: String },
    Other { message: String }
}

// Parsing errors
#[derive(Debug)]
struct ParseError {
    input: String,
    position: usize,
    expected: String,
    found: String,
    line: u32,
    column: u32
}

impl Display for ParseError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Parse error at line {}, column {}: expected {}, found {}",
            self.line, self.column, self.expected, self.found
        )
    }
}

// Validation errors
#[derive(Debug)]
struct ValidationError {
    field: String,
    value: String,
    constraint: String,
    message: String
}

impl Display for ValidationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Validation failed for field '{}': {} (constraint: {})",
            self.field, self.message, self.constraint
        )
    }
}

// Network errors
#[derive(Debug)]
enum NetworkError {
    DNS { hostname: String, error: String },
    HTTP { status: u16, url: String, body: Option<String> },
    SSL { error: String },
    Proxy { error: String },
    InvalidUrl { url: String },
    RequestTimeout { url: String, timeout: Duration },
    TooManyRedirects { url: String, count: u32 }
}
```

### ML-Specific Error Types

```stark
// Tensor operation errors
#[derive(Debug)]
enum TensorError {
    ShapeMismatch { 
        operation: String,
        expected: Vec<i32>, 
        actual: Vec<i32> 
    },
    DeviceMismatch { 
        expected: Device, 
        actual: Device 
    },
    DTypeMismatch { 
        expected: DataType, 
        actual: DataType 
    },
    IndexOutOfBounds { 
        index: Vec<i32>, 
        shape: Vec<i32> 
    },
    InvalidStride { 
        stride: Vec<usize>, 
        shape: Vec<i32> 
    },
    IncompatibleOperation { 
        operation: String, 
        reason: String 
    },
    CudaError { 
        code: i32, 
        message: String 
    },
    OutOfMemory { 
        requested: usize, 
        available: usize, 
        device: Device 
    }
}

impl Display for TensorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ShapeMismatch { operation, expected, actual } =>
                write!(f, "Shape mismatch in {operation}: expected {expected:?}, got {actual:?}"),
            TensorError::DeviceMismatch { expected, actual } =>
                write!(f, "Device mismatch: expected {expected:?}, got {actual:?}"),
            TensorError::DTypeMismatch { expected, actual } =>
                write!(f, "Data type mismatch: expected {expected:?}, got {actual:?}"),
            TensorError::IndexOutOfBounds { index, shape } =>
                write!(f, "Index {index:?} out of bounds for shape {shape:?}"),
            TensorError::InvalidStride { stride, shape } =>
                write!(f, "Invalid stride {stride:?} for shape {shape:?}"),
            TensorError::IncompatibleOperation { operation, reason } =>
                write!(f, "Incompatible operation {operation}: {reason}"),
            TensorError::CudaError { code, message } =>
                write!(f, "CUDA error {code}: {message}"),
            TensorError::OutOfMemory { requested, available, device } =>
                write!(f, "Out of memory on {device:?}: requested {requested}, available {available}")
        }
    }
}

// Model errors
#[derive(Debug)]
enum ModelError {
    LoadingFailed { 
        path: String, 
        format: String, 
        reason: String 
    },
    SavingFailed { 
        path: String, 
        reason: String 
    },
    InvalidArchitecture { 
        expected: String, 
        found: String 
    },
    MissingWeights { 
        layer: String 
    },
    IncompatibleWeights { 
        layer: String, 
        expected_shape: Vec<i32>, 
        actual_shape: Vec<i32> 
    },
    ForwardPassFailed { 
        layer: String, 
        input_shape: Vec<i32>, 
        reason: String 
    },
    BackwardPassFailed { 
        layer: String, 
        gradient_shape: Vec<i32>, 
        reason: String 
    },
    OptimizationFailed { 
        optimizer: String, 
        reason: String 
    },
    ConvergenceError { 
        epoch: u32, 
        loss: f64, 
        threshold: f64 
    }
}

// Training errors
#[derive(Debug)]
enum TrainingError {
    DatasetError { 
        source: Box<dyn Error + Send + Sync> 
    },
    ModelError { 
        source: ModelError 
    },
    OptimizerError { 
        optimizer: String, 
        reason: String 
    },
    LossComputation { 
        predictions_shape: Vec<i32>, 
        targets_shape: Vec<i32> 
    },
    CheckpointError { 
        epoch: u32, 
        path: String, 
        reason: String 
    },
    EarlyStopping { 
        epoch: u32, 
        patience: u32, 
        best_metric: f64, 
        current_metric: f64 
    },
    NumericalInstability { 
        epoch: u32, 
        batch: u32, 
        value: f64 
    }
}

// Dataset errors
#[derive(Debug)]
enum DatasetError {
    LoadingFailed { 
        source: String, 
        reason: String 
    },
    TransformFailed { 
        transform: String, 
        item_index: usize, 
        reason: String 
    },
    BatchingFailed { 
        batch_size: usize, 
        available_items: usize 
    },
    CachingFailed { 
        reason: String 
    },
    SchemaValidation { 
        expected_columns: Vec<String>, 
        found_columns: Vec<String> 
    },
    CorruptedData { 
        item_index: usize, 
        details: String 
    }
}

// Inference errors
#[derive(Debug)]
enum InferenceError {
    ModelNotLoaded,
    InvalidInput { 
        expected_shape: Vec<i32>, 
        actual_shape: Vec<i32> 
    },
    PostprocessingFailed { 
        reason: String 
    },
    BatchSizeMismatch { 
        expected: usize, 
        actual: usize 
    },
    TimeoutExceeded { 
        timeout: Duration, 
        elapsed: Duration 
    },
    ResourceExhausted { 
        resource: String 
    }
}
```

## Error Propagation Patterns

### The Try Operator (?)

```stark
// Basic error propagation
fn process_ml_pipeline() -> Result<ModelMetrics, MLError> {
    let dataset = load_dataset("train.csv")?;          // Propagate DatasetError
    let model = create_model(&config)?;                // Propagate ModelError
    let optimizer = create_optimizer(&opt_config)?;    // Propagate OptimizerError
    
    let metrics = train_model(model, dataset, optimizer)?; // Propagate TrainingError
    save_model(&model, "output.onnx")?;                // Propagate IOError
    
    Ok(metrics)
}

// Error conversion in propagation
impl From<DatasetError> for MLError {
    fn from(err: DatasetError) -> Self {
        MLError::Dataset(err)
    }
}

impl From<ModelError> for MLError {
    fn from(err: ModelError) -> Self {
        MLError::Model(err)
    }
}

// Option propagation
fn find_best_checkpoint(directory: &str) -> Option<String> {
    let entries = std::fs::read_dir(directory).ok()?;
    let mut best_checkpoint = None;
    let mut best_metric = 0.0;
    
    for entry in entries {
        let path = entry.ok()?.path();
        let metadata = parse_checkpoint_metadata(&path)?;
        
        if metadata.validation_accuracy > best_metric {
            best_metric = metadata.validation_accuracy;
            best_checkpoint = Some(path.to_string_lossy().to_string());
        }
    }
    
    best_checkpoint
}

// Early return with try operator
fn validate_training_config(config: &TrainingConfig) -> Result<(), ValidationError> {
    // Validate learning rate
    if config.learning_rate <= 0.0 {
        return Err(ValidationError::new("learning_rate", "must be positive"));
    }
    
    // Validate batch size
    if config.batch_size == 0 {
        return Err(ValidationError::new("batch_size", "must be greater than 0"));
    }
    
    // Validate epochs
    if config.epochs == 0 {
        return Err(ValidationError::new("epochs", "must be greater than 0"));
    }
    
    // Validate model architecture
    validate_model_architecture(&config.model)?;
    
    // Validate optimizer settings
    validate_optimizer_config(&config.optimizer)?;
    
    Ok(())
}
```

### Error Context and Chaining

```stark
// Error context with source chains
#[derive(Debug)]
struct ContextError {
    message: String,
    source: Option<Box<dyn Error + Send + Sync>>
}

impl ContextError {
    fn new<S: Into<String>>(message: S) -> Self {
        ContextError {
            message: message.into(),
            source: None
        }
    }
    
    fn with_source<E: Error + Send + Sync + 'static>(mut self, source: E) -> Self {
        self.source = Some(Box::new(source));
        self
    }
}

impl Display for ContextError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for ContextError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_ref().map(|e| e.as_ref())
    }
}

// Context extension for adding error context
trait ResultExt<T, E> {
    fn context<C: Display>(self, context: C) -> Result<T, ContextError>;
    fn with_context<C: Display, F: FnOnce() -> C>(self, f: F) -> Result<T, ContextError>;
}

impl<T, E: Error + Send + Sync + 'static> ResultExt<T, E> for Result<T, E> {
    fn context<C: Display>(self, context: C) -> Result<T, ContextError> {
        self.map_err(|e| {
            ContextError::new(context.to_string()).with_source(e)
        })
    }
    
    fn with_context<C: Display, F: FnOnce() -> C>(self, f: F) -> Result<T, ContextError> {
        self.map_err(|e| {
            ContextError::new(f().to_string()).with_source(e)
        })
    }
}

// Usage example with rich error context
async fn train_with_context(config: &TrainingConfig) -> Result<ModelMetrics, ContextError> {
    let dataset = Dataset::load(&config.dataset_path)
        .context("Failed to load training dataset")?;
    
    let model = Model::from_config(&config.model)
        .with_context(|| format!("Failed to create model with architecture '{}'", 
                                config.model.architecture))?;
    
    let optimizer = create_optimizer(&config.optimizer)
        .with_context(|| format!("Failed to create {} optimizer", 
                                config.optimizer.name))?;
    
    let metrics = train_model(model, dataset, optimizer).await
        .context("Training failed")?;
    
    save_checkpoint(&model, &config.checkpoint_path)
        .with_context(|| format!("Failed to save checkpoint to '{}'", 
                                config.checkpoint_path))?;
    
    Ok(metrics)
}

// Error chain display
fn display_error_chain(error: &dyn Error) {
    eprintln!("Error: {}", error);
    
    let mut source = error.source();
    let mut level = 1;
    
    while let Some(err) = source {
        eprintln!("  {}: {}", level, err);
        source = err.source();
        level += 1;
    }
}
```

### Error Recovery and Fallbacks

```stark
// Error recovery with multiple attempts
async fn robust_model_loading(paths: &[String]) -> Result<Model, ModelError> {
    let mut last_error = None;
    
    for (i, path) in paths.iter().enumerate() {
        match Model::load(path).await {
            Ok(model) => {
                if i > 0 {
                    warn!("Loaded model from fallback path: {}", path);
                }
                return Ok(model);
            }
            Err(e) => {
                warn!("Failed to load model from {}: {}", path, e);
                last_error = Some(e);
            }
        }
    }
    
    Err(last_error.unwrap_or_else(|| ModelError::LoadingFailed {
        path: "no paths provided".to_string(),
        format: "unknown".to_string(),
        reason: "no model paths specified".to_string()
    }))
}

// Graceful degradation
async fn inference_with_fallback(
    input: Tensor<f32, [?, ?]>,
    primary_model: &Model,
    fallback_model: Option<&Model>
) -> Result<Tensor<f32, [?, ?]>, InferenceError> {
    match primary_model.predict(&input).await {
        Ok(result) => Ok(result),
        Err(e) => {
            warn!("Primary model failed: {}, trying fallback", e);
            
            if let Some(fallback) = fallback_model {
                fallback.predict(&input).await
                    .map_err(|fallback_err| {
                        error!("Both primary and fallback models failed");
                        error!("Primary error: {}", e);
                        error!("Fallback error: {}", fallback_err);
                        fallback_err
                    })
            } else {
                Err(e)
            }
        }
    }
}

// Retry with exponential backoff
struct RetryConfig {
    max_attempts: u32,
    base_delay: Duration,
    max_delay: Duration,
    backoff_factor: f64
}

impl Default for RetryConfig {
    fn default() -> Self {
        RetryConfig {
            max_attempts: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_factor: 2.0
        }
    }
}

async fn retry_with_backoff<T, E, F, Fut>(
    mut operation: F,
    config: RetryConfig
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: std::fmt::Display
{
    let mut attempt = 0;
    let mut delay = config.base_delay;
    
    loop {
        attempt += 1;
        
        match operation().await {
            Ok(result) => return Ok(result),
            Err(error) => {
                if attempt >= config.max_attempts {
                    error!("Operation failed after {} attempts, last error: {}", 
                          attempt, error);
                    return Err(error);
                }
                
                warn!("Attempt {} failed: {}, retrying in {:?}", 
                     attempt, error, delay);
                
                sleep(delay).await;
                
                // Exponential backoff
                delay = std::cmp::min(
                    Duration::from_secs_f64(
                        delay.as_secs_f64() * config.backoff_factor
                    ),
                    config.max_delay
                );
            }
        }
    }
}

// Circuit breaker pattern
struct CircuitBreaker {
    failure_threshold: u32,
    success_threshold: u32,
    timeout: Duration,
    state: CircuitState,
    failure_count: u32,
    success_count: u32,
    last_failure_time: Option<Instant>
}

enum CircuitState {
    Closed,    // Normal operation
    Open,      // Failing, reject requests
    HalfOpen   // Testing if service recovered
}

impl CircuitBreaker {
    fn new(failure_threshold: u32, timeout: Duration) -> Self {
        CircuitBreaker {
            failure_threshold,
            success_threshold: failure_threshold / 2,
            timeout,
            state: CircuitState::Closed,
            failure_count: 0,
            success_count: 0,
            last_failure_time: None
        }
    }
    
    async fn call<T, E, F, Fut>(&mut self, operation: F) -> Result<T, CircuitBreakerError<E>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T, E>>
    {
        match self.state {
            CircuitState::Closed => {
                match operation().await {
                    Ok(result) => {
                        self.on_success();
                        Ok(result)
                    }
                    Err(error) => {
                        self.on_failure();
                        Err(CircuitBreakerError::OperationFailed(error))
                    }
                }
            }
            CircuitState::Open => {
                if let Some(last_failure) = self.last_failure_time {
                    if last_failure.elapsed() > self.timeout {
                        self.state = CircuitState::HalfOpen;
                        self.call(operation).await
                    } else {
                        Err(CircuitBreakerError::CircuitOpen)
                    }
                } else {
                    Err(CircuitBreakerError::CircuitOpen)
                }
            }
            CircuitState::HalfOpen => {
                match operation().await {
                    Ok(result) => {
                        self.success_count += 1;
                        if self.success_count >= self.success_threshold {
                            self.state = CircuitState::Closed;
                            self.failure_count = 0;
                            self.success_count = 0;
                        }
                        Ok(result)
                    }
                    Err(error) => {
                        self.state = CircuitState::Open;
                        self.last_failure_time = Some(Instant::now());
                        Err(CircuitBreakerError::OperationFailed(error))
                    }
                }
            }
        }
    }
    
    fn on_success(&mut self) {
        self.failure_count = 0;
    }
    
    fn on_failure(&mut self) {
        self.failure_count += 1;
        if self.failure_count >= self.failure_threshold {
            self.state = CircuitState::Open;
            self.last_failure_time = Some(Instant::now());
        }
    }
}

#[derive(Debug)]
enum CircuitBreakerError<E> {
    CircuitOpen,
    OperationFailed(E)
}
```

### Error Aggregation and Reporting

```stark
// Collect multiple errors
fn validate_all_inputs(inputs: &[Input]) -> Result<(), ValidationErrors> {
    let mut errors = Vec::new();
    
    for (i, input) in inputs.iter().enumerate() {
        if let Err(e) = validate_input(input) {
            errors.push(IndexedError { index: i, error: e });
        }
    }
    
    if errors.is_empty() {
        Ok(())
    } else {
        Err(ValidationErrors { errors })
    }
}

#[derive(Debug)]
struct ValidationErrors {
    errors: Vec<IndexedError<ValidationError>>
}

#[derive(Debug)]
struct IndexedError<E> {
    index: usize,
    error: E
}

impl Display for ValidationErrors {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "Validation failed with {} errors:", self.errors.len())?;
        for indexed_error in &self.errors {
            writeln!(f, "  [{}]: {}", indexed_error.index, indexed_error.error)?;
        }
        Ok(())
    }
}

// Error reporting and logging
struct ErrorReporter {
    logger: Logger,
    metrics: MetricsCollector,
    alerting: AlertingService
}

impl ErrorReporter {
    async fn report_error(&self, error: &dyn Error, context: ErrorContext) {
        // Log error with full context
        error!("Error occurred: {}", error);
        self.log_error_chain(error);
        
        // Collect metrics
        self.metrics.increment_error_counter(&context.operation, &error.to_string());
        
        // Send alerts for critical errors
        if context.severity >= Severity::Critical {
            let alert = Alert {
                title: format!("Critical error in {}", context.operation),
                description: error.to_string(),
                severity: context.severity,
                timestamp: Utc::now(),
                context: context.clone()
            };
            
            if let Err(e) = self.alerting.send_alert(alert).await {
                warn!("Failed to send alert: {}", e);
            }
        }
    }
    
    fn log_error_chain(&self, error: &dyn Error) {
        for (i, err) in error.iter_chain().enumerate() {
            if i == 0 {
                error!("Error: {}", err);
            } else {
                error!("  Caused by: {}", err);
            }
        }
        
        if let Some(backtrace) = error.backtrace() {
            error!("Backtrace:\n{}", backtrace);
        }
    }
}

#[derive(Clone, Debug)]
struct ErrorContext {
    operation: String,
    severity: Severity,
    user_id: Option<String>,
    request_id: Option<String>,
    additional_context: HashMap<String, String>
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Severity {
    Info = 1,
    Warning = 2,
    Error = 3,
    Critical = 4
}
```

This comprehensive Error Handling System provides:

1. **Type-Safe Error Handling** - Result/Option types with the try operator (?)
2. **Rich Error Hierarchy** - Domain-specific error types for ML/AI workflows  
3. **Error Composition** - Automatic error conversion and chaining
4. **Context and Recovery** - Error context, fallback strategies, and retry mechanisms
5. **Production Features** - Circuit breakers, error aggregation, and monitoring integration
6. **Performance** - Zero-cost error handling for the happy path
7. **Debugging Support** - Error chains, backtraces, and detailed error context