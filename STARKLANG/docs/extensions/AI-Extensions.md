# STARK AI/ML Extensions Overview (Non-Core)

## Overview
This document sketches non-core, AI/ML-focused language extensions. These extensions are not part of the Core v1 language and may evolve independently. A compliant Core v1 implementation may omit these features.

**Tensor and model types are now specified normatively in
[Tensor-Model-Types.md](./Tensor-Model-Types.md)** (extension `tensor` v0.1).
The tensor/model material below is an informal preview of that document;
where they differ, `Tensor-Model-Types.md` wins. The remaining sections
(datasets, LLM blocks) are early sketches without normative semantics.

## Syntax Status
Extension code uses Core v1 syntax plus extension-specific grammar (tensor
shape arguments, `model` items, `@llm` blocks) that a Core v1 parser is not
required to accept. The canonical tensor type syntax for all STARK documents
is:

```
Tensor<ElementType, [Dim1, Dim2, ...]>
```

where the shape is a bracketed dimension list appearing as the second generic
argument. Dimensions are integer constants or symbolic dim variables (see
Tensor-Model-Types.md §3). Older documents using `Tensor<T>[dims]` or
lowercase element types (`f32`) are superseded by this form.

## Tensor Types (Extension)
```stark
// Tensor types (AI primitive); B is a shape variable bound at load time
let features: Tensor<Float32, [B, 128]> = load_data("input.json");
let weights: Tensor<Float32, [128, 10]> = tensor_zeros([128, 10]);
```

## Model Loading (Extension)
```stark
let model: Model = load_pytorch_model("classifier.pt");
let llm: LLMClient = LLMClient::new("openai", "gpt-4");
```

## Dataset Handling (Extension)
```stark
let dataset: Dataset<(Tensor<Float32, [128]>, Int32)> =
    load_dataset("train.csv").batch(32).shuffle(42);
```

## LLM Blocks (Extension)
```stark
@llm as classifier:
    system: "You are a text classifier"
    user: "Classify this text: {{input}}"

match classifier.call({input: text}) {
    "positive" => handle_positive(),
    "negative" => handle_negative(),
    _ => handle_neutral()
}
```

## Tensor Operations (Extension)
```stark
let logits = model.forward(features);
let probabilities = softmax(logits);
let prediction = argmax(probabilities);
```

## AI Error Handling (Extension)
```stark
fn load_model(path: &str) -> Result<Model, ModelError> {
    if !file_exists(path) {
        Err(ModelError::FileNotFound(String::from(path)))
    } else {
        load_pytorch_model(path)
    }
}

fn run_inference(model: &Model, input: TensorAny) -> Result<TensorAny, InferenceError> {
    if input.shape()[0] != model.input_shape()[0] {
        Err(InferenceError::ShapeMismatch)
    } else {
        Ok(model.predict(input)?)
    }
}
```

## Conformance
Implementations may support these features as optional extensions. If supported, implementations should document the extension version and any deviations from this spec.
