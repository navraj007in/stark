/**
 * Every claim on this site, in one place.
 *
 * Kept as data rather than scattered through JSX for one reason: a language homepage is mostly a
 * claims document, and claims drift. When the compiler's position changes, this file is the single
 * thing to edit, and it is short enough to re-read against `COMPILER-STATE.md` in a few minutes.
 *
 * The rule applied throughout: say what is true today, with its limits attached. Nothing here
 * describes an intended future as though it had arrived.
 */

export const REPO = 'https://github.com/navraj007in/stark';

export const hero = {
  title: 'Catch AI pipeline errors before inference begins.',
  lede:
    'STARK is a statically typed language with an ownership-safe core and an optional tensor extension. Shapes, element types, devices and ONNX model signatures are checked at compile time — not discovered in production.',
  status: 'Pre-alpha · active development · expect breaking changes',
};

export const problem = {
  title: 'The errors that survive review',
  body:
    'Inference pipelines connect components that are each individually valid and collectively wrong. These are the failures that reach production, because nothing on the path from source to deployment is in a position to notice them.',
  items: [
    'a model expects NCHW input and receives NHWC',
    'preprocessing emits the wrong element type',
    'a tensor is placed on an incompatible device',
    'an ONNX artifact changed after its declaration was generated',
    'a dynamic dimension is treated as statically known',
    'postprocessing assumes an output shape the model never produces',
  ],
};

export const pillars = [
  {
    title: 'Ownership without a garbage collector',
    body:
      'Moves, borrows and deterministic destruction, checked at compile time. One exclusive borrow or many shared ones. Destructors run exactly once, at a point you can predict from the source.',
  },
  {
    title: 'Failures are specified, not incidental',
    body:
      'Integer overflow, division by zero, out-of-bounds indexing and failing casts trap — in every build mode, release included. A trap carries a category and the source location that caused it.',
  },
  {
    title: 'Shapes checked before the model runs',
    body:
      'The optional tensor extension gives dimensions, element types and devices to the type system, and verifies an ONNX artifact still matches the declaration generated from it.',
  },
  {
    title: 'Small core, optional extensions',
    body:
      'Core v1 is deliberately implementable: no closures, no async, no trait objects, no unsafe. Tensor support is an extension you enable explicitly, and Core compiles as if it did not exist.',
  },
];

/** The genuinely unusual engineering claim, and the one worth explaining carefully. */
export const conformance = {
  title: 'Four engines, one answer',
  body:
    'A STARK program can run four ways: a reference interpreter that defines the semantics, a mid-level IR interpreter, and native binaries built in debug and release. Every maintained conformance case runs through all four, and they must agree — on output, on exit status, on which destructor ran when, and on the exact category and source location of any trap.',
  emphasis:
    'Agreement alone is not the standard. Each case also pins its expected result against the specification, so three engines agreeing on the wrong answer fails.',
  note:
    'That distinction is not theoretical. It is how a bound that every engine ignored equally, and an operation that completed where the specification required a trap, were both found and fixed.',
};

export const example = {
  caption: 'Tensor extension — a typed inference pipeline',
  code: `model Resnet50<N: Dim> {
    input data: Tensor<Float32, [N, 3, 224, 224]>;
    output probabilities: Tensor<Float32, [N, 1000]>;
}

fn preprocess(
    image: Tensor<UInt8, [1, 224, 224, 3]>
) -> Tensor<Float32, [1, 3, 224, 224]> {
    image
        .permute::<[0, 3, 1, 2]>()
        .cast::<Float32>()
}

fn infer(model: Resnet50, raw: TensorAny) -> Result<Tensor<Int64, [1]>, String> {
    // A dynamic tensor must be refined before it can be used as a typed one.
    let image = raw.refine::<UInt8, [1, 224, 224, 3]>()?;
    let input = preprocess(image);
    let output = model.predict(&input);

    Ok(output.softmax::<1>().argmax::<1>())
}`,
};

export const coreExample = {
  caption: 'Core — ownership, traits and pattern matching',
  code: `enum Shape {
    Dot,
    Rect(Int32, Int32),
}

struct Point { x: Int32, y: Int32 }

impl Display for Point {
    fn fmt(&self) -> String {
        String::from("POINT")
    }
}

// Matching through a reference borrows the payload; it is never moved out.
fn area(shape: &Shape) -> Int32 {
    match *shape {
        Shape::Dot => 0,
        Shape::Rect(w, h) => w * h,
    }
}

fn main() {
    let r: Shape = Shape::Rect(6, 7);
    println(area(&r));
    println(area(&r));   // the value survived the first match
}`,
};

export const install = {
  title: 'Install',
  steps: [
    {
      label: 'Build the compiler',
      code: `git clone ${REPO}
cd stark/starkc
cargo build --release --bins`,
    },
    {
      label: 'Install the three binaries',
      code: `PREFIX="$HOME/.local"          # must be on PATH
install -m 755 target/release/stark    "$PREFIX/bin/"
install -m 755 target/release/starkc   "$PREFIX/bin/"
install -m 755 target/release/starkide "$PREFIX/bin/"`,
    },
    {
      label: 'Install the runtime the native backend links',
      code: `rsync -a --exclude target stark-runtime/      "$PREFIX/lib/stark/stark-runtime/"
rsync -a --exclude target stark-provider-abi/ "$PREFIX/lib/stark/stark-provider-abi/"`,
    },
    {
      label: 'Build and run a package',
      code: `stark check
stark run
stark build --release`,
    },
  ],
  footnote:
    'Requires Rust 1.85 or newer. Provider-backed capabilities — clock, filesystem, environment, TCP — need their provider crates installed too; the README has the full layout.',
};

export const state = {
  title: 'Where the compiler actually is',
  lede:
    'Written plainly because a language that checks your assumptions should not misrepresent its own.',
  working: [
    'lexer, parser, name resolution, type checking',
    'ownership, moves, partial moves and borrow checking',
    'generics, traits, associated types, coherence',
    'reference interpreter and mid-level IR interpreter',
    'native compilation, debug and release, on Linux, macOS and Windows',
    'multi-file modules, packages, lock files, semantic versioning',
    'tensor shape/dtype/device analysis and ONNX signature verification',
    'compiler-backed language services (LSP + VS Code extension)',
  ],
  notYet: [
    'closures, async, trait objects and unsafe — not in Core v1',
    'iterator combinators (map, filter, collect, …) are refused by the front end, not silently broken',
    'a complete standard library',
    'a public package registry',
    'training, autodiff and GPU kernel generation',
    'API and language stability guarantees',
  ],
};

export const links = [
  { label: 'Source', href: REPO, note: 'compiler, specification and conformance suite' },
  { label: 'Specification', href: `${REPO}/tree/main/STARKLANG/docs/spec`, note: 'normative Core v1' },
  { label: 'Compiler status', href: `${REPO}/blob/main/COMPILER-STATE.md`, note: 'the authoritative position' },
  { label: 'VS Code extension', href: `${REPO}/tree/main/editors/vscode`, note: 'language server and editor support' },
];
