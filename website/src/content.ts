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

/**
 * The genuinely unusual engineering claim, and the one worth explaining carefully.
 *
 * Wording calibrated by WP-ENGINE-INDEPENDENCE EI6 (2026-08-09) against what EI1-EI5 measured.
 * Three things this copy must keep saying, because each was earned rather than assumed:
 *   - three ENGINES, four CONFIGURATIONS. Native debug and release are one engine at two
 *     optimisation levels, sharing the lowering, the emitted Rust and every semantic authority.
 *   - agreement is not the standard, and where a rule is SHARED agreement cannot corroborate it.
 *   - the shared rules are registered publicly rather than omitted.
 * Do not restore "three independent implementations" or any equivalent: the register does not
 * support it. See STARKLANG/docs/compiler/ENGINE-PUBLIC-CLAIM-CALIBRATION.md.
 */
export const conformance = {
  title: 'Three engines, four configurations, one answer',
  body:
    'A STARK program can run four ways: a reference interpreter that defines the semantics, a mid-level IR interpreter, and a native binary compiled through generated Rust in debug and release. Every maintained conformance case runs through all four, and they must agree — on output, on exit status, on which destructor ran when, and on the exact category and source location of any trap.',
  emphasis:
    'Agreement alone is not the standard. Conformance cases are pinned against the specification, not against each other, so engines agreeing on the wrong answer fails. And where a rule is decided once and shared by every engine — Copy eligibility, destructor eligibility, the trap category vocabulary — agreement cannot corroborate it. Those rules are listed in a public register and checked separately.',
  note:
    'That distinction is not theoretical. It is how a bound that every engine ignored equally, and an operation that completed where the specification required a trap, were both found and fixed.',
  toolchain:
    'The native engine compiles STARK to safe Rust and builds it with rustc, which makes rustc an external check: it rejects generated code that violates Rust\'s borrow and move rules, and has caught real lowering defects that way. It is not a check on meaning — generated Rust can be valid and still say the wrong thing. Where the two languages differ, STARK decides: arithmetic lowers to explicit checked operations rather than relying on the build profile, shifts do not use Rust\'s checked_shl because it validates only the shift count, and destruction order is STARK\'s own plan rather than Rust\'s.',
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
      label: 'Build a release package for this machine',
      code: `git clone ${REPO}
cd stark/starkc
python3 scripts/build-release.py     # py -3 on Windows`,
    },
    {
      label: 'Install it',
      code: `# from the extracted package in target/packages/
./install.sh                         # defaults to ~/.local
.\\install.ps1                        # Windows; updates the user PATH`,
    },
    {
      label: 'Verify the installation',
      code: `stark doctor                         # re-hashes every manifest-listed file
stark doctor --json                  # machine-readable, for CI`,
    },
    {
      label: 'Build and run a package',
      code: `stark check
stark run
stark build --release`,
    },
  ],
  footnote:
    'Requires Rust 1.85 or newer for native builds. The package carries the compiler, its runtime and the provider ABI — not the first-party STARK packages or their native providers, so an HTTP or TLS program still needs those sources installed separately. The README has the layout, and a checkout-only install too.',
  caveat:
    'stark doctor establishes integrity, not authenticity. It detects corruption and a partial extraction; it cannot tell you the manifest came from a STARK release. Archives are unsigned, and a public distribution still needs a signed manifest, a trusted key, verification before installation and platform notarisation.',
};

/** Libraries written in STARK itself — the newest and least expected part of the project. */
export const packages = {
  title: 'Libraries, written in STARK',
  lede:
    'Thirty first-party packages live in the repository, each with its own manifest, lock file and tests, and each exercised by a consumer package that has to actually call the surface it declares. The deepest is an HTTP/1.1 and HTTPS client — written in the language, not bound to a C library.',
  groups: [
    { area: 'Encoding and text', items: 'ascii · base64 · hex · percent · checksum · uuid' },
    { area: 'Data formats', items: 'json · csv · form · mime · query' },
    { area: 'Paths and URLs', items: 'path · glob · url' },
    { area: 'Host access', items: 'time · env · io · random' },
    { area: 'Networking', items: 'net (TCP + DNS) · tls · http-core · http-parser · http-serialize · http-client' },
  ],
  capabilities:
    'Reaching outside the process is derived, envelope-checked, and provider-backed. Capability vocabulary v1 distinguishes filesystem read/write, environment read, network client/listen, clock, randomness, process execution, and native code. The root manifest approves the transitive derived set. The interpreters have no host access, so host-backed code runs through stark build, never stark run.',
  code: `use stark_http_client::default_config;
use stark_http_client::error_text;
use stark_http_client::fetch;
use stark_http_client::new_client;

fn main() {
    let client = new_client(default_config());

    // HTTPS differs from HTTP only in the URL. There is no second API.
    match fetch(&client, "https://example.com/health") {
        Ok(response) => {
            if response.status == 200u16 {
                println("healthy");
            }
        }
        Err(error) => {
            println(error_text(&error).as_str());
        }
    }
}`,
  caption: 'stark-http-client — a verified HTTPS request',
  limits:
    'It was qualified against peers that are adversarial on the wire — 42 executed cases on Linux, macOS and Windows — and the closing packets found four defects, two of them remote-abort vulnerabilities rather than parse errors. What it does not do is written down just as plainly: no HTTP/2, no connection reuse, no decompression, no proxies, no cookie jar, no streaming bodies, and a connect timeout that is accepted and ignored.',
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
    '30 first-party packages, including an HTTP/1.1 and HTTPS client',
    'manifest-declared host capabilities behind native providers',
    'tensor shape/dtype/device analysis and ONNX signature verification',
    'compiler-backed language services (LSP + VS Code extension)',
    'release archives, platform installers and stark doctor verification',
  ],
  notYet: [
    'closures, async, trait objects and unsafe — not in Core v1',
    'iterator combinators (map, filter, collect, …) are refused by the front end, not silently broken',
    'a complete standard library',
    'a signed distribution — the install manifest proves integrity, never authenticity',
    'an offline install that can build an HTTP or TLS program without fetching packages separately',
    'an HTTP server, HTTP/2, structured concurrency, persistent storage',
    'a public package registry — dependencies are local paths',
    'training, autodiff and GPU kernel generation',
    'API and language stability guarantees',
  ],
};

export const links = [
  { label: 'Source', href: REPO, note: 'compiler, packages, specification and conformance suite' },
  { label: 'Specification', href: `${REPO}/tree/main/STARKLANG/docs/spec`, note: 'normative Core v1' },
  { label: 'Compiler status', href: `${REPO}/blob/main/COMPILER-STATE.md`, note: 'the authoritative position' },
  { label: 'Roadmap', href: `${REPO}/blob/main/ROADMAP.md`, note: 'the single live forward plan' },
  {
    label: 'HTTP client limits',
    href: `${REPO}/blob/main/STARKLANG/docs/http-client/HC13-KNOWN-LIMITATIONS.md`,
    note: 'what it refuses, what is absent, what is unproven',
  },
  { label: 'VS Code extension', href: `${REPO}/tree/main/editors/vscode`, note: 'language server and editor support' },
];
