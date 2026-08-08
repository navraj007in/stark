//! Gate 3 tree-walking interpreter for typed STARK HIR.

use crate::ast::{AssignOp, BinOp, Lit, Primitive, UnOp};
use crate::diag::Diagnostic;
use crate::hir::{
    self, BlockId, Builtin, CoreTrait, CoreType, ExprId, Hir, ItemId, LocalId, PatId, Res, StmtId,
};
use crate::literal::{self, LitValue};
use crate::source::Span;
use crate::typecheck::{DisplayPath, DisplayStep, Ty, TypeTables};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::io::{Read, Write};
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct RuntimeError {
    pub message: String,
    pub span: Span,
    /// WP-C7.9 (Packets F and G.3): WHAT KIND of failure this is.
    ///
    /// This used to be a `bool` named `is_trap`, which could express only "a language trap" versus
    /// "something else" — and "something else" had to hold entrypoint-selection errors, interpreter
    /// invariant violations, and host resource exhaustion all at once. They are not the same thing:
    /// the first is a compiler rejection, the second is a compiler DEFECT, and the third is a
    /// property of the machine the program ran on (`LIMIT-RESOURCE-001`). A comparator that cannot
    /// tell them apart either treats a stack overflow as a language outcome or treats a real trap
    /// as noise.
    pub class: FailureClass,
    /// Which named oracle limitation this is, if any. `None` claims none — the ordinary case,
    /// including every other internal invariant.
    pub limitation: Option<OracleLimitation>,
    /// DEV-106 (CD-136): the trap category when the interpreter KNOWS it, rather than leaving it to
    /// be recovered by matching this error's prose.
    ///
    /// Set for `panic(msg)`, whose message is arbitrary USER text that no prose table could
    /// classify — the three-engine harness previously had to reject such a trap outright rather
    /// than risk defaulting it to whatever category the other engines reported. Every other trap
    /// leaves this `None` and keeps its existing prose-matched classification.
    pub trap_category: Option<crate::mir::TrapCategory>,
}

/// This implementation's call-depth capacity (WP-C7.9 Packet F, `LIMIT-RESOURCE-001`).
///
/// **Implementation-defined, and declared rather than discovered.** LIMIT-RESOURCE-001 leaves exact
/// capacities to the implementation and requires only that exhaustion be prevented from becoming
/// host undefined behaviour and be reported. This number is chosen against the *host* stack, not
/// against a language rule: each STARK call consumes several Rust frames
/// (`call_callable` → `eval_block` → `eval_stmt` → `eval_expr` → …), and a Rust test thread's
/// stack is far smaller than the main thread's. A capacity that fits comfortably inside the
/// smallest stack the project runs on is what keeps a deep recursion a *reported* failure on every
/// host rather than an abort on some of them.
///
/// It is deliberately not generous. Ordinary programs — including recursive-descent parsing over
/// realistically nested data — sit far below it; a program that reaches it is recursing without a
/// base case, which is the case this exists to report.
///
/// The MIR interpreter shares this constant, because the two interpreters have equivalent host
/// stack behaviour and a program should not become executable by changing engines.
pub const MAX_CALL_DEPTH: usize = 512;

/// A named oracle limitation, recognised WITHOUT matching prose.
///
/// The differential comparator must tell "this engine cannot execute this construct" from "the
/// engines disagree about the language". Doing that by substring made qualification depend on
/// diagnostic wording — a reword breaks it, and an unrelated internal message could match by
/// accident.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OracleLimitation {
    /// Destruction retains no concrete nominal type arguments, so a generic `Drop`'s parameters
    /// cannot be bound. MIR and native retain them and execute it correctly (DEV-176, A3c-D).
    GenericDrop,
}

/// What kind of failure ended a run (WP-C7.9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureClass {
    /// A language trap: a failure the specification defines, with a category and provenance. The
    /// only class the engines are required to agree on.
    Trap,
    /// Executable-target selection, detected before the entrypoint starts. A compiler rejection.
    Entry,
    /// A host or process resource limit — `LIMIT-RESOURCE-001` names allocation, stack, call depth,
    /// file descriptors and streams, and classifies their exhaustion as a host/process failure
    /// "unless an API returns a specified `Result`". **Never a trap**: capacities are
    /// implementation- and target-defined, so requiring engines to agree on them would be requiring
    /// them to agree on the machine.
    HostResource,
    /// An interpreter invariant that should be unreachable — a defect in the compiler, never in the
    /// program. Surfaced so a harness fails loudly instead of classifying it as a language outcome.
    InternalInvariant,
}

impl RuntimeError {
    /// Whether this is a language trap. Kept as a predicate so the four-way classification has one
    /// definition rather than being re-derived at each call site.
    pub fn is_trap(&self) -> bool {
        matches!(self.class, FailureClass::Trap)
    }

    /// A trap whose category is recovered from its prose by the consumer.
    fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
            class: FailureClass::Trap,
            limitation: None,
            trap_category: None,
        }
    }

    /// WP-C7.9 G.3: an interpreter invariant that should be unreachable. A compiler defect, never a
    /// program outcome — a harness must fail loudly on it rather than classify it as a trap.
    fn internal(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
            class: FailureClass::InternalInvariant,
            limitation: None,
            trap_category: None,
        }
    }

    /// An internal invariant that is a NAMED limitation of this engine.
    fn limitation(message: impl Into<String>, span: Span, limitation: OracleLimitation) -> Self {
        Self {
            message: message.into(),
            span,
            class: FailureClass::InternalInvariant,
            limitation: Some(limitation),
            trap_category: None,
        }
    }

    /// WP-C7.9 Packet F: a host/process resource limit was reached. Not a trap, and deliberately
    /// carries no `TrapCategory` — there is no category for it, because it is not a language
    /// outcome.
    fn host_resource(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
            class: FailureClass::HostResource,
            limitation: None,
            trap_category: None,
        }
    }

    /// DEV-106: a trap whose category the interpreter states outright.
    fn with_category(
        message: impl Into<String>,
        span: Span,
        category: crate::mir::TrapCategory,
    ) -> Self {
        Self {
            message: message.into(),
            span,
            class: FailureClass::Trap,
            limitation: None,
            trap_category: Some(category),
        }
    }

    fn entry(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
            class: FailureClass::Entry,
            limitation: None,
            trap_category: None,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Execution {
    pub output: String,
    /// Core process status produced by normal entrypoint completion.
    pub status: u8,
    /// The program's stderr: every `eprint`/`eprintln` byte it wrote, followed by the
    /// `Err(message)` bytes PROC-EXIT-001 owes on a failing entrypoint completion.
    ///
    /// WP-C7.9 Packet D: `eprint`/`eprintln` used to bypass this field entirely and write straight
    /// to the host process, so this carried only the `Err` completion bytes and the comparator
    /// compared empty-to-empty for every other program.
    pub stderr: String,
}

/// A run's complete observation, including the streams written *before* a failure.
///
/// [`run_with_partial_output`] returns the pre-trap stdout but has no room for the pre-trap
/// stderr, and widening its `Err` tuple would edit a dozen call sites that do not want the third
/// field. This is the superset entry point the comparator uses; that one is now a wrapper over it.
pub struct ExecutionOutcome {
    pub output: String,
    pub stderr: String,
    /// The exit status on normal completion, or the failure that ended the run.
    pub result: Result<u8, RuntimeError>,
}

/// Evaluate every declared constant before execution. This uses the same
/// abstract-machine operations as the interpreter, but only after a closed
/// syntactic-subset check has excluded runtime state and side effects.
pub fn check_constants(hir: &Hir, tables: &TypeTables) -> Vec<Diagnostic> {
    // AS1b-ii-d: the entry source comes from the program's own registry rather than being threaded
    // in beside it. An empty registry means nothing was parsed, so there are no constants either.
    let Some(entry) = hir.sources.entry().cloned() else {
        return Vec::new();
    };
    let mut interpreter = Interpreter::new(hir, entry, tables);
    interpreter.frames.push(Frame::default());
    let mut diagnostics = Vec::new();
    for (index, item) in hir.items.iter().enumerate() {
        let item_id = ItemId(index as u32);
        let hir::ItemKind::Const { value, .. } = &item.kind else {
            continue;
        };
        if let Err((span, message)) = constant_expr_allowed(hir, *value) {
            diagnostics.push(Diagnostic::error(message, span).with_code("E0215"));
            continue;
        }
        // AS1b-ii: DEV-088's swap is gone with DEV-069's three. It existed because a cross-file
        // `const`'s literal was read against the ENTRY file's text — `pub const N: Int32 = 31415;`
        // in a dependency failed "invalid literal". A literal's span now names the file it came
        // from, so the read is right without pointing the interpreter at anything.
        let outcome = interpreter.eval_const_item(item_id);
        if let Err(error) = outcome {
            diagnostics.push(
                Diagnostic::error(
                    format!("constant evaluation failed: {}", error.message),
                    error.span,
                )
                .with_code("E0215"),
            );
        }
    }
    diagnostics
}

fn constant_block_allowed(hir: &Hir, block: BlockId) -> Result<(), (Span, &'static str)> {
    let block = hir.block(block);
    for statement in &block.stmts {
        let statement = hir.stmt(*statement);
        match &statement.kind {
            hir::StmtKind::Empty => {}
            hir::StmtKind::Expr { expr, .. } => constant_expr_allowed(hir, *expr)?,
            _ => {
                return Err((
                    statement.span,
                    "statement is not permitted in a Core constant expression",
                ));
            }
        }
    }
    if let Some(tail) = block.tail {
        constant_expr_allowed(hir, tail)?;
    }
    Ok(())
}

fn constant_expr_allowed(hir: &Hir, expr: ExprId) -> Result<(), (Span, &'static str)> {
    let node = hir.expr(expr);
    match &node.kind {
        hir::ExprKind::Lit(_) => Ok(()),
        hir::ExprKind::Path { res, .. }
            if matches!(
                res,
                Res::Item(item) if matches!(&hir.item(*item).kind, hir::ItemKind::Const { .. })
            ) || matches!(
                res,
                Res::Variant(..)
                    | Res::Builtin(
                        Builtin::None
                            | Builtin::OrderingLess
                            | Builtin::OrderingEqual
                            | Builtin::OrderingGreater
                            | Builtin::IOErrorNotFound
                            | Builtin::IOErrorPermissionDenied
                            | Builtin::IOErrorAlreadyExists
                            | Builtin::IOErrorInvalidInput
                            | Builtin::IOErrorOther
                    )
            ) =>
        {
            Ok(())
        }
        hir::ExprKind::Unary { op, operand } if !matches!(op, UnOp::Ref { .. } | UnOp::Deref) => {
            constant_expr_allowed(hir, *operand)
        }
        hir::ExprKind::Binary { lhs, rhs, .. } => {
            constant_expr_allowed(hir, *lhs)?;
            constant_expr_allowed(hir, *rhs)
        }
        hir::ExprKind::Range { lo, hi, .. } => {
            constant_expr_allowed(hir, *lo)?;
            constant_expr_allowed(hir, *hi)
        }
        hir::ExprKind::Cast { expr, .. } => constant_expr_allowed(hir, *expr),
        hir::ExprKind::Call { callee, args }
            if matches!(
                &hir.expr(*callee).kind,
                hir::ExprKind::Path {
                    res: Res::Variant(..)
                        | Res::Builtin(Builtin::Some | Builtin::Ok | Builtin::Err),
                    ..
                }
            ) =>
        {
            for arg in args {
                constant_expr_allowed(hir, *arg)?;
            }
            Ok(())
        }
        hir::ExprKind::Tuple(values) | hir::ExprKind::Array(values) => {
            for value in values {
                constant_expr_allowed(hir, *value)?;
            }
            Ok(())
        }
        hir::ExprKind::Repeat { value, count } => {
            constant_expr_allowed(hir, *value)?;
            constant_expr_allowed(hir, *count)
        }
        hir::ExprKind::StructLit { fields, .. } => {
            for field in fields {
                let Some(value) = field.expr else {
                    return Err((
                        field.name,
                        "field shorthand is not permitted in a Core constant expression",
                    ));
                };
                constant_expr_allowed(hir, value)?;
            }
            Ok(())
        }
        hir::ExprKind::If {
            cond,
            then_block,
            else_,
        } => {
            constant_expr_allowed(hir, *cond)?;
            constant_block_allowed(hir, *then_block)?;
            if let Some(else_) = else_ {
                constant_expr_allowed(hir, *else_)?;
            }
            Ok(())
        }
        hir::ExprKind::Block(block) => constant_block_allowed(hir, *block),
        _ => Err((
            node.span,
            "expression is not permitted in the Core constant subset",
        )),
    }
}

/// Correction-brief Issue 3: `Value::Float` carries its declared width so the runtime value
/// itself knows whether it's a `Float32` or `Float64`, independent of any static-type-table
/// lookup at the point of use. Before this, every `Float32` value was stored as a plain `f64`
/// with no width marker at all -- correct for arithmetic (Float32 operations are already
/// rounded to `f32` precision after each primitive operation, per the frozen numeric contract,
/// and then widened back to `f64` for uniform storage), but losing the information needed to
/// format a `Float32` value using its own shortest-round-trip digits once it's nested inside a
/// tuple/array/struct/collection and reaches the generic recursive `Display for Value` impl,
/// which has no static-type context to consult. Math builtins (`sqrt`, `sin`, `cos`, ...) are
/// typed `Float64 -> Float64` only (`typecheck/body.rs`'s builtin signatures), so they always
/// produce `F64` and never need to preserve an argument's width.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FloatWidth {
    F32,
    F64,
}

#[derive(Clone, PartialEq)]
enum Value {
    Unit,
    Bool(bool),
    Int(i128),
    Float(f64, FloatWidth),
    Char(char),
    Str(String),
    String(String),
    Tuple(Vec<Option<Value>>),
    Array(Vec<Option<Value>>),
    Struct {
        item: ItemId,
        fields: BTreeMap<String, Option<Value>>,
    },
    Enum {
        item: ItemId,
        variant: u32,
        fields: Vec<Option<Value>>,
        named: BTreeMap<String, Option<Value>>,
    },
    Vec(Vec<Option<Value>>),
    Boxed(Box<Option<Value>>),
    /// **DEV-209: the payload is SLOT-backed, like every other component.**
    ///
    /// `Box<Value>` could not be named by a `Projection`, so PAT-BIND-001's "a binding to a
    /// non-`Copy` component receives `&C`, borrowing the component in place" had no storage to
    /// point at for a prelude payload — and the borrowed matcher fell back to the owned rule,
    /// moving out of a borrow. The rule is uniform over variant payloads, struct fields and tuple
    /// elements; the specialised representation was what failed to preserve a language-level place.
    Option(Option<Box<Option<Value>>>),
    Result(Result<Box<Option<Value>>, Box<Option<Value>>>),
    Range {
        start: i128,
        end: i128,
        inclusive: bool,
    },
    /// WP-C2.2 (DEV-028): an unsized slice is a view into an existing aggregate, not a copied
    /// `Array`. The bounds are half-open indices into `place`.
    Slice(Place, usize, usize),
    Ref(Place),
    Function(FunctionValue),
    CharsIter(String, usize),
    SplitIter(Vec<String>, usize),
    VecIter(Place, usize),
    HashMap(InsertionMap),
    HashSet(InsertionSet),
    HashMapKeysIter(Vec<Option<Value>>, usize),
    HashMapValuesIter(Vec<Option<Value>>, usize),
    HashMapIter(Vec<Option<Value>>, usize),
    HashSetIter(Vec<Option<Value>>, usize),
    MapIter(Box<Value>, ItemId),
    FilterIter(Box<Value>, ItemId),
    /// Simple LCG state (`06-Standard-Library.md` "Random numbers"); the
    /// mutable state is the seed itself, updated in place by `next_int`.
    Random(u64),
    IOError(IOErrorKind),
    File(FileResource),
    /// WP-C2.2 (DEV-027): runtime representation of the prelude `Ordering` enum, mirroring
    /// `IOError`'s builtin-backed pattern (no HIR item; variants resolve to `Builtin`s).
    Ordering(std::cmp::Ordering),
}

/// **The name of a runtime representation, with no payload.** (WP-VALUE-REP-TOTAL, A1.)
///
/// A `Value` cannot be named in a diagnostic or a table without either cloning it or printing its
/// contents, and a representation report must do neither: cloning is what the DEV-121 class is
/// about, and the contents of a value are never the caller's problem when the *shape* is wrong.
/// `ValueKind` is the shape alone.
///
/// **The enum, its display names and `ALL` are generated from one list**, so they cannot disagree.
/// A hand-written `ALL` can both duplicate an entry and omit another while keeping its length — a
/// count assertion passes on that list, which is why the count alone was not enough.
macro_rules! define_value_kinds {
    ($($kind:ident),+ $(,)?) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum ValueKind {
            $($kind),+
        }

        impl ValueKind {
            /// Every kind, in declaration order. Generated, never maintained.
            pub const ALL: &'static [ValueKind] = &[$(ValueKind::$kind),+];

            /// How the kind is written in a diagnostic — the variant's own name, so a renamed
            /// variant cannot keep an out-of-date label.
            pub fn as_str(self) -> &'static str {
                match self {
                    $(ValueKind::$kind => stringify!($kind)),+
                }
            }
        }
    };
}

define_value_kinds!(
    Unit,
    Bool,
    Int,
    Float,
    Char,
    Str,
    String,
    Tuple,
    Array,
    Struct,
    Enum,
    Vec,
    Boxed,
    Option,
    Result,
    Range,
    Slice,
    Ref,
    Function,
    CharsIter,
    SplitIter,
    VecIter,
    HashMap,
    HashSet,
    HashMapKeysIter,
    HashMapValuesIter,
    HashMapIter,
    HashSetIter,
    MapIter,
    FilterIter,
    Random,
    IOError,
    File,
    Ordering,
);

impl Value {
    /// The value's representation, without its contents.
    ///
    /// **No wildcard arm.** See [`ValueKind`] — this match is the forcing function that makes a new
    /// `Value` variant a compile error until the matrix accounts for it.
    pub fn kind(&self) -> ValueKind {
        match self {
            Value::Unit => ValueKind::Unit,
            Value::Bool(_) => ValueKind::Bool,
            Value::Int(_) => ValueKind::Int,
            Value::Float(_, _) => ValueKind::Float,
            Value::Char(_) => ValueKind::Char,
            Value::Str(_) => ValueKind::Str,
            Value::String(_) => ValueKind::String,
            Value::Tuple(_) => ValueKind::Tuple,
            Value::Array(_) => ValueKind::Array,
            Value::Struct { .. } => ValueKind::Struct,
            Value::Enum { .. } => ValueKind::Enum,
            Value::Vec(_) => ValueKind::Vec,
            Value::Boxed(_) => ValueKind::Boxed,
            Value::Option(_) => ValueKind::Option,
            Value::Result(_) => ValueKind::Result,
            Value::Range { .. } => ValueKind::Range,
            Value::Slice(_, _, _) => ValueKind::Slice,
            Value::Ref(_) => ValueKind::Ref,
            Value::Function(_) => ValueKind::Function,
            Value::CharsIter(_, _) => ValueKind::CharsIter,
            Value::SplitIter(_, _) => ValueKind::SplitIter,
            Value::VecIter(_, _) => ValueKind::VecIter,
            Value::HashMap(_) => ValueKind::HashMap,
            Value::HashSet(_) => ValueKind::HashSet,
            Value::HashMapKeysIter(_, _) => ValueKind::HashMapKeysIter,
            Value::HashMapValuesIter(_, _) => ValueKind::HashMapValuesIter,
            Value::HashMapIter(_, _) => ValueKind::HashMapIter,
            Value::HashSetIter(_, _) => ValueKind::HashSetIter,
            Value::MapIter(_, _) => ValueKind::MapIter,
            Value::FilterIter(_, _) => ValueKind::FilterIter,
            Value::Random(_) => ValueKind::Random,
            Value::IOError(_) => ValueKind::IOError,
            Value::File(_) => ValueKind::File,
            Value::Ordering(_) => ValueKind::Ordering,
        }
    }
}

/// Where a value crossed into typed storage. (WP-VALUE-REP-TOTAL, A2.)
///
/// Named rather than a `&str` so a boundary cannot be misspelled at a call site, and so adding a
/// boundary in A4 is a deliberate edit here rather than a new string literal. It appears in the
/// diagnostic because "expected `&[UInt8]`, found owned `Vec`" is far cheaper to act on when it
/// also says whether that happened at a parameter or a return.
/// **How an invocation's generic environment is supplied.** Explicit in every case: a callable with
/// no generics is [`InvocationEnv::Empty`], not missing metadata, so "nothing to install" and
/// "nobody installed it" stay distinguishable — the confusion DEV-197 lived in.
enum InvocationEnv {
    /// Not generic, and encloses no instantiation.
    Empty,
    /// The checker published an instantiation against this call expression.
    Published(ExprId),
    /// Concrete bindings a specialiser produced (bound dispatch, trait defaults).
    Concrete(Vec<(crate::typecheck::GenericBinder, Ty)>),
    /// A function value's captured bindings, already concrete at capture time (DEV-178).
    /// Installed through the pre-existing `push_captured_env`, not a second helper.
    Captured(FunctionValue),
}

/// **Test-only producer mutations — AS3 / DEV-121 class evidence.**
///
/// Twelve wired boundaries that never fire prove nothing: "found no defect" and "is not running"
/// look identical from outside. This is the control that separates them.
///
/// **The mutation is applied to a PRODUCER, never to `check_value_for_ty`.** Corrupting the
/// predicate would only show that the predicate detects an artificial mismatch; corrupting a
/// producer shows that a real value, taking a real path, is stopped by the real funnel at the
/// intended boundary. Each arm below is a place where the interpreter *constructs* a
/// representation, and the mutation makes it construct the wrong one.
///
/// `#[cfg(test)]` throughout: no part of this compiles into a shipped compiler, so there is no
/// runtime switch that could corrupt a real build.
///
/// **Carried on the `Interpreter`, not in a global or a thread-local.** The first attempt used a
/// thread-local and every mutation silently failed to arm: `run` executes the program on a
/// *spawned* thread with a larger stack (`on_interpreter_stack`), so the interpreter never saw
/// what the test thread set. A process-global would have armed correctly and been worse — the
/// test harness runs tests in parallel, so an armed mutation would have corrupted whatever
/// unrelated test happened to be executing beside it. A field on the instance is scoped to exactly
/// the one execution under test.
#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ProducerMutation {
    /// Class 1 — owned/view. `String::as_str` and `Vec::as_slice` emit OWNED storage where the
    /// declared type is a view. The original DEV-121 pairing.
    OwnedForView,
    /// Class 2 — reference. A `&self` receiver binds the pointee BY VALUE instead of a
    /// `Value::Ref` into the caller's place. The materialization defect, deliberately reintroduced.
    OwnedForReference,
    /// Class 3 — function value. A function item coerces to something that is not a function.
    NonFunctionValue,
    /// Class 4 — aggregate. A declared field receives a mis-represented value, injected AFTER the
    /// producer-side boundary has already accepted it so the aggregate boundary is what must catch
    /// it.
    WrongAggregateField,
    /// Audit 10-D — a function value keeps its identity but LOSES its captured generic bindings.
    /// The representation stays a `Value::Function`, so only the environment is corrupted: this is
    /// DEV-178's defect, not DEV-121's.
    StripFunctionValueBindings,
    /// Audit 10-E — a mis-represented value reaches an element/field WRITE, injected after the
    /// producer boundary accepted it so the write boundary is what must catch it. The aggregate
    /// class already has a control at construction; this is the other route into typed storage.
    WrongElementWrite,
}

/// **Test-only environment mutation — AS3 #2 requalification.**
///
/// AS3 criterion 2 claims that every dispatch class installs the checker-selected generic
/// environment. The claim is only meaningful if OMITTING the environment is observable, and
/// DEV-197 is the standing proof that it often was not: nine dispatch sites installed nothing at
/// all and every test passed, because the bodies involved never mentioned their own parameters.
///
/// So the requalification does not assert that a table has an entry. It removes the environment at
/// the single installation point and requires the run to fail — once per dispatch class, on a
/// witness whose behaviour genuinely depends on the instantiation.
#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum EnvMutation {
    /// The environment is selected and delivered, and then not installed — exactly DEV-197's
    /// shape. Not "install the wrong bindings": an absent environment is the failure the
    /// architecture is claimed to prevent.
    DropEnvironment,
}

/// **Where a body's `self` comes from, before it is materialized.**
///
/// Materialization belongs to the invocation authority rather than to each caller, because the
/// binding it produces must match the receiver type the checker published — which is what
/// `RepBoundary::Receiver` will compare against.
///
/// The destructor case is why this exists. Destruction holds an OWNED value, but a
/// `Drop::drop(&mut self)` publishes `&mut Self`. Handing the owned value straight in would make
/// the receiver boundary reject every destructor, and the only sound answers are to materialize a
/// real reference or to weaken `&mut T` — the second would gut DEV-121 exactly when it is being
/// closed.
enum ReceiverSource {
    /// A free function or associated function.
    None,
    /// A method receiver already at a caller-side place.
    Place { kind: hir::Receiver, place: Place },
    /// A destructor's owned value. Materialized into temporary backing storage in the CALLER's
    /// frame, so the body-visible `self` is a genuine `Value::Ref` and the mutated value can be
    /// read back afterwards.
    OwnedForDrop(Value),
}

/// **What happens to the frame when a body finishes.** A destructor differs from an ordinary call
/// in two ways that were previously expressed by having a whole second executor: it hands back the
/// receiver's FINAL value (a `Drop::drop(&mut self)` may legally mutate or replace fields, and the
/// recursive field destruction must see that), and it does not run `cleanup_current_frame`, because
/// it is already inside destruction.
///
/// Making the difference a parameter is what lets one executor serve both — it is not a new
/// semantic rule, it is the existing difference named.
#[derive(Clone)]
enum BodyEpilogue {
    /// Ordinary call: clean the frame's locals, return the body's value.
    Call,
    /// Destructor: return the receiver's final value, read back from the temporary backing
    /// storage its `self` reference pointed at; the enclosing destruction walk owns the remaining
    /// locals. Pairs only with [`ReceiverSource::OwnedForDrop`], which is what creates that
    /// storage — the authority holds the place, so no caller can hand back a different one.
    Destructor,
    /// Method: like `Call`, plus two things a method owes its caller — a `&mut self` receiver is
    /// written back on the error path, and a returned reference derived from `self` is rebased onto
    /// the CALLER's receiver place (DEV-035), because the place it carries points into the frame
    /// that was just popped.
    Method {
        receiver_kind: hir::Receiver,
        receiver_local: LocalId,
        receiver_place: Place,
    },
}

/// A callable selected together with the environment it must run under. Constructing one is the
/// only way to reach [`Interpreter::execute_body`].
struct ResolvedInvocation {
    callable: Callable,
    environment: InvocationEnv,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepBoundary {
    LetBinding,
    Parameter,
    Receiver,
    Return,
    Propagation,
    MatchBinding,
    LoopBinding,
    Assignment,
    FieldWrite,
    ElementWrite,
    AggregateField,
    /// **The producer side.** Every other variant names a place a value comes to REST; this one
    /// names the moment a value is produced by an expression and handed to whatever consumes it.
    /// It is what covers inline values that never bind to a local — a builtin's argument, an
    /// operand of a runtime operation — which the eleven destination boundaries cannot see.
    ExpressionResult,
}

impl RepBoundary {
    /// How the boundary reads in a diagnostic, as a noun phrase that completes
    /// "representation mismatch at ...".
    pub fn as_str(self) -> &'static str {
        match self {
            RepBoundary::LetBinding => "a let binding",
            RepBoundary::Parameter => "a function parameter",
            RepBoundary::Receiver => "a method receiver",
            RepBoundary::Return => "a function return",
            RepBoundary::Propagation => "a propagated return",
            RepBoundary::MatchBinding => "a match binding",
            RepBoundary::LoopBinding => "a loop binding",
            RepBoundary::Assignment => "an assignment",
            RepBoundary::FieldWrite => "a field write",
            RepBoundary::ElementWrite => "an element write",
            RepBoundary::AggregateField => "an aggregate field",
            RepBoundary::ExpressionResult => "an expression result",
        }
    }
}

/// A function ITEM that has become a value, carrying the instantiation it was created with.
/// (WP-VALUE-REP-TOTAL A3c-S2, DEV-178.)
///
/// **The environment travels with the value because it is fixed at the COERCION, not the call.**
/// `let f: fn() -> UInt64 = type_size::<Int32>;` selects `T = Int32` when the item becomes a value;
/// the later `f()` has call-site type `fn() -> UInt64`, which says what the result is and can never
/// say what `T` was. Validating an indirect call against that caller-side type would satisfy a
/// parameter check while leaving the body without its generic context — a patch over DEV-176's
/// defect rather than a repair of it.
///
/// `bindings` are already CONCRETE. A function value may outlive the generic frame that created it,
/// so an unresolved caller parameter stored here would reference a frame that no longer exists.
///
/// A payload change, not a new representation: `Ty::Fn` still maps to `ValueKind::Function`, so
/// §6's matrix is unchanged and `value_matches_ty` is untouched.
#[derive(Clone, Debug)]
struct FunctionValue {
    item: ItemId,
    bindings: Vec<(String, Ty)>,
}

impl PartialEq for FunctionValue {
    /// Equal when they name the same item. Two values of one function at different instantiations
    /// are the same function; `Ty` is not comparable, and comparing bindings would make `f == f`
    /// depend on how each was created.
    fn eq(&self, other: &Self) -> bool {
        self.item == other.item
    }
}

#[derive(Clone)]
struct FileResource(Rc<RefCell<Option<std::fs::File>>>);

impl FileResource {
    fn new(file: std::fs::File) -> Self {
        Self(Rc::new(RefCell::new(Some(file))))
    }

    fn identity(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }
}

impl PartialEq for FileResource {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

/// WP-C2.2 (DEV-032): insertion-ordered map backing `Value::HashMap`, per the normative
/// iteration-order rule (`06-Standard-Library.md` "Iteration Order", CD-009): first insertion
/// appends; re-inserting an existing key updates its value in place without moving it;
/// remove-then-reinsert places the key at the end. Linear key search — performance is not a
/// goal for the reference interpreter; observable ordering semantics are. Equality is
/// content-based (order-independent), preserving the prior `BTreeMap`-era semantics; `Ord`
/// compares canonicalized (sorted) entry lists so it stays consistent with `Eq`.
#[derive(Clone)]
struct InsertionMap(Vec<(Value, Option<Value>)>);

impl InsertionMap {
    fn new() -> Self {
        InsertionMap(Vec::new())
    }
    fn position(&self, key: &Value) -> Option<usize> {
        self.0.iter().position(|(k, _)| k == key)
    }
    fn insert(&mut self, key: Value, value: Option<Value>) -> Option<Option<Value>> {
        match self.position(&key) {
            Some(index) => Some(std::mem::replace(&mut self.0[index].1, value)),
            None => {
                self.0.push((key, value));
                None
            }
        }
    }
    fn get(&self, key: &Value) -> Option<&Option<Value>> {
        self.position(key).map(|index| &self.0[index].1)
    }
    fn remove(&mut self, key: &Value) -> Option<Option<Value>> {
        self.position(key).map(|index| self.0.remove(index).1)
    }
    fn contains_key(&self, key: &Value) -> bool {
        self.position(key).is_some()
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    fn clear(&mut self) {
        self.0.clear();
    }
    fn keys(&self) -> impl Iterator<Item = &Value> {
        self.0.iter().map(|(k, _)| k)
    }
    fn values(&self) -> impl Iterator<Item = &Option<Value>> {
        self.0.iter().map(|(_, v)| v)
    }
    fn values_mut(&mut self) -> impl Iterator<Item = &mut Option<Value>> {
        self.0.iter_mut().map(|(_, v)| v)
    }
    fn iter(&self) -> impl Iterator<Item = (&Value, &Option<Value>)> {
        self.0.iter().map(|(k, v)| (k, v))
    }
    fn sorted_entries(&self) -> Vec<(&Value, &Option<Value>)> {
        let mut entries: Vec<_> = self.0.iter().map(|(k, v)| (k, v)).collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        entries
    }
}

impl PartialEq for InsertionMap {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self
                .0
                .iter()
                .all(|(key, value)| other.get(key) == Some(value))
    }
}

impl InsertionMap {
    fn canonical_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.sorted_entries().cmp(&other.sorted_entries())
    }
}

/// WP-C2.2 (DEV-032): insertion-ordered set backing `Value::HashSet` — same ordering rules and
/// comparison semantics as `InsertionMap`.
#[derive(Clone)]
struct InsertionSet(Vec<Value>);

impl InsertionSet {
    fn new() -> Self {
        InsertionSet(Vec::new())
    }
    fn insert(&mut self, value: Value) -> bool {
        if self.0.contains(&value) {
            false
        } else {
            self.0.push(value);
            true
        }
    }
    fn remove(&mut self, value: &Value) -> bool {
        match self.0.iter().position(|v| v == value) {
            Some(index) => {
                self.0.remove(index);
                true
            }
            None => false,
        }
    }
    fn contains(&self, value: &Value) -> bool {
        self.0.contains(value)
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    fn clear(&mut self) {
        self.0.clear();
    }
    fn iter(&self) -> impl Iterator<Item = &Value> {
        self.0.iter()
    }
    fn sorted_entries(&self) -> Vec<&Value> {
        let mut entries: Vec<_> = self.0.iter().collect();
        entries.sort();
        entries
    }
    fn canonical_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.sorted_entries().cmp(&other.sorted_entries())
    }
}

impl PartialEq for InsertionSet {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.0.iter().all(|value| other.contains(value))
    }
}

/// Mirrors the `IOError` enum in `06-Standard-Library.md`. Given its own
/// runtime representation (like `Value::Option`/`Value::Result`) rather
/// than going through the generic `Value::Enum{item,variant,..}` path,
/// since it has no corresponding real HIR item — `IOError::NotFound` etc.
/// resolve directly to `Builtin` constructors (`resolve.rs`), the same
/// pattern already used for `Some`/`None`/`Ok`/`Err`.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum IOErrorKind {
    NotFound,
    PermissionDenied,
    AlreadyExists,
    InvalidInput,
    Other(String),
}

impl fmt::Display for IOErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IOErrorKind::NotFound => write!(f, "NotFound"),
            IOErrorKind::PermissionDenied => write!(f, "PermissionDenied"),
            IOErrorKind::AlreadyExists => write!(f, "AlreadyExists"),
            IOErrorKind::InvalidInput => write!(f, "InvalidInput"),
            IOErrorKind::Other(msg) => write!(f, "Other({msg})"),
        }
    }
}

impl IOErrorKind {
    /// Map a `std::io::Error` to the closest `IOErrorKind`, matching the
    /// spec's variant set (`NotFound`/`PermissionDenied`/`AlreadyExists`/
    /// `InvalidInput`/`Other`).
    fn from_io_error(error: &std::io::Error) -> Self {
        match error.kind() {
            std::io::ErrorKind::NotFound => IOErrorKind::NotFound,
            std::io::ErrorKind::PermissionDenied => IOErrorKind::PermissionDenied,
            std::io::ErrorKind::AlreadyExists => IOErrorKind::AlreadyExists,
            std::io::ErrorKind::InvalidInput | std::io::ErrorKind::InvalidData => {
                IOErrorKind::InvalidInput
            }
            _ => IOErrorKind::Other(error.to_string()),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Unit => write!(f, "()"),
            Value::Bool(value) => write!(f, "{value}"),
            Value::Int(value) => write!(f, "{value}"),
            Value::Float(value, FloatWidth::F32) => {
                write!(f, "{}", canonical_float32(*value as f32))
            }
            Value::Float(value, FloatWidth::F64) => write!(f, "{}", canonical_float(*value)),
            Value::Char(value) => write!(f, "{value}"),
            Value::Str(value) | Value::String(value) => write!(f, "{value}"),
            Value::Tuple(values) => write_sequence(f, "(", ")", values),
            Value::Array(values) | Value::Vec(values) => write_sequence(f, "[", "]", values),
            Value::Struct { fields, .. } => {
                write!(f, "{{")?;
                for (index, (name, value)) in fields.iter().enumerate() {
                    if index > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{name}: {}", display_slot(value))?;
                }
                write!(f, "}}")
            }
            Value::Enum {
                variant, fields, ..
            } => {
                write!(f, "variant#{variant}")?;
                if !fields.is_empty() {
                    write_sequence(f, "(", ")", fields)?;
                }
                Ok(())
            }
            Value::Boxed(value) => write!(f, "Box({})", display_slot(value)),
            Value::Option(Some(value)) => write!(f, "Some({})", display_slot(value)),
            Value::Option(None) => write!(f, "None"),
            Value::Result(Ok(value)) => write!(f, "Ok({})", display_slot(value)),
            Value::Result(Err(value)) => write!(f, "Err({})", display_slot(value)),
            Value::Range {
                start,
                end,
                inclusive,
            } => write!(f, "{start}..{}{end}", if *inclusive { "=" } else { "" }),
            Value::Slice(_, start, end) => write!(f, "<slice {start}..{end}>"),
            Value::Ref(_) => write!(f, "<reference>"),
            Value::Function(func) => write!(f, "fn#{}", func.item.0),
            Value::CharsIter(..) => write!(f, "<CharsIter>"),
            Value::SplitIter(..) => write!(f, "<SplitIter>"),
            Value::VecIter(..) => write!(f, "<VecIter>"),
            Value::HashMap(map) => {
                write!(f, "HashMap{{")?;
                for (index, (k, v)) in map.iter().enumerate() {
                    if index > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{k}: {}", display_slot(v))?;
                }
                write!(f, "}}")
            }
            Value::HashSet(set) => {
                write!(f, "HashSet{{")?;
                for (index, val) in set.iter().enumerate() {
                    if index > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{val}")?;
                }
                write!(f, "}}")
            }
            Value::HashMapKeysIter(..) => write!(f, "<KeysIter>"),
            Value::HashMapValuesIter(..) => write!(f, "<ValuesIter>"),
            Value::HashMapIter(..) => write!(f, "<HashMapIter>"),
            Value::HashSetIter(..) => write!(f, "<HashSetIter>"),
            Value::MapIter(..) => write!(f, "<MapIter>"),
            Value::FilterIter(..) => write!(f, "<FilterIter>"),
            Value::Random(_) => write!(f, "<Random>"),
            Value::Ordering(ordering) => write!(
                f,
                "{}",
                match ordering {
                    std::cmp::Ordering::Less => "Less",
                    std::cmp::Ordering::Equal => "Equal",
                    std::cmp::Ordering::Greater => "Greater",
                }
            ),
            Value::IOError(kind) => write!(f, "{kind}"),
            Value::File(_) => write!(f, "<File>"),
        }
    }
}

impl Eq for Value {}

impl Ord for Value {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        fn discriminant(val: &Value) -> u8 {
            match val {
                Value::Unit => 0,
                Value::Bool(_) => 1,
                Value::Int(_) => 2,
                Value::Float(..) => 3,
                Value::Char(_) => 4,
                Value::Str(_) => 5,
                Value::String(_) => 6,
                Value::Tuple(_) => 7,
                Value::Array(_) => 8,
                Value::Struct { .. } => 9,
                Value::Enum { .. } => 10,
                Value::Vec(_) => 11,
                Value::Boxed(_) => 12,
                Value::Option(_) => 13,
                Value::Result(_) => 14,
                Value::Range { .. } => 15,
                Value::Slice(..) => 16,
                Value::Ref(_) => 17,
                Value::Function(_) => 18,
                Value::CharsIter(..) => 19,
                Value::SplitIter(..) => 20,
                Value::VecIter(..) => 21,
                Value::HashMap(_) => 22,
                Value::HashSet(_) => 23,
                Value::HashMapKeysIter(..) => 24,
                Value::HashMapValuesIter(..) => 25,
                Value::HashMapIter(..) => 26,
                Value::HashSetIter(..) => 27,
                Value::MapIter(..) => 28,
                Value::FilterIter(..) => 29,
                Value::Random(_) => 30,
                Value::Ordering(_) => 31,
                Value::IOError(_) => 32,
                Value::File(_) => 33,
            }
        }

        let da = discriminant(self);
        let db = discriminant(other);
        if da != db {
            return da.cmp(&db);
        }

        match (self, other) {
            (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
            (Value::Int(a), Value::Int(b)) => a.cmp(b),
            (Value::Float(a, _), Value::Float(b, _)) => a.total_cmp(b),
            (Value::Char(a), Value::Char(b)) => a.cmp(b),
            (Value::Str(a), Value::Str(b)) | (Value::String(a), Value::String(b)) => a.cmp(b),
            (Value::Tuple(a), Value::Tuple(b))
            | (Value::Array(a), Value::Array(b))
            | (Value::Vec(a), Value::Vec(b)) => a.cmp(b),
            (
                Value::Struct {
                    item: ia,
                    fields: fa,
                },
                Value::Struct {
                    item: ib,
                    fields: fb,
                },
            ) => ia.cmp(ib).then_with(|| fa.cmp(fb)),
            (
                Value::Enum {
                    item: ia,
                    variant: va,
                    fields: fa,
                    named: na,
                },
                Value::Enum {
                    item: ib,
                    variant: vb,
                    fields: fb,
                    named: nb,
                },
            ) => ia
                .cmp(ib)
                .then_with(|| va.cmp(vb))
                .then_with(|| fa.cmp(fb))
                .then_with(|| na.cmp(nb)),
            (Value::Boxed(a), Value::Boxed(b)) => a.cmp(b),
            (Value::Option(a), Value::Option(b)) => a.cmp(b),
            (Value::Result(a), Value::Result(b)) => match (a, b) {
                (Ok(va), Ok(vb)) => va.cmp(vb),
                (Err(ea), Err(eb)) => ea.cmp(eb),
                (Ok(_), Err(_)) => std::cmp::Ordering::Less,
                (Err(_), Ok(_)) => std::cmp::Ordering::Greater,
            },
            (
                Value::Range {
                    start: sa,
                    end: ea,
                    inclusive: ia,
                },
                Value::Range {
                    start: sb,
                    end: eb,
                    inclusive: ib,
                },
            ) => sa.cmp(sb).then_with(|| ea.cmp(eb)).then_with(|| ia.cmp(ib)),
            (Value::Slice(pa, sa, ea), Value::Slice(pb, sb, eb)) => pa
                .frame
                .cmp(&pb.frame)
                .then_with(|| pa.local.0.cmp(&pb.local.0))
                .then_with(|| pa.projections.len().cmp(&pb.projections.len()))
                .then_with(|| sa.cmp(sb))
                .then_with(|| ea.cmp(eb)),
            (Value::Ref(a), Value::Ref(b)) => a
                .frame
                .cmp(&b.frame)
                .then_with(|| a.local.0.cmp(&b.local.0))
                .then_with(|| a.projections.len().cmp(&b.projections.len())),
            (Value::Function(a), Value::Function(b)) => a.item.cmp(&b.item),
            (Value::HashMap(a), Value::HashMap(b)) => a.canonical_cmp(b),
            (Value::HashSet(a), Value::HashSet(b)) => a.canonical_cmp(b),
            (Value::MapIter(ia, fa), Value::MapIter(ib, fb)) => ia.cmp(ib).then_with(|| fa.cmp(fb)),
            (Value::FilterIter(ia, fa), Value::FilterIter(ib, fb)) => {
                ia.cmp(ib).then_with(|| fa.cmp(fb))
            }
            (Value::Random(a), Value::Random(b)) => a.cmp(b),
            (Value::Ordering(a), Value::Ordering(b)) => a.cmp(b),
            (Value::IOError(a), Value::IOError(b)) => a.cmp(b),
            (Value::File(a), Value::File(b)) => a.identity().cmp(&b.identity()),
            _ => std::cmp::Ordering::Equal,
        }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// **DEV-209: read a prelude payload that must be present.**
///
/// The slot's `None` is the representation of a MOVE, and the ownership checker is what prevents a
/// program from reading moved storage. So an operation that requires a complete `Some`/`Ok`/`Err`
/// and finds an empty slot has been handed a value the checker should have rejected: that is an
/// `InternalInvariant`, not a new runtime outcome. Introducing a "use of moved value" trap here
/// would invent a language-level category to describe a compiler defect.
fn require_live_payload<'a>(
    slot: &'a Option<Value>,
    context: &'static str,
    span: Span,
) -> Result<&'a Value, RuntimeError> {
    slot.as_ref().ok_or_else(|| {
        RuntimeError::internal(
            format!("{context}: the variant's payload has already been moved out"),
            span,
        )
    })
}

/// **DEV-209: take a prelude payload, for an operation that genuinely consumes it.**
///
/// `unwrap`, `?`, an owned pattern binding, a consuming combinator and destruction are moves, and
/// `Some(v) -> None` is how the move is represented. Only code intentionally performing one should
/// call this; everything else uses [`require_live_payload`].
fn take_payload(
    slot: &mut Option<Value>,
    context: &'static str,
    span: Span,
) -> Result<Value, RuntimeError> {
    slot.take().ok_or_else(|| {
        RuntimeError::internal(
            format!("{context}: the variant's payload has already been moved out"),
            span,
        )
    })
}

/// [`take_payload`] for an OWNED payload box — the by-value match form, which is how most
/// consuming operations receive it.
fn own_payload(
    mut slot: Box<Option<Value>>,
    context: &'static str,
    span: Span,
) -> Result<Value, RuntimeError> {
    take_payload(&mut slot, context, span)
}

fn display_slot(value: &Option<Value>) -> String {
    value
        .as_ref()
        .map(ToString::to_string)
        .unwrap_or_else(|| "<moved>".to_string())
}

fn write_sequence(
    f: &mut fmt::Formatter<'_>,
    open: &str,
    close: &str,
    values: &[Option<Value>],
) -> fmt::Result {
    write!(f, "{open}")?;
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{}", display_slot(value))?;
    }
    write!(f, "{close}")
}

#[derive(Clone, PartialEq)]
enum Projection {
    Field(String),
    Index(usize),
    /// Stable entry position inside an insertion-ordered map. A live reference
    /// prevents structural mutation, so the position cannot change while the
    /// projection is usable.
    MapIndex(usize),
    /// **DEV-209: the payload of a prelude `Option`/`Result`.**
    ///
    /// Named rather than spelled `Index(0)` on purpose. `Index` carries INDEXING semantics — its
    /// projection failure is classified as an index-out-of-bounds trap (`projection_failure`) —
    /// and an absent payload behind a discriminant that already matched is not an index trap. It
    /// is an invariant violation, and the two must not report as the same thing.
    VariantPayload(usize),
}

/// DEV-065: an out-of-range `Index` projection is the language's index-out-of-bounds TRAP
/// (CORE-V1-ABSTRACT-MACHINE), not a moved-field condition — the generic message was
/// misleading for the most common trap a user can hit.
fn projection_failure(projection: &Projection, span: Span) -> RuntimeError {
    match projection {
        // WP-C7.9 G.3: an out-of-range projection IS the language's index trap, so it states its
        // category here rather than leaving a downstream normaliser to recognise the words
        // "index out of bounds".
        Projection::Index(_) | Projection::MapIndex(_) => RuntimeError::with_category(
            "index out of bounds",
            span,
            crate::mir::TrapCategory::IndexOutOfBounds,
        ),
        // Not a language trap: reaching a moved-out field means the compiler let something through
        // that it should have rejected, or the interpreter lost track of a move.
        Projection::Field(_) => RuntimeError::internal("use of moved or invalid field", span),
        // Not an index trap: the discriminant already matched, so the payload must be there.
        Projection::VariantPayload(_) => {
            RuntimeError::internal("a matched variant's payload is absent", span)
        }
    }
}

#[derive(Clone, PartialEq)]
struct Place {
    frame: usize,
    local: LocalId,
    projections: Vec<Projection>,
}

#[derive(Default)]
struct Frame {
    values: HashMap<LocalId, Option<Value>>,
    order: Vec<LocalId>,
}

impl Frame {
    fn insert(&mut self, local: LocalId, value: Option<Value>) {
        if !self.values.contains_key(&local) {
            self.order.push(local);
        }
        self.values.insert(local, value);
    }
}

/// WP-C7.9 Packet C — what a `match` is matching against, decided once per `match` from its own
/// scrutinee (PAT-BIND-001).
///
/// Before this existed, `match_pattern` took a `&Value` and had no place to build a reference
/// from, so every binding was a clone of the referent. That was observationally identical for
/// read-only use — which is why 17 of the rule's 19 cases agreed across all three engines — and
/// wrong the moment a binding was used AS a reference: the oracle failed with "cannot dereference
/// non-reference" where the type checker, MIR and native all said `&String`.
enum PatternSource {
    /// The scrutinee was read as a value: bindings move or copy, exactly as before.
    Owned(Value),
    /// The scrutinee is a place read THROUGH a reference. Non-`Copy` components bind as references
    /// to the original storage; `Copy` components still bind by value, because a `Copy` read moves
    /// nothing.
    Borrowed(Place),
}

enum Flow {
    Value(Value),
    Return(Value),
    Break(Value),
    Continue,
    Propagate(Value),
}

#[derive(Clone)]
struct Callable {
    receiver: Option<(hir::Receiver, LocalId)>,
    params: Vec<LocalId>,
    body: BlockId,
}

fn is_valid_main_return(ty: &Ty) -> bool {
    let unit = Ty::Primitive(Primitive::Unit);
    let int32 = Ty::Primitive(Primitive::Int32);
    let string = Ty::Primitive(Primitive::String);
    ty == &unit
        || ty == &int32
        || matches!(
            ty,
            Ty::Core(CoreType::Result, args)
                if args.len() == 2
                    && (args[0] == unit || args[0] == int32)
                    && args[1] == string
        )
}

fn main_result_to_status(value: Value, span: Span) -> Result<(u8, String), RuntimeError> {
    fn checked_status(value: Value, span: Span) -> Result<u8, RuntimeError> {
        match value {
            Value::Unit => Ok(0),
            Value::Int(value) => {
                // CD-150 CE3: a STATED category, and provenance defined as the entry file at 1:1.
                // PROC-EXIT-001 is violated by the entry's RESULT, not by an expression, so there is
                // no sub-expression the three engines could agree to blame; `Span::point(0)` is the
                // one location all three can report identically.
                u8::try_from(value).map_err(|_| {
                    RuntimeError::with_category(
                        "invalid-exit-status",
                        Span::synthetic(span.source),
                        crate::mir::TrapCategory::InvalidExitStatus,
                    )
                })
            }
            _ => Err(RuntimeError::new(
                "entrypoint returned a value inconsistent with its checked signature",
                span,
            )),
        }
    }

    match value {
        Value::Result(Ok(value)) => Ok((
            checked_status(
                own_payload(value, "the entrypoint's Ok payload", span)?,
                span,
            )?,
            String::new(),
        )),
        Value::Result(Err(message)) => {
            match own_payload(message, "the entrypoint's Err payload", span)? {
                Value::String(message) | Value::Str(message) => Ok((1, format!("{message}\n"))),
                _ => Err(RuntimeError::new("entrypoint error is not a String", span)),
            }
        }
        value => Ok((checked_status(value, span)?, String::new())),
    }
}

/// The host stack the interpreter is given to work in (WP-C7.9 Packet F).
///
/// **Why this exists at all.** The reference interpreter is a tree-walker: one STARK call consumes
/// a chain of Rust frames (`call_callable` → `eval_block` → `eval_stmt` → `eval_expr` → …), and
/// those frames are large in a debug build. Measured on the default 8 MiB main-thread stack, an
/// ordinary recursive function overflowed the process at roughly a hundred STARK frames — a depth
/// real programs reach, and one no language rule says anything about.
///
/// A depth CAP alone cannot fix that: a cap low enough to be safe on a default stack would reject
/// programs the language accepts. So the capacity is made real instead — execution runs on a thread
/// with a stack sized for `MAX_CALL_DEPTH` frames, and the cap then reports exhaustion *before* the
/// host runs out (`LIMIT-RESOURCE-001`: prevent host undefined behaviour, report the classified
/// failure). The reservation is virtual; only touched pages are committed.
pub const INTERPRETER_STACK_BYTES: usize = 256 * 1024 * 1024;

/// Runs `body` on a thread with [`INTERPRETER_STACK_BYTES`] of stack.
///
/// Scoped, so the caller's borrowed HIR and tables need no `'static` bound.
pub fn on_interpreter_stack<T: Send>(body: impl FnOnce() -> T + Send) -> T {
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .stack_size(INTERPRETER_STACK_BYTES)
            .spawn_scoped(scope, body)
            .expect("spawning the interpreter thread failed")
            .join()
            .unwrap_or_else(|payload| std::panic::resume_unwind(payload))
    })
}

pub fn run(
    hir: &Hir,
    file: crate::source::RegisteredSource,
    tables: &TypeTables,
) -> Result<Execution, RuntimeError> {
    on_interpreter_stack(move || run_here(hir, file, tables))
}

/// What a test armed for one execution. Both axes are independent: a producer mutation corrupts a
/// VALUE, an environment mutation removes an INSTANTIATION.
#[cfg(test)]
#[derive(Default, Clone, Copy)]
pub(crate) struct Mutations {
    pub(crate) producer: Option<ProducerMutation>,
    pub(crate) env: Option<EnvMutation>,
}

/// One mutated execution, with the evidence that the mutation was actually reached.
#[cfg(test)]
pub(crate) struct MutatedRun {
    pub(crate) result: Result<Execution, RuntimeError>,
    /// **Why this is reported rather than assumed.** A dispatch-class control whose witness never
    /// reaches the installation point would "detect" nothing and look like a pass. Requiring a
    /// non-zero count is what makes each of the seven classes evidence about ITS OWN path.
    pub(crate) env_mutations_applied: usize,
}

/// [`run`] with mutations armed, for the DEV-121 and AS3 #2 evidence. Test-only.
#[cfg(test)]
pub(crate) fn run_mutated(
    hir: &Hir,
    file: crate::source::RegisteredSource,
    tables: &TypeTables,
    mutations: Mutations,
) -> MutatedRun {
    on_interpreter_stack(move || {
        let mut interpreter = Interpreter::new(hir, file, tables);
        interpreter.mutation = mutations.producer;
        interpreter.env_mutation = mutations.env;
        let outcome = interpreter.run_main();
        let env_mutations_applied = interpreter.env_mutations_applied;
        let result = outcome.map(|(status, exit_stderr)| Execution {
            output: interpreter.output.clone(),
            status,
            stderr: format!("{}{exit_stderr}", interpreter.stderr),
        });
        MutatedRun {
            result,
            env_mutations_applied,
        }
    })
}

/// [`run`] without the stack switch, for a caller that is already on a suitable stack.
fn run_here(
    hir: &Hir,
    file: crate::source::RegisteredSource,
    tables: &TypeTables,
) -> Result<Execution, RuntimeError> {
    let mut interpreter = Interpreter::new(hir, file, tables);
    let (status, exit_stderr) = interpreter.run_main()?;
    // Program stderr first, then the entrypoint's `Err` bytes: the program wrote its own output
    // while running, and the completion message is by definition produced at the end.
    let stderr = format!("{}{exit_stderr}", interpreter.stderr);
    Ok(Execution {
        output: interpreter.output,
        status,
        stderr,
    })
}

impl ExecutionOutcome {
    /// The same outcome in [`run_with_partial_output`]'s shape, for callers that want the older
    /// two-way split. The stderr is preserved separately by the caller when it needs it.
    pub fn into_result(self) -> Result<Execution, (RuntimeError, String)> {
        match self.result {
            Ok(status) => Ok(Execution {
                output: self.output,
                status,
                stderr: self.stderr,
            }),
            Err(error) => Err((error, self.output)),
        }
    }
}

/// Run `main`, keeping both streams whatever the outcome. See [`ExecutionOutcome`].
pub fn run_capturing(
    hir: &Hir,
    file: crate::source::RegisteredSource,
    tables: &TypeTables,
) -> ExecutionOutcome {
    on_interpreter_stack(move || run_capturing_here(hir, file, tables))
}

fn run_capturing_here(
    hir: &Hir,
    file: crate::source::RegisteredSource,
    tables: &TypeTables,
) -> ExecutionOutcome {
    let mut interpreter = Interpreter::new(hir, file, tables);
    match interpreter.run_main() {
        Ok((status, exit_stderr)) => ExecutionOutcome {
            output: interpreter.output,
            stderr: format!("{}{exit_stderr}", interpreter.stderr),
            result: Ok(status),
        },
        Err(error) => ExecutionOutcome {
            output: interpreter.output,
            stderr: interpreter.stderr,
            result: Err(error),
        },
    }
}

/// Like [`run`], but a failure also carries the stdout accumulated before it. The MIR
/// differential comparator (C4.5e-0) needs output equality on trap paths too — two programs
/// printing different prefixes before the same trap are observably different.
pub fn run_with_partial_output(
    hir: &Hir,
    file: crate::source::RegisteredSource,
    tables: &TypeTables,
) -> Result<Execution, (RuntimeError, String)> {
    match run_capturing(hir, file, tables) {
        ExecutionOutcome {
            output,
            stderr,
            result: Ok(status),
        } => Ok(Execution {
            output,
            status,
            stderr,
        }),
        ExecutionOutcome {
            output,
            result: Err(error),
            ..
        } => Err((error, output)),
    }
}

/// Execute a specific zero-argument, receiverless function `item` as the
/// program entry point instead of `main` — used by the test runner
/// (`test_runner::run_test`) to invoke each discovered `test_*` function.
pub fn run_item(
    hir: &Hir,
    file: crate::source::RegisteredSource,
    tables: &TypeTables,
    item: ItemId,
) -> Result<Execution, RuntimeError> {
    on_interpreter_stack(move || run_item_here(hir, file, tables, item))
}

fn run_item_here(
    hir: &Hir,
    file: crate::source::RegisteredSource,
    tables: &TypeTables,
    item: ItemId,
) -> Result<Execution, RuntimeError> {
    let mut interpreter = Interpreter::new(hir, file, tables);
    let span = interpreter.hir.item(item).span;
    let callable = interpreter
        .item_callable(item)
        .ok_or_else(|| RuntimeError::new("item is not executable", span))?;
    interpreter.invoke_callable(
        ResolvedInvocation {
            callable,
            environment: InvocationEnv::Empty,
        },
        ReceiverSource::None,
        Vec::new(),
        span,
    )?;
    Ok(Execution {
        output: interpreter.output,
        status: 0,
        // A test function has no entrypoint completion, so this is the program's own stderr only.
        stderr: interpreter.stderr,
    })
}

struct Interpreter<'a> {
    hir: &'a Hir,
    /// AS1b-ii: the ENTRY source, registered. Frames still record the file that declares each body
    /// (DEV-069) because MIR and diagnostics read it; once every span carries its own source that
    /// per-frame tracking has nothing left to decide.
    file: crate::source::RegisteredSource,
    tables: &'a TypeTables,
    frames: Vec<Frame>,
    output: String,
    /// WP-C7.9 Packet D: the program's own stderr, captured rather than written through to the
    /// host. Ordered independently of `output` — each stream preserves its own order, which is all
    /// PROC-STREAM-001 requires of two separate streams.
    stderr: String,
    copy_items: HashSet<ItemId>,
    /// Nominals with a user destructor, by resolved identity (DEV-210's published authority).
    drop_items: HashSet<ItemId>,
    pending_propagation: Option<Value>,
    const_cache: HashMap<ItemId, Value>,
    const_stack: Vec<ItemId>,
    /// DEV-100: the active generic instantiations, innermost last.
    ///
    /// The ONLY generic machinery the oracle has, and deliberately so — before this it had none at
    /// all, which is why `size_of::<T>()` inside a generic body could not be answered. It carries
    /// call-time type substitutions and nothing else: no specialised bodies, no effect on value
    /// execution, no inference.
    ///
    /// Behind `Rc<RefCell<_>>` so the RAII guard can own a handle instead of borrowing `self` —
    /// a guard holding `&mut self.generic_frames` would conflict with the `&mut self` call it is
    /// meant to wrap.
    generic_frames: std::rc::Rc<std::cell::RefCell<Vec<HashMap<String, Ty>>>>,
    /// The one producer mutation armed for this execution, if any. See [`ProducerMutation`].
    #[cfg(test)]
    mutation: Option<ProducerMutation>,
    /// The environment mutation armed for this execution, if any. See [`EnvMutation`].
    #[cfg(test)]
    env_mutation: Option<EnvMutation>,
    /// How many times the environment mutation actually fired. A dispatch-class control that
    /// never reached the installation point would otherwise "pass" by testing nothing.
    #[cfg(test)]
    env_mutations_applied: usize,
}

/// RAII guard for one entry of [`Interpreter::generic_frames`].
///
/// A guard rather than a manual pop because the oracle's call paths return early through `?` on
/// traps and interpreter errors, and a missed pop would leave a stale instantiation installed for
/// every later query in the run — a silent wrong ANSWER rather than a visible failure.
struct GenericFrame {
    frames: std::rc::Rc<std::cell::RefCell<Vec<HashMap<String, Ty>>>>,
    pushed: bool,
}

impl Drop for GenericFrame {
    fn drop(&mut self) {
        if self.pushed {
            self.frames.borrow_mut().pop();
        }
    }
}

// ---------------------------------------------------------------- the representation model --

/// **The canonical `Ty` → `Value` relation, as a free function.**
///
/// Lifted out of `Interpreter` so the CHECKER can consult the same answer. It depends on nothing
/// but the type, the value and the Copy set — a `&self` receiver was never carrying anything else.
///
/// Two consumers, one authority: the oracle asks *"does this value represent this type"*, and the
/// checker asks *"can this type reach a value boundary at all"* — the second derived from the first
/// by [`ty_is_runtime_representable`] rather than by a second classification.
fn value_matches_ty_with(expected: &Ty, value: &Value, copy_items: &HashSet<ItemId>) -> bool {
    use crate::ast::Primitive;
    let kind = value.kind();
    match expected {
        // ---------------------------------------------------------------- §6.2 scalars/text --
        Ty::Primitive(Primitive::Unit) => kind == ValueKind::Unit,
        Ty::Primitive(Primitive::Bool) => kind == ValueKind::Bool,
        Ty::Primitive(Primitive::Char) => kind == ValueKind::Char,
        // Width is not carried by `Value::Int`, so there is nothing about it to observe here.
        // The payload's numeric domain belongs to checked arithmetic (§6.2.1).
        Ty::Primitive(
            Primitive::Int8
            | Primitive::Int16
            | Primitive::Int32
            | Primitive::Int64
            | Primitive::UInt8
            | Primitive::UInt16
            | Primitive::UInt32
            | Primitive::UInt64,
        ) => kind == ValueKind::Int,
        // `Value::Float` DOES carry a width, so it is checked: this examines information the
        // model genuinely possesses.
        Ty::Primitive(Primitive::Float32) => {
            matches!(value, Value::Float(_, FloatWidth::F32))
        }
        Ty::Primitive(Primitive::Float64) => {
            matches!(value, Value::Float(_, FloatWidth::F64))
        }
        // **Owned `String` has two spellings in the type system**, found by this match refusing
        // to compile without both: the resolver maps the name `String` to
        // `Ty::Primitive(Primitive::String)`, while `Ty::Core(CoreType::String, _)` also occurs.
        // Both are the same owned type and both must permit exactly `Value::String`; covering
        // only the `Core` one would have left every `String` binding unvalidated.
        Ty::Primitive(Primitive::String) => kind == ValueKind::String,
        // An unsized `str` is never a standalone value — only `&str` is (§6.6).
        Ty::Primitive(Primitive::Str) => false,
        // `tensor` extension element types (D3). Not executable in Core v1, so reaching a value
        // boundary with one means extension gating failed.
        Ty::Primitive(Primitive::Float16 | Primitive::BFloat16) => false,

        // ------------------------------------------------------------------ §6.4 references --
        // **`&mut [T]` has the same two representations `&[T]` has.** `Value::Slice(place, ..)`
        // is a view *into a place*: writing through it writes to that place, so it is exactly
        // as much a reference as `Value::Ref` is, and it is what `&mut v[1..3]` produces. The
        // one-line mutable arm predated the slice-view representation and so admitted only
        // `Ref` — asymmetric with `shared_ref_matches` by omission, not by rule.
        Ty::Ref {
            mutable: true,
            inner,
        } => {
            kind == ValueKind::Ref
                || (matches!(inner.as_ref(), Ty::Slice(_)) && kind == ValueKind::Slice)
        }
        Ty::Ref {
            mutable: false,
            inner,
        } => shared_ref_matches_with(inner, value, copy_items),

        // --------------------------------------------------- §6.3 owned aggregates/collections --
        Ty::Tuple(elements) => match value {
            Value::Tuple(slots) => slots.len() == elements.len(),
            _ => false,
        },
        Ty::Array(_, len) => match value {
            Value::Array(slots) => slots.len() as u64 == *len,
            _ => false,
        },
        Ty::Struct(item, _) => match value {
            Value::Struct { item: actual, .. } => actual == item,
            _ => false,
        },
        Ty::Enum(item, _) => match value {
            Value::Enum { item: actual, .. } => actual == item,
            _ => false,
        },
        Ty::Core(core, _) => Interpreter::core_ty_matches(*core, kind),
        Ty::Range(_) => kind == ValueKind::Range,
        Ty::Fn { .. } => kind == ValueKind::Function,

        // -------------------------------------------------------- §6.6 never at a boundary --
        // Listed individually rather than folded into a `_` arm: each is a distinct compiler
        // defect, and a wildcard here would also swallow any `Ty` variant added later.
        Ty::Slice(_) => false,
        Ty::Never => false,
        Ty::Param(_) => false,
        Ty::Infer(_) => false,
        Ty::Error => false,
        // Tensor, model and model-error types live INSIDE `Ty::Extension`; they are not
        // separate `Ty` variants. Not executable in Core v1, so reaching a value boundary with
        // one means extension gating failed.
        Ty::Extension(_) => false,
    }
}

fn shared_ref_matches_with(inner: &Ty, value: &Value, copy_items: &HashSet<ItemId>) -> bool {
    use crate::ast::Primitive;
    let kind = value.kind();

    // `&str`: a detached view, or a reference to text. NOT an owned `String` — that is the
    // DEV-121 pairing, where the static type says borrowed and move behaviour sees owned
    // storage.
    if matches!(inner, Ty::Primitive(Primitive::Str)) {
        return kind == ValueKind::Str || kind == ValueKind::Ref;
    }

    // `&[T]`: a view. A `Ref` to the container is accepted only because real producers make
    // one; `Vec`/`Array` immediately is the owned-storage error again.
    if matches!(inner, Ty::Slice(_)) {
        return kind == ValueKind::Slice || kind == ValueKind::Ref;
    }

    if kind == ValueKind::Ref {
        return true;
    }

    // The bare-value form, licensed ONLY by the pointee being Copy: copying a Copy pointee
    // cannot consume, invalidate or destroy the referent, so the two representations are
    // indistinguishable to any observation the oracle can make. Never extended to non-Copy `T`
    // for convenience.
    pointee_is_copy_with(inner, copy_items) && value_matches_ty_with(inner, value, copy_items)
}

/// Whether a shared reference's pointee is `Copy`. The CHECKER's predicate, never a second one.
fn pointee_is_copy_with(ty: &Ty, copy_items: &HashSet<ItemId>) -> bool {
    crate::typecheck::is_copy_type_with(ty, copy_items)
}

/// **Can a value of type `ty` exist at a runtime value boundary at all?**
///
/// DEV-206 asks a question one step upstream of DEV-121. DEV-121 asks *given a valid runtime type
/// `T`, does `V` represent it*; this asks *should `T` have been allowed to reach a value boundary*.
/// `[T]` is the case that made the difference visible: publishing it for `v[0..2]` is correct —
/// it is a place of unsized type — and letting it escape into `println(...)` is not.
///
/// **Derived, not enumerated.** A second list of "runtime-representable types" beside
/// `value_matches_ty` would be exactly the duplicate semantic authority this campaign removed. So
/// the answer is obtained by ASKING the relation: a type that no representation satisfies cannot
/// exist at a value boundary. The probe below is exhaustive over `ValueKind`, so a new runtime
/// representation forces someone to say what it is here.
pub fn ty_is_runtime_representable(ty: &Ty, copy_items: &HashSet<ItemId>) -> bool {
    ValueKind::ALL
        .iter()
        .any(|kind| value_matches_ty_with(ty, &probe_value(*kind, ty), copy_items))
}

/// One representative value per runtime representation, for [`ty_is_runtime_representable`].
///
/// Contents are irrelevant and deliberately minimal — the relation reads shape, never payload
/// (`a_value_kind_names_the_shape_and_not_the_contents`). Exhaustive on purpose: adding a
/// `ValueKind` without a probe would silently shrink what the property considers representable.
fn probe_value(kind: ValueKind, ty: &Ty) -> Value {
    let place = || Place {
        frame: 0,
        local: LocalId(0),
        projections: Vec::new(),
    };
    match kind {
        ValueKind::Unit => Value::Unit,
        ValueKind::Bool => Value::Bool(false),
        ValueKind::Int => Value::Int(0),
        ValueKind::Float => Value::Float(0.0, FloatWidth::F64),
        ValueKind::Char => Value::Char('a'),
        ValueKind::Str => Value::Str(String::new()),
        ValueKind::String => Value::String(String::new()),
        // **Arity comes from the TYPE.** The relation checks a tuple's arity and an array's
        // length, so a fixed-size probe would report `[Int32; 3]` unrepresentable — a wrong answer
        // about the type rather than about the relation. This supplies a fair witness, not a
        // second opinion about what is representable.
        ValueKind::Tuple => Value::Tuple(match ty {
            Ty::Tuple(elements) => vec![None; elements.len()],
            _ => Vec::new(),
        }),
        ValueKind::Array => Value::Array(match ty {
            Ty::Array(_, len) => vec![None; *len as usize],
            _ => Vec::new(),
        }),
        ValueKind::Struct => Value::Struct {
            item: ItemId(0),
            fields: BTreeMap::new(),
        },
        ValueKind::Enum => Value::Enum {
            item: ItemId(0),
            variant: 0,
            fields: Vec::new(),
            named: BTreeMap::new(),
        },
        ValueKind::Vec => Value::Vec(Vec::new()),
        ValueKind::Boxed => Value::Boxed(Box::new(None)),
        ValueKind::Option => Value::Option(None),
        ValueKind::Result => Value::Result(Ok(Box::new(Some(Value::Unit)))),
        ValueKind::Range => Value::Range {
            start: 0,
            end: 0,
            inclusive: false,
        },
        ValueKind::Slice => Value::Slice(place(), 0, 0),
        ValueKind::Ref => Value::Ref(place()),
        ValueKind::Function => Value::Function(FunctionValue {
            item: ItemId(0),
            bindings: Vec::new(),
        }),
        ValueKind::CharsIter => Value::CharsIter(String::new(), 0),
        ValueKind::SplitIter => Value::SplitIter(Vec::new(), 0),
        ValueKind::VecIter => Value::VecIter(place(), 0),
        ValueKind::HashMap => Value::HashMap(InsertionMap::new()),
        ValueKind::HashSet => Value::HashSet(InsertionSet::new()),
        ValueKind::HashMapKeysIter => Value::HashMapKeysIter(Vec::new(), 0),
        ValueKind::HashMapValuesIter => Value::HashMapValuesIter(Vec::new(), 0),
        ValueKind::HashMapIter => Value::HashMapIter(Vec::new(), 0),
        ValueKind::HashSetIter => Value::HashSetIter(Vec::new(), 0),
        ValueKind::MapIter => Value::MapIter(Box::new(Value::Unit), ItemId(0)),
        ValueKind::FilterIter => Value::FilterIter(Box::new(Value::Unit), ItemId(0)),
        ValueKind::Random => Value::Random(0),
        ValueKind::IOError => Value::IOError(IOErrorKind::Other(String::new())),
        ValueKind::File => Value::File(FileResource(Rc::new(RefCell::new(None)))),
        ValueKind::Ordering => Value::Ordering(std::cmp::Ordering::Equal),
    }
}

impl<'a> Interpreter<'a> {
    /// The registered id of the entry source. Used where a failure has no position of its own —
    /// an interpreter invariant, a missing entrypoint — so it names a real source instead of a
    /// fabricated one.
    fn entry_source(&self) -> crate::source::SourceId {
        self.file.id()
    }

    fn default_value_for(&self, value: &Value) -> Value {
        match value {
            Value::Unit => Value::Unit,
            Value::Bool(_) => Value::Bool(false),
            Value::Int(_) => Value::Int(0),
            Value::Float(_, width) => Value::Float(0.0, *width),
            Value::Char(_) => Value::Char('\0'),
            Value::Str(_) | Value::String(_) => Value::String(String::new()),
            Value::Tuple(elems) => {
                let default_elems = elems
                    .iter()
                    .map(|opt| opt.as_ref().map(|v| self.default_value_for(v)))
                    .collect();
                Value::Tuple(default_elems)
            }
            Value::Array(elems) => {
                let default_elems = elems
                    .iter()
                    .map(|opt| opt.as_ref().map(|v| self.default_value_for(v)))
                    .collect();
                Value::Array(default_elems)
            }
            Value::Struct { item, fields } => {
                let mut default_fields = std::collections::BTreeMap::new();
                for (name, val_opt) in fields {
                    let def_val = val_opt.as_ref().map(|v| self.default_value_for(v));
                    default_fields.insert(name.clone(), def_val);
                }
                Value::Struct {
                    item: *item,
                    fields: default_fields,
                }
            }
            Value::Enum {
                item,
                variant,
                fields,
                named,
            } => {
                let default_fields = fields
                    .iter()
                    .map(|opt| opt.as_ref().map(|v| self.default_value_for(v)))
                    .collect();
                let mut default_named = std::collections::BTreeMap::new();
                for (name, val_opt) in named {
                    let def_val = val_opt.as_ref().map(|v| self.default_value_for(v));
                    default_named.insert(name.clone(), def_val);
                }
                Value::Enum {
                    item: *item,
                    variant: *variant,
                    fields: default_fields,
                    named: default_named,
                }
            }
            Value::Vec(_) => Value::Vec(Vec::new()),
            Value::Boxed(inner) => {
                let default_inner = inner.as_ref().as_ref().map(|v| self.default_value_for(v));
                Value::Boxed(Box::new(default_inner))
            }
            Value::Option(_) => Value::Option(None),
            // A default mirrors the shape, INCLUDING whether the payload is still there: a moved
            // payload's default is still moved.
            Value::Result(res) => match res {
                Ok(val) => Value::Result(Ok(Box::new(
                    (**val).as_ref().map(|value| self.default_value_for(value)),
                ))),
                Err(err) => Value::Result(Err(Box::new(
                    (**err).as_ref().map(|value| self.default_value_for(value)),
                ))),
            },
            Value::Range {
                start: _,
                end: _,
                inclusive,
            } => Value::Range {
                start: 0,
                end: 0,
                inclusive: *inclusive,
            },
            Value::Slice(place, start, end) => Value::Slice(place.clone(), *start, *end),
            Value::Ref(place) => Value::Ref(place.clone()),
            Value::Function(func) => Value::Function(func.clone()),
            Value::CharsIter(..) => Value::CharsIter(String::new(), 0),
            Value::SplitIter(..) => Value::SplitIter(Vec::new(), 0),
            Value::VecIter(place, _) => Value::VecIter(place.clone(), 0),
            Value::HashMap(_) => Value::HashMap(InsertionMap::new()),
            Value::HashSet(_) => Value::HashSet(InsertionSet::new()),
            Value::HashMapKeysIter(..) => Value::HashMapKeysIter(Vec::new(), 0),
            Value::HashMapValuesIter(..) => Value::HashMapValuesIter(Vec::new(), 0),
            Value::HashMapIter(..) => Value::HashMapIter(Vec::new(), 0),
            Value::HashSetIter(..) => Value::HashSetIter(Vec::new(), 0),
            Value::MapIter(inner, item) => {
                Value::MapIter(Box::new(self.default_value_for(inner)), *item)
            }
            Value::FilterIter(inner, item) => {
                Value::FilterIter(Box::new(self.default_value_for(inner)), *item)
            }
            Value::Random(_) => Value::Random(0),
            Value::Ordering(_) => Value::Ordering(std::cmp::Ordering::Equal),
            Value::IOError(_) => Value::IOError(IOErrorKind::NotFound),
            Value::File(resource) => Value::File(resource.clone()),
        }
    }

    fn new(hir: &'a Hir, file: crate::source::RegisteredSource, tables: &'a TypeTables) -> Self {
        // WP-C6.1g-a: the interpreter's Copy set is the same structural+impl eligibility the
        // checker and MIR use (OWN-COPY-001, amended), so all three engines agree.
        let copy_items = crate::typecheck::copy_eligible_types(hir);
        let drop_items = crate::typecheck::nominals_with_destructor(hir);
        Self {
            hir,
            file,
            tables,
            frames: Vec::new(),
            output: String::new(),
            stderr: String::new(),
            copy_items,
            drop_items,
            pending_propagation: None,
            const_cache: HashMap::new(),
            const_stack: Vec::new(),
            generic_frames: std::rc::Rc::new(std::cell::RefCell::new(Vec::new())),
            #[cfg(test)]
            mutation: None,
            #[cfg(test)]
            env_mutation: None,
            #[cfg(test)]
            env_mutations_applied: 0,
        }
    }

    /// Whether `mutation` is the one armed for this execution.
    #[cfg(test)]
    fn mutation_armed(&self, mutation: ProducerMutation) -> bool {
        self.mutation == Some(mutation)
    }

    fn eval_const_item(&mut self, item: ItemId) -> Result<Value, RuntimeError> {
        if let Some(value) = self.const_cache.get(&item) {
            return Ok(value.clone());
        }
        if let Some(start) = self
            .const_stack
            .iter()
            .position(|candidate| *candidate == item)
        {
            let mut names: Vec<String> = self.const_stack[start..]
                .iter()
                .map(|item| self.item_name(*item))
                .collect();
            names.push(self.item_name(item));
            return Err(RuntimeError::new(
                format!("constant dependency cycle: {}", names.join(" -> ")),
                self.hir.item(item).span,
            ));
        }
        let hir::ItemKind::Const { value, .. } = &self.hir.item(item).kind else {
            return Err(RuntimeError::new(
                "item is not a constant",
                self.hir.item(item).span,
            ));
        };
        self.const_stack.push(item);
        let result = self.expect_value(*value);
        self.const_stack.pop();
        let value = result?;
        self.const_cache.insert(item, value.clone());
        Ok(value)
    }

    fn item_name(&self, item: ItemId) -> String {
        match &self.hir.item(item).kind {
            hir::ItemKind::Const { name, .. } => self.text(*name).to_string(),
            _ => format!("#{}", item.0),
        }
    }

    /// Read a span, against the source the span itself names.
    ///
    /// **AS1b-ii: this is the whole point of the packet.** There used to be two readers —
    /// `text`, which indexed the ambient `self.file` and was correct only for the body currently
    /// executing, and `item_text`, which took an `ItemId` to find the right file for a
    /// cross-item read (DEV-069). Choosing between them was a judgement every call site had to
    /// make, and getting it wrong produced a plausible wrong answer rather than an error.
    ///
    /// A span now names its source, so there is nothing to choose. `item_text` delegates here and
    /// its `ItemId` is redundant; the ambient `self.file` no longer participates in reading at all.
    fn text(&self, span: Span) -> &str {
        match self.hir.sources.get(span.source) {
            Some(file) => file
                .src
                .get(span.lo as usize..span.hi as usize)
                .unwrap_or("?"),
            None => "?",
        }
    }

    /// Read a span belonging to `item`.
    ///
    /// Retained only so DEV-069's call sites keep reading as they did; the `item` argument no
    /// longer decides anything, because the span carries its own source.
    fn item_text(&self, _item: ItemId, span: Span) -> &str {
        self.text(span)
    }

    fn run_main(&mut self) -> Result<(u8, String), RuntimeError> {
        let snippet_items;
        let root_items = match &self.hir.root {
            hir::Root::Program(items) => items.as_slice(),
            hir::Root::Snippet { .. } => {
                snippet_items = (0..self.hir.items.len())
                    .map(|index| ItemId(index as u32))
                    .collect::<Vec<_>>();
                snippet_items.as_slice()
            }
        };
        let mains: Vec<ItemId> = root_items
            .iter()
            .copied()
            .filter(|item| match &self.hir.item(*item).kind {
                hir::ItemKind::Fn(def) => self.text(def.sig.name) == "main",
                hir::ItemKind::Const { name, .. }
                | hir::ItemKind::TypeAlias { name, .. }
                | hir::ItemKind::Struct { name, .. }
                | hir::ItemKind::Enum { name, .. }
                | hir::ItemKind::Trait { name, .. }
                | hir::ItemKind::Model(hir::ModelDef { name, .. }) => self.text(*name) == "main",
                _ => false,
            })
            .collect();
        let Some(&main) = mains.first() else {
            return Err(RuntimeError::entry(
                "program has no 'main' function",
                Span::synthetic(self.entry_source()),
            ));
        };
        if mains.len() != 1 {
            return Err(RuntimeError::entry(
                "program must have exactly one root 'main' function",
                self.hir.item(main).span,
            ));
        }
        let hir::ItemKind::Fn(def) = &self.hir.item(main).kind else {
            return Err(RuntimeError::entry(
                "root item 'main' is not a function",
                self.hir.item(main).span,
            ));
        };
        if !def.sig.generics.is_empty() || !def.sig.params.is_empty() || def.sig.receiver.is_some()
        {
            return Err(RuntimeError::entry(
                "'main' must be non-generic and have no parameters",
                def.sig.span,
            ));
        }
        let Some((params, ret_ty)) = self.tables.fn_types.get(&main) else {
            return Err(RuntimeError::entry(
                "missing checked signature for 'main'",
                def.sig.span,
            ));
        };
        if !params.is_empty() || !is_valid_main_return(ret_ty) {
            return Err(RuntimeError::entry(
                "'main' must return Unit, Int32, Result<Unit, String>, or Result<Int32, String>",
                def.sig.span,
            ));
        }
        let callable = self.item_callable(main).ok_or_else(|| {
            RuntimeError::new("'main' is not executable", self.hir.item(main).span)
        })?;
        let result = self.invoke_callable(
            ResolvedInvocation {
                callable,
                environment: InvocationEnv::Empty,
            },
            ReceiverSource::None,
            Vec::new(),
            self.hir.item(main).span,
        )?;
        main_result_to_status(result, self.hir.item(main).span)
    }

    fn item_callable(&self, item: ItemId) -> Option<Callable> {
        let hir::ItemKind::Fn(def) = &self.hir.item(item).kind else {
            return None;
        };
        Some(Callable {
            receiver: None,
            params: def.sig.params.iter().map(|param| param.local).collect(),
            body: def.body,
        })
    }

    /// The published environment for a `Display` render position, keyed the way the plan is.
    fn display_env(&self, root: ExprId, path: &DisplayPath) -> InvocationEnv {
        self.tables
            .display_uses
            .get(&(root, path.clone()))
            .and_then(|id| self.tables.callable_uses.get(id.0 as usize))
            .and_then(|use_| self.env_for_use(use_))
            .unwrap_or(InvocationEnv::Empty)
    }

    /// **The environment a published `CallableUse` calls for.**
    ///
    /// One mapping from the checker's `GenericEnvironment` to the interpreter's `InvocationEnv`,
    /// so every consumer of a published selection installs the same thing. Six method-dispatch
    /// paths installed nothing at all before Packet 1 made the parameter mandatory — operators,
    /// qualified core-trait calls, container `Eq`, `Iterator::next` and both `Display` paths.
    ///
    /// `FromBoundSelection` returns `None`: a bound selection's environment comes from the
    /// specialiser, which produces it atomically with the body, and reconstructing it here would be
    /// the second answer Rule 1 forbids.
    fn env_for_use(&self, use_: &crate::typecheck::CallableUse) -> Option<InvocationEnv> {
        match &use_.environment {
            crate::typecheck::GenericEnvironment::Static(bindings) => {
                Some(InvocationEnv::Concrete(bindings.clone()))
            }
            crate::typecheck::GenericEnvironment::FromBoundSelection => None,
            crate::typecheck::GenericEnvironment::FromFunctionValue => None,
        }
    }

    /// The published environment for a core-trait dispatch at `expr`, or `Empty` when the selection
    /// is a non-generic `Static` one. Used by the operator, iterator and `Display` paths, which all
    /// select through `selected_core_trait_callable`.
    fn core_trait_env(&self, expr: ExprId, core: CoreTrait) -> InvocationEnv {
        self.selected_core_trait_callable(expr, core)
            .as_ref()
            .and_then(|use_| self.env_for_use(use_))
            .unwrap_or(InvocationEnv::Empty)
    }

    /// **DEV-035:** a reference returned from a `&self`/`&mut self` method that was derived from
    /// `self` carries a `Place` pointing into the method's own — just popped — frame, so any later
    /// dereference failed with "dangling reference". Rebase it onto the caller-side receiver place,
    /// preserving projections taken inside the method.
    ///
    /// References into the method's OTHER locals are left untouched: the borrow checker's
    /// return-escape check (E0103) rejects those, and if one ever slipped through the existing
    /// "dangling reference" trap is the correct backstop, not a silent rebase.
    fn rebase_if_method(
        &self,
        mut value: Value,
        body_frame: usize,
        epilogue: &BodyEpilogue,
    ) -> Value {
        if let BodyEpilogue::Method {
            receiver_local,
            receiver_place,
            ..
        } = epilogue
        {
            rebase_frame_refs(&mut value, body_frame, *receiver_local, receiver_place);
        }
        value
    }

    /// **AS3 Packet 1: the ONE invocation authority.**
    ///
    /// Rule 1 — *no executable callable body without an invocation authority*. Body, generic
    /// environment and published signature must be established **atomically** before execution.
    ///
    /// The pattern this replaces —
    ///
    /// ```text
    /// push environment
    /// call_callable(...)
    /// ```
    ///
    /// — put the first step on each call site, and DEV-197 is what that costs: two dispatch paths
    /// omitted it, ran their bodies with `T` unbound, and looked correct because no boundary
    /// consulted the callee's declared types. An environment a caller can forget is not an
    /// invariant.
    ///
    /// [`Self::execute_body`] is the raw executor; this is its only production caller.
    fn invoke_callable(
        &mut self,
        invocation: ResolvedInvocation,
        receiver: ReceiverSource,
        args: Vec<Value>,
        span: Span,
    ) -> Result<Value, RuntimeError> {
        self.invoke_with_epilogue(invocation, receiver, args, BodyEpilogue::Call, span)
    }

    /// The authority proper. [`Self::invoke_callable`] is the ordinary-call spelling; a destructor
    /// differs only in its [`BodyEpilogue`], so it shares this entry rather than owning a second
    /// executor.
    fn invoke_with_epilogue(
        &mut self,
        invocation: ResolvedInvocation,
        receiver: ReceiverSource,
        args: Vec<Value>,
        epilogue: BodyEpilogue,
        span: Span,
    ) -> Result<Value, RuntimeError> {
        let ResolvedInvocation {
            callable,
            environment,
        } = invocation;
        // Installed FIRST and live for the whole call, so every boundary below resolves
        // `Ty::Param` against the callee's own instantiation.
        let _env = self.install_invocation_env(&environment, span)?;
        self.execute_body(callable, receiver, args, epilogue, span)
    }

    /// Install an invocation's environment. Every variant is explicit: "this callable has no
    /// generics" is [`InvocationEnv::Empty`], never absent metadata.
    fn install_invocation_env(
        &mut self,
        environment: &InvocationEnv,
        span: Span,
    ) -> Result<GenericFrame, RuntimeError> {
        // AS3 #2 control: deliver the environment and then fail to install it.
        #[cfg(test)]
        if self.env_mutation == Some(EnvMutation::DropEnvironment) {
            self.env_mutations_applied += 1;
            return Ok(GenericFrame {
                frames: self.generic_frames.clone(),
                pushed: false,
            });
        }
        match environment {
            InvocationEnv::Empty => Ok(GenericFrame {
                frames: self.generic_frames.clone(),
                pushed: false,
            }),
            InvocationEnv::Published(call_expr) => self.push_callable_env(*call_expr, span),
            InvocationEnv::Concrete(bindings) => self.push_resolved_env(bindings, span),
            // **One authority for captured bindings.** `push_captured_env` already existed; a
            // second helper doing the same job was a duplicate introduced during the Return
            // wiring, and is deleted.
            InvocationEnv::Captured(callee) => Ok(self.push_captured_env(callee)),
        }
    }

    /// **The raw body executor.** Its only production caller is [`Self::invoke_callable`], because
    /// reaching it directly is how a body comes to run without its generic environment (DEV-197).
    /// Do not add a second caller; route through the authority.
    fn execute_body(
        &mut self,
        callable: Callable,
        receiver: ReceiverSource,
        args: Vec<Value>,
        epilogue: BodyEpilogue,
        span: Span,
    ) -> Result<Value, RuntimeError> {
        if args.len() != callable.params.len() {
            return Err(RuntimeError::new("runtime argument count mismatch", span));
        }
        // **The declared signature, looked up ONCE, before anything is bound.**
        //
        // A missing signature is an INVARIANT VIOLATION, not an exemption. A3b's claim is that
        // every executable body has a published signature; an `Option` here would reintroduce the
        // "missing metadata means skip validation" pattern AS3 spent this sprint deleting, and
        // would let any future body that loses its entry quietly stop being checked.
        //
        // Hoisted from the return boundary so the receiver and the parameters are read against the
        // SAME published signature the return is. Three boundaries, one lookup: a body cannot be
        // checked on the way out but unchecked on the way in.
        let signature = self
            .tables
            .callable_types
            .get(&callable.body)
            .cloned()
            .ok_or_else(|| {
                RuntimeError::internal(
                    format!(
                        "missing callable signature for executable body {:?} — A3b publishes one \
                         for every executable body, so this is a publication defect, not a \
                         callable class to exempt",
                        callable.body
                    ),
                    span,
                )
            })?;
        // **A3: `pending_propagation` is an intra-expression adapter, never live across a call.**
        //
        // It is interpreter state, not frame state, and `expect_value` parks a propagated value
        // there while returning a dummy `Value::Unit` for the caller to consume immediately. That
        // makes it correct only within one expression's evaluation: a value still parked when a
        // callable boundary is crossed would be attributed to the WRONG function — read against a
        // callee's return type on the way in, or a caller's on the way out.
        //
        // Establishing this as a checked invariant is what makes A4's return validation sound. It
        // is the difference between "validate `Flow::Propagate` against `callable.ret`" being true
        // and being an assumption, and it is why this lands before the wiring rather than with it.
        if self.pending_propagation.is_some() {
            return Err(RuntimeError::internal(
                "DEV-121: a pending propagation entered a callable boundary — `?` must be consumed \
                 within the expression that produced it",
                span,
            ));
        }
        // WP-C7.9 Packet F: the interpreter's own call-depth capacity, checked BEFORE the frame is
        // pushed. Without it, a deeply recursive STARK program consumed the host's Rust stack and
        // the process aborted — taking the test runner with it, with no classification and no way
        // for a harness to tell "the program recursed too deeply" from "the interpreter crashed".
        // `LIMIT-RESOURCE-001` already names call depth and already says what this is: a
        // host/process failure, reported rather than crashed into, with an implementation-defined
        // capacity.
        if self.frames.len() >= MAX_CALL_DEPTH {
            return Err(RuntimeError::host_resource(
                format!(
                    "call depth limit reached ({MAX_CALL_DEPTH} frames): the program exceeded this \
                     implementation's call-depth capacity"
                ),
                span,
            ));
        }
        // **Materialization.** The binding produced here must match the receiver type the checker
        // published, which is what `RepBoundary::Receiver` compares against.
        let mut drop_backing = None;
        let materialized = match receiver {
            ReceiverSource::None => None,
            // WP-C2.2 (DEV-034): a by-value receiver CONSUMES the already-resolved place — proper
            // move semantics, including partial moves out of fields — rather than re-evaluating the
            // receiver expression, which was a confirmed double-evaluation bug.
            ReceiverSource::Place {
                kind: hir::Receiver::Value,
                place,
            } => Some(self.take_place(&place, span)?),
            // Class 2 mutation: bind the pointee BY VALUE, which is the materialization defect
            // the destructor case was repaired for.
            #[cfg(test)]
            ReceiverSource::Place { ref place, .. }
                if self.mutation_armed(ProducerMutation::OwnedForReference) =>
            {
                Some(self.place_value(place, span)?.clone())
            }
            // DEV-070 (A2): `&self`/`&mut self` bind a genuine REFERENCE to the caller's place,
            // not a clone.
            ReceiverSource::Place { place, .. } => Some(Value::Ref(place)),
            // The destructor case: an owned value becomes a real `&mut Self` by giving it backing
            // storage in the CALLER's frame, which outlives the body and can be read back.
            ReceiverSource::OwnedForDrop(value) => {
                let backing = self.promote_to_owned_temp_place(value, span)?;
                drop_backing = Some(backing.clone());
                Some(Value::Ref(backing))
            }
        };
        let mut frame = Frame::default();
        if let (Some((_, local)), Some(value)) = (callable.receiver, materialized) {
            // **The RECEIVER boundary.** `callable_types` records the receiver *as the body binds
            // it* — `Self` for `self`, `&Self` for `&self`, `&mut Self` for `&mut self` — so this
            // compares the materialized binding against exactly what the body will read.
            //
            // A destructor passes here with no `Drop`-shaped exception: its `ReceiverSource::
            // OwnedForDrop` became a genuine `Value::Ref` into caller-frame backing storage, which
            // is what `&mut Self` means. That is the whole reason materialization moved into the
            // authority.
            let declared = signature.receiver.as_ref().ok_or_else(|| {
                RuntimeError::internal(
                    "a callable with a runtime receiver has no receiver in its published \
                     signature — A3b forms both from the same declaration, so they cannot \
                     legitimately disagree",
                    span,
                )
            })?;
            self.check_value_for_ty(declared, &value, span, RepBoundary::Receiver)?;
            frame.insert(local, Some(value));
        }
        // **The PARAMETER boundary** — the second uncovered site DEV-121 named. Read against the
        // published signature rather than `local_types`, so a parameter is checked by the contract
        // the CALLEE declared, under the callee's own instantiation.
        if signature.params.len() != callable.params.len() {
            return Err(RuntimeError::internal(
                format!(
                    "published signature for body {:?} declares {} parameters but the callable \
                     binds {} — A3b forms both from the same declaration",
                    callable.body,
                    signature.params.len(),
                    callable.params.len()
                ),
                span,
            ));
        }
        for ((local, declared), value) in callable
            .params
            .iter()
            .copied()
            .zip(signature.params.iter())
            .zip(args)
        {
            self.check_value_for_ty(declared, &value, span, RepBoundary::Parameter)?;
            frame.insert(local, Some(value));
        }
        self.frames.push(frame);
        let body_frame = self.frames.len() - 1;
        // AS1b-ii: DEV-069's per-call file swap is GONE. It existed so `self.file` named the
        // executing body's file while `text()` sliced against it; `text()` now slices against the
        // source the span itself names, so there is nothing for the swap to keep correct.
        // DEV-113-B stamped the raising file onto the error here, because a trap inside a
        // dependency was otherwise attributed to the entry file. AS1b-ii-d deleted the stamp: it
        // was derived from `error.span.source`, so it was a copy of something the error already
        // carried, and a copy is a thing that can disagree.
        let result = self.eval_block(callable.body);
        if let BodyEpilogue::Destructor = epilogue {
            // The pairing is structural: only `OwnedForDrop` creates backing storage, so a
            // `Destructor` epilogue without it is a miswired call site, not a program error.
            let Some(backing) = drop_backing else {
                self.frames.pop();
                return Err(RuntimeError::internal(
                    "a `Destructor` epilogue ran without an owned receiver to give back",
                    span,
                ));
            };
            // The destructor's own locals belong to the enclosing destruction walk, so no
            // `cleanup_current_frame` here — and the receiver comes back because `drop()` may have
            // mutated or replaced fields that the recursive field destruction must then see.
            self.frames.pop();
            result?;
            // Read the (possibly mutated) value back out of the backing storage the body's `self`
            // referenced. `Drop::drop(&mut self)` may replace fields, and the recursive field
            // destruction that follows must see that.
            let restored = self.take_place(&backing, span).ok();
            // **Not `unwrap_or(Unit)`.** The callable has a receiver, it was inserted, and a
            // `Drop` body cannot legitimately make its own receiver binding vanish — so `None`
            // here is an interpreter defect. Falling back to `Unit` would let a representation or
            // lifetime bug erase the receiver and have destruction continue on nothing, silently
            // skipping the recursive field destruction that follows.
            return restored.ok_or_else(|| {
                RuntimeError::internal(
                    "the `Drop` receiver disappeared while its destructor executed",
                    span,
                )
            });
        }
        if let Err(error) = result {
            if let BodyEpilogue::Method {
                receiver_kind,
                receiver_local,
                ref receiver_place,
            } = epilogue
            {
                // A `&mut self` receiver is written back even on the error path: the caller's place
                // was emptied to make the binding, and leaving it empty would lose the value.
                let restored = self
                    .frame_mut()
                    .values
                    .get_mut(&receiver_local)
                    .and_then(Option::take);
                let place = receiver_place.clone();
                self.frames.pop();
                if let (hir::Receiver::RefMut, Some(restored)) = (receiver_kind, restored) {
                    self.place_slot_mut(&place, span)?.replace(restored);
                }
                return Err(error);
            }
            self.frames.pop();
            return Err(error);
        }
        let flow = result?;
        // The other half of the invariant: nothing may still be parked on the way out either. A
        // value left here would be picked up by whatever the CALLER evaluates next, silently
        // becoming that expression's propagation.
        if self.pending_propagation.is_some() {
            return Err(RuntimeError::internal(
                "DEV-121: a pending propagation escaped expression handling and reached a callable \
                 boundary",
                span,
            ));
        }
        // A `&self` receiver is taken back out before cleanup so the method's own locals are
        // destroyed without the borrowed receiver among them — the ordering `call_user_method`
        // established, preserved here rather than restated there.
        if let BodyEpilogue::Method {
            receiver_kind: hir::Receiver::Ref,
            receiver_local,
            ..
        } = epilogue
        {
            let _restored = self
                .frame_mut()
                .values
                .get_mut(&receiver_local)
                .and_then(Option::take);
        }
        self.cleanup_current_frame()?;
        self.frames.pop();
        let declared_ret = &signature.ret;
        match flow {
            // The RETURN boundary, against the same published signature the receiver and the
            // parameters were read against.
            Flow::Value(value) | Flow::Return(value) => {
                self.check_value_for_ty(declared_ret, &value, span, RepBoundary::Return)?;
                Ok(self.rebase_if_method(value, body_frame, &epilogue))
            }
            // **The PROPAGATION boundary.** A `?` that leaves the body IS the body's return value —
            // §6.5 requires the error type to match, so the propagated `Result::Err`/`Option::None`
            // is read against the declared return type exactly as an explicit `return` is. Leaving
            // it unchecked meant `?` was the one way out of a function that no boundary observed.
            Flow::Propagate(value) => {
                self.check_value_for_ty(declared_ret, &value, span, RepBoundary::Propagation)?;
                Ok(self.rebase_if_method(value, body_frame, &epilogue))
            }
            Flow::Break(_) | Flow::Continue => {
                Err(RuntimeError::new("loop control escaped a function", span))
            }
        }
    }

    fn eval_block(&mut self, block_id: BlockId) -> Result<Flow, RuntimeError> {
        let block = self.hir.block(block_id);
        let mut locals = Vec::new();
        for stmt in &block.stmts {
            if let hir::StmtKind::Let { local, .. } = self.hir.stmt(*stmt).kind {
                locals.push(local);
            }
            let flow = self.eval_stmt(*stmt)?;
            if !matches!(flow, Flow::Value(_)) {
                self.cleanup_locals(&locals)?;
                return Ok(flow);
            }
        }
        let flow = if let Some(tail) = block.tail {
            self.eval_expr(tail)?
        } else {
            Flow::Value(Value::Unit)
        };
        self.cleanup_locals(&locals)?;
        Ok(flow)
    }

    fn eval_stmt(&mut self, stmt_id: StmtId) -> Result<Flow, RuntimeError> {
        let stmt = self.hir.stmt(stmt_id);
        match &stmt.kind {
            hir::StmtKind::Empty => Ok(Flow::Value(Value::Unit)),
            hir::StmtKind::Expr { expr, .. } => match self.eval_expr(*expr)? {
                Flow::Value(value) => {
                    self.drop_value(value)?;
                    Ok(Flow::Value(Value::Unit))
                }
                flow => Ok(flow),
            },
            hir::StmtKind::Let { local, init, .. } => {
                let value = if let Some(init) = init {
                    let value = self.expect_value(*init)?;
                    if let Some(propagated) = self.pending_propagation.take() {
                        return Ok(Flow::Propagate(propagated));
                    }
                    Some(value)
                } else {
                    None
                };
                // A `let` with no initialiser binds nothing yet — definite assignment (§4) is
                // what guarantees the read cannot precede the write, so there is no value to check
                // and an empty slot is the correct state, not an unchecked one.
                match value {
                    Some(value) => {
                        self.bind_typed_local(*local, value, stmt.span, RepBoundary::LetBinding)?
                    }
                    None => {
                        self.frame_mut().insert(*local, None);
                    }
                }
                Ok(Flow::Value(Value::Unit))
            }
            hir::StmtKind::Return(expr) => {
                let value = if let Some(expr) = expr {
                    self.expect_value(*expr)?
                } else {
                    Value::Unit
                };
                Ok(Flow::Return(
                    self.pending_propagation.take().unwrap_or(value),
                ))
            }
            hir::StmtKind::Break(expr) => Ok(Flow::Break(if let Some(expr) = expr {
                self.expect_value(*expr)?
            } else {
                Value::Unit
            })),
            hir::StmtKind::Continue => Ok(Flow::Continue),
            hir::StmtKind::Item(_) => Ok(Flow::Value(Value::Unit)),
            hir::StmtKind::Error => Err(RuntimeError::new("invalid statement", stmt.span)),
        }
    }

    /// **The one entry point every boundary uses.** (WP-VALUE-REP-TOTAL, A2 §7.)
    ///
    /// Normalises `expected` against the active instantiation, then asks the relation. A refusal is
    /// `FailureClass::InternalInvariant` — a compiler defect, never a language trap and never a
    /// user type error — so the differential harness fails loudly rather than accepting an oracle
    /// bug as a program outcome and pressuring MIR and native into reproducing it.
    ///
    /// **The diagnostic names the shape and never the contents.** Printing the value would leak
    /// program data into compiler output, and describing it would mean cloning or borrowing it,
    /// which is the behaviour this class of check exists to police.
    ///
    /// Wired to no boundary yet: A4 does that. At A2 this is reachable only from tests, which is
    /// deliberate — the relation is provable before it is enforced.
    fn check_value_for_ty(
        &self,
        expected: &Ty,
        value: &Value,
        span: Span,
        boundary: RepBoundary,
    ) -> Result<(), RuntimeError> {
        // The resolution failure carries no boundary of its own, and "an unsubstituted parameter
        // reached a value boundary" is unactionable without knowing WHICH — so name it here.
        let concrete = self
            .concrete_runtime_ty(expected, span)
            .map_err(|mut error| {
                error.message = format!("{} [at {}]", error.message, boundary.as_str());
                error
            })?;
        if self.value_matches_ty(&concrete, value) {
            return Ok(());
        }
        Err(RuntimeError::internal(
            format!(
                "DEV-121 representation mismatch at {}: expected `{concrete:?}`, found `{}`",
                boundary.as_str(),
                value.kind().as_str()
            ),
            span,
        ))
    }

    /// **The one way a value comes to rest in a local.** AS3 Packet 3.
    ///
    /// `let`, a `match` arm's pattern bindings, and both `for` forms each did their own
    /// `frame_mut().insert(local, Some(value))`, and only two of the four checked anything. That is
    /// how DEV-121 survived: the check was a thing a site could remember to do, so the sites that
    /// forgot were indistinguishable from the sites with nothing to check.
    ///
    /// Every caller names its own [`RepBoundary`], because "a value entered a local" is not
    /// actionable — which of the four is. The expected type is `local_types[local]`, the checker's
    /// answer for that binding, never anything reconstructed from the value.
    ///
    /// **A missing `local_types` entry is an `InternalInvariant`, not a skip.** Every caller here
    /// is a LANGUAGE-level binding — a `let`, a `match` arm's pattern binding, a `for` loop item —
    /// and the checker types all of them. Inheriting
    /// [`Self::check_local_value_if_typed`]'s permissiveness would have left a missing-metadata
    /// escape inside a wire the inventory reports as `Wired`: structurally present, silently
    /// inert for any binding whose entry went missing. That is the pattern this sprint exists to
    /// delete, and a funnel is exactly where it would be least visible.
    fn bind_typed_local(
        &mut self,
        local: LocalId,
        value: Value,
        span: Span,
        boundary: RepBoundary,
    ) -> Result<(), RuntimeError> {
        let expected = self
            .tables
            .local_types
            .get(&local)
            .cloned()
            .ok_or_else(|| {
                RuntimeError::internal(
                    format!(
                        "missing checker-published local type at {} — every language-level \
                         binding is typed, so this is a publication defect, not a binding to \
                         exempt",
                        boundary.as_str()
                    ),
                    span,
                )
            })?;
        self.check_value_for_ty(&expected, &value, span, boundary)?;
        self.frame_mut().insert(local, Some(value));
        Ok(())
    }

    /// [`check_value_for_ty`] against an expression's published type.
    ///
    /// A missing entry is `internal`: the checker types every expression it accepts, so an absent
    /// one means the tables and the tree disagree.
    fn check_expr_value(&self, expr: ExprId, value: &Value) -> Result<(), RuntimeError> {
        let span = self.hir.expr(expr).span;
        let declared = self.tables.expr_types.get(&expr).cloned().ok_or_else(|| {
            RuntimeError::internal(
                "no published type for an evaluated expression — the checker types every \
                 expression it accepts, so this is a table/tree disagreement",
                span,
            )
        })?;
        self.check_value_for_ty(&declared, value, span, RepBoundary::ExpressionResult)
    }

    /// The relation of WP-VALUE-REP-TOTAL §6, as executable code.
    ///
    /// **Returns whether the pairing is PERMITTED. It never converts.** A validator that turned a
    /// `String` into a `Str`, a `Vec` into a `Slice`, or dereferenced a `Ref` to make a check pass
    /// would destroy the evidence it exists to expose; the repair belongs at the producer.
    ///
    /// The `Ty` match is exhaustive with no permissive wildcard. Where a type admits more than one
    /// representation the alternatives are named individually, so "permitted" is always a closed
    /// set and never an absence of opinion.
    fn value_matches_ty(&self, expected: &Ty, value: &Value) -> bool {
        value_matches_ty_with(expected, value, &self.copy_items)
    }

    /// §6.5. One representation each, named individually — a "these are all iterators" row would be
    /// a wildcard.
    fn core_ty_matches(core: hir::CoreType, kind: ValueKind) -> bool {
        use hir::CoreType;
        match core {
            CoreType::String => kind == ValueKind::String,
            CoreType::Vec => kind == ValueKind::Vec,
            CoreType::Box => kind == ValueKind::Boxed,
            CoreType::Option => kind == ValueKind::Option,
            CoreType::Result => kind == ValueKind::Result,
            CoreType::Range | CoreType::RangeInclusive => kind == ValueKind::Range,
            CoreType::CharsIter => kind == ValueKind::CharsIter,
            CoreType::SplitIter => kind == ValueKind::SplitIter,
            CoreType::VecIter => kind == ValueKind::VecIter,
            CoreType::HashMap => kind == ValueKind::HashMap,
            CoreType::HashSet => kind == ValueKind::HashSet,
            CoreType::KeysIter => kind == ValueKind::HashMapKeysIter,
            CoreType::ValuesIter => kind == ValueKind::HashMapValuesIter,
            // One core type serves both containers; the pair is closed and both are named.
            CoreType::Iter => kind == ValueKind::HashMapIter || kind == ValueKind::HashSetIter,
            CoreType::MapIter => kind == ValueKind::MapIter,
            CoreType::FilterIter => kind == ValueKind::FilterIter,
            CoreType::Random => kind == ValueKind::Random,
            CoreType::IOError => kind == ValueKind::IOError,
            CoreType::File => kind == ValueKind::File,
            CoreType::Ordering => kind == ValueKind::Ordering,
        }
    }

    /// Normalise `ty` against the active generic instantiation, or refuse.
    ///
    /// **A2 (WP-VALUE-REP-TOTAL §8).** Validation is meaningless against a type that still contains
    /// a parameter: `T` permits every representation, so a relation asked about `T` can only answer
    /// "yes". Refusing is the point — an unsubstituted parameter reaching a value boundary means
    /// the instantiation frame did not cover it, which is a compiler defect and not a program
    /// outcome.
    ///
    /// It reuses `typecheck::substitute_ty` and `ty_contains_param` rather than walking types
    /// again here, because a second substitution algorithm is a second answer to what a generic
    /// instantiation means.
    fn concrete_runtime_ty(&self, ty: &Ty, span: Span) -> Result<Ty, RuntimeError> {
        let concrete = match self.generic_frames.borrow().last() {
            Some(map) => crate::typecheck::substitute_ty(ty, map),
            None => ty.clone(),
        };
        // **Then discharge associated-type projections.** `fn first<T: Holder>(t: T) -> T::Item`
        // publishes `Param("T::Item")`, and substitution cannot touch it: the environment binds
        // `T`, not `T::Item`. Once `T` is concrete the projection has exactly one answer, and the
        // checker already computed it — `tables.assoc_projections`, keyed by (implementing
        // nominal, associated name). Consulted rather than re-derived; an oracle-local scan of the
        // impl set would be a third authority for a question `normalize_projections` and MIR's
        // `ProgramMeta` already answer.
        let concrete = self.resolve_projections(&concrete);
        if crate::typecheck::ty_contains_param(&concrete) {
            return Err(RuntimeError::internal(
                format!(
                    "DEV-121: `{concrete:?}` still contains an unsubstituted generic parameter at a \
                     value boundary — the active instantiation did not cover it"
                ),
                span,
            ));
        }
        Ok(concrete)
    }

    /// Replace every resolvable `Param("Base::Assoc")` in `ty` with the checker's binding.
    ///
    /// `Base` is looked up in the active generic frame; its nominal selects the impl, and
    /// `assoc_projections` gives the impl's `type Assoc = ...`. A projection whose base is still
    /// parametric is left alone — `ty_contains_param` then reports it, which is the correct
    /// outcome: an unresolvable projection at a value boundary is a missing instantiation, not
    /// something to guess at.
    fn resolve_projections(&self, ty: &Ty) -> Ty {
        let mut projections = std::collections::BTreeSet::new();
        crate::typecheck::collect_ty_params(ty, &mut projections);
        let frames = self.generic_frames.borrow();
        let bindings = frames.last();
        let mut map: HashMap<String, Ty> = HashMap::new();
        for name in projections {
            let Some((base, assoc)) = name.split_once("::") else {
                continue;
            };
            // The base may be bound in the environment (`T`) or already concrete in the type.
            let base_ty = match bindings.and_then(|b| b.get(base)) {
                Some(ty) => ty.clone(),
                None => continue,
            };
            let Some(nominal) = nominal_item_of_ty(&base_ty) else {
                continue;
            };
            if let Some(bound) = self
                .tables
                .assoc_projections
                .get(&(nominal, assoc.to_string()))
            {
                map.insert(name.clone(), bound.clone());
            }
        }
        if map.is_empty() {
            return ty.clone();
        }
        crate::typecheck::substitute_ty(ty, &map)
    }

    /// WP-FMT-001: pack a source-level format specification into the runtime's spec word.
    fn format_spec_word(spec: &crate::ast::FormatSpec) -> (u64, char) {
        use crate::ast::FormatKind;
        use stark_runtime::fmt_spec::{Align, Kind, Sign, Spec};
        let align = match spec.align {
            None => Align::Default,
            Some(crate::ast::FormatAlign::Left) => Align::Left,
            Some(crate::ast::FormatAlign::Right) => Align::Right,
            Some(crate::ast::FormatAlign::Center) => Align::Center,
        };
        let sign = match spec.sign {
            None | Some(crate::ast::FormatSign::Minus) => Sign::Minus,
            Some(crate::ast::FormatSign::Plus) => Sign::Plus,
            Some(crate::ast::FormatSign::Space) => Sign::Space,
        };
        let kind = match spec.kind {
            None => Kind::Display,
            Some(FormatKind::Bin) => Kind::Bin,
            Some(FormatKind::Oct) => Kind::Oct,
            Some(FormatKind::LowerHex) => Kind::LowerHex,
            Some(FormatKind::UpperHex) => Kind::UpperHex,
            Some(FormatKind::Fixed) => Kind::Fixed,
        };
        let word = Spec::pack(
            spec.width.unwrap_or(0),
            spec.precision.map(|p| p as u16),
            align,
            sign,
            spec.alternate,
            spec.zero_pad,
            kind,
        );
        (word, spec.fill.unwrap_or(' '))
    }

    /// WP-FMT-001: whether a specification asks for a NUMERIC rendering, which owns sign placement
    /// and radix and therefore takes the value rather than its `Display` text.
    fn format_spec_is_numeric(spec: &crate::ast::FormatSpec) -> bool {
        spec.kind.is_some()
            || spec.precision.is_some()
            || spec.sign.is_some()
            || spec.alternate
            || spec.zero_pad
    }

    /// WP-FMT-001: whether this expression denotes a PLACE.
    ///
    /// A place is borrowed for formatting and left alone; anything else is a temporary this
    /// evaluation created and must destroy. That distinction is what makes `f"{x}"` twice, then
    /// `use_value(x)`, legal while still dropping `f"{make_value()}"`'s temporary exactly once.
    fn format_field_is_place(&self, expr: ExprId) -> bool {
        match &self.hir.expr(expr).kind {
            hir::ExprKind::Path { res, .. } => {
                matches!(res, Res::Local(_) | Res::SelfValue(_))
            }
            hir::ExprKind::Field { .. } | hir::ExprKind::TupleField { .. } => true,
            hir::ExprKind::Unary {
                op: UnOp::Deref, ..
            } => true,
            _ => false,
        }
    }

    /// WP-FMT-001: render one field.
    ///
    /// A numeric specification renders the VALUE through `stark_runtime::fmt_spec` — the same
    /// functions the MIR interpreter and generated native code call, so there is no
    /// interpreter-local padding or rounding rule. Everything else renders through `display_text`,
    /// the very path `println` uses, and is then padded; that is what makes `f"{x}"` and `x.fmt()`
    /// agree by construction rather than by coincidence.
    fn render_format_field(
        &mut self,
        // AS3 Boundary 4: the field expression is its own Display root — the checker keyed the
        // plan on exactly this id, from `check_format_field`.
        field: ExprId,
        value: Value,
        spec: &crate::ast::FormatSpec,
        span: Span,
        owned: bool,
    ) -> Result<String, RuntimeError> {
        let (word, fill) = Self::format_spec_word(spec);
        if Self::format_spec_is_numeric(spec) {
            match &value {
                Value::Int(v) => {
                    let text = if *v < 0 {
                        let narrowed = i64::try_from(*v).map_err(|_| {
                            RuntimeError::new("integer out of range for formatting", span)
                        })?;
                        stark_runtime::fmt_spec::fmt_int_spec(narrowed, word, fill)
                    } else {
                        let narrowed = u64::try_from(*v).map_err(|_| {
                            RuntimeError::new("integer out of range for formatting", span)
                        })?;
                        stark_runtime::fmt_spec::fmt_uint_spec(narrowed, word, fill)
                    };
                    return Ok(text);
                }
                Value::Float(f, width) => {
                    let text = match width {
                        FloatWidth::F32 => {
                            stark_runtime::fmt_spec::fmt_float32_spec(*f as f32, word, fill)
                        }
                        FloatWidth::F64 => {
                            stark_runtime::fmt_spec::fmt_float64_spec(*f, word, fill)
                        }
                    };
                    return Ok(text);
                }
                _ => {}
            }
        }
        let (text, arg_place) = self.display_text(field, value, span)?;
        // A BORROWED field must not run the value's destructor: `display_text` promoted a copy of
        // a place's contents, and the place still owns the original.
        if owned {
            self.finish_display(arg_place, span)?;
        } else if let Some(place) = arg_place {
            let _ = self.take_place(&place, span);
        }
        Ok(stark_runtime::fmt_spec::fmt_pad_spec(&text, word, fill))
    }

    fn eval_expr(&mut self, expr_id: ExprId) -> Result<Flow, RuntimeError> {
        let expr = self.hir.expr(expr_id);
        match &expr.kind {
            hir::ExprKind::Lit(lit) => {
                let value = self.eval_lit(*lit, expr.span)?;
                Ok(Flow::Value(
                    self.normalize_numeric(value, expr_id, expr.span)?,
                ))
            }
            // WP-FMT-001: fields are evaluated in source order and exactly ONCE each. A place is
            // borrowed (its value cloned for rendering, the place untouched); anything else is a
            // temporary this evaluation owns and destroys after its bytes are appended.
            hir::ExprKind::FormatString { segments } => {
                let segments = segments.clone();
                let mut out = String::new();
                for segment in &segments {
                    match segment {
                        hir::FormatSegment::Literal { text, .. } => out.push_str(text),
                        hir::FormatSegment::Field {
                            expr,
                            spec,
                            expr_span,
                            ..
                        } => {
                            let is_place = self.format_field_is_place(*expr);
                            let (value, owned) = if is_place {
                                let place = self.expr_place(*expr)?;
                                let place = self.deref_place(place, *expr_span)?;
                                (self.clone_place_value(&place, *expr_span)?, false)
                            } else {
                                match self.eval_expr(*expr)? {
                                    // **DEV-203: this consumed an expression result unchecked.**
                                    // An interpolated field is precisely the "inline value entering
                                    // a runtime operation" class `ExpressionResult` exists for — it
                                    // never binds to a local, so no destination boundary sees it —
                                    // and it reached the renderer through a direct `eval_expr`
                                    // rather than through `expect_value`.
                                    Flow::Value(value) => {
                                        self.check_expr_value(*expr, &value)?;
                                        (value, true)
                                    }
                                    // Not a boundary: control flow leaving the interpolation is
                                    // returned to the enclosing expression, carrying no value that
                                    // comes to rest here.
                                    other => return Ok(other),
                                }
                            };
                            let rendered =
                                self.render_format_field(*expr, value, spec, *expr_span, owned)?;
                            out.push_str(&rendered);
                        }
                    }
                }
                Ok(Flow::Value(Value::String(out)))
            }
            hir::ExprKind::Path { res, .. } => Ok(Flow::Value(self.eval_path(*res, expr_id)?)),
            hir::ExprKind::Unary { op, operand } => {
                let value = match op {
                    UnOp::Ref { .. } => {
                        let place = self.expr_place(*operand)?;
                        // A range-index place is a synthetic slot containing a slice view.
                        // The language value for `&base[a..b]` is the view itself, not a
                        // reference to that synthetic slot (which dies with a method frame).
                        match self.place_value(&place, expr.span)?.clone() {
                            Value::Slice(base, start, end) => Value::Slice(base, start, end),
                            _ => Value::Ref(place),
                        }
                    }
                    UnOp::Deref => {
                        let reference = self.expect_value(*operand)?;
                        let Value::Ref(place) = reference else {
                            return Err(RuntimeError::new(
                                "cannot dereference non-reference",
                                expr.span,
                            ));
                        };
                        self.clone_place_value(&place, expr.span)?
                    }
                    _ => self.expect_value(*operand)?,
                };
                Ok(Flow::Value(
                    self.eval_unary(*op, value, expr_id, expr.span)?,
                ))
            }
            hir::ExprKind::Binary { op, lhs, rhs } => {
                if *op == BinOp::And {
                    let left = self.expect_value(*lhs)?;
                    if let Some(propagated) = self.pending_propagation.take() {
                        return Ok(Flow::Propagate(propagated));
                    }
                    let Value::Bool(left) = left else {
                        return Err(RuntimeError::new("expected Bool", self.hir.expr(*lhs).span));
                    };
                    if !left {
                        return Ok(Flow::Value(Value::Bool(false)));
                    }
                    let right = self.expect_bool(*rhs)?;
                    if let Some(propagated) = self.pending_propagation.take() {
                        return Ok(Flow::Propagate(propagated));
                    }
                    return Ok(Flow::Value(Value::Bool(right)));
                }
                if *op == BinOp::Or {
                    let left = self.expect_value(*lhs)?;
                    if let Some(propagated) = self.pending_propagation.take() {
                        return Ok(Flow::Propagate(propagated));
                    }
                    let Value::Bool(left) = left else {
                        return Err(RuntimeError::new("expected Bool", self.hir.expr(*lhs).span));
                    };
                    if left {
                        return Ok(Flow::Value(Value::Bool(true)));
                    }
                    let right = self.expect_bool(*rhs)?;
                    if let Some(propagated) = self.pending_propagation.take() {
                        return Ok(Flow::Propagate(propagated));
                    }
                    return Ok(Flow::Value(Value::Bool(right)));
                }
                // Equality/ordering desugar to `Eq::eq(&self, &other)`/`Ord::cmp(&self, &other)`
                // (03-Type-System.md "Operators and Traits"): both operands are borrowed, not
                // consumed. Evaluating a place operand (a local, field, index, or deref target)
                // through the ordinary move-or-copy path would move a non-`Copy` value out of
                // its storage just to compare it, making it unusable afterward even though the
                // comparison never took ownership. `expect_value_borrowed` clones place operands
                // instead; non-place operands (call results, literals) have no other owner, so
                // ordinary evaluation is unaffected.
                //
                // Both branches check `pending_propagation` after the left operand, before the
                // right operand ever evaluates: `?` in `lhs` must stop `rhs` from running at all,
                // not silently continue with a dummy `Value::Unit` left operand. The comparison
                // branch also threads real operand *places* through to `eval_binary` (Correction
                // brief Issue 2): passing only cloned values, as `expect_value_borrowed` alone
                // would, loses the original storage identity that `Eq::eq`/`Ord::cmp` dispatch
                // needs to borrow rather than duplicate.
                let (left, right, left_place, right_place) = if matches!(
                    op,
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
                ) {
                    let (left, left_place) = self.resolve_comparison_operand(*lhs)?;
                    if let Some(propagated) = self.pending_propagation.take() {
                        return Ok(Flow::Propagate(propagated));
                    }
                    let (right, right_place) = self.resolve_comparison_operand(*rhs)?;
                    (left, right, left_place, right_place)
                } else {
                    let left = self.expect_value(*lhs)?;
                    if let Some(propagated) = self.pending_propagation.take() {
                        return Ok(Flow::Propagate(propagated));
                    }
                    let right = self.expect_value(*rhs)?;
                    (left, right, None, None)
                };
                if let Some(propagated) = self.pending_propagation.take() {
                    // The right operand itself propagated; already-evaluated `left` was either a
                    // borrow (nothing owned to clean up) or a fresh temporary with no destructor
                    // side effects distinguishable from ordinary drop order, so no explicit
                    // cleanup beyond normal drop semantics is required here.
                    return Ok(Flow::Propagate(propagated));
                }
                Ok(Flow::Value(self.eval_binary(
                    *op,
                    (left, left_place),
                    (right, right_place),
                    expr_id,
                    expr.span,
                )?))
            }
            hir::ExprKind::Assign { op, lhs, rhs } => {
                let right = self.expect_value(*rhs)?;
                if let Some(propagated) = self.pending_propagation.take() {
                    return Ok(Flow::Propagate(propagated));
                }
                let place = self.expr_place(*lhs)?;
                let value = if *op == AssignOp::Assign {
                    right
                } else {
                    let current = self.take_place(&place, expr.span)?;
                    // Compound-assignment operators (`+=`, `-=`, ...) never desugar to `Eq`/`Ord`
                    // dispatch (there is no `==`-assignment form), so no operand place is needed.
                    //
                    // **WP-C7.9 Packet A: the type-carrying expression is the LHS, not the
                    // assignment.** `eval_binary` range-checks its result against the width the
                    // type tables give the expression it is passed, and an assignment expression's
                    // type is `Unit` — which has no width, so `fits_target_integer_width` returned
                    // `true` and the check passed vacuously. `acc /= -1` on `Int32::MIN` therefore
                    // COMPLETED in the oracle, storing 2147483648 in an `Int32`, while MIR trapped.
                    // Found by this packet's compound-assignment case; no maintained case had ever
                    // overflowed through a compound assignment.
                    self.eval_binary(
                        assign_binop(*op),
                        (current, None),
                        (right, None),
                        *lhs,
                        expr.span,
                    )?
                };
                // Audit 10-E: injected after the producer boundary accepted the value, so the
                // WRITE boundary is what must refuse it.
                #[cfg(test)]
                let value = if self.mutation_armed(ProducerMutation::WrongElementWrite) {
                    Value::Unit
                } else {
                    value
                };
                self.write_place(&place, value, *lhs, expr.span)?;
                Ok(Flow::Value(Value::Unit))
            }
            hir::ExprKind::Range { lo, hi, inclusive } => {
                let start = self.expect_int(*lo)?;
                if let Some(propagated) = self.pending_propagation.take() {
                    return Ok(Flow::Propagate(propagated));
                }
                let end = self.expect_int(*hi)?;
                if let Some(propagated) = self.pending_propagation.take() {
                    return Ok(Flow::Propagate(propagated));
                }
                Ok(Flow::Value(Value::Range {
                    start,
                    end,
                    inclusive: *inclusive,
                }))
            }
            hir::ExprKind::Cast { expr: value, .. } => {
                let value = self.expect_value(*value)?;
                Ok(Flow::Value(self.eval_cast(value, expr_id, expr.span)?))
            }
            hir::ExprKind::Call { callee, args } => {
                self.eval_call(expr_id, *callee, args, expr.span)
            }
            hir::ExprKind::Field { .. } | hir::ExprKind::TupleField { .. } => {
                let place = self.expr_place(expr_id)?;
                Ok(Flow::Value(self.take_place(&place, expr.span)?))
            }
            hir::ExprKind::Index { base: _, index } => {
                if matches!(self.tables.expr_types.get(index), Some(Ty::Range(_))) {
                    let place = self.expr_place(expr_id)?;
                    return Ok(Flow::Value(self.clone_place_value(&place, expr.span)?));
                }
                let place = self.expr_place(expr_id)?;
                Ok(Flow::Value(self.take_place(&place, expr.span)?))
            }
            hir::ExprKind::Try(inner) => {
                let value = self.expect_value(*inner)?;
                match value {
                    Value::Option(Some(value)) => Ok(Flow::Value(own_payload(
                        value,
                        "`?` on a `Some`",
                        expr.span,
                    )?)),
                    Value::Option(None) => Ok(Flow::Propagate(Value::Option(None))),
                    Value::Result(Ok(value)) => Ok(Flow::Value(own_payload(
                        value,
                        "`?` on an `Ok`",
                        expr.span,
                    )?)),
                    Value::Result(Err(value)) => Ok(Flow::Propagate(Value::Result(Err(value)))),
                    _ => Err(RuntimeError::new(
                        "'?' requires Option or Result",
                        expr.span,
                    )),
                }
            }
            hir::ExprKind::Tuple(values) => match self.eval_aggregate_elements(values)? {
                // DEV-112 / TYPE-PRIM-001: `()` IS `Unit`, one type with two spellings, so it
                // evaluates to the Unit VALUE rather than to an empty tuple. Both other engines
                // canonicalise the same way (`unit_or_tuple` in the checker, `Constant::Unit` in
                // lowering); an oracle that produced `Tuple([])` here would make `Ok(())` fail
                // `main_result_to_status`, which is exactly how this surfaced.
                Ok(values) if values.is_empty() => Ok(Flow::Value(Value::Unit)),
                Ok(values) => Ok(Flow::Value(Value::Tuple(values))),
                Err(propagated) => Ok(Flow::Propagate(propagated)),
            },
            hir::ExprKind::Array(values) => match self.eval_aggregate_elements(values)? {
                Ok(values) => Ok(Flow::Value(Value::Array(values))),
                Err(propagated) => Ok(Flow::Propagate(propagated)),
            },
            hir::ExprKind::Repeat { value, count } => {
                let value = self.expect_value(*value)?;
                if let Some(propagated) = self.pending_propagation.take() {
                    return Ok(Flow::Propagate(propagated));
                }
                let count_value = self.expect_int(*count)?;
                if let Some(propagated) = self.pending_propagation.take() {
                    self.drop_value(value)?;
                    return Ok(Flow::Propagate(propagated));
                }
                let count = usize::try_from(count_value).map_err(|_| {
                    RuntimeError::new("invalid repeat count", self.hir.expr(*count).span)
                })?;
                Ok(Flow::Value(Value::Array(vec![Some(value); count])))
            }
            hir::ExprKind::StructLit { res, fields, .. } => {
                self.eval_struct_lit(expr_id, *res, fields, expr.span)
            }
            hir::ExprKind::If {
                cond,
                then_block,
                else_,
            } => {
                let cond_value = self.expect_bool(*cond)?;
                if let Some(propagated) = self.pending_propagation.take() {
                    return Ok(Flow::Propagate(propagated));
                }
                if cond_value {
                    self.eval_block(*then_block)
                } else if let Some(else_expr) = else_ {
                    self.eval_expr(*else_expr)
                } else {
                    Ok(Flow::Value(Value::Unit))
                }
            }
            hir::ExprKind::Match { scrutinee, arms } => {
                // WP-C7.9 Packet C / PAT-BIND-001: the binding mode is decided ONCE, here, from
                // this match's own scrutinee — not per pattern and not inherited. A scrutinee that
                // is a place read through a reference keeps its place, so a non-`Copy` component
                // can bind as a reference to the ORIGINAL storage; every other scrutinee is read
                // as an owned value exactly as before.
                let source = if self.scrutinee_reads_through_ref(*scrutinee) {
                    let place = self.expr_place(*scrutinee)?;
                    if let Some(propagated) = self.pending_propagation.take() {
                        return Ok(Flow::Propagate(propagated));
                    }
                    PatternSource::Borrowed(place)
                } else {
                    let value = self.expect_value(*scrutinee)?;
                    if let Some(propagated) = self.pending_propagation.take() {
                        return Ok(Flow::Propagate(propagated));
                    }
                    PatternSource::Owned(value)
                };
                for arm in arms {
                    let mut bindings = Vec::new();
                    if self.match_source(arm.pat, &source, &mut bindings)? {
                        for (local, value) in &bindings {
                            self.bind_typed_local(
                                *local,
                                value.clone(),
                                expr.span,
                                RepBoundary::MatchBinding,
                            )?;
                        }
                        let flow = self.eval_expr(arm.body)?;
                        let locals: Vec<_> = bindings.iter().map(|(local, _)| *local).collect();
                        self.cleanup_locals(&locals)?;
                        // WP-C2.2 (DEV-030): the match consumed the scrutinee; anything the
                        // matched pattern did not bind must still be dropped exactly once.
                        // Runs after the arm body and after the bindings' own cleanup,
                        // mirroring "the scrutinee temporary outlives the arm" scoping.
                        //
                        // PAT-BIND-001: a BORROWED read consumes nothing — the referent is still
                        // owned by whoever owned it before the match — so there is nothing to
                        // drop, and dropping here would destroy a live value.
                        if let PatternSource::Owned(value) = source {
                            self.drop_unbound(arm.pat, value)?;
                        }
                        return Ok(flow);
                    }
                }
                Err(RuntimeError::new("non-exhaustive match reached", expr.span))
            }
            hir::ExprKind::Loop { body } => loop {
                match self.eval_block(*body)? {
                    Flow::Value(_) | Flow::Continue => {}
                    Flow::Break(value) => break Ok(Flow::Value(value)),
                    flow => break Ok(flow),
                }
            },
            hir::ExprKind::While { cond, body } => {
                loop {
                    let cond_value = self.expect_bool(*cond)?;
                    if let Some(propagated) = self.pending_propagation.take() {
                        return Ok(Flow::Propagate(propagated));
                    }
                    if !cond_value {
                        break;
                    }
                    match self.eval_block(*body)? {
                        Flow::Value(_) | Flow::Continue => {}
                        Flow::Break(_) => break,
                        flow => return Ok(flow),
                    }
                }
                Ok(Flow::Value(Value::Unit))
            }
            hir::ExprKind::For {
                local, iter, body, ..
            } => {
                let iterable = self.expect_value(*iter)?;
                // **`for x in &v` is `for x in v.iter()`, in this engine too.**
                //
                // The borrow form evaluates to a `Ref` at the Vec's place, and `v.iter()` evaluates
                // to `VecIter` at that same place — so building the cursor here makes the two one
                // iteration rather than two implementations that must be kept in step. MIR lowers
                // both to `VecIterNew`/`VecIterNext` for the same reason.
                //
                // Unconditional on `Ref` because the front end admits no other reference as an
                // iterable: a `&Vec<T>` type-checks and every other `&T` is E0001. A `Ref` to
                // something else would fail in `iterator_step` with its own diagnostic rather than
                // being silently iterated as something it is not.
                let iterable = match iterable {
                    Value::Ref(place) => Value::VecIter(place, 0),
                    other => other,
                };
                match iterable {
                    Value::Range { .. } | Value::Array(_) | Value::Vec(_) | Value::Slice(..) => {
                        let mut remaining = self.iter_values(iterable, expr.span)?.into_iter();
                        while let Some(value) = remaining.next() {
                            // INV-VALUE-REP-001 at the LOOP ITEM — the blind spot DEV-121
                            // UPDATE 2 named. Both known instances of the class arrived here.
                            self.bind_typed_local(
                                *local,
                                value,
                                expr.span,
                                RepBoundary::LoopBinding,
                            )?;
                            let flow = self.eval_block(*body)?;
                            self.cleanup_locals(&[*local])?;
                            match flow {
                                Flow::Value(_) | Flow::Continue => {}
                                Flow::Break(value) => {
                                    self.drop_value(value)?;
                                    for value in remaining.rev() {
                                        self.drop_value(value)?;
                                    }
                                    break;
                                }
                                flow => {
                                    for value in remaining.rev() {
                                        self.drop_value(value)?;
                                    }
                                    return Ok(flow);
                                }
                            }
                        }
                    }
                    iterator => {
                        let iterator_place = self.promote_to_temp_place(iterator, expr.span)?;
                        let mut escaped = None;
                        while let Some(value) =
                            self.next_for_iterator(&iterator_place, expr_id, expr.span)?
                        {
                            // The USER-iterator form checked nothing at all — the same loop
                            // boundary as the branch above, reached through `Iterator::next`
                            // instead of a built-in iterable. Two spellings of one binding, and
                            // only one of them was covered.
                            self.bind_typed_local(
                                *local,
                                value,
                                expr.span,
                                RepBoundary::LoopBinding,
                            )?;
                            let flow = self.eval_block(*body)?;
                            self.cleanup_locals(&[*local])?;
                            match flow {
                                Flow::Value(_) | Flow::Continue => {}
                                Flow::Break(value) => {
                                    self.drop_value(value)?;
                                    break;
                                }
                                flow => {
                                    escaped = Some(flow);
                                    break;
                                }
                            }
                        }
                        let iterator = self.take_place(&iterator_place, expr.span)?;
                        self.drop_value(iterator)?;
                        if let Some(flow) = escaped {
                            return Ok(flow);
                        }
                    }
                }
                Ok(Flow::Value(Value::Unit))
            }
            hir::ExprKind::Block(block) => self.eval_block(*block),
            hir::ExprKind::Error => Err(RuntimeError::new("invalid expression", expr.span)),
        }
    }

    fn expect_value(&mut self, expr: ExprId) -> Result<Value, RuntimeError> {
        match self.eval_expr(expr)? {
            // **The PRODUCER-side boundary.** AS3 Packet 6.
            //
            // The eleven destination boundaries see values that come to rest. A value handed
            // straight to a builtin or a runtime operation never binds to anything, so none of them
            // sees it — the gap the inventory recorded and could not name, because `RepBoundary`
            // had no variant for it. `expect_value` is the funnel every such value passes through,
            // and it carries the `ExprId`, so `expr_types[expr]` is the checker's own answer.
            //
            // Defence in depth, not a second authority: this and every destination check consume
            // the same `check_value_for_ty`.
            Flow::Value(value) => {
                self.check_expr_value(expr, &value)?;
                Ok(value)
            }
            // NOT checked against this expression: the parked value is a PROPAGATION, whose type is
            // the enclosing function's return type, and the `Unit` handed back is a placeholder the
            // caller discards. `RepBoundary::Propagation` reads it against the right type at the
            // body boundary.
            Flow::Propagate(value) => {
                self.pending_propagation = Some(value);
                Ok(Value::Unit)
            }
            _ => Err(RuntimeError::new(
                "control flow used where a value was required",
                self.hir.expr(expr).span,
            )),
        }
    }

    /// Evaluates an operand for a borrowing context (currently: comparison operators). If
    /// `expr` is a place expression (a local, field, tuple field, index, or deref target), its
    /// value is cloned rather than moved, leaving the original storage usable afterward. Other
    /// expressions (calls, literals, freshly built aggregates) have no other owner, so ordinary
    /// `expect_value` evaluation is used instead. Deliberately does not delegate to
    /// `expr_place`, whose non-place fallback arm evaluates-and-stashes into a synthetic temp
    /// local that nothing ever cleans up -- correct for its own callers (which immediately
    /// consume or write through the returned place) but not safe to reuse here.
    /// Resolves a comparison operand (`==`/`!=`/`<`/`<=`/`>`/`>=`), returning both a value (for
    /// the structural-equality fallback used by primitives and `Ty::Core` container types, which
    /// never dispatches to user code and so never needs place identity) and, for a place
    /// expression, the *real* place itself. Nominal `Eq`/`Ord` dispatch (`eval_binary`) uses that
    /// real place to pass `Value::Ref(place)` -- a genuine borrow of the original storage -- to
    /// the user's `eq`/`cmp` method, instead of the value returned here (a clone, needed only for
    /// the non-dispatching structural-comparison path, which never involves user-code execution
    /// or frame cleanup and so has no drop-timing hazard). Passing the *clone* as if it were the
    /// real operand is exactly the correction-brief Issue 2 bug: the callee's own per-parameter
    /// cleanup then destroys what should have been a mere reference. Non-place operands (call
    /// results, literals) have no other owner and no place to borrow, so `None` is returned for
    /// them; the caller promotes a fresh temporary only if nominal dispatch actually needs one.
    fn resolve_comparison_operand(
        &mut self,
        expr: ExprId,
    ) -> Result<(Value, Option<Place>), RuntimeError> {
        let is_place = matches!(
            self.hir.expr(expr).kind,
            hir::ExprKind::Path {
                res: Res::Local(_) | Res::SelfValue(_),
                ..
            } | hir::ExprKind::Field { .. }
                | hir::ExprKind::TupleField { .. }
                | hir::ExprKind::Index { .. }
                | hir::ExprKind::Unary {
                    op: UnOp::Deref,
                    ..
                }
        );
        if is_place {
            let span = self.hir.expr(expr).span;
            let place = self.expr_place(expr)?;
            let value = self.clone_place_value(&place, span)?;
            Ok((value, Some(place)))
        } else {
            Ok((self.expect_value(expr)?, None))
        }
    }

    fn expect_bool(&mut self, expr: ExprId) -> Result<bool, RuntimeError> {
        match self.expect_value(expr)? {
            Value::Bool(value) => Ok(value),
            // If `?` inside `expr` just propagated, `expect_value` returned a dummy
            // `Value::Unit` and left `pending_propagation` set -- pass a placeholder through
            // rather than reporting a misleading "expected Bool" trap; the caller is required
            // to check `pending_propagation` immediately after this call (every call site does)
            // and will correctly convert it to `Flow::Propagate` before the placeholder value
            // could ever be observed or acted on.
            _ if self.pending_propagation.is_some() => Ok(false),
            _ => Err(RuntimeError::new("expected Bool", self.hir.expr(expr).span)),
        }
    }

    fn expect_int(&mut self, expr: ExprId) -> Result<i128, RuntimeError> {
        match self.expect_value(expr)? {
            Value::Int(value) => Ok(value),
            // See `expect_bool`'s matching arm: a pending propagation must reach the caller's
            // own `pending_propagation` check, not this function's type-mismatch error path.
            _ if self.pending_propagation.is_some() => Ok(0),
            _ => Err(RuntimeError::new(
                "expected integer",
                self.hir.expr(expr).span,
            )),
        }
    }

    fn eval_lit(&self, lit: Lit, span: Span) -> Result<Value, RuntimeError> {
        let text = self.text(span);
        let value = literal::eval_lit_value(lit, text, &self.hir.str_lits)
            .ok_or_else(|| RuntimeError::new("invalid literal", span))?;
        // WP-C1.5 (DEV-015): defense-in-depth mirror of the checker's suffixed-literal
        // magnitude check (`check_expr`'s `Lit::Int` arm) -- re-verified here in case a literal
        // ever reaches evaluation without having gone through that check (e.g. a future
        // alternate entry point). Unsuffixed-literal-vs-inferred-type magnitude is not
        // re-checked here since that requires the type table the checker already consulted;
        // trusting the already-validated static type for that half is the same trust boundary
        // `check_integer_range` (used elsewhere in this file) already relies on.
        if let (
            LitValue::Int(value),
            Lit::Int {
                suffix: Some(s), ..
            },
        ) = (&value, lit)
        {
            if !literal::int_suffix_range_contains(s, *value) {
                return Err(RuntimeError::new("integer literal out of range", span));
            }
        }
        match value {
            LitValue::Bool(value) => Ok(Value::Bool(value)),
            LitValue::Char(value) => Ok(Value::Char(value)),
            LitValue::Str(value) => Ok(Value::Str(value)),
            LitValue::Int(value) => Ok(Value::Int(value)),
            LitValue::Float(value) => {
                let width = match lit {
                    Lit::Float {
                        suffix: Some(crate::lexer::FloatSuffix::F32),
                    } => FloatWidth::F32,
                    _ => FloatWidth::F64,
                };
                Ok(Value::Float(value, width))
            }
        }
    }

    fn eval_path(&mut self, res: Res, expr: ExprId) -> Result<Value, RuntimeError> {
        match res {
            Res::Local(local) | Res::SelfValue(local) => {
                let place = Place {
                    frame: self.frames.len() - 1,
                    local,
                    projections: Vec::new(),
                };
                self.take_place(&place, self.hir.expr(expr).span)
            }
            Res::Item(item) => match &self.hir.item(item).kind {
                // **DEV-178: capture the instantiation here, where it is selected.** The later
                // call cannot recover it — see `FunctionValue`. Concretised against the ACTIVE
                // frame before storage, because the value may outlive that frame.
                hir::ItemKind::Fn(_) => {
                    // Class 3 mutation: a function item coerces to a non-function value.
                    #[cfg(test)]
                    if self.mutation_armed(ProducerMutation::NonFunctionValue) {
                        return Ok(Value::Unit);
                    }
                    Ok(Value::Function(self.capture_function_value(
                        item,
                        expr,
                        self.hir.expr(expr).span,
                    )?))
                }
                hir::ItemKind::Const { .. } => self.eval_const_item(item),
                _ => Err(RuntimeError::new(
                    "item is not a runtime value",
                    self.hir.expr(expr).span,
                )),
            },
            Res::Variant(item, variant) => Ok(Value::Enum {
                item,
                variant,
                fields: Vec::new(),
                named: BTreeMap::new(),
            }),
            Res::Builtin(Builtin::None) => Ok(Value::Option(None)),
            Res::Builtin(Builtin::MathPi) => {
                Ok(Value::Float(std::f64::consts::PI, FloatWidth::F64))
            }
            Res::Builtin(Builtin::MathE) => Ok(Value::Float(std::f64::consts::E, FloatWidth::F64)),
            Res::Builtin(Builtin::IOErrorNotFound) => Ok(Value::IOError(IOErrorKind::NotFound)),
            Res::Builtin(Builtin::IOErrorPermissionDenied) => {
                Ok(Value::IOError(IOErrorKind::PermissionDenied))
            }
            Res::Builtin(Builtin::IOErrorAlreadyExists) => {
                Ok(Value::IOError(IOErrorKind::AlreadyExists))
            }
            Res::Builtin(Builtin::IOErrorInvalidInput) => {
                Ok(Value::IOError(IOErrorKind::InvalidInput))
            }
            Res::Builtin(Builtin::OrderingLess) => Ok(Value::Ordering(std::cmp::Ordering::Less)),
            Res::Builtin(Builtin::OrderingEqual) => Ok(Value::Ordering(std::cmp::Ordering::Equal)),
            Res::Builtin(Builtin::OrderingGreater) => {
                Ok(Value::Ordering(std::cmp::Ordering::Greater))
            }
            _ => Err(RuntimeError::new(
                "path is not a runtime value",
                self.hir.expr(expr).span,
            )),
        }
    }

    fn eval_unary(
        &self,
        op: UnOp,
        value: Value,
        expr: ExprId,
        span: Span,
    ) -> Result<Value, RuntimeError> {
        match (op, value) {
            (UnOp::Neg, Value::Int(value)) => value
                .checked_neg()
                .map(Value::Int)
                .ok_or_else(|| {
                    RuntimeError::with_category(
                        "integer overflow",
                        span,
                        crate::mir::TrapCategory::IntegerOverflow,
                    )
                })
                .and_then(|value| self.normalize_numeric(value, expr, span)),
            (UnOp::Neg, Value::Float(value, width)) => {
                self.normalize_numeric(Value::Float(-value, width), expr, span)
            }
            (UnOp::Not, Value::Bool(value)) => Ok(Value::Bool(!value)),
            (UnOp::BitNot, Value::Int(value)) => {
                let value = match self.tables.expr_types.get(&expr) {
                    Some(Ty::Primitive(
                        Primitive::UInt8
                        | Primitive::UInt16
                        | Primitive::UInt32
                        | Primitive::UInt64,
                    )) => {
                        let width = integer_width(self.tables.expr_types.get(&expr)).unwrap();
                        (!value) & ((1_i128 << width) - 1)
                    }
                    _ => !value,
                };
                self.normalize_numeric(Value::Int(value), expr, span)
            }
            (UnOp::Ref { .. } | UnOp::Deref, value) => Ok(value),
            _ => Err(RuntimeError::new("invalid unary operation", span)),
        }
    }

    /// `left`/`right` bundle each operand's value with its real place, when the operand is a
    /// place expression (see `resolve_comparison_operand`) -- grouped into one tuple parameter
    /// per side to keep the parameter count under clippy's `too_many_arguments` threshold rather
    /// than passing four related values separately.
    /// Resolve a published `Bound` obligation at this call, if there is one.
    ///
    /// AS3 Boundary 4c. `Self` is concrete here — it is the receiver's own type — so the
    /// specialiser has what it needs. Returns `None` when the checker published no bound use for
    /// this expression, which is every ordinary concrete method call.
    /// **AS3 Boundary 4 (DEV-192): the body behind an operator whose operand is a bounded
    /// generic parameter.**
    ///
    /// `a == b` / `a < b` inside `fn f<T: Eq>` / `fn f<T: Ord>` publishes a `Bound` use whose
    /// `self_ty` is the parameter itself. Substituting it through the active generic frame gives
    /// the concrete `Self` — arguments included, which is what a generic impl needs to unify.
    ///
    /// The operator path cannot use [`Self::specialised_bound_callable`]: `eval_binary` receives
    /// evaluated operand VALUES, not their expressions, and a runtime `Value` carries no type
    /// arguments. The published `self_ty` does.
    fn specialised_operator_callable(
        &self,
        expr: ExprId,
        core: CoreTrait,
        span: Span,
    ) -> Option<Callable> {
        let ids = self.tables.callable_uses_by_expr.get(&expr)?;
        let use_ = ids
            .iter()
            .filter_map(|id| self.tables.callable_uses.get(id.0 as usize))
            .find(|u| {
                matches!(
                    u.provenance,
                    crate::typecheck::DispatchProvenance::Bound {
                        trait_: hir::BoundTrait::Core(c)
                    } if c == core
                )
            })?;
        let crate::typecheck::CalleeSelection::Bound {
            trait_,
            member,
            self_ty,
            trait_args,
            method_args,
        } = &use_.selection
        else {
            return None;
        };
        let mut self_ty = self.concrete_runtime_ty(self_ty, span).ok()?;
        while let Ty::Ref { inner, .. } = self_ty {
            self_ty = *inner;
        }
        let resolved = crate::bound_dispatch::specialize_bound_callable(
            &self.tables.trait_impls,
            &self.tables.callable_types,
            *trait_,
            member,
            &self_ty,
            trait_args,
            method_args,
        )?;
        self.callable_for_body(resolved.body)
    }

    /// **AS3 Boundary 4: consume the checker's `Static` selection for a method call.**
    ///
    /// The census found this as an engine asymmetry: MIR added `static_selected_key` and the
    /// interpreter kept scanning by name, so the two engines answered an ordinary method call by
    /// different means. They agreed — which is the state DEV-192 was hiding in.
    ///
    /// Only `Inherent`, `TraitImpl` and `Qualified` provenances are eligible; an operator's
    /// `CoreTrait` use can be published against the same expression and names a different callable.
    /// The Static environment published with the selection at `expr`, if any.
    fn static_selected_env(&self, expr: ExprId) -> Vec<(crate::typecheck::GenericBinder, Ty)> {
        let Some(ids) = self.tables.callable_uses_by_expr.get(&expr) else {
            return Vec::new();
        };
        ids.iter()
            .filter_map(|id| self.tables.callable_uses.get(id.0 as usize))
            .find_map(|u| match &u.environment {
                crate::typecheck::GenericEnvironment::Static(bindings) if !bindings.is_empty() => {
                    Some(bindings.clone())
                }
                _ => None,
            })
            .unwrap_or_default()
    }

    fn static_selected_callable(&self, expr: ExprId) -> Option<Callable> {
        let ids = self.tables.callable_uses_by_expr.get(&expr)?;
        let use_ = ids
            .iter()
            .filter_map(|id| self.tables.callable_uses.get(id.0 as usize))
            .find(|u| {
                matches!(
                    u.provenance,
                    crate::typecheck::DispatchProvenance::Inherent
                        | crate::typecheck::DispatchProvenance::TraitImpl { .. }
                        | crate::typecheck::DispatchProvenance::Qualified { .. }
                )
            })?;
        let crate::typecheck::CalleeSelection::Static { body, .. } = &use_.selection else {
            return None;
        };
        self.callable_for_body(*body)
    }

    fn specialised_bound_callable(
        &self,
        expr: ExprId,
        base: ExprId,
    ) -> Option<(Callable, Vec<(crate::typecheck::GenericBinder, Ty)>)> {
        let ids = self.tables.callable_uses_by_expr.get(&expr)?;
        let use_ = ids
            .iter()
            .filter_map(|id| self.tables.callable_uses.get(id.0 as usize))
            .find(|u| matches!(u.selection, crate::typecheck::CalleeSelection::Bound { .. }))?;
        let crate::typecheck::CalleeSelection::Bound {
            trait_,
            member,
            trait_args,
            method_args,
            ..
        } = &use_.selection
        else {
            return None;
        };
        // **DEV-187's repair: the concrete `Self` INCLUDING its type arguments.**
        //
        // This passed the bare nominal head, taken from the runtime `Value` — which carries no type
        // arguments, so `impl<T> Describe for W2<T>` never matched and both engines silently fell
        // back to their old scans. The checker knows the receiver's type at this expression, and
        // `concrete_runtime_ty` substitutes it through the active generic frame using the shared
        // `substitute_ty`. No new machinery, and no change to the runtime representation.
        let declared = self.tables.expr_types.get(&base)?;
        let self_ty = self
            .concrete_runtime_ty(declared, self.hir.expr(base).span)
            .ok()?;
        // Auto-deref: a receiver of `&T` selects on `T`.
        let mut self_ty = self_ty;
        while let Ty::Ref { inner, .. } = self_ty {
            self_ty = *inner;
        }
        let resolved = crate::bound_dispatch::specialize_bound_callable(
            &self.tables.trait_impls,
            &self.tables.callable_types,
            *trait_,
            member,
            &self_ty,
            trait_args,
            method_args,
        )?;
        let callable = self.callable_for_body(resolved.body)?;
        Some((callable, resolved.environment))
    }

    /// The runnable form of a body the checker already selected.
    ///
    /// **AS3 Boundary 3: this is a LOOKUP keyed on the checker's answer, not a selection.**
    /// `find_method` searches by NAME across every impl on a nominal and decides which body wins;
    /// this takes the body the checker published and finds the locals needed to run it. The
    /// difference is the whole packet: one asks "which body does `a == b` mean", the other asks
    /// "how do I enter this body".
    fn callable_for_body(&self, body: BlockId) -> Option<Callable> {
        for (idx, item) in self.hir.items.iter().enumerate() {
            let owner = ItemId(idx as u32);
            let _ = owner;
            match &item.kind {
                hir::ItemKind::Fn(def) if def.body == body => {
                    return Some(Callable {
                        receiver: def.sig.receiver.zip(def.sig.receiver_local),
                        params: def.sig.params.iter().map(|param| param.local).collect(),
                        body: def.body,
                    })
                }
                hir::ItemKind::Impl { items, .. } => {
                    for impl_item in items {
                        if let hir::ImplItem::Fn { def, .. } = impl_item {
                            if def.body == body {
                                return Some(Callable {
                                    receiver: def.sig.receiver.zip(def.sig.receiver_local),
                                    params: def
                                        .sig
                                        .params
                                        .iter()
                                        .map(|param| param.local)
                                        .collect(),
                                    body: def.body,
                                });
                            }
                        }
                    }
                }
                hir::ItemKind::Trait { items, .. } => {
                    for trait_item in items {
                        if let hir::TraitItem::Method { sig, body: Some(b) } = trait_item {
                            if *b == body {
                                return Some(Callable {
                                    receiver: sig.receiver.zip(sig.receiver_local),
                                    params: sig.params.iter().map(|param| param.local).collect(),
                                    body: *b,
                                });
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// The callable the checker selected for a compiler-known trait operation at `expr`.
    ///
    /// **AS3 Boundary 3: consumption.** This replaces `find_method(nominal, "eq", Some(CoreTrait))`
    /// at the operator sites — a scan of every impl on the nominal, run again at execution time
    /// after the checker had already decided. Choosing among published records is consumption;
    /// scanning the HIR and re-running selection is what this removes.
    ///
    /// `None` means the checker published nothing for this expression, which for an operator means
    /// it reaches no user body — a primitive comparison. The caller keeps its built-in path.
    fn selected_core_trait_callable(
        &self,
        expr: ExprId,
        core: CoreTrait,
    ) -> Option<crate::typecheck::CallableUse> {
        let ids = self.tables.callable_uses_by_expr.get(&expr)?;
        ids.iter()
            .filter_map(|id| self.tables.callable_uses.get(id.0 as usize))
            .find(|use_| {
                matches!(
                    use_.provenance,
                    crate::typecheck::DispatchProvenance::CoreTrait { core: c } if c == core
                )
            })
            .cloned()
    }

    fn eval_binary(
        &mut self,
        op: BinOp,
        left: (Value, Option<Place>),
        right: (Value, Option<Place>),
        expr: ExprId,
        span: Span,
    ) -> Result<Value, RuntimeError> {
        let (left, left_place) = left;
        let (right, right_place) = right;
        let left = self.deref_value(left, span)?;
        let right = self.deref_value(right, span)?;
        if matches!(op, BinOp::Eq | BinOp::Ne) {
            // WP-C1.3 (2026-07-17): dispatch to a user-defined `impl Eq for T`'s `eq` method
            // when one exists, per 03-Type-System.md "Operators and Traits" (`==`/`!=` desugar
            // to `Eq::eq`) -- structural `Value` equality was previously used unconditionally,
            // even for struct/enum values whose type has a real, type-checker-verified `impl Eq`
            // with custom comparison logic (typecheck/body.rs's `require_operator_bound` already
            // requires such an impl to exist for any struct/enum `==`, so this dispatch cannot
            // find a program that type-checks but has no matching impl). Primitives and
            // Ty::Core container types (Option/Result/Vec/Box/String) have no user-overridable
            // Eq impl in Core v1 (operator overloading is a future extension per the spec), so
            // structural comparison remains exactly correct for them -- only struct/enum values
            // are looked up here. See COMPILER-STATE.md DEV-008.
            if let Some(nominal) = nominal_item(&left) {
                // AS3 Boundary 3: CONSUME the checker's selection instead of scanning for a
                // method named "eq". `nominal` is retained only to decide whether a user body is
                // involved at all — the choice of body is the checker's, published at this
                // expression.
                let _ = nominal;
                if let Some(method) = self
                    .selected_core_trait_callable(expr, hir::CoreTrait::Eq)
                    .and_then(|use_| match use_.selection {
                        crate::typecheck::CalleeSelection::Static { body, .. } => {
                            self.callable_for_body(body)
                        }
                        crate::typecheck::CalleeSelection::Bound { .. }
                        | crate::typecheck::CalleeSelection::FunctionValue => None,
                    })
                    // **AS3 Boundary 4 (DEV-192): the late-bound equality case.** When the operand
                    // is a bounded generic parameter the selection is `Bound`, and this fell
                    // through to the primitive path — which for `==` meant STRUCTURAL comparison
                    // silently replacing the user's `impl Eq`, and for `<` meant the trap "invalid
                    // binary operation". Now the shared specialiser answers it.
                    .or_else(|| self.specialised_operator_callable(expr, hir::CoreTrait::Eq, span))
                {
                    // Correction-brief Issue 2: `Eq::eq(&self, &other)` borrows both operands --
                    // it never takes ownership. Passing owned clones here (the pre-fix
                    // behavior) is observably wrong two different ways: the receiver's clone
                    // silently vanished via ordinary Rust-level drop with no STARK-level
                    // `Drop::drop` call at all (data loss for any `Drop`-observable identity),
                    // while the argument's clone got a *real*, extra `Drop::drop` call fired by
                    // the callee's own normal per-parameter cleanup, at the wrong time relative
                    // to the original operand's own destruction. `Value::Ref(place)` for a real
                    // place operand fixes both: the callee's `self`/`other` locals hold genuine
                    // references, so cleanup of either is a no-op (`drop_value` treats `Ref` as
                    // borrowed, never owned), and the *real* value being compared is the
                    // original operand's own storage -- never duplicated at all. A non-place
                    // operand (a call result, with no other owner) still needs a temporary to
                    // point the reference at; that temporary's own eventual cleanup is unchanged
                    // from before this fix (naturally scoped to the enclosing frame).
                    let receiver_place = match left_place {
                        Some(place) => place,
                        None => self.promote_to_owned_temp_place(left.clone(), span)?,
                    };
                    let argument_place = match right_place {
                        Some(place) => place,
                        None => self.promote_to_owned_temp_place(right.clone(), span)?,
                    };
                    let result = self.call_user_method(
                        method,
                        receiver_place.clone(),
                        self.core_trait_env(expr, hir::CoreTrait::Eq),
                        vec![Value::Ref(argument_place)],
                        span,
                    )?;
                    let equal = matches!(result, Value::Bool(true));
                    return Ok(Value::Bool(if op == BinOp::Eq { equal } else { !equal }));
                }
            }
            // DEV-130: the inline `Str`/`String` pairing that used to live here is now
            // `values_equal`, shared with `assert_eq`, `language_equal` and literal patterns —
            // all of which lacked it.
            let equal = values_equal(&left, &right);
            return Ok(Value::Bool(if op == BinOp::Eq { equal } else { !equal }));
        }
        if matches!(op, BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge) {
            // WP-C2.2 (DEV-027): comparison operators on nominal user types dispatch through
            // `Ord::cmp`, just as equality above dispatches through `Eq::eq`. The type checker
            // has already required a matching `Ord` implementation.
            if let Some(nominal) = nominal_item(&left) {
                // AS3 Boundary 3: as above, for ordering.
                let _ = nominal;
                if let Some(method) = self
                    .selected_core_trait_callable(expr, hir::CoreTrait::Ord)
                    .and_then(|use_| match use_.selection {
                        crate::typecheck::CalleeSelection::Static { body, .. } => {
                            self.callable_for_body(body)
                        }
                        crate::typecheck::CalleeSelection::Bound { .. }
                        | crate::typecheck::CalleeSelection::FunctionValue => None,
                    })
                    // **AS3 Boundary 4 (DEV-192): the late-bound ordering case.** When the operand
                    // is a bounded generic parameter the selection is `Bound`, and this fell
                    // through to the primitive path — which for `==` meant STRUCTURAL comparison
                    // silently replacing the user's `impl Eq`, and for `<` meant the trap "invalid
                    // binary operation". Now the shared specialiser answers it.
                    .or_else(|| self.specialised_operator_callable(expr, hir::CoreTrait::Ord, span))
                {
                    // Same fix as the `Eq::eq` dispatch above: `Ord::cmp(&self, &other)` borrows
                    // both operands.
                    let receiver_place = match left_place {
                        Some(place) => place,
                        None => self.promote_to_owned_temp_place(left.clone(), span)?,
                    };
                    let argument_place = match right_place {
                        Some(place) => place,
                        None => self.promote_to_owned_temp_place(right.clone(), span)?,
                    };
                    let ordering = self.call_user_method(
                        method,
                        receiver_place.clone(),
                        self.core_trait_env(expr, hir::CoreTrait::Ord),
                        vec![Value::Ref(argument_place)],
                        span,
                    )?;
                    let Value::Ordering(ordering) = ordering else {
                        return Err(RuntimeError::new("Ord::cmp must return Ordering", span));
                    };
                    let result = match op {
                        BinOp::Lt => ordering == std::cmp::Ordering::Less,
                        BinOp::Le => ordering != std::cmp::Ordering::Greater,
                        BinOp::Gt => ordering == std::cmp::Ordering::Greater,
                        BinOp::Ge => ordering != std::cmp::Ordering::Less,
                        _ => unreachable!(),
                    };
                    return Ok(Value::Bool(result));
                }
            }
        }
        match (left, right) {
            (Value::Int(left), Value::Int(right)) => {
                if matches!(op, BinOp::Shl | BinOp::Shr) {
                    let width = integer_width(self.tables.expr_types.get(&expr)).unwrap_or(128);
                    if right < 0 || right >= i128::from(width) {
                        return Err(RuntimeError::with_category(
                            "invalid shift count",
                            span,
                            crate::mir::TrapCategory::InvalidShift,
                        ));
                    }
                }
                // `MIN % -1` traps even though its mathematical result (0) is representable:
                // the operation is undefined at the CPU instruction level, matching `MIN / -1`
                // (which already traps here via the post-hoc range check below, since the wider
                // `i128` carrier lets `checked_div`/`checked_rem` succeed where the declared
                // width would overflow). `Rem` alone needs this explicit guard because its
                // mathematical result always happens to fit back into the declared width.
                if op == BinOp::Rem && right == -1 {
                    if let Some(min) = signed_integer_min(self.tables.expr_types.get(&expr)) {
                        if left == min {
                            return Err(RuntimeError::with_category(
                                "integer overflow",
                                span,
                                crate::mir::TrapCategory::IntegerOverflow,
                            ));
                        }
                    }
                }
                let value = match op {
                    BinOp::Add => left.checked_add(right),
                    BinOp::Sub => left.checked_sub(right),
                    BinOp::Mul => left.checked_mul(right),
                    BinOp::Div => left.checked_div(right),
                    BinOp::Rem => left.checked_rem(right),
                    BinOp::Pow => u32::try_from(right)
                        .ok()
                        .and_then(|power| left.checked_pow(power)),
                    BinOp::BitAnd => Some(left & right),
                    BinOp::BitOr => Some(left | right),
                    BinOp::BitXor => Some(left ^ right),
                    BinOp::Shl => u32::try_from(right)
                        .ok()
                        .and_then(|shift| left.checked_shl(shift)),
                    BinOp::Shr => u32::try_from(right)
                        .ok()
                        .and_then(|shift| left.checked_shr(shift)),
                    BinOp::Lt => return Ok(Value::Bool(left < right)),
                    BinOp::Le => return Ok(Value::Bool(left <= right)),
                    BinOp::Gt => return Ok(Value::Bool(left > right)),
                    BinOp::Ge => return Ok(Value::Bool(left >= right)),
                    _ => None,
                }
                .ok_or_else(|| {
                    // WP-C7.9 G.3: the CATEGORY is decided by the same condition that picks the
                    // wording, at the raise site, instead of being recovered downstream by matching
                    // the wording. Rewording this message can no longer reclassify the trap.
                    let (message, category) = if right == 0 && matches!(op, BinOp::Div | BinOp::Rem)
                    {
                        ("division by zero", crate::mir::TrapCategory::DivideByZero)
                    } else {
                        (
                            "integer overflow",
                            crate::mir::TrapCategory::IntegerOverflow,
                        )
                    };
                    RuntimeError::with_category(message, span, category)
                })?;
                self.check_integer_range(value, expr, span).map(Value::Int)
            }
            (Value::Float(left, width), Value::Float(right, _)) => match op {
                BinOp::Add => canonicalize_float_result(self.normalize_numeric(
                    Value::Float(left + right, width),
                    expr,
                    span,
                )),
                BinOp::Sub => canonicalize_float_result(self.normalize_numeric(
                    Value::Float(left - right, width),
                    expr,
                    span,
                )),
                BinOp::Mul => canonicalize_float_result(self.normalize_numeric(
                    Value::Float(left * right, width),
                    expr,
                    span,
                )),
                BinOp::Div => canonicalize_float_result(self.normalize_numeric(
                    Value::Float(left / right, width),
                    expr,
                    span,
                )),
                BinOp::Rem => canonicalize_float_result(self.normalize_numeric(
                    Value::Float(left % right, width),
                    expr,
                    span,
                )),
                BinOp::Pow => Err(RuntimeError::new(
                    "floating-point `**` is not a Core v1 operation",
                    span,
                )),
                BinOp::Lt => Ok(Value::Bool(left < right)),
                BinOp::Le => Ok(Value::Bool(left <= right)),
                BinOp::Gt => Ok(Value::Bool(left > right)),
                BinOp::Ge => Ok(Value::Bool(left >= right)),
                _ => Err(RuntimeError::new("invalid floating-point operation", span)),
            },
            (Value::String(mut left), Value::String(right)) if op == BinOp::Add => {
                left.push_str(&right);
                Ok(Value::String(left))
            }
            (Value::String(left) | Value::Str(left), Value::String(right) | Value::Str(right)) => {
                match op {
                    BinOp::Lt => Ok(Value::Bool(left < right)),
                    BinOp::Le => Ok(Value::Bool(left <= right)),
                    BinOp::Gt => Ok(Value::Bool(left > right)),
                    BinOp::Ge => Ok(Value::Bool(left >= right)),
                    _ => Err(RuntimeError::new("invalid string operation", span)),
                }
            }
            // DEV-075 (owner specification decision, 2026-07-20): `Char` has a total order by
            // UNICODE SCALAR VALUE — not locale-sensitive or linguistic collation. The oracle
            // rejected all four ordered operators on `Char` ("invalid binary operation") while
            // MIR executed them correctly, which was an engine DIVERGENCE, not merely a gap.
            // Rust's `char: Ord` is scalar-value order, so this matches MIR by construction.
            // (`Bool` is deliberately absent: per the same decision `Bool` is NOT `Ord`, and its
            // ordered operators are rejected at type-check time.)
            (Value::Char(left), Value::Char(right)) => match op {
                BinOp::Lt => Ok(Value::Bool(left < right)),
                BinOp::Le => Ok(Value::Bool(left <= right)),
                BinOp::Gt => Ok(Value::Bool(left > right)),
                BinOp::Ge => Ok(Value::Bool(left >= right)),
                _ => Err(RuntimeError::new("invalid Char operation", span)),
            },
            _ => Err(RuntimeError::new("invalid binary operation", span)),
        }
    }

    fn eval_cast(&self, value: Value, expr: ExprId, span: Span) -> Result<Value, RuntimeError> {
        let target = self
            .tables
            .expr_types
            .get(&expr)
            .cloned()
            .unwrap_or(Ty::Error);
        match value {
            Value::Int(value) if matches!(target, Ty::Primitive(p) if is_integer(p)) => {
                self.check_cast_range(value, expr, span).map(Value::Int)
            }
            Value::Int(value) if matches!(target, Ty::Primitive(p) if is_float(p)) => {
                self.normalize_numeric(Value::Float(value as f64, FloatWidth::F64), expr, span)
            }
            Value::Float(value, width) if matches!(target, Ty::Primitive(p) if is_float(p)) => {
                self.normalize_numeric(Value::Float(value, width), expr, span)
            }
            Value::Float(value, _) if matches!(target, Ty::Primitive(p) if is_integer(p)) => {
                // A finite float-to-integer cast truncates toward zero, then traps only when
                // the truncated result is unrepresentable in the target width (not merely
                // because the source had a nonzero fractional part). NaN and infinities always
                // trap. `.trunc() as i128` truncates finite f64 values toward zero exactly (i128
                // covers every integral f64 magnitude a cast can produce, including 2^64); the
                // subsequent `check_cast_range` call performs the actual representability check
                // against the target's declared width, in exact integer arithmetic.
                if !value.is_finite() {
                    return Err(RuntimeError::with_category(
                        "numeric cast out of range",
                        span,
                        crate::mir::TrapCategory::CastFailure,
                    ));
                }
                let truncated = value.trunc() as i128;
                self.check_cast_range(truncated, expr, span).map(Value::Int)
            }
            _ => Err(RuntimeError::with_category(
                "invalid numeric cast",
                span,
                crate::mir::TrapCategory::CastFailure,
            )),
        }
    }

    /// The range check for a failing `as` CAST. Identical representability test to
    /// [`Self::check_integer_range`], different trap: a failing cast is `TrapCategory::
    /// CastFailure`, not `IntegerOverflow` -- 03-Type-System.md enumerates them as distinct
    /// always-trap causes, and the MIR interpreter and native backend both classify a failing
    /// cast as `CastFailure`. Both cast arms previously routed through `check_integer_range` and
    /// so reported a cast failure with the ARITHMETIC overflow message, making the HIR oracle
    /// disagree with the other two engines on category for every out-of-range cast at any width.
    /// The message matches the one this function's non-finite sibling case already used, and is
    /// what the differential comparator maps to `CastFailure`.
    fn check_cast_range(
        &self,
        value: i128,
        expr: ExprId,
        span: Span,
    ) -> Result<i128, RuntimeError> {
        if self.fits_target_integer_width(value, expr) {
            Ok(value)
        } else {
            Err(RuntimeError::with_category(
                "numeric cast out of range",
                span,
                crate::mir::TrapCategory::CastFailure,
            ))
        }
    }

    fn check_integer_range(
        &self,
        value: i128,
        expr: ExprId,
        span: Span,
    ) -> Result<i128, RuntimeError> {
        if self.fits_target_integer_width(value, expr) {
            Ok(value)
        } else {
            Err(RuntimeError::with_category(
                "integer overflow",
                span,
                crate::mir::TrapCategory::IntegerOverflow,
            ))
        }
    }

    /// Whether `value` is representable in the integer width the type tables assign to `expr`.
    /// Shared by the arithmetic-overflow and cast-failure checks so the two can never drift on
    /// WHICH values are in range while differing, correctly, on which trap they raise.
    fn fits_target_integer_width(&self, value: i128, expr: ExprId) -> bool {
        let ty = self.tables.expr_types.get(&expr);
        match ty {
            Some(Ty::Primitive(Primitive::Int8)) => i8::try_from(value).is_ok(),
            Some(Ty::Primitive(Primitive::Int16)) => i16::try_from(value).is_ok(),
            Some(Ty::Primitive(Primitive::Int32)) => i32::try_from(value).is_ok(),
            Some(Ty::Primitive(Primitive::Int64)) => i64::try_from(value).is_ok(),
            Some(Ty::Primitive(Primitive::UInt8)) => u8::try_from(value).is_ok(),
            Some(Ty::Primitive(Primitive::UInt16)) => u16::try_from(value).is_ok(),
            Some(Ty::Primitive(Primitive::UInt32)) => u32::try_from(value).is_ok(),
            Some(Ty::Primitive(Primitive::UInt64)) => u64::try_from(value).is_ok(),
            _ => true,
        }
    }

    fn normalize_numeric(
        &self,
        value: Value,
        expr: ExprId,
        span: Span,
    ) -> Result<Value, RuntimeError> {
        match value {
            Value::Int(value) => self.check_integer_range(value, expr, span).map(Value::Int),
            Value::Float(value, _) => {
                if matches!(
                    self.tables.expr_types.get(&expr),
                    Some(Ty::Primitive(Primitive::Float32))
                ) {
                    Ok(Value::Float((value as f32) as f64, FloatWidth::F32))
                } else {
                    Ok(Value::Float(value, FloatWidth::F64))
                }
            }
            value => Ok(value),
        }
    }

    fn eval_call(
        &mut self,
        expr_id: ExprId,
        callee: ExprId,
        args: &[ExprId],
        span: Span,
    ) -> Result<Flow, RuntimeError> {
        match &self.hir.expr(callee).kind {
            hir::ExprKind::Path { res, .. } => match res {
                // WP-C5.3e: a layout query is intercepted here rather than in `call_builtin`,
                // because the answer depends on the TYPE ARGUMENT and only the callee expression
                // identifies it. `call_builtin` receives evaluated values and cannot.
                Res::Builtin(b @ (Builtin::SizeOf | Builtin::AlignOf)) => {
                    self.layout_query(*b, callee, span).map(Flow::Value)
                }
                Res::Builtin(builtin) => match self.eval_call_arguments(args)? {
                    Ok(values) => self
                        .call_builtin(*builtin, values, args, span)
                        .map(Flow::Value),
                    Err(propagated) => Ok(Flow::Propagate(propagated)),
                },
                Res::Item(item) => match self.eval_call_arguments(args)? {
                    Ok(values) => {
                        let callable = self.item_callable(*item).ok_or_else(|| {
                            RuntimeError::new("item is not callable", self.hir.expr(callee).span)
                        })?;
                        // DEV-100: a call to a GENERIC item installs its instantiation for the
                        // duration of the callee, so a layout query inside the body can resolve
                        // `Ty::Param`. Pushed and popped around the call on every path — the
                        // guard's `Drop` covers traps and interpreter errors too, which a manual
                        // pop after `?` would not.
                        // A3c-S: one installer for every callable kind. The old
                        // `push_generic_frame` bound only a free function's own parameters and did
                        // not compose with an enclosing instantiation; `push_callable_env` does
                        // both, so a generic call inside a generic body now resolves too.
                        self.invoke_callable(
                            ResolvedInvocation {
                                callable,
                                environment: InvocationEnv::Published(callee),
                            },
                            ReceiverSource::None,
                            values,
                            span,
                        )
                        .map(Flow::Value)
                    }
                    Err(propagated) => Ok(Flow::Propagate(propagated)),
                },
                Res::Variant(item, variant) => {
                    // A positional enum-variant constructor (`Some(x)`, `MyEnum::Variant(a, b)`)
                    // is aggregate construction via call syntax, the same construct
                    // `eval_aggregate_elements` already covers for tuple/array literals.
                    match self.eval_aggregate_elements(args)? {
                        Ok(values) => Ok(Flow::Value(Value::Enum {
                            item: *item,
                            variant: *variant,
                            fields: values,
                            named: BTreeMap::new(),
                        })),
                        Err(propagated) => Ok(Flow::Propagate(propagated)),
                    }
                }
                Res::TraitMember(trait_id, member) => {
                    self.call_qualified_trait(expr_id, *trait_id, *member, args, span)
                }
                Res::CoreTraitMember(core_trait, _) => {
                    self.call_qualified_core_trait(expr_id, *core_trait, args, span)
                }
                Res::AssociatedFn(item, name) => match self.eval_call_arguments(args)? {
                    Ok(values) => {
                        let callable = self
                            .find_associated_fn(*item, self.text(*name))
                            .ok_or_else(|| {
                                RuntimeError::new("associated function not found", span)
                            })?;
                        // **A4: install the instantiation, as the `Res::Item` path already does.**
                        //
                        // This path pushed no generic environment, so `Stack::identity<T>(6)` ran
                        // its body with `T` unbound. Nothing observed it while no boundary consulted
                        // the declared type; the RETURN boundary did, immediately, on its first run.
                        // AS3 criterion 2 says every dispatch installs the checker-selected
                        // environment — this one did not.
                        self.invoke_callable(
                            ResolvedInvocation {
                                callable,
                                environment: InvocationEnv::Published(callee),
                            },
                            ReceiverSource::None,
                            values,
                            span,
                        )
                        .map(Flow::Value)
                    }
                    Err(propagated) => Ok(Flow::Propagate(propagated)),
                },
                // DEV-061: an indirect call through a function-value local or `self`
                // (`let f: fn(Int32) -> Int32 = double; f(x)`, or a fn-typed parameter).
                // These previously fell into the "expression is not callable" arm below even
                // though the general value-dispatch machinery (the non-Path fallback of the
                // outer match) already handled exactly this for non-path callee expressions.
                Res::Local(_) | Res::SelfValue(_) => {
                    let function = self.expect_value(callee)?;
                    if let Some(propagated) = self.pending_propagation.take() {
                        return Ok(Flow::Propagate(propagated));
                    }
                    let Value::Function(callee) = function else {
                        return Err(RuntimeError::new("expression is not callable", span));
                    };
                    match self.eval_call_arguments(args)? {
                        Ok(values) => {
                            let callable = self.item_callable(callee.item).ok_or_else(|| {
                                RuntimeError::new("expression is not callable", span)
                            })?;
                            // **A4: install the instantiation the VALUE carries.**
                            //
                            // DEV-178 put `bindings` on `FunctionValue` precisely because a
                            // function value's instantiation cannot be recovered from the call
                            // site — `Ty::Fn` cannot say which one produced it. This path then
                            // discarded them, so `let f: fn(Int32) -> Int32 = identity;` ran
                            // `identity<T>`'s body with `T` unbound. Invisible until the RETURN
                            // boundary consulted the declared type.
                            self.invoke_callable(
                                ResolvedInvocation {
                                    callable,
                                    environment: InvocationEnv::Captured(callee.clone()),
                                },
                                ReceiverSource::None,
                                values,
                                span,
                            )
                            .map(Flow::Value)
                        }
                        Err(propagated) => Ok(Flow::Propagate(propagated)),
                    }
                }
                _ => Err(RuntimeError::new("expression is not callable", span)),
            },
            hir::ExprKind::Field { base, name, .. } => {
                self.call_method(expr_id, *base, self.text(*name).to_string(), args, span)
            }
            _ => {
                let function = self.expect_value(callee)?;
                if let Some(propagated) = self.pending_propagation.take() {
                    return Ok(Flow::Propagate(propagated));
                }
                let Value::Function(callee) = function else {
                    return Err(RuntimeError::new("expression is not callable", span));
                };
                match self.eval_call_arguments(args)? {
                    Ok(values) => {
                        let callable = self
                            .item_callable(callee.item)
                            .ok_or_else(|| RuntimeError::new("expression is not callable", span))?;
                        self.invoke_callable(
                            ResolvedInvocation {
                                callable,
                                environment: InvocationEnv::Captured(callee.clone()),
                            },
                            ReceiverSource::None,
                            values,
                            span,
                        )
                        .map(Flow::Value)
                    }
                    Err(propagated) => Ok(Flow::Propagate(propagated)),
                }
            }
        }
    }

    /// WP-C5.3e (CD-067) / DEV-100: answer `size_of::<T>()` / `align_of::<T>()` from the selected
    /// named target CONTRACT, over the queried type with the ACTIVE generic instantiation applied.
    ///
    /// The oracle previously returned a hardcoded `8` for every query without reading the queried
    /// type at all. It still does not own a type walker: the checker publishes `LayoutTables`,
    /// because it owns the declaration-ordered nominal tables and generic parameter names, and a
    /// second walker here would be a fourth derivation of machinery that exists.
    ///
    /// An unsubstituted parameter surviving to here is an oracle DEFECT, not a fallback: answering
    /// a layout query for an unknown type would be inventing an observable value.
    fn layout_query(
        &mut self,
        builtin: Builtin,
        callee: ExprId,
        span: Span,
    ) -> Result<Value, RuntimeError> {
        let Some(queried) = self.tables.layout_queries.get(&callee).cloned() else {
            return Err(RuntimeError::new(
                "layout query with no recorded type argument",
                span,
            ));
        };
        let concrete = match self.generic_frames.borrow().last() {
            Some(map) => crate::typecheck::substitute_ty(&queried, map),
            None => queried,
        };
        if crate::typecheck::ty_contains_param(&concrete) {
            // **DEV-176: a compiler defect, not a language trap.**
            //
            // An accepted program reached execution with a type parameter the oracle could not
            // resolve, because `push_generic_frame` installs a substitution frame only for direct
            // free-function calls — impl, method and trait generics and `Self` are never bound. The
            // program is valid; the oracle lacks the context.
            //
            // Classified `internal` because the HIR interpreter is the behavioural oracle: a
            // trap-classified oracle defect is one the differential harness can accept as a
            // legitimate program outcome and then pressure MIR and native into reproducing.
            //
            // Deliberately NARROW. Only this condition changes — ordinary `layout_of` refusals keep
            // their classification until each is individually judged, because some of them do
            // correspond to genuinely invalid programs and folding those into `InternalInvariant`
            // would make the class meaningless.
            return Err(RuntimeError::internal(
                format!(
                    "DEV-176: layout query on {concrete:?} still contains an unsubstituted generic \
                     parameter: the oracle installs generic context only for direct free-function \
                     calls, so an impl, method or trait parameter is never bound"
                ),
                span,
            ));
        }
        let layout = self
            .tables
            .layout
            .layout_of(&concrete)
            .map_err(|e| RuntimeError::new(e.0, span))?;
        Ok(Value::Int(i128::from(if builtin == Builtin::SizeOf {
            layout.size
        } else {
            layout.align
        })))
    }

    /// DEV-100: install the generic instantiation recorded for this call site, if the callee is
    /// generic, returning a guard that removes it again.
    ///
    /// Deliberately narrow. This is a call-time substitution CONTEXT and nothing more: it does not
    /// clone or specialise HIR bodies, does not touch value execution, does not infer missing
    /// arguments, and never falls back to a partial map. A generic item whose call site has no
    /// recorded instantiation, or whose arity disagrees, installs NOTHING — the query then fails
    /// as an unsubstituted parameter rather than silently answering from a stale or partial frame.
    /// Build a function value, capturing the environment the checker selected at this use.
    ///
    /// A non-generic function captures nothing, which is the common case and costs a map lookup.
    /// A generic one captures its bindings CONCRETISED against the active frame, so
    /// `fn outer<T>() { let f: fn() -> UInt64 = type_size::<T>; }` stores `T`'s caller-resolved
    /// value rather than the parameter.
    fn capture_function_value(
        &self,
        item: ItemId,
        use_expr: ExprId,
        span: Span,
    ) -> Result<FunctionValue, RuntimeError> {
        let Some(env) = self.tables.callable_instantiations.get(&use_expr) else {
            // **DEV-204: an absent instantiation must not silently mean "no generics".**
            //
            // This returned an empty-binding `FunctionValue` for ANY missing entry, which is the
            // DEV-178 defect written as a fallback: a generic function coerced to a value would
            // carry nothing, and `Ty::Fn` cannot say which instantiation produced it, so nothing
            // downstream could recover it. The two meanings of absence are separated by
            // information we already have — whether the item declares generics at all.
            let generics = match &self.hir.item(item).kind {
                hir::ItemKind::Fn(def) => def.sig.generics.len(),
                _ => 0,
            };
            if generics > 0 {
                return Err(RuntimeError::internal(
                    format!(
                        "DEV-178: a generic function with {generics} parameter(s) was coerced to a                          function value with no published instantiation — the bindings are fixed                          at the coercion and cannot be recovered from `Ty::Fn` at the call"
                    ),
                    span,
                ));
            }
            return Ok(FunctionValue {
                item,
                bindings: Vec::new(),
            });
        };
        // Audit 10-D: keep the value, lose the instantiation.
        #[cfg(test)]
        if self.mutation_armed(ProducerMutation::StripFunctionValueBindings) {
            return Ok(FunctionValue {
                item,
                bindings: Vec::new(),
            });
        }
        // The published environment must belong to the body this item actually runs — the
        // signature is body-keyed and the environment call-site-keyed, so their agreement is
        // asserted rather than assumed.
        if let hir::ItemKind::Fn(def) = &self.hir.item(item).kind {
            if def.body != env.body {
                return Err(RuntimeError::internal(
                    format!(
                        "DEV-178: the environment published for this use names body {:?}, but the \
                         function item executes {:?}",
                        env.body, def.body
                    ),
                    span,
                ));
            }
        }
        let mut bindings = Vec::new();
        for (binder, ty) in &env.bindings {
            bindings.push((
                binder.name().to_string(),
                self.concrete_runtime_ty(ty, span)?,
            ));
        }
        Ok(FunctionValue { item, bindings })
    }

    /// **A3c-S: install the checker-selected generic environment for one callable use.**
    ///
    /// `push_generic_frame` binds only a free function's own parameters, which is DEV-176: impl
    /// generics, method generics, trait generics and `Self` were never bound, so a generic method
    /// body executed with no idea what its parameters stood for.
    ///
    /// **Composition happens BEFORE the push, and that ordering is load-bearing.** Published
    /// bindings may themselves contain `Ty::Param` when the CALLER is generic — `fn outer<T>(w:
    /// Wrapper<T>)` publishes `impl T -> T`. Each value is concretised against the currently active
    /// frame first; pushing first and substituting after would resolve `T -> T` against itself and
    /// silently keep the parameter.
    ///
    /// The interpreter CONSUMES this environment and never reconstructs one from names, runtime
    /// values or impl scanning — a second instantiation algorithm would be a second answer to what
    /// a generic call means.
    fn push_callable_env(
        &mut self,
        callee: ExprId,
        span: Span,
    ) -> Result<GenericFrame, RuntimeError> {
        let Some(env) = self.tables.callable_instantiations.get(&callee).cloned() else {
            return Ok(GenericFrame {
                frames: self.generic_frames.clone(),
                pushed: false,
            });
        };
        if env.bindings.is_empty() {
            return Ok(GenericFrame {
                frames: self.generic_frames.clone(),
                pushed: false,
            });
        }
        // **Two passes, because the bindings have a dependency order.** `Self` is published as the
        // impl's self type — `Wrapper<T>` — which references the impl's OWN parameters, so it
        // cannot be resolved against the caller's frame alone. The parameters are concretised
        // first, then `Self` is substituted through them before its own concretisation. A flat
        // single-pass loop leaves `Self = Wrapper<Param("T")>` and fails at the first value
        // boundary.
        self.push_bindings(&env.bindings, span)
    }

    /// **AS3 Boundary 4: install a generic environment the SPECIALISER produced.**
    ///
    /// A `Bound` call's environment cannot be published: the body is only chosen once `Self` is
    /// concrete, which is here. `specialize_bound_callable` returns those bindings, and without
    /// installing them a trait DEFAULT body reached through a bound runs with no `Self` at all —
    /// so `self.name()` inside it resolves nothing. That was the `pkg/07-traits` regression: the
    /// name scan used to paper over the missing environment by finding `name` on the runtime
    /// value's nominal.
    fn push_resolved_env(
        &mut self,
        bindings: &[(crate::typecheck::GenericBinder, Ty)],
        span: Span,
    ) -> Result<GenericFrame, RuntimeError> {
        if bindings.is_empty() {
            return Ok(GenericFrame {
                frames: self.generic_frames.clone(),
                pushed: false,
            });
        }
        self.push_bindings(bindings, span)
    }

    /// The two-pass installation shared by both. **`Self` is resolved second, on purpose:** it is
    /// published as the impl's self type (`Wrapper<T>`), which mentions the impl's own parameters,
    /// so it cannot be concretised until they are. A flat single pass leaves
    /// `Self = Wrapper<Param("T")>` and fails at the first value boundary.
    fn push_bindings(
        &mut self,
        bindings: &[(crate::typecheck::GenericBinder, Ty)],
        span: Span,
    ) -> Result<GenericFrame, RuntimeError> {
        let mut concrete: HashMap<String, Ty> = HashMap::new();
        for (binder, ty) in bindings {
            if matches!(binder, crate::typecheck::GenericBinder::SelfType) {
                continue;
            }
            concrete.insert(
                binder.name().to_string(),
                self.concrete_runtime_ty(ty, span)?,
            );
        }
        for (binder, ty) in bindings {
            if !matches!(binder, crate::typecheck::GenericBinder::SelfType) {
                continue;
            }
            let through_params = crate::typecheck::substitute_ty(ty, &concrete);
            concrete.insert(
                binder.name().to_string(),
                self.concrete_runtime_ty(&through_params, span)?,
            );
        }
        self.generic_frames.borrow_mut().push(concrete);
        Ok(GenericFrame {
            frames: self.generic_frames.clone(),
            pushed: true,
        })
    }

    /// **DEV-126: flatten a reference argument that refers to a STRING.**
    ///
    /// `as_str` yields the receiver's place rather than a detached copy, so `s.as_str()` arrives at
    /// a builtin or core method as a `Value::Ref`. `string_arg` is a free function with no `&self`
    /// and therefore no way to follow a place, so it rejected those as "expected string argument".
    ///
    /// The condition is the REFERENT'S KIND, not the callee's name. Keying on names is what the
    /// `remove`/`contains_key`/`contains` special case does, and it only ever covered the three
    /// that had been reported; every string-taking entry point has the same requirement. Keying on
    /// "the referent is a string" also cannot disturb a `&mut Vec`/`&mut HashMap` argument, whose
    /// whole purpose is to stay a reference — those referents are not strings.
    fn flatten_string_refs(&mut self, args: &mut [Value], span: Span) -> Result<(), RuntimeError> {
        for argument in args {
            if !matches!(argument, Value::Ref(_)) {
                continue;
            }
            let flattened = self.deref_value(argument.clone(), span)?;
            if matches!(flattened, Value::Str(_) | Value::String(_)) {
                *argument = flattened;
            }
        }
        Ok(())
    }

    /// **DEV-131: deref a string argument at the sites that read its CONTENT, and only those.**
    ///
    /// DEV-126 first solved this by flattening every reference-to-string argument on the way into
    /// `call_builtin`. That was too broad and broke `take(&mut a)`: `take` needs the REFERENCE, and
    /// flattening handed it the string. A blanket rule cannot tell "reads the text" from "needs the
    /// place" because `Value::Ref` does not record which the caller meant.
    ///
    /// So the deref moves to the five sites that call `string_arg` — the ones that demonstrably
    /// want text. Anything wanting a place is untouched by construction rather than by exemption.
    fn string_arg_deref(
        &mut self,
        value: Option<Value>,
        span: Span,
    ) -> Result<String, RuntimeError> {
        let value = match value {
            Some(value) => Some(self.deref_value(value, span)?),
            None => None,
        };
        string_arg(value, span)
    }

    fn call_builtin(
        &mut self,
        builtin: Builtin,
        mut args: Vec<Value>,
        // AS3 Boundary 4: the ARGUMENT expressions, so the print family can key the Display plan
        // on the same root the checker used. `call_builtin` receives evaluated values, which carry
        // no identity of their own.
        arg_exprs: &[ExprId],
        span: Span,
    ) -> Result<Value, RuntimeError> {
        match builtin {
            Builtin::Print | Builtin::Println => {
                let value = args.pop().unwrap_or(Value::Unit);
                let deref = self.deref_value(value, span)?;
                let root = arg_exprs.first().copied();
                let (text, arg_place) = match root {
                    Some(root) => self.display_text(root, deref, span)?,
                    // No argument expression: `print()` with nothing to render.
                    None => (self.format_runtime_value(&deref, span)?, None),
                };
                self.output.push_str(&text);
                if builtin == Builtin::Println {
                    self.output.push('\n');
                }
                self.finish_display(arg_place, span)?;
                Ok(Value::Unit)
            }
            Builtin::Panic => {
                let value = args.pop().unwrap_or(Value::Unit);
                let deref = self.deref_value(value, span)?;
                // DEV-106: the message is arbitrary USER text, so the category is stated rather
                // than left to be inferred from prose.
                Err(RuntimeError::with_category(
                    self.format_runtime_value(&deref, span)?,
                    span,
                    crate::mir::TrapCategory::Panic,
                ))
            }
            Builtin::Assert => match args.pop() {
                Some(Value::Bool(true)) => Ok(Value::Unit),
                Some(Value::Bool(false)) => Err(RuntimeError::with_category(
                    "assertion failed",
                    span,
                    crate::mir::TrapCategory::AssertFailure,
                )),
                _ => Err(RuntimeError::new("assert expects Bool", span)),
            },
            Builtin::AssertEq | Builtin::AssertNe => {
                let right = args.pop().ok_or_else(|| {
                    RuntimeError::new("assert_eq/assert_ne expects two arguments", span)
                })?;
                let left = args.pop().ok_or_else(|| {
                    RuntimeError::new("assert_eq/assert_ne expects two arguments", span)
                })?;
                let left = self.deref_value(left, span)?;
                let right = self.deref_value(right, span)?;
                // DEV-130: was a raw `==`, so `assert_eq(s.as_str(), "beta")` failed with
                // `left: beta, right: beta` — identical text in different wrappers.
                let equal = values_equal(&left, &right);
                let want_eq = builtin == Builtin::AssertEq;
                if equal == want_eq {
                    Ok(Value::Unit)
                } else if want_eq {
                    Err(RuntimeError::with_category(
                        format!("assertion failed: `(left == right)`\n  left: `{left}`\n right: `{right}`"),
                        span,
                        crate::mir::TrapCategory::AssertFailure,
                    ))
                } else {
                    Err(RuntimeError::with_category(
                        format!("assertion failed: `(left != right)`\n  left: `{left}`\n right: `{right}`"),
                        span,
                        crate::mir::TrapCategory::AssertFailure,
                    ))
                }
            }
            // Transcendental domain errors produce NaN rather than a language trap (the
            // standard-library math contract, distinct from the numeric-trap rules governing
            // integer overflow/division and float-to-int casts). `f64::sqrt` already returns
            // NaN for negative finite inputs, so no domain branch is needed.
            Builtin::Sqrt => match args.pop() {
                Some(Value::Float(value, _)) => Ok(Value::Float(
                    canonicalize_nan(value.sqrt(), FloatWidth::F64),
                    FloatWidth::F64,
                )),
                _ => Err(RuntimeError::new("sqrt expects Float64", span)),
            },
            Builtin::Drop => {
                if let Some(value) = args.pop() {
                    self.drop_value(value)?;
                }
                Ok(Value::Unit)
            }
            // WP-C5.3e: unreachable -- `eval_call` intercepts layout queries, which need the
            // type argument this entry point does not receive. Kept as a loud failure rather
            // than deleted, so a new call path that misses the interception is caught.
            Builtin::SizeOf | Builtin::AlignOf => Err(RuntimeError::new(
                "layout query reached call_builtin, which has no type argument",
                span,
            )),
            Builtin::Swap => {
                let b = args
                    .pop()
                    .ok_or_else(|| RuntimeError::new("swap expects two arguments", span))?;
                let a = args
                    .pop()
                    .ok_or_else(|| RuntimeError::new("swap expects two arguments", span))?;
                if let (Value::Ref(place_a), Value::Ref(place_b)) = (a, b) {
                    let slot_a = self.place_slot_mut(&place_a, span)?;
                    let val_a = slot_a
                        .take()
                        .ok_or_else(|| RuntimeError::new("use of moved value", span))?;

                    let slot_b = self.place_slot_mut(&place_b, span)?;
                    let val_b = slot_b
                        .take()
                        .ok_or_else(|| RuntimeError::new("use of moved value", span))?;

                    let slot_a = self.place_slot_mut(&place_a, span)?;
                    *slot_a = Some(val_b);

                    let slot_b = self.place_slot_mut(&place_b, span)?;
                    *slot_b = Some(val_a);

                    Ok(Value::Unit)
                } else {
                    Err(RuntimeError::new("swap expects mutable references", span))
                }
            }
            Builtin::Replace => {
                let src = args
                    .pop()
                    .ok_or_else(|| RuntimeError::new("replace expects two arguments", span))?;
                let dest = args
                    .pop()
                    .ok_or_else(|| RuntimeError::new("replace expects two arguments", span))?;
                if let Value::Ref(place_dest) = dest {
                    let slot = self.place_slot_mut(&place_dest, span)?;
                    let old_val = slot
                        .replace(src)
                        .ok_or_else(|| RuntimeError::new("use of moved value", span))?;
                    Ok(old_val)
                } else {
                    Err(RuntimeError::new("replace expects mutable reference", span))
                }
            }
            Builtin::Take => {
                let dest = args
                    .pop()
                    .ok_or_else(|| RuntimeError::new("take expects one argument", span))?;
                if let Value::Ref(place_dest) = dest {
                    let old_val = self.place_value(&place_dest, span)?.clone();
                    let def_val = self.default_value_for(&old_val);
                    let slot = self.place_slot_mut(&place_dest, span)?;
                    let _ = slot
                        .replace(def_val)
                        .ok_or_else(|| RuntimeError::new("use of moved value", span))?;
                    Ok(old_val)
                } else {
                    Err(RuntimeError::new("take expects mutable reference", span))
                }
            }
            Builtin::StringFrom => {
                let text = self.string_arg_deref(args.pop(), span)?;
                Ok(Value::String(text))
            }
            Builtin::StringNew => Ok(Value::String(String::new())),
            Builtin::StringWithCapacity => {
                let capacity = usize_arg(args.pop(), span)?;
                Ok(Value::String(String::with_capacity(capacity)))
            }
            Builtin::CharFromU32 => {
                let code = u32_arg(args.pop(), span)?;
                Ok(Value::Option(
                    char::from_u32(code).map(|ch| Box::new(Some(Value::Char(ch)))),
                ))
            }
            Builtin::VecNew => Ok(Value::Vec(Vec::new())),
            Builtin::VecWithCapacity => {
                let capacity = usize_arg(args.pop(), span)?;
                Ok(Value::Vec(Vec::with_capacity(capacity)))
            }
            Builtin::HashMapNew => Ok(Value::HashMap(InsertionMap::new())),
            Builtin::HashMapWithCapacity => {
                let _capacity = usize_arg(args.pop(), span)?;
                Ok(Value::HashMap(InsertionMap::new()))
            }
            Builtin::HashSetNew => Ok(Value::HashSet(InsertionSet::new())),
            Builtin::BoxNew => Ok(Value::Boxed(Box::new(args.pop()))),
            Builtin::BoxIntoInner => match args.pop() {
                Some(Value::Boxed(value)) => Ok((*value).unwrap_or(Value::Unit)),
                _ => Err(RuntimeError::new("Box::into_inner expects Box", span)),
            },
            Builtin::Some => Ok(Value::Option(args.pop().map(|value| Box::new(Some(value))))),
            Builtin::None => Ok(Value::Option(None)),
            Builtin::Ok => Ok(Value::Result(Ok(Box::new(Some(
                args.pop().unwrap_or(Value::Unit),
            ))))),
            Builtin::Err => Ok(Value::Result(Err(Box::new(Some(
                args.pop().unwrap_or(Value::Unit),
            ))))),
            Builtin::ReadFile => {
                let path = self.string_arg_deref(args.pop(), span)?;
                Ok(match std::fs::read_to_string(path) {
                    Ok(value) => Value::Result(Ok(Box::new(Some(Value::String(value))))),
                    Err(error) => Value::Result(Err(Box::new(Some(Value::IOError(
                        IOErrorKind::from_io_error(&error),
                    ))))),
                })
            }
            Builtin::WriteFile => {
                if args.len() != 2 {
                    return Err(RuntimeError::new("write_file expects two arguments", span));
                }
                let content = self.string_arg_deref(args.pop(), span)?;
                let path = self.string_arg_deref(args.pop(), span)?;
                Ok(match std::fs::write(path, content) {
                    Ok(()) => Value::Result(Ok(Box::new(Some(Value::Unit)))),
                    Err(error) => Value::Result(Err(Box::new(Some(Value::IOError(
                        IOErrorKind::from_io_error(&error),
                    ))))),
                })
            }
            Builtin::FileOpen | Builtin::FileCreate => {
                let path = self.string_arg_deref(args.pop(), span)?;
                let result = if builtin == Builtin::FileOpen {
                    std::fs::File::open(path)
                } else {
                    std::fs::File::create(path)
                };
                Ok(match result {
                    Ok(file) => {
                        Value::Result(Ok(Box::new(Some(Value::File(FileResource::new(file))))))
                    }
                    Err(error) => Value::Result(Err(Box::new(Some(Value::IOError(
                        IOErrorKind::from_io_error(&error),
                    ))))),
                })
            }
            // -- Phase 4E: Math constants and functions --
            Builtin::MathAbs => match args.pop() {
                Some(Value::Int(value)) => value.checked_abs().map(Value::Int).ok_or_else(|| {
                    RuntimeError::with_category(
                        "integer overflow",
                        span,
                        crate::mir::TrapCategory::IntegerOverflow,
                    )
                }),
                Some(Value::Float(value, width)) => {
                    Ok(Value::Float(canonicalize_nan(value.abs(), width), width))
                }
                _ => Err(RuntimeError::new("abs expects Int or Float", span)),
            },
            Builtin::MathMin | Builtin::MathMax => {
                let b = args.pop();
                let a = args.pop();
                let ord = numeric_cmp(&a, &b, span)?;
                let want = if builtin == Builtin::MathMin {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                };
                Ok(if ord == want || ord == std::cmp::Ordering::Equal {
                    a.unwrap()
                } else {
                    b.unwrap()
                })
            }
            Builtin::MathClamp => {
                let max = args.pop();
                let min = args.pop();
                let value = args.pop();
                if numeric_cmp(&value, &min, span)? == std::cmp::Ordering::Less {
                    Ok(min.unwrap())
                } else if numeric_cmp(&value, &max, span)? == std::cmp::Ordering::Greater {
                    Ok(max.unwrap())
                } else {
                    Ok(value.unwrap())
                }
            }
            Builtin::Pow => {
                let exp = float_arg(args.pop(), span)?;
                let base = float_arg(args.pop(), span)?;
                Ok(Value::Float(
                    canonicalize_nan(base.powf(exp), FloatWidth::F64),
                    FloatWidth::F64,
                ))
            }
            Builtin::Atan2 => {
                let x = float_arg(args.pop(), span)?;
                let y = float_arg(args.pop(), span)?;
                Ok(Value::Float(
                    canonicalize_nan(y.atan2(x), FloatWidth::F64),
                    FloatWidth::F64,
                ))
            }
            Builtin::Log => Ok(Value::Float(
                canonicalize_nan(float_arg(args.pop(), span)?.ln(), FloatWidth::F64),
                FloatWidth::F64,
            )),
            Builtin::Log10 => Ok(Value::Float(
                canonicalize_nan(float_arg(args.pop(), span)?.log10(), FloatWidth::F64),
                FloatWidth::F64,
            )),
            Builtin::Exp => Ok(Value::Float(
                canonicalize_nan(float_arg(args.pop(), span)?.exp(), FloatWidth::F64),
                FloatWidth::F64,
            )),
            Builtin::Sin => Ok(Value::Float(
                canonicalize_nan(float_arg(args.pop(), span)?.sin(), FloatWidth::F64),
                FloatWidth::F64,
            )),
            Builtin::Cos => Ok(Value::Float(
                canonicalize_nan(float_arg(args.pop(), span)?.cos(), FloatWidth::F64),
                FloatWidth::F64,
            )),
            Builtin::Tan => Ok(Value::Float(
                canonicalize_nan(float_arg(args.pop(), span)?.tan(), FloatWidth::F64),
                FloatWidth::F64,
            )),
            Builtin::Asin => Ok(Value::Float(
                canonicalize_nan(float_arg(args.pop(), span)?.asin(), FloatWidth::F64),
                FloatWidth::F64,
            )),
            Builtin::Acos => Ok(Value::Float(
                canonicalize_nan(float_arg(args.pop(), span)?.acos(), FloatWidth::F64),
                FloatWidth::F64,
            )),
            Builtin::Atan => Ok(Value::Float(
                canonicalize_nan(float_arg(args.pop(), span)?.atan(), FloatWidth::F64),
                FloatWidth::F64,
            )),
            Builtin::Floor => Ok(Value::Float(
                canonicalize_nan(float_arg(args.pop(), span)?.floor(), FloatWidth::F64),
                FloatWidth::F64,
            )),
            Builtin::Ceil => Ok(Value::Float(
                canonicalize_nan(float_arg(args.pop(), span)?.ceil(), FloatWidth::F64),
                FloatWidth::F64,
            )),
            Builtin::Round => Ok(Value::Float(
                canonicalize_nan(float_arg(args.pop(), span)?.round(), FloatWidth::F64),
                FloatWidth::F64,
            )),
            Builtin::Trunc => Ok(Value::Float(
                canonicalize_nan(float_arg(args.pop(), span)?.trunc(), FloatWidth::F64),
                FloatWidth::F64,
            )),
            // -- Phase 4E: stderr; WP-C7.9 Packet D --
            //
            // These used to write through to the HOST process's stderr with `eprint!`. Two things
            // followed, both bad: the bytes never reached `Execution.stderr`, so no comparator
            // could see them and a case could "agree" while nobody observed the operation at all;
            // and a STARK program's stderr landed in the Rust test runner's own stderr, mixed with
            // whatever the harness was saying. They are captured now, exactly as `print`/`println`
            // are, and the channel is compared.
            Builtin::Eprint | Builtin::Eprintln => {
                let value = args.pop().unwrap_or(Value::Unit);
                let deref = self.deref_value(value, span)?;
                let root = arg_exprs.first().copied();
                let (text, arg_place) = match root {
                    Some(root) => self.display_text(root, deref, span)?,
                    // No argument expression: `print()` with nothing to render.
                    None => (self.format_runtime_value(&deref, span)?, None),
                };
                self.stderr.push_str(&text);
                if builtin == Builtin::Eprintln {
                    self.stderr.push('\n');
                }
                self.finish_display(arg_place, span)?;
                Ok(Value::Unit)
            }
            // -- Phase 4E: Random --
            Builtin::RandomNew => Ok(Value::Random(u64_arg(args.pop(), span)?)),
            // -- Phase 4E: IOError --
            Builtin::OrderingLess => Ok(Value::Ordering(std::cmp::Ordering::Less)),
            Builtin::OrderingEqual => Ok(Value::Ordering(std::cmp::Ordering::Equal)),
            Builtin::OrderingGreater => Ok(Value::Ordering(std::cmp::Ordering::Greater)),
            Builtin::IOErrorNotFound => Ok(Value::IOError(IOErrorKind::NotFound)),
            Builtin::IOErrorPermissionDenied => Ok(Value::IOError(IOErrorKind::PermissionDenied)),
            Builtin::IOErrorAlreadyExists => Ok(Value::IOError(IOErrorKind::AlreadyExists)),
            Builtin::IOErrorInvalidInput => Ok(Value::IOError(IOErrorKind::InvalidInput)),
            Builtin::IOErrorOther => Ok(Value::IOError(IOErrorKind::Other(string_arg(
                args.pop(),
                span,
            )?))),
            Builtin::MathPi | Builtin::MathE => {
                Err(RuntimeError::new("PI/E are constants, not callable", span))
            }
            // AS6: one refusal, not thirty-three patterns for it. The oracle has no tensor
            // runtime at all, so the answer does not vary by operation — and enumerating them here
            // made Core's interpreter carry the extension's catalogue to say so.
            Builtin::Tensor(_) => Err(RuntimeError::new(
                "tensor operations are not supported in the Core interpreter",
                span,
            )),
        }
    }

    fn call_method(
        &mut self,
        expr_id: ExprId,
        base: ExprId,
        name: String,
        args: &[ExprId],
        span: Span,
    ) -> Result<Flow, RuntimeError> {
        if self.is_core_value(base) {
            let result = self.call_core_method(Some(expr_id), base, &name, args, span)?;
            return Ok(match self.pending_propagation.take() {
                Some(propagated) => Flow::Propagate(propagated),
                None => Flow::Value(result),
            });
        }
        // WP-C2.2 (DEV-034): resolve the receiver to a place exactly once, before anything else.
        // A place expression resolves without re-running its subexpressions later; a non-place
        // expression (e.g. `make_thing().consume()`) evaluates once here into a synthetic temp
        // in the caller's frame. Previously the by-value receiver path re-evaluated the original
        // receiver expression a second time inside `call_user_method` (confirmed empirically:
        // a `println` inside a receiver-constructing function printed twice for one call), and
        // the `&mut self` path re-resolved the place (re-running index subexpressions). The
        // resolved place is also what DEV-035's returned-reference rebasing targets.
        let receiver_place = self.core_receiver_place(base, span)?;
        let receiver_value = self.clone_place_value(&receiver_place, span)?;
        let nominal = nominal_item(&receiver_value);
        // DEV-BOUND-TRAIT-IDENTITY: when this call resolved through a generic parameter's bound,
        // the checker recorded WHICH trait supplied the method. Selecting by name alone ran the
        // first impl on this nominal declaring it, so two bounds naming two different same-named
        // traits both reached the same implementation.
        let trait_filter = self.tables.bound_trait_calls.get(&expr_id).copied();
        // **AS3 Boundary 4c: consume the shared bound specialiser.**
        //
        // A `Bound` use fixes the obligation; the body becomes knowable here, where the receiver is
        // a concrete value. Resolving it through `bound_dispatch` rather than `find_method` means
        // the interpreter and MIR ask ONE authority the same question — and it is the only path
        // that reaches a trait DEFAULT body by construction rather than by the scan happening to
        // fall through to one.
        // A `Bound` resolution also yields the ENVIRONMENT the body must run under; a `Static`
        // one's environment is published and installed by `push_callable_env`.
        let mut bound_env: Option<Vec<(crate::typecheck::GenericBinder, Ty)>> = None;
        let specialised = self.static_selected_callable(expr_id).or_else(|| {
            self.specialised_bound_callable(expr_id, base)
                .map(|(callable, env)| {
                    bound_env = Some(env);
                    callable
                })
        });
        // **AS3 Boundary 4: no name-scan fallback.** This ended in
        // `.or_else(|| self.find_method(nominal, &name, trait_filter))`. Instrumented across nine
        // suites — both differentials, iterators, bound identity, adversarial trait impls,
        // cross-package generics, associated types and Display dispatch — it fired **zero** times
        // once the `Static` selection was consumed. Deleted rather than annotated: unreached in the
        // suites you ran is not unreachable, and an annotation is what let DEV-191 hide.
        let _ = (&name, trait_filter);
        let method = match specialised {
            Some(method) => method,
            // DEV-DISPLAY-DISPATCH: the receiver's STATIC type is a generic parameter, so
            // `is_core_value` above could not classify it — `Ty::Param` names no shape. The
            // RUNTIME value settles it: a value with no nominal item (an `Int32`, a `String`, a
            // `Vec`) is a Core value, and a `T: Display` bound made `x.fmt()` legal for exactly
            // that instantiation. Dispatch it through the Core surface with the place already in
            // hand. A generic parameter instantiated at a user nominal never reaches here —
            // `find_method` resolves its `impl` — so this is additive, not a redirection.
            None if nominal.is_none() && self.receiver_is_type_param(base) => {
                let result = self.call_core_method_at(
                    Some(expr_id),
                    receiver_place,
                    base,
                    &name,
                    args,
                    span,
                )?;
                return Ok(match self.pending_propagation.take() {
                    Some(propagated) => Flow::Propagate(propagated),
                    None => Flow::Value(result),
                });
            }
            None => {
                return Err(RuntimeError::new(
                    format!("method '{name}' not found at runtime"),
                    span,
                ))
            }
        };
        match self.eval_call_arguments(args)? {
            Ok(values) => {
                // **Packet 1: an explicit environment, chosen once.** A `Bound` selection carries
                // the specialiser's concrete bindings; everything else consumes what the checker
                // published against this call expression.
                //
                // **DEV-202: this site used to INSTALL it as well, and then pass it on.** The
                // authority installs it too, so every method call pushed the callee's
                // instantiation twice — and, worse, the outer guard was live while the CALLER's
                // receiver place was still being resolved and materialized. Caller-side work under
                // the callee's environment is the same scope error P6 exists to prevent, running
                // in the other direction. Choosing the environment here and installing it there is
                // the split the authority was created for.
                let environment = match bound_env {
                    Some(bindings) => InvocationEnv::Concrete(bindings),
                    None => InvocationEnv::Published(expr_id),
                };
                self.call_user_method(method, receiver_place, environment, values, span)
                    .map(Flow::Value)
            }
            Err(propagated) => Ok(Flow::Propagate(propagated)),
        }
    }

    fn call_qualified_trait(
        &mut self,
        // AS3 Boundary 4: the call expression, so the checker's `Qualified` selection can be read.
        call_expr: ExprId,
        trait_id: ItemId,
        member: u32,
        args: &[ExprId],
        span: Span,
    ) -> Result<Flow, RuntimeError> {
        let method_name = match &self.hir.item(trait_id).kind {
            hir::ItemKind::Trait { items, .. } => match items.get(member as usize) {
                Some(hir::TraitItem::Method { sig, .. }) => self.text(sig.name).to_string(),
                _ => return Err(RuntimeError::new("trait member is not callable", span)),
            },
            _ => return Err(RuntimeError::new("invalid trait call", span)),
        };
        let Some((first, rest)) = args.split_first() else {
            return Err(RuntimeError::new("trait call requires receiver", span));
        };
        // WP-C2.2 (DEV-034/DEV-035): same single-resolution receiver handling as `call_method`.
        let receiver_place = self.core_receiver_place(*first, span)?;
        // `<T as Tr>::m()` is a `Qualified` dispatch the checker already resolved; scanning the
        // receiver's nominal for the member name re-decided it.
        let method = self
            .static_selected_callable(call_expr)
            .or_else(|| {
                self.specialised_bound_callable(call_expr, *first)
                    .map(|(c, _)| c)
            })
            .ok_or_else(|| RuntimeError::new("trait implementation not found", span))?;
        let _ = (&method_name, trait_id);
        // A qualified call may land on a trait DEFAULT body, which runs with `Self` parametric.
        // Install the binding the checker published, or `self.other()` inside it resolves nothing.
        let bindings = self.static_selected_env(call_expr);
        let _env = self.push_resolved_env(&bindings, span)?;
        match self.eval_call_arguments(rest)? {
            Ok(values) => self
                .call_user_method(
                    method,
                    receiver_place,
                    InvocationEnv::Concrete(bindings),
                    values,
                    span,
                )
                .map(Flow::Value),
            Err(propagated) => Ok(Flow::Propagate(propagated)),
        }
    }

    /// DEV-052: `Eq::eq(&a, &b)`-style qualified calls to a compiler-known `CoreTrait`'s method.
    /// Unlike `call_qualified_trait` (a user-declared trait, whose HIR item must be scanned for
    /// the member's declared name), a `CoreTrait` has no declaration item at all -- its single
    /// callable method name is fixed per trait (`resolve.rs`'s `core_trait_method_name`, shared
    /// so both modules agree), and dispatch reuses the exact same `find_method(..., Some(Res::
    /// CoreTrait(core_trait)))` lookup the `==`/`<`/etc. operator sugar already uses for these
    /// traits (`eval_binary`'s nominal Eq/Ord dispatch) -- a qualified call is just an explicit
    /// spelling of the same dispatch, not a separate mechanism.
    fn call_qualified_core_trait(
        &mut self,
        call_expr: ExprId,
        core_trait: CoreTrait,
        args: &[ExprId],
        span: Span,
    ) -> Result<Flow, RuntimeError> {
        let method_name = crate::resolve::core_trait_method_name(core_trait)
            .ok_or_else(|| RuntimeError::new("invalid trait call", span))?;
        let Some((first, rest)) = args.split_first() else {
            return Err(RuntimeError::new("trait call requires receiver", span));
        };
        let receiver_place = self.core_receiver_place(*first, span)?;
        // A qualified core-trait call publishes through `publish_operator_use`, so it carries
        // `CoreTrait` provenance — the same record `a == b` produces, because it is the same
        // dispatch spelled explicitly.
        let method = self
            .selected_core_trait_callable(call_expr, core_trait)
            .and_then(|use_| match use_.selection {
                crate::typecheck::CalleeSelection::Static { body, .. } => {
                    self.callable_for_body(body)
                }
                crate::typecheck::CalleeSelection::Bound { .. }
                | crate::typecheck::CalleeSelection::FunctionValue => None,
            })
            .or_else(|| self.specialised_operator_callable(call_expr, core_trait, span))
            .ok_or_else(|| RuntimeError::new("trait implementation not found", span))?;
        let environment = self.core_trait_env(call_expr, core_trait);
        let _ = method_name;
        match self.eval_call_arguments(rest)? {
            Ok(values) => self
                .call_user_method(method, receiver_place, environment, values, span)
                .map(Flow::Value),
            Err(propagated) => Ok(Flow::Propagate(propagated)),
        }
    }

    fn find_associated_fn(&self, nominal: ItemId, name: &str) -> Option<Callable> {
        let mut inherent = Vec::new();
        let mut trait_candidates = Vec::new();
        for (idx, item) in self.hir.items.iter().enumerate() {
            let impl_id = ItemId(idx as u32);
            let hir::ItemKind::Impl {
                trait_,
                self_ty,
                items,
                ..
            } = &item.kind
            else {
                continue;
            };
            if !matches!(
                &self.hir.ty(*self_ty).kind,
                hir::TypeKind::Path { res: Res::Item(item), .. } if *item == nominal
            ) {
                continue;
            }
            let candidate = items.iter().find_map(|item| match item {
                hir::ImplItem::Fn { def, .. }
                    if def.sig.receiver.is_none()
                        && self.item_text(impl_id, def.sig.name) == name =>
                {
                    Some(Callable {
                        receiver: None,
                        params: def.sig.params.iter().map(|param| param.local).collect(),
                        body: def.body,
                    })
                }
                _ => None,
            });
            if let Some(candidate) = candidate {
                if trait_.is_none() {
                    inherent.push(candidate);
                } else {
                    trait_candidates.push(candidate);
                }
            }
        }
        if inherent.len() == 1 {
            inherent.pop()
        } else if inherent.is_empty() && trait_candidates.len() == 1 {
            trait_candidates.pop()
        } else {
            None
        }
    }

    /// **A method invocation, through the one authority.**
    ///
    /// This was the last of three body-execution funnels: it pushed its own frame, called
    /// `eval_block` and popped, duplicating `execute_body` so that a method could carry its own
    /// epilogue. That epilogue is now [`BodyEpilogue::Method`], so the difference is a parameter
    /// and the executor is shared.
    ///
    /// The environment is supplied by the CALLER, which is what selected the body — a `Bound`
    /// selection carries the specialiser's bindings, everything else the checker's published
    /// instantiation.
    fn call_user_method(
        &mut self,
        callable: Callable,
        receiver_place: Place,
        environment: InvocationEnv,
        args: Vec<Value>,
        span: Span,
    ) -> Result<Value, RuntimeError> {
        let Some((receiver_kind, receiver_local)) = callable.receiver else {
            return Err(RuntimeError::new("method has no receiver", span));
        };
        self.invoke_with_epilogue(
            ResolvedInvocation {
                callable,
                environment,
            },
            ReceiverSource::Place {
                kind: receiver_kind,
                place: receiver_place.clone(),
            },
            args,
            BodyEpilogue::Method {
                receiver_kind,
                receiver_local,
                receiver_place,
            },
            span,
        )
    }

    /// DEV-DISPLAY-DISPATCH: whether the receiver expression's static type is (a reference to) a
    /// generic parameter. Such a receiver has no shape until the call is instantiated, so
    /// [`Self::is_core_value`] cannot classify it and dispatch has to consult the runtime value.
    fn receiver_is_type_param(&self, expr: ExprId) -> bool {
        let mut ty = self.tables.expr_types.get(&expr);
        while let Some(Ty::Ref { inner, .. }) = ty {
            ty = Some(inner.as_ref());
        }
        matches!(ty, Some(Ty::Param(_)))
    }

    fn is_core_value(&self, expr: ExprId) -> bool {
        let mut ty = self.tables.expr_types.get(&expr);
        while let Some(Ty::Ref { inner, .. }) = ty {
            ty = Some(inner.as_ref());
        }
        matches!(
            ty,
            Some(Ty::Primitive(..) | Ty::Core(..) | Ty::Array(..) | Ty::Slice(..) | Ty::Tuple(..))
        )
    }

    /// Execute collection operations that either require language-level `Eq`
    /// or discard owned values. They must run outside the generic `&mut
    /// target` match so discarded values can be routed through `drop_value`.
    fn call_collection_ownership_method(
        &mut self,
        // AS3 Boundary 4: the originating call, for the element `Eq::eq` lookup.
        call_expr: Option<ExprId>,
        receiver_place: &Place,
        name: &str,
        arguments: &mut Vec<Value>,
        span: Span,
    ) -> Result<Option<Value>, RuntimeError> {
        let snapshot = self.clone_place_value(receiver_place, span)?;
        match snapshot {
            Value::Vec(_) if name == "clear" => {
                let removed = match self.place_value_mut(receiver_place, span)? {
                    Value::Vec(values) => std::mem::take(values),
                    _ => unreachable!(),
                };
                for value in removed.into_iter().rev().flatten() {
                    self.drop_value(value)?;
                }
                Ok(Some(Value::Unit))
            }
            Value::HashMap(map) => {
                let key = arguments.first().cloned();
                let position = if let Some(key) = key.as_ref() {
                    let keys = map.keys().cloned().collect::<Vec<_>>();
                    self.language_position(call_expr, &keys, key, span)?
                } else {
                    None
                };
                match name {
                    "get" | "get_mut" => {
                        let Some(index) = position else {
                            return Ok(Some(Value::Option(None)));
                        };
                        let mut place = receiver_place.clone();
                        place.projections.push(Projection::MapIndex(index));
                        Ok(Some(Value::Option(Some(Box::new(Some(Value::Ref(place)))))))
                    }
                    "insert" => {
                        if arguments.len() < 2 {
                            return Err(RuntimeError::new(
                                "HashMap::insert expects key and value",
                                span,
                            ));
                        }
                        let key = arguments.remove(0);
                        let value = arguments.remove(0);
                        if let Some(index) = position {
                            let old = match self.place_value_mut(receiver_place, span)? {
                                Value::HashMap(map) => map.0[index].1.replace(value),
                                _ => unreachable!(),
                            };
                            // The stored key remains; ownership of the newly supplied equal key
                            // is consumed by the call and must be destroyed.
                            self.drop_value(key)?;
                            Ok(Some(Value::Option(old.map(|value| Box::new(Some(value))))))
                        } else {
                            match self.place_value_mut(receiver_place, span)? {
                                Value::HashMap(map) => map.0.push((key, Some(value))),
                                _ => unreachable!(),
                            }
                            Ok(Some(Value::Option(None)))
                        }
                    }
                    "remove" => {
                        let Some(index) = position else {
                            return Ok(Some(Value::Option(None)));
                        };
                        let (stored_key, value) =
                            match self.place_value_mut(receiver_place, span)? {
                                Value::HashMap(map) => map.0.remove(index),
                                _ => unreachable!(),
                            };
                        self.drop_value(stored_key)?;
                        Ok(Some(Value::Option(
                            value.map(|value| Box::new(Some(value))),
                        )))
                    }
                    "contains_key" => Ok(Some(Value::Bool(position.is_some()))),
                    "clear" => {
                        let removed = match self.place_value_mut(receiver_place, span)? {
                            Value::HashMap(map) => std::mem::take(&mut map.0),
                            _ => unreachable!(),
                        };
                        for (key, value) in removed.into_iter().rev() {
                            if let Some(value) = value {
                                self.drop_value(value)?;
                            }
                            self.drop_value(key)?;
                        }
                        Ok(Some(Value::Unit))
                    }
                    _ => Ok(None),
                }
            }
            Value::HashSet(set) => {
                let value = arguments.first().cloned();
                let position = if let Some(value) = value.as_ref() {
                    self.language_position(call_expr, &set.0, value, span)?
                } else {
                    None
                };
                match name {
                    "insert" => {
                        if arguments.is_empty() {
                            return Err(RuntimeError::new("HashSet::insert expects value", span));
                        }
                        let value = arguments.remove(0);
                        if position.is_some() {
                            self.drop_value(value)?;
                            Ok(Some(Value::Bool(false)))
                        } else {
                            match self.place_value_mut(receiver_place, span)? {
                                Value::HashSet(set) => set.0.push(value),
                                _ => unreachable!(),
                            }
                            Ok(Some(Value::Bool(true)))
                        }
                    }
                    "remove" => {
                        let Some(index) = position else {
                            return Ok(Some(Value::Bool(false)));
                        };
                        let stored = match self.place_value_mut(receiver_place, span)? {
                            Value::HashSet(set) => set.0.remove(index),
                            _ => unreachable!(),
                        };
                        self.drop_value(stored)?;
                        Ok(Some(Value::Bool(true)))
                    }
                    "contains" => Ok(Some(Value::Bool(position.is_some()))),
                    "clear" => {
                        let removed = match self.place_value_mut(receiver_place, span)? {
                            Value::HashSet(set) => std::mem::take(&mut set.0),
                            _ => unreachable!(),
                        };
                        for value in removed.into_iter().rev() {
                            self.drop_value(value)?;
                        }
                        Ok(Some(Value::Unit))
                    }
                    _ => Ok(None),
                }
            }
            _ => Ok(None),
        }
    }

    fn language_position(
        &mut self,
        // AS3 Boundary 4: the container call this comparison serves. `Vec::contains`,
        // `HashSet::insert`, `HashMap::get` — the checker publishes the element's `Eq::eq` against
        // exactly this expression, which is what let this path stop scanning.
        call_expr: Option<ExprId>,
        values: &[Value],
        needle: &Value,
        span: Span,
    ) -> Result<Option<usize>, RuntimeError> {
        for (index, value) in values.iter().enumerate() {
            if self.language_equal(call_expr, value.clone(), needle.clone(), span)? {
                return Ok(Some(index));
            }
        }
        Ok(None)
    }

    fn language_equal(
        &mut self,
        call_expr: Option<ExprId>,
        left: Value,
        right: Value,
        span: Span,
    ) -> Result<bool, RuntimeError> {
        let left = self.deref_value(left, span)?;
        let right = self.deref_value(right, span)?;
        if nominal_item(&left).is_some() {
            // **AS3 Boundary 4: the follow-up this site recorded is done.**
            //
            // It used to scan for a member named `eq`, because `language_equal` is reached from a
            // collection lookup with runtime values and no expression id — the case
            // `WP-CALLABLE-USE-TOTAL.md` §3.2 named when it rejected `ExprId` as the sole key.
            // The answer was not a new key: it was to thread the ORIGINATING call down and have
            // the checker publish the element's `Eq::eq` against it
            // (`publish_core_element_eq_use`). Inventing an id to look up would have been a
            // fabricated correspondence; threading the real one is not.
            if let Some(method) = call_expr.and_then(|expr| {
                self.selected_core_trait_callable(expr, hir::CoreTrait::Eq)
                    .and_then(|use_| match use_.selection {
                        crate::typecheck::CalleeSelection::Static { body, .. } => {
                            self.callable_for_body(body)
                        }
                        crate::typecheck::CalleeSelection::Bound { .. }
                        | crate::typecheck::CalleeSelection::FunctionValue => None,
                    })
                    .or_else(|| self.specialised_operator_callable(expr, CoreTrait::Eq, span))
            }) {
                let receiver_place = self.promote_to_temp_place(left.clone(), span)?;
                let argument_place = self.promote_to_temp_place(right, span)?;
                return match self.call_user_method(
                    method,
                    receiver_place,
                    call_expr
                        .map(|e| self.core_trait_env(e, CoreTrait::Eq))
                        .unwrap_or(InvocationEnv::Empty),
                    vec![Value::Ref(argument_place)],
                    span,
                )? {
                    Value::Bool(value) => Ok(value),
                    _ => Err(RuntimeError::new("Eq::eq must return Bool", span)),
                };
            }
        }
        // DEV-130: reached by `Vec::contains` and friends, and it had the same raw `==`.
        Ok(values_equal(&left, &right))
    }

    fn call_core_method(
        &mut self,
        expr_id: Option<ExprId>,
        base: ExprId,
        name: &str,
        args: &[ExprId],
        span: Span,
    ) -> Result<Value, RuntimeError> {
        // WP-C2.2 (DEV-033): resolve the receiver place exactly once, BEFORE evaluating any
        // argument — the normative evaluation order (03-Type-System.md "Evaluation Order",
        // CD-007/CD-010) is receiver-before-arguments, but this path previously evaluated all
        // arguments first and resolved the receiver lazily inside each method-name branch
        // (also re-resolving it — and re-running index subexpressions — once per use).
        let receiver_place = self.core_receiver_place(base, span)?;
        self.call_core_method_at(expr_id, receiver_place, base, name, args, span)
    }

    /// The Core method surface, entered with the receiver place ALREADY resolved.
    ///
    /// DEV-DISPLAY-DISPATCH split this out so the generic-receiver path in [`Self::call_method`]
    /// can reach the Core surface without resolving the receiver a second time. Resolving twice
    /// would re-run a non-place receiver expression (`make_thing().fmt()`), which is exactly the
    /// double evaluation WP-C2.2/DEV-033 removed.
    fn call_core_method_at(
        &mut self,
        expr_id: Option<ExprId>,
        receiver_place: Place,
        base: ExprId,
        name: &str,
        args: &[ExprId],
        span: Span,
    ) -> Result<Value, RuntimeError> {
        // Copied out before the receiver's storage is borrowed mutably below; the seams for the
        // view producers sit inside those borrows and cannot re-borrow `self`.
        #[cfg(test)]
        let armed = self.mutation;
        let receiver_ty = self
            .tables
            .expr_types
            .get(&base)
            .cloned()
            .unwrap_or(Ty::Error);
        // This function has exactly one caller (`call_method`), which checks
        // `pending_propagation` immediately after calling it and before doing anything else --
        // the "function-boundary adapter" exception to routing propagation through `Flow`
        // directly (this dispatcher's body is too large to convert every internal `Ok(value)`
        // return to `Ok(Flow::Value(value))` without a much larger, riskier rewrite for no
        // behavioral benefit, since propagation can only originate here from argument
        // evaluation, never from the dispatcher body itself). Re-arming
        // `pending_propagation` here and returning a dummy value mirrors `expect_value`'s own
        // existing convention exactly, just with a caller that is guaranteed to check it
        // immediately rather than one that might not.
        let mut arguments = match self.eval_call_arguments(args)? {
            Ok(values) => values,
            Err(propagated) => {
                self.pending_propagation = Some(propagated);
                return Ok(Value::Unit);
            }
        };
        if (name == "remove" || name == "contains_key" || name == "contains")
            && !arguments.is_empty()
        {
            arguments[0] = self.deref_value(arguments[0].clone(), span)?;
        }
        self.flatten_string_refs(&mut arguments, span)?;
        if let Some(result) = self.call_collection_ownership_method(
            expr_id,
            &receiver_place,
            name,
            &mut arguments,
            span,
        )? {
            return Ok(result);
        }
        let mut values = arguments.into_iter();
        let mutating = matches!(
            name,
            "push"
                | "push_str"
                | "pop"
                | "clear"
                | "insert"
                | "remove"
                | "append"
                | "get_mut"
                | "extend"
                | "next"
                | "count"
                | "collect"
                | "map"
                | "filter"
                | "fold"
                | "reduce"
                | "any"
                | "all"
                | "find"
                | "next_int"
                | "next_float"
                | "range"
                | "read_to_string"
                | "write"
                | "write_str"
        );
        // DEV-077 (WP-C4.7-6.1): `Box::into_inner` CONSUMES the box and transfers the contained
        // value to the caller. It must therefore take from the real place, exactly like `close`
        // below — the borrowing path further down operates on a CLONE of the receiver, so
        // `.take()` there emptied the clone while the original box kept the value and dropped it
        // again at end of scope. With a `Drop` payload that was an observable DOUBLE DROP
        // (violating EXEC-ONCE-001) and it disagreed with MIR, which drops exactly once.
        if name == "into_inner" {
            let value = self.take_place(&receiver_place, span)?;
            let Value::Boxed(mut inner) = value else {
                return Err(RuntimeError::new("Box::into_inner expects Box", span));
            };
            return Ok(inner.take().unwrap_or(Value::Unit));
        }
        if name == "close" {
            let value = self.take_place(&receiver_place, span)?;
            let Value::File(resource) = value else {
                return Err(RuntimeError::new("File::close expects File", span));
            };
            resource.0.borrow_mut().take();
            return Ok(Value::Result(Ok(Box::new(Some(Value::Unit)))));
        }
        // DEV-076 (WP-C4.7-8.1 prerequisite): `unwrap_or` CONSUMES the receiver and discards
        // exactly one of the two values — the payload or the default. It has to be intercepted
        // here for the same reason `into_inner` is: the borrowing path below operates on a CLONE,
        // so taking the payload there left the ORIGINAL Option holding it, and it was destroyed a
        // second time at end of scope. The default fared worse — being discarded on the `Some`
        // path, its destructor never ran at all. Both halves violate EXEC-ONCE-001 (every value's
        // destructor runs exactly once), and MIR refused the construct entirely, so the
        // divergence was invisible to the differential.
        //
        // Correct semantics: consume the receiver from the real place; yield the payload and drop
        // the (already-evaluated, since Core has no laziness) default on the `Some`/`Ok` path;
        // yield the default on the `None`/`Err` path, dropping the `Err` payload it displaces.
        if name == "unwrap_or"
            && matches!(
                self.place_value(&receiver_place, span)?,
                Value::Option(_) | Value::Result(_)
            )
        {
            let receiver = self.take_place(&receiver_place, span)?;
            let default = values.next().unwrap_or(Value::Unit);
            return match receiver {
                Value::Option(Some(payload)) => {
                    self.drop_value(default)?;
                    own_payload(payload, "`unwrap_or` on a `Some`", span)
                }
                Value::Option(None) => Ok(default),
                Value::Result(Ok(payload)) => {
                    self.drop_value(default)?;
                    own_payload(payload, "`unwrap_or` on an `Ok`", span)
                }
                Value::Result(Err(error)) => {
                    let error = own_payload(error, "`unwrap_or` on an `Err`", span)?;
                    self.drop_value(error)?;
                    Ok(default)
                }
                other => Err(RuntimeError::new(
                    format!("unwrap_or expects Option or Result, found {other}"),
                    span,
                )),
            };
        }
        // DEV-063: the fn-value-consuming `Option`/`Result` combinators
        // (06-Standard-Library.md §Option/§Result). Intercepted here — before the borrowing
        // receiver match below — because they consume `self` (take_place) and re-enter the
        // interpreter to call the user's function value, which must happen with no receiver
        // borrow outstanding. Gated on the receiver actually being an Option/Result so the
        // iterator `.map` (lazy MapIter) path below is unaffected.
        if matches!(name, "map" | "and_then" | "map_err")
            && matches!(
                self.place_value(&receiver_place, span)?,
                Value::Option(_) | Value::Result(_)
            )
        {
            let func = values.next().ok_or_else(|| {
                RuntimeError::new(format!("{name} expects a function argument"), span)
            })?;
            let Value::Function(callee) = func else {
                return Err(RuntimeError::new(
                    format!("{name} expects a function value"),
                    span,
                ));
            };
            let callable = self
                .item_callable(callee.item)
                .ok_or_else(|| RuntimeError::new("expression is not callable", span))?;
            let receiver = self.take_place(&receiver_place, span)?;
            return match (receiver, name) {
                (Value::Option(option), "map") => match option {
                    Some(value) => {
                        let mapped = self.invoke_callable(
                            ResolvedInvocation {
                                callable: callable.clone(),
                                environment: InvocationEnv::Captured(callee.clone()),
                            },
                            ReceiverSource::None,
                            vec![own_payload(
                                value,
                                "a consuming combinator's payload",
                                span,
                            )?],
                            span,
                        )?;
                        Ok(Value::Option(Some(Box::new(Some(mapped)))))
                    }
                    None => Ok(Value::Option(None)),
                },
                (Value::Option(option), "and_then") => match option {
                    Some(value) => self.invoke_callable(
                        ResolvedInvocation {
                            callable: callable.clone(),
                            environment: InvocationEnv::Captured(callee.clone()),
                        },
                        ReceiverSource::None,
                        vec![own_payload(
                            value,
                            "a consuming combinator's payload",
                            span,
                        )?],
                        span,
                    ),
                    None => Ok(Value::Option(None)),
                },
                (Value::Result(result), "map") => match result {
                    Ok(value) => {
                        let mapped = self.invoke_callable(
                            ResolvedInvocation {
                                callable: callable.clone(),
                                environment: InvocationEnv::Captured(callee.clone()),
                            },
                            ReceiverSource::None,
                            vec![own_payload(
                                value,
                                "a consuming combinator's payload",
                                span,
                            )?],
                            span,
                        )?;
                        Ok(Value::Result(Ok(Box::new(Some(mapped)))))
                    }
                    Err(error) => Ok(Value::Result(Err(error))),
                },
                (Value::Result(result), "map_err") => match result {
                    Ok(value) => Ok(Value::Result(Ok(value))),
                    Err(error) => {
                        let mapped = self.invoke_callable(
                            ResolvedInvocation {
                                callable: callable.clone(),
                                environment: InvocationEnv::Captured(callee.clone()),
                            },
                            ReceiverSource::None,
                            vec![own_payload(
                                error,
                                "a consuming combinator's payload",
                                span,
                            )?],
                            span,
                        )?;
                        Ok(Value::Result(Err(Box::new(Some(mapped)))))
                    }
                },
                (Value::Result(result), "and_then") => match result {
                    Ok(value) => self.invoke_callable(
                        ResolvedInvocation {
                            callable: callable.clone(),
                            environment: InvocationEnv::Captured(callee.clone()),
                        },
                        ReceiverSource::None,
                        vec![own_payload(
                            value,
                            "a consuming combinator's payload",
                            span,
                        )?],
                        span,
                    ),
                    Err(error) => Ok(Value::Result(Err(error))),
                },
                (_, _) => Err(RuntimeError::new(
                    format!("unsupported combinator '{name}' for this receiver"),
                    span,
                )),
            };
        }
        if name == "fmt" {
            let receiver = self.place_value(&receiver_place, span)?;
            // `Value::Float` carries its own `FloatWidth`, so `Display for Value` already picks
            // the right shortest-round-trip digits -- no external static-type lookup needed here.
            return Ok(Value::String(receiver.to_string()));
        }
        if name == "hash" {
            let receiver = self.place_value(&receiver_place, span)?;
            return Ok(Value::Int(
                standard_hash(receiver, &receiver_ty, self.entry_source())? as i128,
            ));
        }
        // WP-C4.7-6.2: `Ord::cmp` on a primitive receiver (06's `impl Ord for Int32` and
        // "similar for other types"). The checker admits this only for totally-ordered
        // primitives — floats are excluded per CD-015 — so by the time execution reaches here
        // the comparison is well defined. `canonical_cmp` is the SAME comparison the existing
        // `<`/`>` operator path and sorted-collection iteration already use (`Ord for Value`),
        // so `a.cmp(&b)` and `a < b` cannot disagree.
        if name == "cmp" {
            let receiver = self.place_value(&receiver_place, span)?.clone();
            let mut values = self
                .eval_call_arguments(args)?
                .map_err(|_| RuntimeError::new("cmp argument propagated an error", span))?;
            let other = values
                .pop()
                .ok_or_else(|| RuntimeError::new("cmp expects one argument", span))?;
            // The argument is `&Self`; compare the referents, not the references.
            let other = self.deref_value(other, span)?;
            let receiver = self.deref_value(receiver, span)?;
            return Ok(Value::Ordering(Ord::cmp(&receiver, &other)));
        }
        if matches!(name, "read_to_string" | "write" | "write_str") {
            let Value::File(resource) = self.place_value(&receiver_place, span)? else {
                return Err(RuntimeError::new("file method expects File", span));
            };
            let resource = resource.clone();
            let mut file = resource.0.borrow_mut();
            let Some(file) = file.as_mut() else {
                return Ok(Value::Result(Err(Box::new(Some(Value::IOError(
                    IOErrorKind::InvalidInput,
                ))))));
            };
            let io_result: Result<(), std::io::Error> = match name {
                "read_to_string" => {
                    let mut bytes = Vec::new();
                    match file.read_to_end(&mut bytes) {
                        Ok(_) => match String::from_utf8(bytes) {
                            Ok(text) => {
                                return Ok(Value::Result(Ok(Box::new(Some(Value::String(text))))))
                            }
                            Err(_) => Err(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                "file content is not valid UTF-8",
                            )),
                        },
                        Err(error) => Err(error),
                    }
                }
                "write_str" => {
                    let text = string_arg(values.next(), span)?;
                    match file.write(text.as_bytes()) {
                        Ok(count) => {
                            return Ok(Value::Result(Ok(Box::new(Some(Value::Int(count as i128))))))
                        }
                        Err(error) => Err(error),
                    }
                }
                "write" => {
                    let bytes = self.file_bytes_arg(values.next(), span)?;
                    match file.write(&bytes) {
                        Ok(count) => {
                            return Ok(Value::Result(Ok(Box::new(Some(Value::Int(count as i128))))))
                        }
                        Err(error) => Err(error),
                    }
                }
                _ => unreachable!(),
            };
            return Ok(Value::Result(Err(Box::new(Some(Value::IOError(
                IOErrorKind::from_io_error(&io_result.unwrap_err()),
            ))))));
        }
        // WP-C1.3 (2026-07-17): generic `.clone()` for every core-type value. `Value` already
        // derives Rust `Clone` (a deep/structural copy, which is exactly STARK's Clone semantics
        // for these built-in collection/string/option/result types -- none of them are
        // user-overridable per 03-Type-System.md "operator overloading for user-defined types...
        // is a future extension", so there is no alternate `clone()` body to dispatch to, unlike
        // struct/enum Clone impls which go through the ordinary call_method/find_method path).
        // The type-checker (`core_method_signature`) only accepts "clone" for the value-like
        // core types listed there; this mirrors that set. See COMPILER-STATE.md DEV-013.
        if name == "clone" {
            let receiver_place = receiver_place.clone();
            let receiver_val = self.place_value(&receiver_place, span)?;
            if matches!(
                receiver_val,
                Value::String(_)
                    | Value::Str(_)
                    | Value::Vec(_)
                    | Value::Boxed(_)
                    | Value::Option(_)
                    | Value::Result(_)
                    | Value::HashMap(_)
                    | Value::HashSet(_)
                    | Value::Range { .. }
                    | Value::IOError(_)
            ) {
                return Ok(receiver_val.clone());
            }
        }
        if name == "get" {
            let receiver_place = receiver_place.clone();
            let receiver_val = self.place_value(&receiver_place, span)?;
            match receiver_val {
                Value::HashMap(map) => {
                    let key_arg = values
                        .next()
                        .ok_or_else(|| RuntimeError::new("expected key argument", span))?;
                    let key = self.deref_value(key_arg, span)?;
                    let mut place = receiver_place;
                    let keys = map.keys().cloned().collect::<Vec<_>>();
                    let position = self.language_position(expr_id, &keys, &key, span)?;
                    if let Some(index) = position {
                        place.projections.push(Projection::MapIndex(index));
                    }
                    return Ok(Value::Option(
                        position.map(|_| Box::new(Some(Value::Ref(place)))),
                    ));
                }
                _ => {
                    let index = usize_arg(values.next(), span)?;
                    let mut place = receiver_place;
                    place.projections.push(Projection::Index(index));
                    return Ok(Value::Option(
                        self.place_value(&place, span)
                            .ok()
                            .map(|_| Box::new(Some(Value::Ref(place)))),
                    ));
                }
            }
        }
        if name == "get_mut" {
            let receiver_place = receiver_place.clone();
            let receiver_val = self.place_value(&receiver_place, span)?;
            match receiver_val {
                Value::HashMap(map) => {
                    let key_arg = values
                        .next()
                        .ok_or_else(|| RuntimeError::new("expected key argument", span))?;
                    let key = self.deref_value(key_arg, span)?;
                    let mut place = receiver_place;
                    let keys = map.keys().cloned().collect::<Vec<_>>();
                    let position = self.language_position(expr_id, &keys, &key, span)?;
                    if let Some(index) = position {
                        place.projections.push(Projection::MapIndex(index));
                    }
                    return Ok(Value::Option(
                        position.map(|_| Box::new(Some(Value::Ref(place)))),
                    ));
                }
                _ => {
                    let index = usize_arg(values.next(), span)?;
                    let mut place = receiver_place;
                    place.projections.push(Projection::Index(index));
                    return Ok(Value::Option(
                        self.place_value(&place, span)
                            .ok()
                            .map(|_| Box::new(Some(Value::Ref(place)))),
                    ));
                }
            }
        }
        if name == "iter" {
            let receiver_place = receiver_place.clone();
            let receiver_val = self.place_value(&receiver_place, span)?;
            match receiver_val {
                Value::HashMap(map) => {
                    let pairs = map
                        .iter()
                        .map(|(k, v)| {
                            Some(Value::Tuple(vec![
                                Some(k.clone()),
                                Some(v.clone().unwrap_or(Value::Unit)),
                            ]))
                        })
                        .collect();
                    return Ok(Value::HashMapIter(pairs, 0));
                }
                Value::HashSet(set) => {
                    let items = set.iter().cloned().map(Some).collect();
                    return Ok(Value::HashSetIter(items, 0));
                }
                _ => {
                    return Ok(Value::VecIter(receiver_place, 0));
                }
            }
        }
        if name == "keys" {
            let receiver_place = receiver_place.clone();
            let receiver_val = self.place_value(&receiver_place, span)?;
            if let Value::HashMap(map) = receiver_val {
                let keys = map.keys().cloned().map(Some).collect();
                return Ok(Value::HashMapKeysIter(keys, 0));
            }
        }
        if name == "values" {
            let receiver_place = receiver_place.clone();
            let receiver_val = self.place_value(&receiver_place, span)?;
            if let Value::HashMap(map) = receiver_val {
                let values = map.values().cloned().collect();
                return Ok(Value::HashMapValuesIter(values, 0));
            }
        }
        if matches!(name, "len" | "is_empty") {
            if let Value::Slice(_, start, end) = self.place_value(&receiver_place, span)?.clone() {
                return Ok(if name == "len" {
                    Value::Int((end - start) as i128)
                } else {
                    Value::Bool(start == end)
                });
            }
        }
        if name == "next" {
            let place = receiver_place.clone();
            let iter_val = self.place_value(&place, span)?.clone();
            let (next_val, updated_iter) = self.iterator_step(iter_val, Some(&place), span)?;
            let iter_mut = self.place_value_mut(&place, span)?;
            *iter_mut = updated_iter;
            return Ok(Value::Option(next_val.map(|value| Box::new(Some(value)))));
        }
        if name == "extend" {
            let iter_arg = values
                .next()
                .ok_or_else(|| RuntimeError::new("extend expects an iterator", span))?;
            if let Value::Ref(iter_place) = iter_arg {
                let place = receiver_place.clone();
                loop {
                    let iter_val = self.place_value(&iter_place, span)?.clone();
                    let (next_val, updated_iter) =
                        self.iterator_step(iter_val, Some(&iter_place), span)?;
                    let iter_mut = self.place_value_mut(&iter_place, span)?;
                    *iter_mut = updated_iter;

                    if let Some(val) = next_val {
                        let mut deref_val = self.deref_value(val, span)?;
                        if let Value::Tuple(ref mut pair) = deref_val {
                            if pair.len() == 2 {
                                let k =
                                    self.deref_value(pair[0].clone().unwrap_or(Value::Unit), span)?;
                                let v =
                                    self.deref_value(pair[1].clone().unwrap_or(Value::Unit), span)?;
                                pair[0] = Some(k);
                                pair[1] = Some(v);
                            }
                        }
                        match self.clone_place_value(&place, span)? {
                            Value::Vec(_) => {
                                let Value::Vec(items) = self.place_value_mut(&place, span)? else {
                                    unreachable!()
                                };
                                items.push(Some(deref_val));
                            }
                            Value::HashMap(_) => {
                                let Value::Tuple(mut pair) = deref_val else {
                                    continue;
                                };
                                if pair.len() == 2 {
                                    let key = pair[0].take().unwrap_or(Value::Unit);
                                    let value = pair[1].take().unwrap_or(Value::Unit);
                                    let mut arguments = vec![key, value];
                                    let returned = self
                                        .call_collection_ownership_method(
                                            expr_id,
                                            &place,
                                            "insert",
                                            &mut arguments,
                                            span,
                                        )?
                                        .expect("HashMap insert is handled");
                                    // `extend` does not expose a replaced value.
                                    self.drop_value(returned)?;
                                }
                            }
                            Value::HashSet(_) => {
                                let mut arguments = vec![deref_val];
                                self.call_collection_ownership_method(
                                    expr_id,
                                    &place,
                                    "insert",
                                    &mut arguments,
                                    span,
                                )?;
                            }
                            _ => {}
                        }
                    } else {
                        break;
                    }
                }
                return Ok(Value::Unit);
            } else {
                return Err(RuntimeError::new(
                    "extend expects reference to iterator",
                    span,
                ));
            }
        }
        if name == "count" {
            let place = receiver_place.clone();
            let mut iter_val = self.place_value(&place, span)?.clone();
            let mut cnt = 0u64;
            loop {
                let (next_val, updated_iter) = self.iterator_step(iter_val, Some(&place), span)?;
                iter_val = updated_iter;
                if next_val.is_some() {
                    cnt += 1;
                } else {
                    break;
                }
            }
            let iter_mut = self.place_value_mut(&place, span)?;
            *iter_mut = iter_val;
            return Ok(Value::Int(cnt as i128));
        }
        if name == "collect" {
            let place = receiver_place.clone();
            let mut iter_val = self.place_value(&place, span)?.clone();
            let mut items = Vec::new();
            loop {
                let (next_val, updated_iter) = self.iterator_step(iter_val, Some(&place), span)?;
                iter_val = updated_iter;
                if let Some(x) = next_val {
                    items.push(Some(x));
                } else {
                    break;
                }
            }
            let iter_mut = self.place_value_mut(&place, span)?;
            *iter_mut = iter_val;

            let mut is_hashset = false;
            let mut is_hashmap = false;
            if let Some(expr_id) = expr_id {
                if let Some(ty) = self.tables.expr_types.get(&expr_id) {
                    let mut current_ty = ty;
                    while let Ty::Ref { inner, .. } = current_ty {
                        current_ty = &**inner;
                    }
                    if let Ty::Core(crate::hir::CoreType::HashSet, _) = current_ty {
                        is_hashset = true;
                    } else if let Ty::Core(crate::hir::CoreType::HashMap, _) = current_ty {
                        is_hashmap = true;
                    }
                }
            }

            let is_all_pairs = !items.is_empty()
                && items.iter().all(|item| {
                    if let Some(Value::Tuple(p)) = item {
                        p.len() == 2
                    } else {
                        false
                    }
                });

            if is_hashset {
                let mut set = InsertionSet::new();
                for x in items.into_iter().flatten() {
                    let value = self.deref_value(x, span)?;
                    if self
                        .language_position(expr_id, &set.0, &value, span)?
                        .is_some()
                    {
                        self.drop_value(value)?;
                    } else {
                        set.0.push(value);
                    }
                }
                return Ok(Value::HashSet(set));
            } else if is_hashmap || is_all_pairs {
                let mut map = InsertionMap::new();
                for item in items {
                    if let Some(Value::Tuple(p)) = item {
                        let k = self.deref_value(p[0].clone().unwrap_or(Value::Unit), span)?;
                        let v = self.deref_value(p[1].clone().unwrap_or(Value::Unit), span)?;
                        let keys = map.keys().cloned().collect::<Vec<_>>();
                        if let Some(index) = self.language_position(expr_id, &keys, &k, span)? {
                            let old = map.0[index].1.replace(v);
                            self.drop_value(k)?;
                            if let Some(old) = old {
                                self.drop_value(old)?;
                            }
                        } else {
                            map.0.push((k, Some(v)));
                        }
                    }
                }
                return Ok(Value::HashMap(map));
            } else {
                let mut deref_items = Vec::new();
                for x in items.into_iter().flatten() {
                    deref_items.push(Some(self.deref_value(x, span)?));
                }
                return Ok(Value::Vec(deref_items));
            }
        }
        if name == "fold" {
            let place = receiver_place.clone();
            let mut iter_val = self.place_value(&place, span)?.clone();
            let mut init = values
                .next()
                .ok_or_else(|| RuntimeError::new("fold expects init value", span))?;
            let f = values
                .next()
                .ok_or_else(|| RuntimeError::new("fold expects function argument", span))?;
            loop {
                let (next_val, updated_iter) = self.iterator_step(iter_val, Some(&place), span)?;
                iter_val = updated_iter;
                if let Some(x) = next_val {
                    init = self.call_function_pointer(f.clone(), vec![init, x], span)?;
                } else {
                    break;
                }
            }
            let iter_mut = self.place_value_mut(&place, span)?;
            *iter_mut = iter_val;
            return Ok(init);
        }
        if name == "reduce" {
            let place = receiver_place.clone();
            let mut iter_val = self.place_value(&place, span)?.clone();
            let f = values
                .next()
                .ok_or_else(|| RuntimeError::new("reduce expects function argument", span))?;
            let (first, updated_iter) = self.iterator_step(iter_val, Some(&place), span)?;
            iter_val = updated_iter;
            if let Some(first_val) = first {
                let mut acc = first_val;
                loop {
                    let (next_val, updated_iter) =
                        self.iterator_step(iter_val, Some(&place), span)?;
                    iter_val = updated_iter;
                    if let Some(x) = next_val {
                        acc = self.call_function_pointer(f.clone(), vec![acc, x], span)?;
                    } else {
                        break;
                    }
                }
                let iter_mut = self.place_value_mut(&place, span)?;
                *iter_mut = iter_val;
                return Ok(Value::Option(Some(Box::new(Some(acc)))));
            } else {
                let iter_mut = self.place_value_mut(&place, span)?;
                *iter_mut = iter_val;
                return Ok(Value::Option(None));
            }
        }
        if name == "any" {
            let place = receiver_place.clone();
            let mut iter_val = self.place_value(&place, span)?.clone();
            let f = values
                .next()
                .ok_or_else(|| RuntimeError::new("any expects function argument", span))?;
            let mut found = false;
            loop {
                let (next_val, updated_iter) = self.iterator_step(iter_val, Some(&place), span)?;
                iter_val = updated_iter;
                if let Some(x) = next_val {
                    let res = self.call_function_pointer(f.clone(), vec![x], span)?;
                    if let Value::Bool(true) = res {
                        found = true;
                        break;
                    }
                } else {
                    break;
                }
            }
            let iter_mut = self.place_value_mut(&place, span)?;
            *iter_mut = iter_val;
            return Ok(Value::Bool(found));
        }
        if name == "all" {
            let place = receiver_place.clone();
            let mut iter_val = self.place_value(&place, span)?.clone();
            let f = values
                .next()
                .ok_or_else(|| RuntimeError::new("all expects function argument", span))?;
            let mut all_true = true;
            loop {
                let (next_val, updated_iter) = self.iterator_step(iter_val, Some(&place), span)?;
                iter_val = updated_iter;
                if let Some(x) = next_val {
                    let res = self.call_function_pointer(f.clone(), vec![x], span)?;
                    if let Value::Bool(false) = res {
                        all_true = false;
                        break;
                    }
                } else {
                    break;
                }
            }
            let iter_mut = self.place_value_mut(&place, span)?;
            *iter_mut = iter_val;
            return Ok(Value::Bool(all_true));
        }
        if name == "find" && is_iterator_ty(&receiver_ty) {
            let place = receiver_place.clone();
            let mut iter_val = self.place_value(&place, span)?.clone();
            let f = values
                .next()
                .ok_or_else(|| RuntimeError::new("find expects function argument", span))?;
            let mut found = None;
            loop {
                let (next_val, updated_iter) = self.iterator_step(iter_val, Some(&place), span)?;
                iter_val = updated_iter;
                if let Some(x) = next_val {
                    let x_ref = Value::Ref(self.promote_to_temp_place(x.clone(), span)?);
                    let res = self.call_function_pointer(f.clone(), vec![x_ref], span)?;
                    if let Value::Bool(true) = res {
                        found = Some(x);
                        break;
                    }
                } else {
                    break;
                }
            }
            let iter_mut = self.place_value_mut(&place, span)?;
            *iter_mut = iter_val;
            return Ok(Value::Option(found.map(|value| Box::new(Some(value)))));
        }
        if name == "map" {
            let place = receiver_place.clone();
            let f = values
                .next()
                .ok_or_else(|| RuntimeError::new("map expects function argument", span))?;
            let Value::Function(callee) = f else {
                return Err(RuntimeError::new("expected function pointer for map", span));
            };
            let iter_val = self.place_value(&place, span)?.clone();
            let iter_mut = self.place_value_mut(&place, span)?;
            *iter_mut = Value::MapIter(Box::new(iter_val), callee.item);
            return Ok(self.place_value(&place, span)?.clone());
        }
        if name == "filter" {
            let place = receiver_place.clone();
            let f = values
                .next()
                .ok_or_else(|| RuntimeError::new("filter expects function argument", span))?;
            let Value::Function(pred_callee) = f else {
                return Err(RuntimeError::new(
                    "expected function pointer for filter",
                    span,
                ));
            };
            let iter_val = self.place_value(&place, span)?.clone();
            let iter_mut = self.place_value_mut(&place, span)?;
            *iter_mut = Value::FilterIter(Box::new(iter_val), pred_callee.item);
            return Ok(self.place_value(&place, span)?.clone());
        }
        if name == "append" {
            let Some(Value::Ref(other_place)) = values.next() else {
                return Err(RuntimeError::new(
                    "Vec::append expects a mutable Vec reference",
                    span,
                ));
            };
            let mut other = match self.place_value_mut(&other_place, span)? {
                Value::Vec(other) => std::mem::take(other),
                _ => {
                    return Err(RuntimeError::new(
                        "Vec::append expects a mutable Vec reference",
                        span,
                    ));
                }
            };
            let Value::Vec(receiver) = self.place_value_mut(&receiver_place, span)? else {
                return Err(RuntimeError::new("Vec::append expects Vec receiver", span));
            };
            receiver.append(&mut other);
            return Ok(Value::Unit);
        }
        let mut owned;
        let target = if mutating {
            let place = receiver_place.clone();
            self.place_value_mut(&place, span)?
        } else {
            // WP-C2.2 (DEV-033): read through the already-resolved (and already ref-chain-
            // dereferenced) receiver place instead of re-resolving the receiver expression.
            owned = self.clone_place_value(&receiver_place, span)?;
            &mut owned
        };
        match target {
            Value::String(string) | Value::Str(string) => match name {
                "len" => Ok(Value::Int(string.len() as i128)),
                "is_empty" => Ok(Value::Bool(string.is_empty())),
                "push" => match values.next() {
                    Some(Value::Char(ch)) => {
                        string.push(ch);
                        Ok(Value::Unit)
                    }
                    _ => Err(RuntimeError::new("String::push expects Char", span)),
                },
                "push_str" => {
                    string.push_str(&string_arg(values.next(), span)?);
                    Ok(Value::Unit)
                }
                "pop" => Ok(Value::Option(
                    string.pop().map(|ch| Box::new(Some(Value::Char(ch)))),
                )),
                "clear" => {
                    string.clear();
                    Ok(Value::Unit)
                }
                // **DEV-126: `as_str` is a BORROW, so it must yield the receiver's place.**
                //
                // This cloned the string, producing a value with no link to what it was a view of.
                // Every consumer that needs the OWNER — `bytes()`, which anchors its materialised
                // byte storage to the owner's frame — then had nothing to anchor to, and
                // `expr_place`'s fallback promoted the detached copy into the running frame. So
                // `fn f(c: &C) -> &[UInt8] { c.input.as_str().bytes() }` dangled on return while
                // the direct `c.input.bytes()` worked: identical types, different provenance.
                //
                // `Value::Ref` is what a `&str` of a place actually is, and `deref_place` /
                // `deref_value` already normalise through it, so the receiver of a chained call
                // resolves back to the `String`'s own place.
                "as_str" => {
                    // Class 1 mutation: emit the owned `String` where `&str` is declared.
                    #[cfg(test)]
                    if armed == Some(ProducerMutation::OwnedForView) {
                        return Ok(Value::String(string.clone()));
                    }
                    Ok(Value::Ref(receiver_place.clone()))
                }
                "trim" => Ok(Value::Str(string.trim().to_string())),
                "contains" => Ok(Value::Bool(
                    string.contains(&string_arg(values.next(), span)?),
                )),
                "starts_with" => Ok(Value::Bool(
                    string.starts_with(&string_arg(values.next(), span)?),
                )),
                "ends_with" => Ok(Value::Bool(
                    string.ends_with(&string_arg(values.next(), span)?),
                )),
                "find" => Ok(Value::Option(
                    string
                        .find(&string_arg(values.next(), span)?)
                        .map(|index| Box::new(Some(Value::Int(index as i128)))),
                )),
                "replace" => {
                    let from = string_arg(values.next(), span)?;
                    let to = string_arg(values.next(), span)?;
                    Ok(Value::String(string.replace(&from, &to)))
                }
                "substring" => {
                    let start = usize_arg(values.next(), span)?;
                    let end = usize_arg(values.next(), span)?;
                    if start > end
                        || end > string.len()
                        || !string.is_char_boundary(start)
                        || !string.is_char_boundary(end)
                    {
                        return Err(RuntimeError::new(
                            "String::substring range is not on valid UTF-8 boundaries",
                            span,
                        ));
                    }
                    Ok(Value::Str(string[start..end].to_string()))
                }
                "chars" => Ok(Value::CharsIter(string.clone(), 0)),
                // **`bytes` and `into_bytes` have DIFFERENT types and must have different runtime
                // representations.** They shared this arm, and both produced `Value::Vec`.
                //
                //   bytes()       -> `&[UInt8]`      a SHARED SLICE, and shared references are Copy
                //   into_bytes()  -> `Vec<UInt8>`    an OWNED vector, correctly not Copy
                //
                // `Value::Vec` is classified non-`Copy` — rightly, it owns its elements. So the
                // view returned by `bytes()` was CONSUMED when passed to a function, and any later
                // use of the caller's binding read an emptied slot: "use of unavailable value", at
                // run time, on a program the checker and MIR both accept.
                //
                // This is DEV-087's defect in a second producer. That fix classified
                // `Value::Slice` as `Copy` and its comment describes this exact symptom —
                // `total(shared); shared[0]` failing in the oracle alone. The classification was
                // right; `bytes()` simply never produced the classified thing. Two representations
                // claimed to be `&[UInt8]` and only one obeyed the ownership contract.
                //
                // Found in `stark-mime`, whose `slice_to_string(bytes, ..)` passes the view;
                // `stark-percent` only ever indexed it, which is why one package worked and the
                // other did not for what looked like an unrelated reason.
                "bytes" => {
                    let elements: Vec<Option<Value>> = string
                        .bytes()
                        .map(|b| Some(Value::Int(b as i128)))
                        .collect();
                    // The receiver borrow ends here, before the frame is touched.
                    let len = elements.len();
                    // Into the RECEIVER's frame, not the current one: the bytes are a view of that
                    // string and must live as long as it does. Promoting into the running frame
                    // dangles the moment the view is returned from the function that made it.
                    let owner_frame = receiver_place.frame.min(self.frames.len() - 1);
                    let place = self.promote_to_temp_place_in(owner_frame, Value::Vec(elements))?;
                    Ok(Value::Slice(place, 0, len))
                }
                "into_bytes" => {
                    let bytes_val = string
                        .bytes()
                        .map(|b| Some(Value::Int(b as i128)))
                        .collect();
                    Ok(Value::Vec(bytes_val))
                }
                "split" => {
                    let delimiter = string_arg(values.next(), span)?;
                    let parts = if string.is_empty() {
                        Vec::new()
                    } else if delimiter.is_empty() {
                        string.chars().map(|scalar| scalar.to_string()).collect()
                    } else {
                        string.split(&delimiter).map(str::to_string).collect()
                    };
                    Ok(Value::SplitIter(parts, 0))
                }
                "to_string" => Ok(Value::String(string.clone())),
                "to_lowercase" => Ok(Value::String(string.to_lowercase())),
                "to_uppercase" => Ok(Value::String(string.to_uppercase())),
                _ => Err(RuntimeError::new(
                    format!("unsupported String method '{name}'"),
                    span,
                )),
            },
            Value::Vec(vector) => match name {
                "len" => Ok(Value::Int(vector.len() as i128)),
                "capacity" => Ok(Value::Int(vector.capacity() as i128)),
                "is_empty" => Ok(Value::Bool(vector.is_empty())),
                "push" => {
                    vector.push(values.next());
                    Ok(Value::Unit)
                }
                "pop" => Ok(Value::Option(
                    vector.pop().flatten().map(|value| Box::new(Some(value))),
                )),
                "insert" => {
                    let index = usize_arg(values.next(), span)?;
                    if index > vector.len() {
                        return Err(RuntimeError::with_category(
                            "Vec insertion index out of bounds",
                            span,
                            crate::mir::TrapCategory::IndexOutOfBounds,
                        ));
                    }
                    vector.insert(index, values.next());
                    Ok(Value::Unit)
                }
                "remove" => {
                    let index = usize_arg(values.next(), span)?;
                    if index >= vector.len() {
                        return Err(RuntimeError::with_category(
                            "Vec removal index out of bounds",
                            span,
                            crate::mir::TrapCategory::IndexOutOfBounds,
                        ));
                    }
                    Ok(vector.remove(index).unwrap_or(Value::Unit))
                }
                "clear" => {
                    vector.clear();
                    Ok(Value::Unit)
                }
                "as_slice" => {
                    // Class 1 mutation: emit the owned `Vec` where `&[T]` is declared.
                    #[cfg(test)]
                    if armed == Some(ProducerMutation::OwnedForView) {
                        return Ok(Value::Vec(vector.clone()));
                    }
                    Ok(Value::Slice(receiver_place.clone(), 0, vector.len()))
                }
                _ => Err(RuntimeError::new(
                    format!("unsupported Vec method '{name}'"),
                    span,
                )),
            },
            Value::Array(array) => match name {
                "len" => Ok(Value::Int(array.len() as i128)),
                "is_empty" => Ok(Value::Bool(array.is_empty())),
                _ => Err(RuntimeError::new(
                    format!("unsupported Array method '{name}'"),
                    span,
                )),
            },
            Value::Option(option) => match name {
                "is_some" => Ok(Value::Bool(option.is_some())),
                "is_none" => Ok(Value::Bool(option.is_none())),
                "unwrap" => option.take().and_then(|value| *value).ok_or_else(|| {
                    // R-01/CD-141 shape: STATE the category rather than leaving it to prose
                    // matching. `oracle_category` refused these outright with a stale "Option/Result
                    // are WP-C5.3c" message, so a corpus case for OPT-UNWRAP could not be compared
                    // at all until the raise site said what it was raising.
                    RuntimeError::with_category(
                        "called unwrap on None",
                        span,
                        crate::mir::TrapCategory::UnwrapNone,
                    )
                }),
                "unwrap_or" => Ok(option
                    .take()
                    .and_then(|value| *value)
                    .unwrap_or_else(|| values.next().unwrap_or(Value::Unit))),
                _ => Err(RuntimeError::new(
                    format!("unsupported Option method '{name}'"),
                    span,
                )),
            },
            Value::Result(result) => match name {
                "is_ok" => Ok(Value::Bool(result.is_ok())),
                "is_err" => Ok(Value::Bool(result.is_err())),
                "unwrap" => match std::mem::replace(result, Ok(Box::new(Some(Value::Unit)))) {
                    Ok(value) => own_payload(value, "`unwrap` on an `Ok`", span),
                    Err(error) => Err(RuntimeError::with_category(
                        format!("called unwrap on Err({})", display_slot(&error)),
                        span,
                        crate::mir::TrapCategory::UnwrapErr,
                    )),
                },
                "unwrap_or" => match std::mem::replace(result, Ok(Box::new(Some(Value::Unit)))) {
                    Ok(value) => own_payload(value, "`unwrap_or` on an `Ok`", span),
                    Err(_) => Ok(values.next().unwrap_or(Value::Unit)),
                },
                _ => Err(RuntimeError::new(
                    format!("unsupported Result method '{name}'"),
                    span,
                )),
            },
            Value::HashMap(map) => match name {
                "insert" => {
                    let k = values
                        .next()
                        .ok_or_else(|| RuntimeError::new("HashMap::insert expects key", span))?;
                    let v = values
                        .next()
                        .ok_or_else(|| RuntimeError::new("HashMap::insert expects value", span))?;
                    Ok(Value::Option(
                        map.insert(k, Some(v))
                            .flatten()
                            .map(|value| Box::new(Some(value))),
                    ))
                }
                "remove" => {
                    let k = values.next().ok_or_else(|| {
                        RuntimeError::new("HashMap::remove expects key ref", span)
                    })?;
                    Ok(Value::Option(
                        map.remove(&k).flatten().map(|value| Box::new(Some(value))),
                    ))
                }
                "contains_key" => {
                    let k = values.next().ok_or_else(|| {
                        RuntimeError::new("HashMap::contains_key expects key ref", span)
                    })?;
                    Ok(Value::Bool(map.contains_key(&k)))
                }
                "len" => Ok(Value::Int(map.len() as i128)),
                "is_empty" => Ok(Value::Bool(map.is_empty())),
                "clear" => {
                    map.clear();
                    Ok(Value::Unit)
                }
                _ => Err(RuntimeError::new(
                    format!("unsupported HashMap method '{name}'"),
                    span,
                )),
            },
            Value::HashSet(set) => match name {
                "insert" => {
                    let val = values
                        .next()
                        .ok_or_else(|| RuntimeError::new("HashSet::insert expects value", span))?;
                    Ok(Value::Bool(set.insert(val)))
                }
                "remove" => {
                    let val = values.next().ok_or_else(|| {
                        RuntimeError::new("HashSet::remove expects value ref", span)
                    })?;
                    Ok(Value::Bool(set.remove(&val)))
                }
                "contains" => {
                    let val = values.next().ok_or_else(|| {
                        RuntimeError::new("HashSet::contains expects value ref", span)
                    })?;
                    Ok(Value::Bool(set.contains(&val)))
                }
                "len" => Ok(Value::Int(set.len() as i128)),
                "is_empty" => Ok(Value::Bool(set.is_empty())),
                "clear" => {
                    set.clear();
                    Ok(Value::Unit)
                }
                _ => Err(RuntimeError::new(
                    format!("unsupported HashSet method '{name}'"),
                    span,
                )),
            },
            // Phase 4E: `Random` (simple LCG; MMIX/Knuth multiplier and
            // increment — any full-period 64-bit LCG constants satisfy the
            // spec's "simple linear congruential generator").
            Value::Random(seed) => match name {
                "next_int" => {
                    *seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    Ok(Value::Int(*seed as i128))
                }
                "next_float" => {
                    *seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    Ok(Value::Float(
                        *seed as f64 / (u64::MAX as f64 + 1.0),
                        FloatWidth::F64,
                    ))
                }
                "range" => {
                    let min = match values.next() {
                        Some(Value::Int(v)) => v,
                        _ => return Err(RuntimeError::new("range expects Int32 min", span)),
                    };
                    let max = match values.next() {
                        Some(Value::Int(v)) => v,
                        _ => return Err(RuntimeError::new("range expects Int32 max", span)),
                    };
                    if max <= min {
                        return Err(RuntimeError::new("Random::range requires max > min", span));
                    }
                    *seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let span_size = (max - min) as u64;
                    let offset = (*seed % span_size) as i128;
                    Ok(Value::Int(min + offset))
                }
                _ => Err(RuntimeError::new(
                    format!("unsupported Random method '{name}'"),
                    span,
                )),
            },
            _ => Err(RuntimeError::new(
                format!("method '{name}' is unavailable for this value"),
                span,
            )),
        }
    }

    fn file_bytes_arg(&self, value: Option<Value>, span: Span) -> Result<Vec<u8>, RuntimeError> {
        fn slots_to_bytes(slots: &[Option<Value>], span: Span) -> Result<Vec<u8>, RuntimeError> {
            slots
                .iter()
                .map(|slot| match slot {
                    Some(Value::Int(value)) => u8::try_from(*value)
                        .map_err(|_| RuntimeError::new("file byte is outside UInt8", span)),
                    _ => Err(RuntimeError::new("File::write expects &[UInt8]", span)),
                })
                .collect()
        }

        match value {
            Some(Value::Array(values)) | Some(Value::Vec(values)) => slots_to_bytes(&values, span),
            Some(Value::Slice(place, start, end)) => {
                let source = self.place_value(&place, span)?;
                match source {
                    Value::Array(values) | Value::Vec(values) => {
                        let range = values.get(start..end).ok_or_else(|| {
                            RuntimeError::with_category(
                                "slice out of bounds",
                                span,
                                crate::mir::TrapCategory::IndexOutOfBounds,
                            )
                        })?;
                        slots_to_bytes(range, span)
                    }
                    _ => Err(RuntimeError::new("File::write expects &[UInt8]", span)),
                }
            }
            Some(Value::Ref(place)) => match self.place_value(&place, span)? {
                Value::Array(values) | Value::Vec(values) => slots_to_bytes(values, span),
                _ => Err(RuntimeError::new("File::write expects &[UInt8]", span)),
            },
            _ => Err(RuntimeError::new("File::write expects &[UInt8]", span)),
        }
    }

    /// Evaluates aggregate element expressions (tuple/array elements) left to right, stopping
    /// immediately -- without evaluating any later element at all -- when one element's
    /// evaluation triggers early transfer via `?` (`Flow::Propagate`). `Ok(Ok(values))` is
    /// ordinary completion; `Ok(Err(propagated))` means early transfer occurred, already
    /// -completed elements were destroyed in reverse completion order (matching ordinary
    /// failed-aggregate-construction cleanup per the abstract machine), and `propagated` is the
    /// value the caller should wrap in `Flow::Propagate`. `expect_value`'s existing
    /// `pending_propagation` channel is reused rather than introducing a new control-flow
    /// representation; a genuine Rust-level trap (`Err(RuntimeError)`) still unwinds immediately
    /// via `?` with no cleanup, unchanged from before this fix -- traps abort without running
    /// pending destructors.
    /// Evaluates call-argument expressions left to right, stopping immediately -- without
    /// evaluating any later argument at all -- when one argument's evaluation triggers early
    /// transfer via `?`. Shares `eval_aggregate_elements`'s stop-and-clean-up-in-reverse
    /// contract (see its doc comment) but returns plain owned values rather than
    /// aggregate-storage `Option<Value>` slots, matching how call arguments are consumed
    /// (moved into the callee's parameter locals, not tracked as move-able aggregate fields).
    /// Used by every call-argument-evaluating site (ordinary/associated/builtin function calls,
    /// user-method and qualified-trait-method calls, core/builtin-type method calls) so `?`
    /// inside any argument position stops evaluation of every later argument and the call
    /// itself, instead of `expect_value`'s dummy-`Value::Unit`-on-propagation swallowing it.
    fn eval_call_arguments(
        &mut self,
        args: &[ExprId],
    ) -> Result<Result<Vec<Value>, Value>, RuntimeError> {
        let mut completed: Vec<Value> = Vec::with_capacity(args.len());
        for arg in args {
            let evaluated = self.expect_value(*arg)?;
            if let Some(propagated) = self.pending_propagation.take() {
                for value in completed.into_iter().rev() {
                    self.drop_value(value)?;
                }
                return Ok(Err(propagated));
            }
            completed.push(evaluated);
        }
        Ok(Ok(completed))
    }

    fn eval_aggregate_elements(
        &mut self,
        values: &[ExprId],
    ) -> Result<Result<Vec<Option<Value>>, Value>, RuntimeError> {
        let mut completed: Vec<Value> = Vec::with_capacity(values.len());
        for value in values {
            let evaluated = self.expect_value(*value)?;
            if let Some(propagated) = self.pending_propagation.take() {
                for value in completed.into_iter().rev() {
                    self.drop_value(value)?;
                }
                return Ok(Err(propagated));
            }
            completed.push(evaluated);
        }
        Ok(Ok(completed.into_iter().map(Some).collect()))
    }

    /// **The `AggregateField` boundary lives here.** AS3 Packet 5.
    ///
    /// The expected type is the field's DECLARED type, instantiated for this literal —
    /// `aggregate_field_types[lit]`, published by the checker at the same point it unified the
    /// initialisers against it. Deliberately not `expr_types[init]`: that is the type of the
    /// expression that produced the value, so comparing the value against it would assert nothing,
    /// and it does not exist at all for a shorthand field (`W { v }`), which has no initialiser
    /// expression. The field NAME is the key, so shorthand is covered by the same lookup.
    fn eval_struct_lit(
        &mut self,
        lit: ExprId,
        res: Res,
        fields: &[hir::FieldInit],
        span: Span,
    ) -> Result<Flow, RuntimeError> {
        let declared = self.tables.aggregate_field_types.get(&lit).ok_or_else(|| {
            RuntimeError::internal(
                "no published field types for an aggregate literal — the checker publishes them \
                 wherever it checks the initialisers, so this is a publication defect",
                span,
            )
        })?;
        let declared: BTreeMap<String, Ty> = declared
            .iter()
            .map(|(name, ty)| (name.clone(), ty.clone()))
            .collect();
        let mut values = BTreeMap::new();
        let mut completed_order: Vec<String> = Vec::new();
        for field in fields {
            let name = self.text(field.name).to_string();
            let value = if let Some(expr) = field.expr {
                let value = self.expect_value(expr)?;
                // See `eval_aggregate_elements` for the same stop-and-clean-up-in-reverse
                // pattern applied here to named struct/enum-struct-variant fields.
                if let Some(propagated) = self.pending_propagation.take() {
                    for name in completed_order.into_iter().rev() {
                        if let Some(Some(value)) = values.remove(&name) {
                            self.drop_value(value)?;
                        }
                    }
                    return Ok(Flow::Propagate(propagated));
                }
                value
            } else {
                let local = self.find_local_by_name(&name).ok_or_else(|| {
                    RuntimeError::new(format!("unknown shorthand field '{name}'"), field.name)
                })?;
                self.take_place(
                    &Place {
                        frame: self.frames.len() - 1,
                        local,
                        projections: Vec::new(),
                    },
                    field.name,
                )?
            };
            // Class 4 mutation: injected AFTER the producer-side boundary has accepted the value,
            // so `AggregateField` is what must catch it rather than `ExpressionResult`.
            #[cfg(test)]
            let value = if self.mutation_armed(ProducerMutation::WrongAggregateField) {
                Value::Unit
            } else {
                value
            };
            // An initialiser for a field the nominal does not declare is a checker error the
            // program cannot reach execution with, so an absent entry here is a table/tree
            // disagreement rather than a field to skip.
            let field_ty = declared.get(&name).ok_or_else(|| {
                RuntimeError::internal(
                    format!("aggregate literal initialises undeclared field '{name}'"),
                    field.name,
                )
            })?;
            self.check_value_for_ty(
                &field_ty.clone(),
                &value,
                field.name,
                RepBoundary::AggregateField,
            )?;
            completed_order.push(name.clone());
            values.insert(name, Some(value));
        }
        match res {
            Res::Item(item) => Ok(Flow::Value(Value::Struct {
                item,
                fields: values,
            })),
            Res::Variant(item, variant) => Ok(Flow::Value(Value::Enum {
                item,
                variant,
                fields: Vec::new(),
                named: values,
            })),
            _ => Err(RuntimeError::new("invalid aggregate constructor", span)),
        }
    }

    /// PAT-BIND-001's predicate, in the interpreter: is this scrutinee a place read *through a
    /// reference*?
    ///
    /// Deliberately the same shape as the type checker's `scrutinee_reads_through_ref`, over the
    /// same HIR and the same expression types. The two must decide identically — the checker gives
    /// the binding its `&C` type and this gives it its `&C` value, and a disagreement between them
    /// is exactly the class of defect this packet closes.
    fn scrutinee_reads_through_ref(&self, expr: ExprId) -> bool {
        match &self.hir.expr(expr).kind {
            hir::ExprKind::Unary {
                op: UnOp::Deref, ..
            } => true,
            hir::ExprKind::Field { base, .. } | hir::ExprKind::TupleField { base, .. } => {
                matches!(self.tables.expr_types.get(base), Some(Ty::Ref { .. }))
                    || self.scrutinee_reads_through_ref(*base)
            }
            _ => false,
        }
    }

    /// Match `pat` against a source whose binding mode the caller already decided.
    fn match_source(
        &mut self,
        pat: PatId,
        source: &PatternSource,
        bindings: &mut Vec<(LocalId, Value)>,
    ) -> Result<bool, RuntimeError> {
        match source {
            PatternSource::Owned(value) => {
                let value = value.clone();
                self.match_pattern(pat, &value, bindings)
            }
            PatternSource::Borrowed(place) => {
                let place = place.clone();
                self.match_pattern_borrowed(pat, &place, bindings)
            }
        }
    }

    /// Pattern matching against a PLACE, for a scrutinee read through a reference.
    ///
    /// Structurally parallel to [`Self::match_pattern`], with one difference that is the whole
    /// point: where the owned matcher clones a component to bind it, this one projects the
    /// component's place and binds `Value::Ref` to it when the component is not `Copy`. Tests
    /// (literals, discriminants, arities) read through the place and are unaffected — a test never
    /// moves anything, so reading it by value was never the defect.
    ///
    /// **Uniform over every component (DEV-209).** Struct field, tuple element, array element,
    /// user-enum payload — and, since the prelude payload became slot-backed,
    /// `Option`/`Result` too. That last one was a stated narrowing for as long as `Box<Value>` had
    /// no `Projection` to name it, and the narrowing was the defect: PAT-BIND-001 is uniform over
    /// variant payloads, struct fields and tuple elements, so a specialised runtime representation
    /// that could not participate was the thing that had to change.
    fn match_pattern_borrowed(
        &mut self,
        pat: PatId,
        place: &Place,
        bindings: &mut Vec<(LocalId, Value)>,
    ) -> Result<bool, RuntimeError> {
        let pattern = self.hir.pat(pat);
        let span = pattern.span;
        match &pattern.kind {
            hir::PatKind::Wild => Ok(true),
            hir::PatKind::Binding { local, .. } => {
                let local = *local;
                let value = self.place_value(place, span)?.clone();
                if self.value_is_copy(&value) {
                    bindings.push((local, value));
                } else {
                    bindings.push((local, Value::Ref(place.clone())));
                }
                Ok(true)
            }
            hir::PatKind::Tuple(pats) => {
                let pats = pats.clone();
                let arity = match self.place_value(place, span)? {
                    Value::Tuple(values) => values.len(),
                    _ => return Ok(false),
                };
                self.match_elements_borrowed(&pats, place, arity, bindings)
            }
            hir::PatKind::Array(pats) => {
                let pats = pats.clone();
                let arity = match self.place_value(place, span)? {
                    Value::Array(values) | Value::Vec(values) => values.len(),
                    _ => return Ok(false),
                };
                self.match_elements_borrowed(&pats, place, arity, bindings)
            }
            hir::PatKind::TupleVariant { res, pats, .. } => {
                let res = *res;
                let pats = pats.clone();
                let value = self.place_value(place, span)?.clone();
                match (res, &value) {
                    (
                        Res::Variant(item, variant),
                        Value::Enum {
                            item: actual,
                            variant: actual_variant,
                            fields,
                            ..
                        },
                    ) if item == *actual && variant == *actual_variant => {
                        if pats.len() != fields.len() {
                            return Ok(false);
                        }
                        for (index, sub) in pats.iter().enumerate() {
                            let mut child = place.clone();
                            child.projections.push(Projection::Index(index));
                            if !self.match_pattern_borrowed(*sub, &child, bindings)? {
                                return Ok(false);
                            }
                        }
                        Ok(true)
                    }
                    // **DEV-209: the prelude payload borrows in place, like every other
                    // component.** It used to fall back to the owned rule here — moving out of a
                    // borrow — because `Box<Value>` had no `Projection` to name it. It is
                    // slot-backed now, so PAT-BIND-001 has the storage it always required.
                    (Res::Builtin(Builtin::Some), Value::Option(Some(_)))
                    | (Res::Builtin(Builtin::Ok), Value::Result(Ok(_))) => {
                        let Some(sub) = pats.first() else {
                            return Ok(true);
                        };
                        let mut child = place.clone();
                        child.projections.push(Projection::VariantPayload(0));
                        self.match_pattern_borrowed(*sub, &child, bindings)
                    }
                    (Res::Builtin(Builtin::Err), Value::Result(Err(_))) => {
                        let Some(sub) = pats.first() else {
                            return Ok(true);
                        };
                        let mut child = place.clone();
                        child.projections.push(Projection::VariantPayload(1));
                        self.match_pattern_borrowed(*sub, &child, bindings)
                    }
                    // A discriminant that does not match, and every non-prelude shape, still reads
                    // through the owned matcher: a test moves nothing.
                    _ => self.match_pattern(pat, &value, bindings),
                }
            }
            hir::PatKind::Struct { res, fields, .. } => {
                let res = *res;
                // `FieldPat` is not `Clone`, and the HIR borrow ends at the first `&mut self` call
                // below — so the three `Copy` fields each entry contributes are taken here.
                let fields: Vec<(Span, Option<PatId>, Option<LocalId>)> =
                    fields.iter().map(|f| (f.name, f.pat, f.local)).collect();
                let value = self.place_value(place, span)?.clone();
                let named_ok = match (res, &value) {
                    (Res::Item(item), Value::Struct { item: actual, .. }) => item == *actual,
                    (
                        Res::Variant(item, variant),
                        Value::Enum {
                            item: actual,
                            variant: actual_variant,
                            ..
                        },
                    ) => item == *actual && variant == *actual_variant,
                    _ => false,
                };
                if !named_ok {
                    return Ok(false);
                }
                for (field_name, field_pat, field_local) in fields {
                    let name = self.text(field_name).to_string();
                    let mut child = place.clone();
                    child.projections.push(Projection::Field(name));
                    // A field the value does not have, or one already moved out, is not a match.
                    if self.place_value(&child, span).is_err() {
                        return Ok(false);
                    }
                    if let Some(sub) = field_pat {
                        if !self.match_pattern_borrowed(sub, &child, bindings)? {
                            return Ok(false);
                        }
                    } else if let Some(local) = field_local {
                        let component = self.place_value(&child, span)?.clone();
                        if self.value_is_copy(&component) {
                            bindings.push((local, component));
                        } else {
                            bindings.push((local, Value::Ref(child)));
                        }
                    }
                }
                Ok(true)
            }
            // Tests read the component's value; nothing is bound, so the owned matcher's logic is
            // exactly right and is reused rather than restated.
            hir::PatKind::Lit(_) | hir::PatKind::Path { .. } => {
                let value = self.place_value(place, span)?.clone();
                self.match_pattern(pat, &value, bindings)
            }
            hir::PatKind::Error => Err(RuntimeError::new("invalid pattern", span)),
        }
    }

    /// Element patterns applied to the projected element places of a borrowed sequence — the
    /// tuple and array arms of [`Self::match_pattern_borrowed`].
    fn match_elements_borrowed(
        &mut self,
        patterns: &[PatId],
        place: &Place,
        arity: usize,
        bindings: &mut Vec<(LocalId, Value)>,
    ) -> Result<bool, RuntimeError> {
        if patterns.len() != arity {
            return Ok(false);
        }
        for (index, sub) in patterns.iter().enumerate() {
            let mut child = place.clone();
            child.projections.push(Projection::Index(index));
            if !self.match_pattern_borrowed(*sub, &child, bindings)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn match_pattern(
        &mut self,
        pat: PatId,
        value: &Value,
        bindings: &mut Vec<(LocalId, Value)>,
    ) -> Result<bool, RuntimeError> {
        let pattern = self.hir.pat(pat);
        match &pattern.kind {
            hir::PatKind::Wild => Ok(true),
            hir::PatKind::Binding { local, .. } => {
                bindings.push((*local, value.clone()));
                Ok(true)
            }
            // **DEV-129: a literal pattern compares CONTENT, so it reads through a reference.**
            //
            // `match s.as_str() { "beta" => ... }` is normative — a `&str` scrutinee against string
            // literals compares by content, never structurally. Since DEV-126 made `as_str` yield
            // the receiver's place, the scrutinee arrives as `Value::Ref` and every arm missed,
            // silently falling through to `_`: the oracle printed 0 where MIR printed 2. Silent is
            // the operative word — no arm is "wrong" to fail, so nothing could report this except
            // the differential that caught it.
            hir::PatKind::Lit(lit) => {
                let scrutinee = self.deref_value(value.clone(), pattern.span)?;
                let expected = self.eval_lit(*lit, pattern.span)?;
                Ok(values_equal(&expected, &scrutinee))
            }
            hir::PatKind::Path { res, .. } => match (res, value) {
                (Res::Item(item), actual) => match &self.hir.item(*item).kind {
                    // A const pattern is a literal pattern with a name, so it reads through a
                    // reference for the same reason. Only this sub-case: the variant arms below
                    // must NOT deref, because a reference-typed enum scrutinee is a type error
                    // (PAT-BIND-001/CD-303) and quietly accepting one here would re-open it.
                    hir::ItemKind::Const {
                        value: initializer, ..
                    } => {
                        let actual = self.deref_value(actual.clone(), pattern.span)?;
                        let expected = self.expect_value(*initializer)?;
                        Ok(values_equal(&expected, &actual))
                    }
                    _ => Ok(false),
                },
                (
                    Res::Variant(item, variant),
                    Value::Enum {
                        item: actual,
                        variant: actual_variant,
                        ..
                    },
                ) => Ok(item == actual && variant == actual_variant),
                (Res::Builtin(Builtin::None), Value::Option(None)) => Ok(true),
                (
                    Res::Builtin(Builtin::OrderingLess),
                    Value::Ordering(std::cmp::Ordering::Less),
                )
                | (
                    Res::Builtin(Builtin::OrderingEqual),
                    Value::Ordering(std::cmp::Ordering::Equal),
                )
                | (
                    Res::Builtin(Builtin::OrderingGreater),
                    Value::Ordering(std::cmp::Ordering::Greater),
                ) => Ok(true),
                (Res::Builtin(Builtin::IOErrorNotFound), Value::IOError(IOErrorKind::NotFound)) => {
                    Ok(true)
                }
                (
                    Res::Builtin(Builtin::IOErrorPermissionDenied),
                    Value::IOError(IOErrorKind::PermissionDenied),
                ) => Ok(true),
                (
                    Res::Builtin(Builtin::IOErrorAlreadyExists),
                    Value::IOError(IOErrorKind::AlreadyExists),
                ) => Ok(true),
                (
                    Res::Builtin(Builtin::IOErrorInvalidInput),
                    Value::IOError(IOErrorKind::InvalidInput),
                ) => Ok(true),
                _ => Ok(false),
            },
            hir::PatKind::TupleVariant { res, pats, .. } => {
                let fields: Vec<Option<Value>> = match (res, value) {
                    (
                        Res::Variant(item, variant),
                        Value::Enum {
                            item: actual,
                            variant: actual_variant,
                            fields,
                            ..
                        },
                    ) if item == actual && variant == actual_variant => fields.clone(),
                    (Res::Builtin(Builtin::Some), Value::Option(Some(value))) => {
                        vec![(**value).clone()]
                    }
                    (Res::Builtin(Builtin::Ok), Value::Result(Ok(value))) => {
                        vec![(**value).clone()]
                    }
                    (Res::Builtin(Builtin::Err), Value::Result(Err(value))) => {
                        vec![(**value).clone()]
                    }
                    (
                        Res::Builtin(Builtin::IOErrorOther),
                        Value::IOError(IOErrorKind::Other(msg)),
                    ) => {
                        vec![Some(Value::String(msg.clone()))]
                    }
                    _ => return Ok(false),
                };
                self.match_sequence(pats, &fields, bindings)
            }
            hir::PatKind::Struct { res, fields, .. } => {
                let actual = match (res, value) {
                    (
                        Res::Item(item),
                        Value::Struct {
                            item: actual,
                            fields,
                        },
                    ) if item == actual => fields,
                    (
                        Res::Variant(item, variant),
                        Value::Enum {
                            item: actual,
                            variant: actual_variant,
                            named,
                            ..
                        },
                    ) if item == actual && variant == actual_variant => named,
                    _ => return Ok(false),
                };
                for field in fields {
                    let name = self.text(field.name);
                    let Some(Some(value)) = actual.get(name) else {
                        return Ok(false);
                    };
                    if let Some(pat) = field.pat {
                        if !self.match_pattern(pat, value, bindings)? {
                            return Ok(false);
                        }
                    } else if let Some(local) = field.local {
                        bindings.push((local, value.clone()));
                    }
                }
                Ok(true)
            }
            hir::PatKind::Tuple(pats) => match value {
                Value::Tuple(values) => self.match_sequence(pats, values, bindings),
                _ => Ok(false),
            },
            hir::PatKind::Array(pats) => match value {
                Value::Array(values) | Value::Vec(values) => {
                    self.match_sequence(pats, values, bindings)
                }
                _ => Ok(false),
            },
            hir::PatKind::Error => Err(RuntimeError::new("invalid pattern", pattern.span)),
        }
    }

    fn match_sequence(
        &mut self,
        patterns: &[PatId],
        values: &[Option<Value>],
        bindings: &mut Vec<(LocalId, Value)>,
    ) -> Result<bool, RuntimeError> {
        if patterns.len() != values.len() {
            return Ok(false);
        }
        for (pattern, value) in patterns.iter().zip(values) {
            let Some(value) = value else {
                return Ok(false);
            };
            if !self.match_pattern(*pattern, value, bindings)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Whether any `Binding` (including a struct-pattern shorthand field) occurs anywhere in
    /// this pattern subtree. Used by `drop_unbound` to decide between dropping a matched
    /// sub-value whole (no bindings: fully consumed unbound, the container's own `Drop` runs
    /// via `drop_value`) and recursing structurally (bindings present: a partial move, so the
    /// container's own `Drop` must not run — borrowck forbids partial moves out of
    /// `Drop`-implementing types, making that combination unrepresentable in checked code).
    fn pattern_binds(&self, pat: PatId) -> bool {
        match &self.hir.pat(pat).kind {
            hir::PatKind::Binding { .. } => true,
            hir::PatKind::Wild
            | hir::PatKind::Lit(_)
            | hir::PatKind::Path { .. }
            | hir::PatKind::Error => false,
            hir::PatKind::TupleVariant { pats, .. }
            | hir::PatKind::Tuple(pats)
            | hir::PatKind::Array(pats) => pats.iter().any(|&p| self.pattern_binds(p)),
            hir::PatKind::Struct { fields, .. } => fields.iter().any(|field| {
                field.local.is_some() || field.pat.is_some_and(|p| self.pattern_binds(p))
            }),
        }
    }

    /// WP-C2.2 (DEV-030): drop the portions of an owned, consumed match scrutinee that the
    /// matched pattern did NOT bind. Matching moves the scrutinee into the match; a `Binding`
    /// transfers each bound sub-value's ownership to its new local (whose normal end-of-scope
    /// cleanup drops it), but everything matched by `_`, a literal/path pattern, or an
    /// unmentioned struct field was previously just abandoned — its destructor never ran, for
    /// the rest of the program (confirmed empirically; a silent violation of
    /// `03-Type-System.md`'s "destructors run exactly once" invariant). Reference scrutinees
    /// are borrows, not owners, and are never dropped through.
    fn drop_unbound(&mut self, pat: PatId, value: Value) -> Result<(), RuntimeError> {
        if matches!(value, Value::Ref(_)) {
            return Ok(());
        }
        if !self.pattern_binds(pat) {
            // Fully unbound subtree: the sub-value was consumed whole; run its full drop
            // (including the container's own `Drop` impl, if any).
            return self.drop_value(value);
        }
        // **DEV-212: a nominal with its OWN destructor is destroyed whole, never decomposed.**
        //
        // PAT-DROP-001 destroys "every still-owned, unbound component of the hidden scrutinee",
        // and the decomposition below implements exactly that — but for a type carrying its own
        // `Drop`, decomposing is what makes the type's destructor never run at all. `match e` on
        // an `impl Drop` enum printed the arm and nothing else, in BOTH engines.
        //
        // Sound because DEV-211 now refuses any binding that would MOVE a component out of such a
        // value: the only bindings that reach here copied a `Copy` component, which PAT-DROP-001
        // says "remains initialized in the hidden scrutinee". So the value really is complete, and
        // complete is exactly what its destructor requires.
        let kind = &self.hir.pat(pat).kind;
        // A bound value MOVED into the binding; the arm's scope destroys it. Decided BEFORE the
        // whole-value rule below — the first version of that rule ran first and destroyed a bound
        // `Drop` element a second time.
        if matches!(kind, hir::PatKind::Binding { .. }) {
            return Ok(());
        }
        // **DEV-212: a nominal with its OWN destructor is destroyed whole, not decomposed.**
        //
        // PAT-DROP-001 destroys "every still-owned, unbound component of the hidden scrutinee", and
        // the decomposition below implements that — but for a type carrying its own `Drop`,
        // decomposing is what makes that destructor never run at all.
        //
        // Sound because DEV-211 refuses any binding that would MOVE a component out of such a
        // value: every binding reaching here copied a `Copy` component, which PAT-DROP-001 says
        // "remains initialized in the hidden scrutinee". The value is complete, which is what its
        // destructor requires.
        if nominal_item(&value).is_some_and(|item| self.drop_items.contains(&item)) {
            return self.drop_value(value);
        }
        match kind {
            hir::PatKind::Binding { .. } => Ok(()),
            hir::PatKind::TupleVariant { pats, .. } => {
                let pats = pats.clone();
                let payloads: Vec<Option<Value>> = match value {
                    Value::Enum { fields, .. } => fields,
                    Value::Option(Some(inner)) => vec![*inner],
                    Value::Result(Ok(inner)) | Value::Result(Err(inner)) => vec![*inner],
                    Value::IOError(IOErrorKind::Other(msg)) => vec![Some(Value::String(msg))],
                    _ => return Ok(()),
                };
                for (pat, payload) in pats.iter().zip(payloads).rev() {
                    if let Some(payload) = payload {
                        self.drop_unbound(*pat, payload)?;
                    }
                }
                Ok(())
            }
            hir::PatKind::Struct { fields, .. } => {
                let field_pats: Vec<(String, Option<PatId>, bool)> = fields
                    .iter()
                    .map(|field| {
                        (
                            self.text(field.name).to_string(),
                            field.pat,
                            field.local.is_some(),
                        )
                    })
                    .collect();
                let (mut value_fields, mut names) = match value {
                    Value::Struct { item, fields } => {
                        (fields, self.declared_field_order(item, None))
                    }
                    Value::Enum {
                        item,
                        variant,
                        named,
                        ..
                    } => (named, self.declared_field_order(item, Some(variant))),
                    _ => return Ok(()),
                };
                // Unmentioned fields drop in reverse declaration order where the declaration
                // is known; the map's own (alphabetical) order is only a fallback for values
                // with no recoverable declaration order. Mentioned-with-subpattern fields
                // recurse; shorthand-bound fields transferred ownership to their binding.
                for name in value_fields.keys() {
                    if !names.contains(name) {
                        names.push(name.clone());
                    }
                }
                for name in names.into_iter().rev() {
                    let Some(field_value) = value_fields.remove(&name).flatten() else {
                        continue;
                    };
                    match field_pats.iter().find(|(n, _, _)| *n == name) {
                        Some((_, Some(sub_pat), _)) => self.drop_unbound(*sub_pat, field_value)?,
                        Some((_, None, true)) => {}
                        Some((_, None, false)) | None => self.drop_value(field_value)?,
                    }
                }
                Ok(())
            }
            hir::PatKind::Tuple(pats) => {
                let pats = pats.clone();
                let values = match value {
                    Value::Tuple(values) => values,
                    _ => return Ok(()),
                };
                for (pat, item) in pats.iter().zip(values).rev() {
                    if let Some(item) = item {
                        self.drop_unbound(*pat, item)?;
                    }
                }
                Ok(())
            }
            hir::PatKind::Array(pats) => {
                let pats = pats.clone();
                let values = match value {
                    Value::Array(values) | Value::Vec(values) => values,
                    _ => return Ok(()),
                };
                for (pat, item) in pats.iter().zip(values).rev() {
                    if let Some(item) = item {
                        self.drop_unbound(*pat, item)?;
                    }
                }
                Ok(())
            }
            // Wild/Lit/Path bind nothing and are handled by the fully-unbound fast path above.
            hir::PatKind::Wild
            | hir::PatKind::Lit(_)
            | hir::PatKind::Path { .. }
            | hir::PatKind::Error => Ok(()),
        }
    }

    fn iter_values(&self, value: Value, span: Span) -> Result<Vec<Value>, RuntimeError> {
        match value {
            Value::Range {
                start,
                end,
                inclusive,
            } => {
                let final_end = if inclusive {
                    end.checked_add(1)
                        .ok_or_else(|| RuntimeError::new("range overflow", span))?
                } else {
                    end
                };
                Ok((start..final_end).map(Value::Int).collect())
            }
            Value::Array(values) | Value::Vec(values) => Ok(values.into_iter().flatten().collect()),
            Value::Slice(place, start, end) => Ok((start..end)
                .map(|index| {
                    let mut item = place.clone();
                    item.projections.push(Projection::Index(index));
                    Value::Ref(item)
                })
                .collect()),
            _ => Err(RuntimeError::new("value is not directly iterable", span)),
        }
    }

    /// WP-C2.2 (DEV-031): advance a standard or nominal iterator exactly once. `for` calls this
    /// between body executions, preserving the observable `next`/body interleaving and allowing
    /// `break` to stop without eagerly exhausting the iterator.
    fn next_for_iterator(
        &mut self,
        iterator_place: &Place,
        for_expr: ExprId,
        span: Span,
    ) -> Result<Option<Value>, RuntimeError> {
        let current = self.clone_place_value(iterator_place, span)?;
        let next = if nominal_item(&current).is_some() {
            // AS3 Boundary 4: CONSUME the checker's `Iterator::next` selection. The scan below
            // is transitional and goes when `find_method` does.
            let method = self
                .selected_core_trait_callable(for_expr, hir::CoreTrait::Iterator)
                .and_then(|use_| match use_.selection {
                    crate::typecheck::CalleeSelection::Static { body, .. } => {
                        self.callable_for_body(body)
                    }
                    // AS3 Boundary 4: `Bound` needs the shared specialiser, not this path.
                    crate::typecheck::CalleeSelection::Bound { .. }
                    | crate::typecheck::CalleeSelection::FunctionValue => None,
                })
                // A late-bound iterator (`for x in it` where `it: T` and `T: Iterator`) resolves
                // through the shared specialiser, not through a name scan.
                .or_else(|| {
                    self.specialised_bound_callable(for_expr, for_expr)
                        .map(|(c, _)| c)
                })
                .ok_or_else(|| RuntimeError::new("value is not iterable", span))?;
            let iterator_env = self.core_trait_env(for_expr, hir::CoreTrait::Iterator);
            self.call_user_method(
                method,
                iterator_place.clone(),
                iterator_env,
                Vec::new(),
                span,
            )?
        } else {
            let (next, updated) = self.iterator_step(current, Some(iterator_place), span)?;
            *self.place_value_mut(iterator_place, span)? = updated;
            Value::Option(next.map(|value| Box::new(Some(value))))
        };
        match next {
            Value::Option(Some(value)) => Ok(Some(own_payload(
                value,
                "`Iterator::next` returned a `Some`",
                span,
            )?)),
            Value::Option(None) => Ok(None),
            _ => Err(RuntimeError::new("Iterator::next must return Option", span)),
        }
    }

    fn slice_bounds(
        &self,
        range: Value,
        length: usize,
        span: Span,
    ) -> Result<(usize, usize), RuntimeError> {
        let Value::Range {
            start,
            end,
            inclusive,
        } = range
        else {
            return Err(RuntimeError::new("slice index must be a range", span));
        };
        let start = usize::try_from(start).map_err(|_| {
            RuntimeError::with_category(
                "slice range out of bounds (negative start)",
                span,
                crate::mir::TrapCategory::IndexOutOfBounds,
            )
        })?;
        let mut end = usize::try_from(end).map_err(|_| {
            RuntimeError::with_category(
                "slice range out of bounds (negative end)",
                span,
                crate::mir::TrapCategory::IndexOutOfBounds,
            )
        })?;
        if inclusive {
            end = end.checked_add(1).ok_or_else(|| {
                RuntimeError::with_category(
                    "slice range out of bounds (inclusive end overflow)",
                    span,
                    crate::mir::TrapCategory::IndexOutOfBounds,
                )
            })?;
        }
        if start > end || end > length {
            return Err(RuntimeError::with_category(
                "slice range out of bounds",
                span,
                crate::mir::TrapCategory::IndexOutOfBounds,
            ));
        }
        Ok((start, end))
    }

    fn expr_place(&mut self, expr: ExprId) -> Result<Place, RuntimeError> {
        let node = self.hir.expr(expr);
        match &node.kind {
            hir::ExprKind::Path {
                res: Res::Local(local) | Res::SelfValue(local),
                ..
            } => Ok(Place {
                frame: self.frames.len() - 1,
                local: *local,
                projections: Vec::new(),
            }),
            // WP-C2.2 (DEV-037): each projection arm normalizes the base place through any
            // `Value::Ref` chain before projecting. Field/index access through a reference
            // (`r.v` for `r: &Inner`) type-checks per the auto-deref rule but previously tried
            // to project directly on the stored `Value::Ref`, failing at runtime with "use of
            // moved or invalid field" — confirmed pre-existing at Gate C1 close, found while
            // fixing DEV-035 (whose nested-accessor case routes a method-returned reference
            // through exactly this path).
            hir::ExprKind::Field { base, name, .. } => {
                let place = self.expr_place(*base)?;
                let mut place = self.deref_place(place, node.span)?;
                place
                    .projections
                    .push(Projection::Field(self.text(*name).to_string()));
                Ok(place)
            }
            hir::ExprKind::TupleField { base, index } => {
                let place = self.expr_place(*base)?;
                let mut place = self.deref_place(place, node.span)?;
                let index = self
                    .text(*index)
                    .parse::<usize>()
                    .map_err(|_| RuntimeError::new("invalid tuple index", *index))?;
                place.projections.push(Projection::Index(index));
                Ok(place)
            }
            hir::ExprKind::Index { base, index } => {
                let place = self.expr_place(*base)?;
                let place = self.deref_place(place, node.span)?;
                if matches!(self.tables.expr_types.get(index), Some(Ty::Range(_))) {
                    let range = self.expect_value(*index)?;
                    let (base_place, base_start, base_end) = match self
                        .place_value(&place, node.span)?
                        .clone()
                    {
                        Value::Slice(base_place, start, end) => (base_place, start, end),
                        Value::Array(values) | Value::Vec(values) => (place, 0, values.len()),
                        _ => return Err(RuntimeError::new("value cannot be sliced", node.span)),
                    };
                    let (start, end) =
                        self.slice_bounds(range, base_end - base_start, node.span)?;
                    return self.promote_to_temp_place(
                        Value::Slice(base_place, base_start + start, base_start + end),
                        node.span,
                    );
                }
                let index_value = self.expect_int(*index)?;
                if self.pending_propagation.take().is_some() {
                    // `expr_place` returns a bare `Place`, not `Flow`, so it cannot itself
                    // signal early transfer to its caller (a real, documented architectural
                    // gap -- see DEV-045's follow-up notes). Fail loudly rather than silently
                    // using `expect_int`'s placeholder `0` as a real index: this is no worse
                    // than this site's pre-existing "negative index" rejection for a genuinely
                    // out-of-range value, just reached via a different condition.
                    return Err(RuntimeError::new(
                        "index expression did not produce a value",
                        self.hir.expr(*index).span,
                    ));
                }
                // DEV-097: BOTH ends of one bounds check report the SAME site. This used to
                // use the index OPERAND's span while the out-of-range arm below uses `node.span`
                // (the whole index expression), so the oracle blamed two different columns for
                // two ends of the same check -- and disagreed with MIR and the native backend on
                // one of them. Found by the three-engine harness's negative-index case; no
                // corpus or inline case had ever indexed with a negative value.
                let index = usize::try_from(index_value).map_err(|_| {
                    RuntimeError::with_category(
                        "negative index",
                        node.span,
                        crate::mir::TrapCategory::IndexOutOfBounds,
                    )
                })?;
                if let Value::Slice(base_place, start, end) =
                    self.place_value(&place, node.span)?.clone()
                {
                    if index >= end - start {
                        return Err(RuntimeError::with_category(
                            "index out of bounds",
                            node.span,
                            crate::mir::TrapCategory::IndexOutOfBounds,
                        ));
                    }
                    let mut item = base_place;
                    item.projections.push(Projection::Index(start + index));
                    return Ok(item);
                }
                let mut place = place;
                place.projections.push(Projection::Index(index));
                Ok(place)
            }
            hir::ExprKind::Unary {
                op: UnOp::Deref,
                operand,
            } => match self.expect_value(*operand)? {
                Value::Ref(place) => Ok(place),
                _ => Err(RuntimeError::new(
                    "cannot dereference non-reference",
                    node.span,
                )),
            },
            _ => {
                let value = self.expect_value(expr)?;
                let local_id = LocalId(1000000 + self.frame().values.len() as u32);
                self.frame_mut().values.insert(local_id, Some(value));
                Ok(Place {
                    frame: self.frames.len() - 1,
                    local: local_id,
                    projections: Vec::new(),
                })
            }
        }
    }

    fn promote_to_temp_place(&mut self, value: Value, _span: Span) -> Result<Place, RuntimeError> {
        let frame = self.frames.len() - 1;
        self.promote_to_temp_place_in(frame, value)
    }

    /// `promote_to_temp_place`, with the owning frame chosen explicitly.
    ///
    /// **The frame decides the backing storage's lifetime, so a view must be promoted into the
    /// frame of what it is a view OF — not into whichever frame happens to be running.**
    ///
    /// CD-305 made `String::bytes()` return a `Value::Slice` over materialised bytes, and promoted
    /// them into the CURRENT frame. That is correct while the view is used locally and wrong the
    /// moment it escapes: `fn borrow_of(s: &String) -> &[UInt8] { s.bytes() }` materialised into
    /// `borrow_of`'s frame, which pops on return, leaving the returned slice pointing at storage
    /// that no longer exists — "dangling reference". Core v1 admits that function (a returned
    /// reference deriving from a reference parameter), so the program is valid and the
    /// representation was not.
    ///
    /// Promoting into the receiver's frame ties the bytes to the same frame as the string they
    /// came from, which is precisely the lifetime the borrow already has. Found by
    /// WP-COPY-CANON's matrix, whose ordinary-language producer controls cover the escaping case
    /// that CD-305's own regression tests did not.
    fn promote_to_temp_place_in(
        &mut self,
        frame: usize,
        value: Value,
    ) -> Result<Place, RuntimeError> {
        let Some(target) = self.frames.get_mut(frame) else {
            return Err(RuntimeError::new(
                "internal: promotion into a frame that does not exist",
                Span::synthetic(self.entry_source()),
            ));
        };
        let local_id = LocalId(1000000 + target.values.len() as u32);
        // `values.insert`, NOT `Frame::insert`: the latter also appends to `order`, which is the
        // frame's DROP ORDER. A promoted temp is a view's backing storage, not a value the frame
        // owns and destroys — registering it made promoted temps participate in destruction and
        // broke `collection_replacement_and_removal_drop_consumed_keys`. The original spelling was
        // load-bearing and the refactor lost it.
        target.values.insert(local_id, Some(value));
        Ok(Place {
            frame,
            local: local_id,
            projections: Vec::new(),
        })
    }

    /// Like `promote_to_temp_place`, but for a value this call actually creates sole,
    /// no-other-owner storage for (as opposed to `promote_to_temp_place`'s many existing uses,
    /// which momentarily wrap a *view* into data still separately owned elsewhere -- e.g.
    /// iterator snapshots that clone a value out for `Value::Ref` wrapping while the iterator's
    /// own backing storage still holds it too, or the collection-probe key path). Registers the
    /// new local through `Frame::insert` so it participates in `Frame::order` and is correctly
    /// destroyed exactly once when the frame is cleaned up -- `promote_to_temp_place` bypasses
    /// `Frame::insert` (a raw `.values.insert(...)`), so anything placed there is never in
    /// `order` and is silently discarded via ordinary Rust-level deallocation with no
    /// STARK-level `Drop::drop` call at all when the frame is popped. Confirmed empirically:
    /// swapping a non-place comparison operand's temp through this instead of the plain helper
    /// is what makes `comparison_of_temporary_operands_evaluates_each_once_and_drops_after_call`
    /// (Correction brief Issue 2) actually observe the destructor. Using this helper at one of
    /// `promote_to_temp_place`'s existing view-only call sites would double-drop the underlying
    /// data; use it only where this call is the value's sole and complete owner.
    fn promote_to_owned_temp_place(
        &mut self,
        value: Value,
        _span: Span,
    ) -> Result<Place, RuntimeError> {
        let local_id = LocalId(1000000 + self.frame().values.len() as u32);
        self.frame_mut().insert(local_id, Some(value));
        Ok(Place {
            frame: self.frames.len() - 1,
            local: local_id,
            projections: Vec::new(),
        })
    }

    fn call_function_pointer(
        &mut self,
        func: Value,
        values: Vec<Value>,
        span: Span,
    ) -> Result<Value, RuntimeError> {
        let Value::Function(callee) = func else {
            return Err(RuntimeError::new("expected a function pointer", span));
        };
        let callable = self
            .item_callable(callee.item)
            .ok_or_else(|| RuntimeError::new("expression is not callable", span))?;
        // **DEV-178: install the environment the VALUE carries, never one looked up on this call
        // expression.** The instantiation was selected at the coercion; this call site knows only
        // the function's type. `_env` lives for the whole call and its `Drop` covers traps,
        // propagation and internal failures.
        self.invoke_callable(
            ResolvedInvocation {
                callable,
                environment: InvocationEnv::Captured(callee),
            },
            ReceiverSource::None,
            values,
            span,
        )
    }

    /// Install a function value's captured environment for the duration of its call.
    ///
    /// Already concrete — `capture_function_value` resolved it against the frame that created the
    /// value, which may since have gone. Nothing is substituted here.
    fn push_captured_env(&mut self, callee: &FunctionValue) -> GenericFrame {
        if callee.bindings.is_empty() {
            return GenericFrame {
                frames: self.generic_frames.clone(),
                pushed: false,
            };
        }
        let map: HashMap<String, Ty> = callee.bindings.iter().cloned().collect();
        self.generic_frames.borrow_mut().push(map);
        GenericFrame {
            frames: self.generic_frames.clone(),
            pushed: true,
        }
    }

    fn iterator_step(
        &mut self,
        mut iter: Value,
        iter_place: Option<&Place>,
        span: Span,
    ) -> Result<(Option<Value>, Value), RuntimeError> {
        match &mut iter {
            // `*idx` is a BYTE offset into `s`, always on a scalar boundary because it only ever
            // advances by `len_utf8()`.
            //
            // It used to be a SCALAR index compared against `s.len()`, which is a byte count. For
            // any string containing a multi-byte scalar the two disagree: `"Stark語"` is 6 scalars
            // in 8 bytes, so the loop ran twice too many times and `nth(6).unwrap()` panicked the
            // host process. ASCII-only strings hid it exactly, because there the two counts are
            // equal. The MIR interpreter and the native runtime were both already correct, so this
            // was an oracle-only divergence — and one no differential could observe, because a
            // panicking host produces no observation to compare.
            //
            // The byte cursor also removes an O(n^2) walk: `nth(idx)` re-scanned from the start of
            // the string on every step.
            Value::CharsIter(s, ref mut idx) => {
                // §4.6: no `unwrap` on a value derived from user content. A cursor off a scalar
                // boundary would be an interpreter invariant violation, not a program error, so it
                // is reported as one instead of panicking the host.
                let Some(rest) = s.get(*idx..) else {
                    return Err(RuntimeError::internal(
                        "chars iterator cursor is not on a UTF-8 scalar boundary",
                        span,
                    ));
                };
                let opt = match rest.chars().next() {
                    Some(ch) => {
                        *idx += ch.len_utf8();
                        Some(Value::Char(ch))
                    }
                    None => None,
                };
                Ok((opt, iter))
            }
            Value::SplitIter(parts, ref mut idx) => {
                // DEV-138 (a DEV-121 instance): `SplitIter`'s item is `&str`
                // (`06-Standard-Library.md`), so it must be `Value::Str` — which `value_is_copy`
                // reports as Copy — and NOT `Value::String`, which is owned and therefore
                // consumed by its first use. Yielding the owned form meant
                // `String::from(word)` twice in one iteration trapped "use of unavailable value"
                // on the second, while the checker and MIR both saw a `Copy` shared reference.
                //
                // This is DEV-121's governing rule verbatim: Copy/move behaviour, and the runtime
                // representation carrying it, follow the normalized semantic type and never the
                // expression that produced the value. `trim` and `substring` already yield
                // `Value::Str` for the same declared type; `split` was the outlier.
                let opt = if *idx < parts.len() {
                    let s = parts[*idx].clone();
                    *idx += 1;
                    Some(Value::Str(s))
                } else {
                    None
                };
                Ok((opt, iter))
            }
            Value::VecIter(place, ref mut idx) => {
                let vec_val = self.place_value(place, span)?;
                if let Value::Vec(items) = vec_val {
                    let opt = if *idx < items.len() {
                        let mut item_place = place.clone();
                        item_place.projections.push(Projection::Index(*idx));
                        *idx += 1;
                        Some(Value::Ref(item_place))
                    } else {
                        None
                    };
                    Ok((opt, iter))
                } else {
                    Err(RuntimeError::new("expected Vec", span))
                }
            }
            Value::HashMapKeysIter(keys, ref mut idx) => {
                let opt = if *idx < keys.len() {
                    let opt_val = if let Some(place) = iter_place {
                        let mut item_place = place.clone();
                        item_place.projections.push(Projection::Index(*idx));
                        *idx += 1;
                        Value::Ref(item_place)
                    } else {
                        let val = keys[*idx].clone().unwrap_or(Value::Unit);
                        *idx += 1;
                        Value::Ref(self.promote_to_temp_place(val, span)?)
                    };
                    Some(opt_val)
                } else {
                    None
                };
                Ok((opt, iter))
            }
            Value::HashMapValuesIter(values, ref mut idx) => {
                let opt = if *idx < values.len() {
                    let opt_val = if let Some(place) = iter_place {
                        let mut item_place = place.clone();
                        item_place.projections.push(Projection::Index(*idx));
                        *idx += 1;
                        Value::Ref(item_place)
                    } else {
                        let val = values[*idx].clone().unwrap_or(Value::Unit);
                        *idx += 1;
                        Value::Ref(self.promote_to_temp_place(val, span)?)
                    };
                    Some(opt_val)
                } else {
                    None
                };
                Ok((opt, iter))
            }
            Value::HashMapIter(pairs, ref mut idx) => {
                let opt = if *idx < pairs.len() {
                    let opt_val = if let Some(place) = iter_place {
                        let item_place = place.clone();
                        let mut k_place = item_place.clone();
                        k_place.projections.push(Projection::Index(*idx));
                        k_place.projections.push(Projection::Index(0));

                        let mut v_place = item_place.clone();
                        v_place.projections.push(Projection::Index(*idx));
                        v_place.projections.push(Projection::Index(1));

                        let tuple_val = Value::Tuple(vec![
                            Some(Value::Ref(k_place)),
                            Some(Value::Ref(v_place)),
                        ]);
                        *idx += 1;
                        tuple_val
                    } else {
                        let pair = pairs[*idx].clone().unwrap_or(Value::Unit);
                        *idx += 1;
                        if let Value::Tuple(mut elems) = pair {
                            if elems.len() == 2 {
                                let k = elems[0].clone().unwrap_or(Value::Unit);
                                let v = elems[1].clone().unwrap_or(Value::Unit);
                                let k_ref = Value::Ref(self.promote_to_temp_place(k, span)?);
                                let v_ref = Value::Ref(self.promote_to_temp_place(v, span)?);
                                elems[0] = Some(k_ref);
                                elems[1] = Some(v_ref);
                            }
                            Value::Tuple(elems)
                        } else {
                            pair
                        }
                    };
                    Some(opt_val)
                } else {
                    None
                };
                Ok((opt, iter))
            }
            Value::HashSetIter(items, ref mut idx) => {
                let opt = if *idx < items.len() {
                    let opt_val = if let Some(place) = iter_place {
                        let mut item_place = place.clone();
                        item_place.projections.push(Projection::Index(*idx));
                        *idx += 1;
                        Value::Ref(item_place)
                    } else {
                        let val = items[*idx].clone().unwrap_or(Value::Unit);
                        *idx += 1;
                        Value::Ref(self.promote_to_temp_place(val, span)?)
                    };
                    Some(opt_val)
                } else {
                    None
                };
                Ok((opt, iter))
            }
            Value::MapIter(inner, func) => {
                let (next_opt, updated_inner) = self.iterator_step(*inner.clone(), None, span)?;
                **inner = updated_inner;
                if let Some(x) = next_opt {
                    let called =
                        // DEV-179 (DORMANT): sound only while E0105 makes MapIter/FilterIter unreachable
                        // from accepted Core v1 source. Before lifting E0105, retain the callback's
                        // captured FunctionValue rather than reconstructing it with empty bindings.
                        self.call_function_pointer(Value::Function(FunctionValue { item: *func, bindings: Vec::new() }), vec![x], span)?;
                    Ok((Some(called), iter))
                } else {
                    Ok((None, iter))
                }
            }
            Value::FilterIter(inner, pred) => {
                let mut current_inner = *inner.clone();
                loop {
                    let (next_opt, updated_inner) =
                        self.iterator_step(current_inner, None, span)?;
                    current_inner = updated_inner;
                    if let Some(x) = next_opt {
                        let x_ref = Value::Ref(self.promote_to_temp_place(x.clone(), span)?);
                        let res =
                            // DEV-179 (DORMANT): see the `map` site above — empty bindings are sound only
                            // while E0105 keeps this unreachable.
                            self.call_function_pointer(Value::Function(FunctionValue { item: *pred, bindings: Vec::new() }), vec![x_ref], span)?;
                        if let Value::Bool(true) = res {
                            **inner = current_inner;
                            return Ok((Some(x), iter));
                        }
                    } else {
                        **inner = current_inner;
                        return Ok((None, iter));
                    }
                }
            }
            _ => Err(RuntimeError::new("expected an iterator", span)),
        }
    }

    fn core_receiver_place(&mut self, expr: ExprId, span: Span) -> Result<Place, RuntimeError> {
        let place = self.expr_place(expr)?;
        self.deref_place(place, span)
    }

    /// Normalize a place through any chain of `Value::Ref` values stored at it, yielding the
    /// place of the ultimate referent. A no-op for places whose value is not a reference.
    fn deref_place(&self, mut place: Place, span: Span) -> Result<Place, RuntimeError> {
        loop {
            match self.place_value(&place, span)? {
                Value::Ref(referent) => place = referent.clone(),
                _ => return Ok(place),
            }
        }
    }

    fn clone_place_value(&self, place: &Place, span: Span) -> Result<Value, RuntimeError> {
        self.place_value(place, span).cloned()
    }

    fn deref_value(&self, mut value: Value, span: Span) -> Result<Value, RuntimeError> {
        while let Value::Ref(place) = value {
            value = self.clone_place_value(&place, span)?;
        }
        Ok(value)
    }

    fn format_runtime_value(&self, value: &Value, span: Span) -> Result<String, RuntimeError> {
        let Value::Slice(place, start, end) = value else {
            return Ok(value.to_string());
        };
        let elements = match self.place_value(place, span)? {
            Value::Array(elements) | Value::Vec(elements) => elements,
            _ => return Err(RuntimeError::new("slice base is unavailable", span)),
        };
        if start > end || *end > elements.len() {
            return Err(RuntimeError::with_category(
                "slice range out of bounds",
                span,
                crate::mir::TrapCategory::IndexOutOfBounds,
            ));
        }
        let rendered = elements[*start..*end]
            .iter()
            .map(|element| {
                element
                    .as_ref()
                    .map_or_else(|| "<moved>".to_string(), ToString::to_string)
            })
            .collect::<Vec<_>>()
            .join(", ");
        Ok(format!("[{rendered}]"))
    }

    /// DEV-089 (WP-C4.7 close-out): render a by-value `print`/`println`/`eprint`/`eprintln`
    /// argument through its language-level `Display`. A user nominal (struct/enum) that has its
    /// own coherent `Display` impl runs that impl's `fmt(&self) -> String`, and the returned
    /// String's bytes are what is printed — the internal aggregate/debug rendering
    /// (`format_runtime_value`) is NOT language-level `Display` and no longer reaches a user
    /// type here (the checker's E0500 guarantees any type printed either is a standard `Display`
    /// type or has an impl). Standard-library display types keep their built-in formatting,
    /// which is observationally identical to their canonical `Display`.
    ///
    /// Ownership: the argument arrives owned (moved into the call). `fmt` borrows it via `&self`
    /// (the canonical receiver). The returned bytes must be SUBMITTED to the output stream before
    /// the by-value argument is destroyed (§2.4/§2.6), so this returns the rendered text plus the
    /// still-live temp place holding the argument; the caller submits the bytes, then calls
    /// `finish_display` to run the argument's destructor (ordinary by-value call ownership). If
    /// `fmt` traps, the `?` propagates and the argument is not dropped (traps abort; destructors
    /// do not run).
    /// The rendered root's concrete static type, substituted through the active generic frame.
    ///
    /// `None` when the checker recorded no type for the expression, which the walk below treats as
    /// "cannot decide the step" rather than as a licence to guess.
    fn display_root_ty(&self, root: ExprId, span: Span) -> Result<Option<Ty>, RuntimeError> {
        let Some(declared) = self.tables.expr_types.get(&root) else {
            return Ok(None);
        };
        Ok(Some(self.concrete_runtime_ty(declared, span)?))
    }

    /// The static type one step down, mirroring the checker's own walk.
    fn display_child_ty(ty: Option<&Ty>, step: DisplayStep) -> Option<Ty> {
        let mut ty = ty?;
        // A reference renders as its referent, exactly as the checker's walk peels it.
        while let Ty::Ref { inner, .. } = ty {
            ty = inner;
        }
        match (step, ty) {
            (DisplayStep::TupleField(index), Ty::Tuple(elems)) => {
                elems.get(index as usize).cloned()
            }
            (DisplayStep::ArrayElement, Ty::Array(elem, _))
            | (DisplayStep::SliceElement, Ty::Slice(elem)) => Some((**elem).clone()),
            (DisplayStep::VecElement, Ty::Core(crate::hir::CoreType::Vec, args))
            | (DisplayStep::OptionSome, Ty::Core(crate::hir::CoreType::Option, args))
            | (DisplayStep::ResultOk, Ty::Core(crate::hir::CoreType::Result, args)) => {
                args.first().cloned()
            }
            (DisplayStep::ResultErr, Ty::Core(crate::hir::CoreType::Result, args)) => {
                args.get(1).cloned()
            }
            _ => None,
        }
    }

    /// Which element step a sequence value's STATIC type calls for. `Value::Array` and `Value::Vec`
    /// are one runtime shape but three static ones, and the checker keyed them apart.
    fn display_element_step(ty: Option<&Ty>) -> DisplayStep {
        let mut ty = ty;
        while let Some(Ty::Ref { inner, .. }) = ty {
            ty = Some(inner);
        }
        match ty {
            Some(Ty::Array(..)) => DisplayStep::ArrayElement,
            Some(Ty::Slice(..)) => DisplayStep::SliceElement,
            _ => DisplayStep::VecElement,
        }
    }

    /// **The published body for one render position.**
    ///
    /// `Static` names it outright. `Bound` is late-bound — `println(x)` inside
    /// `fn show<T: Display>` cannot name a body at check time — so it goes through the shared
    /// specialiser with `Self` taken from the published parametric type, substituted through the
    /// active generic frame. Same resolution the operator path uses; no second algorithm.
    fn display_callable(
        &self,
        root: ExprId,
        path: &DisplayPath,
        span: Span,
    ) -> Result<Option<Callable>, RuntimeError> {
        let Some(id) = self.tables.display_uses.get(&(root, path.clone())) else {
            return Ok(None);
        };
        let Some(use_) = self.tables.callable_uses.get(id.0 as usize) else {
            return Ok(None);
        };
        match &use_.selection {
            crate::typecheck::CalleeSelection::Static { body, .. } => {
                Ok(self.callable_for_body(*body))
            }
            crate::typecheck::CalleeSelection::Bound {
                trait_,
                member,
                self_ty,
                trait_args,
                method_args,
            } => {
                let mut self_ty = self.concrete_runtime_ty(self_ty, span)?;
                while let Ty::Ref { inner, .. } = self_ty {
                    self_ty = *inner;
                }
                let Some(resolved) = crate::bound_dispatch::specialize_bound_callable(
                    &self.tables.trait_impls,
                    &self.tables.callable_types,
                    *trait_,
                    member,
                    &self_ty,
                    trait_args,
                    method_args,
                ) else {
                    return Ok(None);
                };
                Ok(self.callable_for_body(resolved.body))
            }
            crate::typecheck::CalleeSelection::FunctionValue => Ok(None),
        }
    }

    fn display_text(
        &mut self,
        // **AS3 Boundary 4: the root the checker keyed its dispatch plan on.** A `println`
        // argument or an interpolation field; both are roots in their own right.
        root: ExprId,
        value: Value,
        span: Span,
    ) -> Result<(String, Option<Place>), RuntimeError> {
        if let Value::Struct { .. } | Value::Enum { .. } = &value {
            // **No improvisation here either.** This branch used to fall through to
            // `format_runtime_value` — the aggregate debug form — whenever the lookup missed, which
            // is the same defect as `display_deep`'s old `value.to_string()`, one level up. It
            // survived the first mutation pass because the top-level path was still silently
            // answering for itself; the mutation is what exposed it. A user nominal that reached
            // here passed E0500, so a missing publication is an internal invariant violation.
            let Some(callable) = self.display_callable(root, &DisplayPath::default(), span)? else {
                return Err(RuntimeError::new(
                    "internal invariant: no Display use published for a checked user nominal",
                    span,
                ));
            };
            {
                // Give the by-value argument its own storage so `&self` can borrow it.
                let place = self.promote_to_owned_temp_place(value, span)?;
                let result = self.call_user_method(
                    callable,
                    place.clone(),
                    self.display_env(root, &DisplayPath::default()),
                    Vec::new(),
                    span,
                );
                let text = match result? {
                    Value::String(text) => text,
                    _ => {
                        return Err(RuntimeError::new(
                            "Display::fmt did not return a String",
                            span,
                        ))
                    }
                };
                return Ok((text, Some(place)));
            }
        }
        // CD-123: a COMPOSITE argument renders through language-level `Display` at EVERY depth — a
        // nested user nominal runs its own `Display::fmt`, NOT the aggregate `{field: value}` debug
        // form — matching the native lowering (`emit_display_value`). The whole composite is promoted
        // to a place so it is dropped exactly once after its bytes are submitted (Contract C).
        // **DEV-207: a slice VIEW is a composite too, and it was not in this list.**
        //
        // So it fell to `format_runtime_value` — the structural debug form — and a `struct X` with
        // its own `Display` printed `{n: 1}` instead of running `X::fmt`. The checker had published
        // `DisplayPath([SliceElement])` for the position all along; nothing consumed it.
        //
        // Handled BEFORE the block below rather than added to it: a slice borrows its elements, so
        // there is no owned composite to promote and nothing for the caller to drop afterwards.
        // Promoting it would give a view its own drop-registered storage.
        if let Value::Slice(..) = &value {
            let ty = self.display_root_ty(root, span)?;
            let text =
                self.display_deep(root, &value, ty.as_ref(), DisplayPath::default(), span)?;
            return Ok((text, None));
        }
        if let Value::Tuple(_)
        | Value::Array(_)
        | Value::Vec(_)
        | Value::Option(_)
        | Value::Result(_) = &value
        {
            let place = self.promote_to_owned_temp_place(value, span)?;
            let snapshot = self.clone_place_value(&place, span)?;
            let ty = self.display_root_ty(root, span)?;
            let text =
                self.display_deep(root, &snapshot, ty.as_ref(), DisplayPath::default(), span)?;
            return Ok((text, Some(place)));
        }
        Ok((self.format_runtime_value(&value, span)?, None))
    }

    /// CD-123: render a value through language-level `Display` recursively. A user nominal at ANY
    /// depth runs its own `Display::fmt`; a composite renders element-by-element with the same
    /// delimiters the native lowering emits (`(a, b)` / `[a, b]` / `Some(v)` / `Ok(v)`), so the HIR
    /// oracle and the native binary agree at every nesting level. Everything else (primitives,
    /// `String`/`str`) uses its `Display for Value`.
    ///
    /// A nested nominal is CLONED to give `fmt` a `&self` place; a Rust clone runs no STARK
    /// destructor, and the clone is discarded WITHOUT `drop_value`, so the real element is still
    /// dropped exactly once by its owning composite (Contract C) — never a double destructor.
    fn display_deep(
        &mut self,
        root: ExprId,
        value: &Value,
        ty: Option<&Ty>,
        path: DisplayPath,
        span: Span,
    ) -> Result<String, RuntimeError> {
        match value {
            Value::Struct { .. } | Value::Enum { .. } => {
                // **AS3 Boundary 4: consume, and do not improvise on absence.**
                //
                // This scanned the nominal's impls for a member named `fmt`, and on failure
                // returned `value.to_string()` — the aggregate debug form. That defensive arm is
                // gone. The checker published a body for every position it will render (E0500
                // rejects the rest), so a missing entry HERE is an internal invariant violation,
                // not permission to substitute a different rendering. Silently answering with a
                // second algorithm when the published one is missing is exactly what DEV-192 cost.
                let Some(callable) = self.display_callable(root, &path, span)? else {
                    return Err(RuntimeError::new(
                        format!(
                            "internal invariant: no Display use published at {path:?} for a \
                             checked user nominal"
                        ),
                        span,
                    ));
                };
                let place = self.promote_to_owned_temp_place(value.clone(), span)?;
                let text = match self.call_user_method(
                    callable,
                    place.clone(),
                    self.display_env(root, &path),
                    Vec::new(),
                    span,
                )? {
                    Value::String(text) => text,
                    _ => {
                        return Err(RuntimeError::new(
                            "Display::fmt did not return a String",
                            span,
                        ))
                    }
                };
                // Discard the rendering temp WITHOUT its STARK destructor (see the doc-comment).
                let _ = self.take_place(&place, span);
                Ok(text)
            }
            Value::Tuple(elems) => {
                let mut parts = Vec::with_capacity(elems.len());
                for (index, slot) in elems.iter().enumerate() {
                    let step = DisplayStep::TupleField(index as u32);
                    let elem_ty = Self::display_child_ty(ty, step);
                    parts.push(self.display_slot(
                        root,
                        slot,
                        elem_ty.as_ref(),
                        path.child(step),
                        span,
                    )?);
                }
                Ok(format!("({})", parts.join(", ")))
            }
            // **The static type decides the step, not the value.** `Value::Array` and `Value::Vec`
            // share this arm — the runtime representation does not distinguish them — while the
            // checker keyed `ArrayElement` and `VecElement` separately. Reading the step off the
            // type is what keeps the two walks in step; guessing here, or trying both keys, would
            // let a mismatch pass as a hit.
            Value::Array(elems) | Value::Vec(elems) => {
                let step = Self::display_element_step(ty);
                let elem_ty = Self::display_child_ty(ty, step);
                let mut parts = Vec::with_capacity(elems.len());
                for slot in elems.iter() {
                    // One published position, executed once per element: the plan is static, the
                    // loop is the renderer's. No record per runtime element.
                    parts.push(self.display_slot(
                        root,
                        slot,
                        elem_ty.as_ref(),
                        path.child(step),
                        span,
                    )?);
                }
                Ok(format!("[{}]", parts.join(", ")))
            }
            // **DEV-207: a slice VIEW renders through the plan, not structurally.**
            //
            // The checker publishes `DisplayPath([SliceElement])` for this position, and the walk
            // had no arm for `Value::Slice` — so the value fell through to `format_runtime_value`,
            // which renders structurally. A `struct X` with its own `Display` printed `{n: 1}`
            // instead of calling `X::fmt`: a published selection the engine did not consume, which
            // is the AS3 Boundary 4 class exactly.
            //
            // Unreachable until DEV-206, because `&[T]` was refused by `Display` eligibility
            // outright; bare `[T]` was accepted and rendered structurally, so the same defect was
            // there and could not be seen through a correct program.
            Value::Slice(place, start, end) => {
                let elements = match self.place_value(place, span)? {
                    Value::Array(elements) | Value::Vec(elements) => elements.clone(),
                    _ => return Err(RuntimeError::new("slice base is unavailable", span)),
                };
                if start > end || *end > elements.len() {
                    return Err(RuntimeError::new("slice range out of bounds", span));
                }
                let step = DisplayStep::SliceElement;
                let elem_ty = Self::display_child_ty(ty, step);
                let mut parts = Vec::with_capacity(end - start);
                for slot in &elements[*start..*end] {
                    parts.push(self.display_slot(
                        root,
                        slot,
                        elem_ty.as_ref(),
                        path.child(step),
                        span,
                    )?);
                }
                Ok(format!("[{}]", parts.join(", ")))
            }
            Value::Option(Some(inner)) => {
                let step = DisplayStep::OptionSome;
                let inner_ty = Self::display_child_ty(ty, step);
                let inner = require_live_payload(inner, "rendering a `Some`", span)?;
                let text =
                    self.display_deep(root, inner, inner_ty.as_ref(), path.child(step), span)?;
                Ok(format!("Some({text})"))
            }
            Value::Option(None) => Ok("None".to_string()),
            Value::Result(Ok(inner)) => {
                let step = DisplayStep::ResultOk;
                let inner_ty = Self::display_child_ty(ty, step);
                let inner = require_live_payload(inner, "rendering an `Ok`", span)?;
                let text =
                    self.display_deep(root, inner, inner_ty.as_ref(), path.child(step), span)?;
                Ok(format!("Ok({text})"))
            }
            Value::Result(Err(inner)) => {
                let step = DisplayStep::ResultErr;
                let inner_ty = Self::display_child_ty(ty, step);
                let inner = require_live_payload(inner, "rendering an `Err`", span)?;
                let text =
                    self.display_deep(root, inner, inner_ty.as_ref(), path.child(step), span)?;
                Ok(format!("Err({text})"))
            }
            other => Ok(other.to_string()),
        }
    }

    /// One tuple/array/Vec slot, or `<moved>` for an emptied one.
    fn display_slot(
        &mut self,
        root: ExprId,
        slot: &Option<Value>,
        ty: Option<&Ty>,
        path: DisplayPath,
        span: Span,
    ) -> Result<String, RuntimeError> {
        match slot {
            Some(value) => self.display_deep(root, value, ty, path, span),
            None => Ok("<moved>".to_string()),
        }
    }

    /// DEV-089: destroy a `print`/`println` by-value argument AFTER its formatted bytes have been
    /// submitted. A no-op for the standard-formatting path (which leaves no live temp).
    fn finish_display(&mut self, arg_place: Option<Place>, span: Span) -> Result<(), RuntimeError> {
        if let Some(place) = arg_place {
            if let Ok(arg) = self.take_place(&place, span) {
                self.drop_value(arg)?;
            }
        }
        Ok(())
    }

    fn place_value(&self, place: &Place, span: Span) -> Result<&Value, RuntimeError> {
        let mut value = self
            .frames
            .get(place.frame)
            .ok_or_else(|| RuntimeError::new("dangling reference", span))?
            .values
            .get(&place.local)
            .and_then(Option::as_ref)
            .ok_or_else(|| RuntimeError::new("use of unavailable value", span))?;
        for projection in &place.projections {
            value = project(value, projection)
                .and_then(Option::as_ref)
                .ok_or_else(|| projection_failure(projection, span))?;
        }
        Ok(value)
    }

    fn place_value_mut(&mut self, place: &Place, span: Span) -> Result<&mut Value, RuntimeError> {
        let mut value = self
            .frames
            .get_mut(place.frame)
            .ok_or_else(|| RuntimeError::new("dangling reference", span))?
            .values
            .get_mut(&place.local)
            .and_then(Option::as_mut)
            .ok_or_else(|| RuntimeError::new("use of unavailable value", span))?;
        for projection in &place.projections {
            value = project_mut(value, projection)
                .and_then(Option::as_mut)
                .ok_or_else(|| projection_failure(projection, span))?;
        }
        Ok(value)
    }

    fn take_place(&mut self, place: &Place, span: Span) -> Result<Value, RuntimeError> {
        let value = self.place_value(place, span)?.clone();
        if self.value_is_copy(&value) {
            return Ok(value);
        }
        let slot = self.place_slot_mut(place, span)?;
        slot.take()
            .ok_or_else(|| RuntimeError::new("use of moved value", span))
    }

    /// **The one path by which a value is written into existing storage.** AS3 Packet 4.
    ///
    /// Three of the eleven boundaries live here — `Assignment`, `FieldWrite` and `ElementWrite` —
    /// and which one this write *is* follows from the place's last projection, not from the caller.
    /// The earlier framing assumed a field or an element could not be checked because neither has a
    /// local to key on; both do have a checker-published type, because both are named by an
    /// EXPRESSION, and `expr_types[target]` is the checker's answer for it whatever the projection
    /// depth.
    ///
    /// A missing `expr_types` entry is an `internal` invariant failure. The checker types every
    /// expression it accepts, and the only caller is a real `Assign` left-hand side — so an absent
    /// entry means the tables and the tree disagree, which is not a case to skip validation for.
    fn write_place(
        &mut self,
        place: &Place,
        value: Value,
        target: ExprId,
        span: Span,
    ) -> Result<(), RuntimeError> {
        let boundary = match place.projections.last() {
            None => RepBoundary::Assignment,
            Some(Projection::Field(_)) => RepBoundary::FieldWrite,
            // A map entry is an element of a container exactly as an indexed slot is; the two
            // differ in how the position is found, not in what the write means.
            Some(Projection::Index(_) | Projection::MapIndex(_)) => RepBoundary::ElementWrite,
            // DEV-209: a variant's payload is a COMPONENT of an aggregate, named positionally
            // rather than by identifier — the same kind of write as a struct field, and not an
            // element of a container, whose count varies at runtime.
            Some(Projection::VariantPayload(_)) => RepBoundary::FieldWrite,
        };
        let declared = self
            .tables
            .expr_types
            .get(&target)
            .cloned()
            .ok_or_else(|| {
                RuntimeError::internal(
                    format!(
                    "no published type for the target of {} — the checker types every expression \
                     it accepts, so this is a table/tree disagreement, not a write to exempt",
                    boundary.as_str()
                ),
                    span,
                )
            })?;
        self.check_value_for_ty(&declared, &value, span, boundary)?;
        let previous = self.place_slot_mut(place, span)?.replace(value);
        if let Some(previous) = previous {
            self.drop_value(previous)?;
        }
        Ok(())
    }

    fn place_slot_mut(
        &mut self,
        place: &Place,
        span: Span,
    ) -> Result<&mut Option<Value>, RuntimeError> {
        let mut slot = self
            .frames
            .get_mut(place.frame)
            .ok_or_else(|| RuntimeError::new("dangling reference", span))?
            .values
            .get_mut(&place.local)
            .ok_or_else(|| RuntimeError::new("unknown local", span))?;
        for projection in &place.projections {
            let value = slot
                .as_mut()
                .ok_or_else(|| RuntimeError::new("use of moved value", span))?;
            slot = project_mut(value, projection)
                .ok_or_else(|| RuntimeError::new("index or field out of bounds", span))?;
        }
        Ok(slot)
    }

    fn value_is_copy(&self, value: &Value) -> bool {
        match value {
            Value::Unit
            | Value::Bool(_)
            | Value::Int(_)
            | Value::Float(..)
            | Value::Char(_)
            | Value::Str(_)
            | Value::Ref(_)
            | Value::Function(_) => true,
            Value::Tuple(values) | Value::Array(values) => values
                .iter()
                .flatten()
                .all(|value| self.value_is_copy(value)),
            // Eligible AND every field/payload value is itself Copy — the value-level analog of the
            // type-level per-instance rule (an eligible `H` at `H<String>` is Move).
            Value::Struct { item, fields } => {
                self.copy_items.contains(item)
                    && fields
                        .values()
                        .flatten()
                        .all(|value| self.value_is_copy(value))
            }
            Value::Enum {
                item,
                fields,
                named,
                ..
            } => {
                self.copy_items.contains(item)
                    && fields
                        .iter()
                        .flatten()
                        .all(|value| self.value_is_copy(value))
                    && named
                        .values()
                        .flatten()
                        .all(|value| self.value_is_copy(value))
            }
            Value::Option(value) => value
                .as_deref()
                .and_then(Option::as_ref)
                .is_none_or(|value| self.value_is_copy(value)),
            Value::Result(value) => match value {
                Ok(value) | Err(value) => (**value)
                    .as_ref()
                    .is_none_or(|value| self.value_is_copy(value)),
            },
            // DEV-087 (WP-C4.7-9 corpus): a `Value::Slice` IS a shared reference — `&[T]` — and
            // shared references are `Copy` (03-Type-System; `Value::Ref` above is treated the
            // same way). Classifying it non-`Copy` made passing a slice to a function CONSUME
            // the caller's binding, so `total(shared); shared[0]` failed "use of unavailable
            // value" in the oracle while the checker accepted it and MIR ran it. Exclusive
            // (`&mut [T]`) views are not distinguished here because the interpreter's slice
            // value carries no mutability — write permission is a static property the front end
            // and the verifier enforce, exactly as for `Value::Ref`.
            Value::Slice(..) => true,
            Value::String(_)
            | Value::Vec(_)
            | Value::Boxed(_)
            | Value::Range { .. }
            | Value::CharsIter(..)
            | Value::SplitIter(..)
            | Value::VecIter(..)
            | Value::HashMap(_)
            | Value::HashSet(_)
            | Value::HashMapKeysIter(..)
            | Value::HashMapValuesIter(..)
            | Value::HashMapIter(..)
            | Value::HashSetIter(..)
            | Value::MapIter(..)
            | Value::FilterIter(..)
            | Value::Random(_)
            | Value::Ordering(_)
            | Value::IOError(_)
            | Value::File(_) => false,
        }
    }

    fn cleanup_locals(&mut self, locals: &[LocalId]) -> Result<(), RuntimeError> {
        for local in locals.iter().rev() {
            if let Some(value) = self
                .frame_mut()
                .values
                .get_mut(local)
                .and_then(Option::take)
            {
                self.drop_value(value)?;
            }
        }
        Ok(())
    }

    fn cleanup_current_frame(&mut self) -> Result<(), RuntimeError> {
        let order = self.frame().order.clone();
        self.cleanup_locals(&order)
    }

    fn drop_value(&mut self, mut value: Value) -> Result<(), RuntimeError> {
        if let Some(item) = nominal_item(&value) {
            // **A3c-D: a GENERIC `Drop` implementation is refused, not guessed at.**
            //
            // Destruction reaches here with a `Value` and recovers the nominal through
            // `nominal_item`, so `Wrapper<String>` and `Wrapper<Int32>` are indistinguishable — the
            // type arguments that selected the impl are gone. Every way of proceeding is a guess:
            // an empty generic frame, inference from the runtime fields, or scanning impls and
            // hoping. A destructor is the last place to guess, because running the wrong one or
            // running one with the wrong parameter bindings corrupts silently.
            //
            // Refused BEFORE the body executes, so no side effect happens at all — a partially run
            // destructor is worse than none. Classified `internal` because it is a limitation of
            // this engine, not a property of the program: MIR and native retain the arguments and
            // execute it correctly, so calling it a language trap would tell the differential
            // harness the program is at fault.
            //
            // Recorded rather than repaired (DEV-176's ledger entry): threading a concrete `Ty`
            // through 44 `drop_value` call sites, or retaining type arguments in `Value`, is
            // disproportionate to 0 `Drop` impls in the first-party packages and 2 generic-`Drop`
            // fixtures in the whole corpus. The refusal is the signal to build it when that changes.
            if self.drop_impl_is_generic(item) {
                return Err(RuntimeError::limitation(
                    "the HIR oracle cannot execute a generic `Drop` implementation: destruction \
                     retains no concrete nominal type arguments, so the destructor's generic \
                     parameters cannot be bound (DEV-176, A3c-D)",
                    self.hir.item(item).span,
                    OracleLimitation::GenericDrop,
                ));
            }
            if let Some(callable) = self.find_drop(item) {
                // **Packet 1: the destructor body goes through the invocation authority.**
                //
                // This was the THIRD body-execution funnel — its own frame push, `eval_block` and
                // pop, alongside `execute_body` and `call_user_method`. A destructor running with
                // the wrong environment is the DEV-176 shape exactly, so it is the last funnel that
                // should have had its own copy.
                //
                // **The environment is provably `Empty`, not assumed.** A generic `Drop` impl is
                // refused immediately above (DEV-176/A3c-D), so a destructor body only ever runs
                // for a non-generic impl, which has no parameters to bind.
                //
                // The receiver is MOVED in, not cloned: `Drop::drop(&mut self)` may mutate or
                // replace fields, and the recursive field destruction below must see that. The
                // `Destructor` epilogue hands it back.
                if callable.receiver.is_none() {
                    return Err(RuntimeError::internal(
                        "a `Drop::drop` implementation without a receiver reached destruction",
                        self.hir.item(item).span,
                    ));
                }
                // The value is MOVED into backing storage by the authority, and the body's `self`
                // becomes a genuine `&mut Self` reference to it — so the published receiver type
                // and the runtime binding agree without a `Drop`-shaped exemption.
                let dtor_span = self.hir.item(item).span;
                value = self.invoke_with_epilogue(
                    ResolvedInvocation {
                        callable,
                        environment: InvocationEnv::Empty,
                    },
                    ReceiverSource::OwnedForDrop(value),
                    Vec::new(),
                    BodyEpilogue::Destructor,
                    dtor_span,
                )?;
            }
        }
        match &mut value {
            Value::Tuple(values) | Value::Array(values) | Value::Vec(values) => {
                for child in values.iter_mut().rev().filter_map(Option::take) {
                    self.drop_value(child)?;
                }
            }
            // WP-C2.2 (DEV-029): named fields drop in REVERSE DECLARATION order per
            // 05-Memory-Model.md "Drop Order" (made explicit for fields under CD-011).
            // The `BTreeMap` representation iterates alphabetically, so the declaration
            // order is recovered from the HIR item; any field name the HIR doesn't list
            // (unreachable for well-typed values) falls back to map order afterwards.
            Value::Struct { item, fields } => {
                let item = *item;
                let order = self.declared_field_order(item, None);
                let mut fields = std::mem::take(fields);
                for name in order.iter().rev() {
                    if let Some(child) = fields.remove(name).flatten() {
                        self.drop_value(child)?;
                    }
                }
                for child in fields.into_values().flatten() {
                    self.drop_value(child)?;
                }
            }
            Value::Enum {
                item,
                variant,
                fields,
                named,
            } => {
                let (item, variant) = (*item, *variant);
                for child in fields.iter_mut().rev().filter_map(Option::take) {
                    self.drop_value(child)?;
                }
                let order = self.declared_field_order(item, Some(variant));
                let mut named = std::mem::take(named);
                for name in order.iter().rev() {
                    if let Some(child) = named.remove(name).flatten() {
                        self.drop_value(child)?;
                    }
                }
                for child in named.into_values().flatten() {
                    self.drop_value(child)?;
                }
            }
            Value::Boxed(child) => {
                if let Some(child) = child.take() {
                    self.drop_value(child)?;
                }
            }
            Value::Option(child) => {
                if let Some(mut child) = child.take() {
                    if let Some(child) = child.take() {
                        self.drop_value(child)?;
                    }
                }
            }
            Value::Result(result) => {
                match std::mem::replace(result, Ok(Box::new(Some(Value::Unit)))) {
                    Ok(mut child) | Err(mut child) => {
                        if let Some(child) = child.take() {
                            self.drop_value(child)?;
                        }
                    }
                }
            }
            Value::HashMap(map) => {
                for (key, child) in std::mem::take(&mut map.0).into_iter().rev() {
                    if let Some(child) = child {
                        self.drop_value(child)?;
                    }
                    self.drop_value(key)?;
                }
            }
            Value::HashSet(set) => {
                for child in std::mem::take(&mut set.0).into_iter().rev() {
                    self.drop_value(child)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// WP-C2.2 (DEV-029): a struct's (or enum struct-like variant's) field names in source
    /// declaration order, recovered from the HIR item — the runtime `BTreeMap` representation
    /// only preserves alphabetical order.
    fn declared_field_order(&self, item: ItemId, variant: Option<u32>) -> Vec<String> {
        match (&self.hir.item(item).kind, variant) {
            (hir::ItemKind::Struct { fields, .. }, None) => fields
                .iter()
                .map(|field| self.text(field.name).to_string())
                .collect(),
            (hir::ItemKind::Enum { variants, .. }, Some(index)) => variants
                .get(index as usize)
                .map(|variant| match &variant.kind {
                    hir::VariantKind::Struct(fields) => fields
                        .iter()
                        .map(|field| self.text(field.name).to_string())
                        .collect(),
                    _ => Vec::new(),
                })
                .unwrap_or_default(),
            _ => Vec::new(),
        }
    }

    /// Whether the `Drop` implementation selected for `item` declares generic parameters.
    ///
    /// Matched the same way `find_drop` matches, so the two cannot disagree about which impl is in
    /// question — a check that looked at a different impl than the one about to run would be worse
    /// than no check.
    fn drop_impl_is_generic(&self, item: ItemId) -> bool {
        self.hir.items.iter().enumerate().any(|(idx, candidate)| {
            let hir::ItemKind::Impl {
                generics,
                trait_: Some(reference),
                self_ty,
                ..
            } = &candidate.kind
            else {
                return false;
            };
            let _ = idx;
            reference.res == Res::CoreTrait(hir::CoreTrait::Drop)
                && matches!(&self.hir.ty(*self_ty).kind,
                    hir::TypeKind::Path { res: Res::Item(actual), .. } if *actual == item)
                && !generics.is_empty()
        })
    }

    fn find_drop(&self, item: ItemId) -> Option<Callable> {
        // DEV-069: a `Drop` impl may live in a different file from the type's user; the
        // destructor's method name and body both belong to the impl's own file.
        self.hir.items.iter().enumerate().find_map(|(idx, candidate)| {
            let impl_id = ItemId(idx as u32);
            let hir::ItemKind::Impl { trait_: Some(reference), self_ty, items, .. } = &candidate.kind else { return None; };
            if reference.res != Res::CoreTrait(hir::CoreTrait::Drop) || !matches!(&self.hir.ty(*self_ty).kind, hir::TypeKind::Path { res: Res::Item(actual), .. } if *actual == item) { return None; }
            items.iter().find_map(|item| match item {
                hir::ImplItem::Fn { def, .. } if self.item_text(impl_id, def.sig.name) == "drop" => Some(Callable { receiver: def.sig.receiver.zip(def.sig.receiver_local), params: def.sig.params.iter().map(|param| param.local).collect(), body: def.body }),
                _ => None,
            })
        })
    }

    fn find_local_by_name(&self, name: &str) -> Option<LocalId> {
        self.frame().values.keys().copied().find(|local| {
            self.tables.local_types.contains_key(local)
                && self.hir.exprs.iter().any(|expr| matches!(&expr.kind, hir::ExprKind::Path { path, res: Res::Local(found), .. } if found == local && path.segments.last().is_some_and(|segment| self.text(segment.span) == name)))
        })
    }

    fn frame(&self) -> &Frame {
        self.frames.last().expect("runtime frame exists")
    }
    fn frame_mut(&mut self) -> &mut Frame {
        self.frames.last_mut().expect("runtime frame exists")
    }
}

/// WP-C2.2 (DEV-035): rewrite every `Place` reachable inside `value` that points at
/// `(popped_frame, receiver_local)` — i.e. at the `self` slot of a method call frame that is
/// about to become (or already is) invalid — so it points at the caller-side receiver place
/// instead, with any projections taken inside the method appended after the receiver place's
/// own. Places into *other* locals of the popped frame are deliberately left untouched: the
/// borrow checker rejects returning references to method-body locals (E0103), and the runtime
/// "dangling reference" trap remains the correct backstop for anything that slips through.
/// `BTreeMap`/`BTreeSet` *keys* are not rewritten (they cannot be mutated in place without
/// breaking the container's ordering invariant); a key containing a frame-local reference is
/// not constructible from well-typed STARK source, and the dangling-reference backstop covers
/// it regardless.
fn rebase_frame_refs(
    value: &mut Value,
    popped_frame: usize,
    receiver_local: LocalId,
    receiver_place: &Place,
) {
    let rebase_place = |place: &mut Place| {
        if place.frame == popped_frame && place.local == receiver_local {
            let mut rebased = receiver_place.clone();
            rebased.projections.append(&mut place.projections);
            *place = rebased;
        }
    };
    match value {
        Value::Ref(place) => rebase_place(place),
        Value::VecIter(place, _) => rebase_place(place),
        Value::Tuple(items)
        | Value::Array(items)
        | Value::Vec(items)
        | Value::HashMapKeysIter(items, _)
        | Value::HashMapValuesIter(items, _)
        | Value::HashMapIter(items, _)
        | Value::HashSetIter(items, _) => {
            for item in items.iter_mut().flatten() {
                rebase_frame_refs(item, popped_frame, receiver_local, receiver_place);
            }
        }
        Value::Enum { fields, named, .. } => {
            for field in fields.iter_mut().flatten() {
                rebase_frame_refs(field, popped_frame, receiver_local, receiver_place);
            }
            for field in named.values_mut().flatten() {
                rebase_frame_refs(field, popped_frame, receiver_local, receiver_place);
            }
        }
        Value::Struct { fields, .. } => {
            for field in fields.values_mut().flatten() {
                rebase_frame_refs(field, popped_frame, receiver_local, receiver_place);
            }
        }
        Value::Boxed(inner) => {
            if let Some(inner) = inner.as_mut() {
                rebase_frame_refs(inner, popped_frame, receiver_local, receiver_place);
            }
        }
        Value::Option(Some(inner)) => {
            if let Some(inner) = inner.as_mut() {
                rebase_frame_refs(inner, popped_frame, receiver_local, receiver_place);
            }
        }
        Value::Result(Ok(inner)) | Value::Result(Err(inner)) => {
            if let Some(inner) = inner.as_mut() {
                rebase_frame_refs(inner, popped_frame, receiver_local, receiver_place);
            }
        }
        Value::MapIter(inner, _) | Value::FilterIter(inner, _) => {
            rebase_frame_refs(inner, popped_frame, receiver_local, receiver_place);
        }
        Value::Slice(place, _, _) => {
            if place.frame == popped_frame && place.local == receiver_local {
                let mut rebased = receiver_place.clone();
                rebased.projections.extend(place.projections.clone());
                *place = rebased;
            }
        }
        Value::HashMap(map) => {
            for entry in map.values_mut().flatten() {
                rebase_frame_refs(entry, popped_frame, receiver_local, receiver_place);
            }
        }
        _ => {}
    }
}

fn project<'a>(value: &'a Value, projection: &Projection) -> Option<&'a Option<Value>> {
    match (value, projection) {
        // DEV-209: the prelude payload, now a slot like every other component.
        (Value::Option(Some(slot)), Projection::VariantPayload(0)) => Some(slot),
        (Value::Result(Ok(slot)), Projection::VariantPayload(0)) => Some(slot),
        (Value::Result(Err(slot)), Projection::VariantPayload(1)) => Some(slot),
        (Value::Struct { fields, .. }, Projection::Field(name))
        | (Value::Enum { named: fields, .. }, Projection::Field(name)) => fields.get(name),
        // WP-C7.9 Packet C: a tuple-variant payload is slot-backed like any other component, but
        // nothing named it — `Index` reached tuples, arrays and `Vec`s only, so an enum payload
        // could not be borrowed in place and PAT-BIND-001 had no storage to point at.
        (Value::Enum { fields, .. }, Projection::Index(index)) => fields.get(*index),
        (
            Value::Tuple(values)
            | Value::Array(values)
            | Value::Vec(values)
            | Value::HashMapKeysIter(values, _)
            | Value::HashMapValuesIter(values, _)
            | Value::HashMapIter(values, _)
            | Value::HashSetIter(values, _),
            Projection::Index(index),
        ) => values.get(*index),
        (Value::HashMap(map), Projection::MapIndex(index)) => {
            map.0.get(*index).map(|(_, value)| value)
        }
        _ => None,
    }
}

fn project_mut<'a>(value: &'a mut Value, projection: &Projection) -> Option<&'a mut Option<Value>> {
    match (value, projection) {
        // DEV-209, matching `project`.
        (Value::Option(Some(slot)), Projection::VariantPayload(0)) => Some(slot),
        (Value::Result(Ok(slot)), Projection::VariantPayload(0)) => Some(slot),
        (Value::Result(Err(slot)), Projection::VariantPayload(1)) => Some(slot),
        (Value::Struct { fields, .. }, Projection::Field(name))
        | (Value::Enum { named: fields, .. }, Projection::Field(name)) => fields.get_mut(name),
        // WP-C7.9 Packet C, matching `project`: the positional payload of a tuple variant.
        (Value::Enum { fields, .. }, Projection::Index(index)) => fields.get_mut(*index),
        (
            Value::Tuple(values)
            | Value::Array(values)
            | Value::Vec(values)
            | Value::HashMapKeysIter(values, _)
            | Value::HashMapValuesIter(values, _)
            | Value::HashMapIter(values, _)
            | Value::HashSetIter(values, _),
            Projection::Index(index),
        ) => values.get_mut(*index),
        (Value::HashMap(map), Projection::MapIndex(index)) => {
            map.0.get_mut(*index).map(|(_, value)| value)
        }
        _ => None,
    }
}

/// The nominal an associated-type projection's base selects. `nominal_item` answers the same
/// question for a runtime `Value`; this one answers it for a `Ty`, which is what a projection has.
/// References are looked through — `&H::Item` projects through `H` — and anything without a
/// nominal (a primitive, a tuple, a core container) implements no user trait with an associated
/// type, so `None` is the honest answer rather than a guess.
fn nominal_item_of_ty(ty: &Ty) -> Option<ItemId> {
    match ty {
        Ty::Struct(item, _) | Ty::Enum(item, _) => Some(*item),
        Ty::Ref { inner, .. } => nominal_item_of_ty(inner),
        _ => None,
    }
}

fn nominal_item(value: &Value) -> Option<ItemId> {
    match value {
        Value::Struct { item, .. } | Value::Enum { item, .. } => Some(*item),
        _ => None,
    }
}

fn assign_binop(op: AssignOp) -> BinOp {
    match op {
        AssignOp::Assign => unreachable!(),
        AssignOp::AddAssign => BinOp::Add,
        AssignOp::SubAssign => BinOp::Sub,
        AssignOp::MulAssign => BinOp::Mul,
        AssignOp::DivAssign => BinOp::Div,
        AssignOp::RemAssign => BinOp::Rem,
        AssignOp::PowAssign => BinOp::Pow,
        AssignOp::BitAndAssign => BinOp::BitAnd,
        AssignOp::BitOrAssign => BinOp::BitOr,
        AssignOp::BitXorAssign => BinOp::BitXor,
        AssignOp::ShlAssign => BinOp::Shl,
        AssignOp::ShrAssign => BinOp::Shr,
    }
}

fn is_integer(primitive: Primitive) -> bool {
    matches!(
        primitive,
        Primitive::Int8
            | Primitive::Int16
            | Primitive::Int32
            | Primitive::Int64
            | Primitive::UInt8
            | Primitive::UInt16
            | Primitive::UInt32
            | Primitive::UInt64
    )
}

fn signed_integer_min(ty: Option<&Ty>) -> Option<i128> {
    match ty {
        Some(Ty::Primitive(Primitive::Int8)) => Some(i8::MIN as i128),
        Some(Ty::Primitive(Primitive::Int16)) => Some(i16::MIN as i128),
        Some(Ty::Primitive(Primitive::Int32)) => Some(i32::MIN as i128),
        Some(Ty::Primitive(Primitive::Int64)) => Some(i64::MIN as i128),
        _ => None,
    }
}

fn integer_width(ty: Option<&Ty>) -> Option<u32> {
    match ty {
        Some(Ty::Primitive(Primitive::Int8 | Primitive::UInt8)) => Some(8),
        Some(Ty::Primitive(Primitive::Int16 | Primitive::UInt16)) => Some(16),
        Some(Ty::Primitive(Primitive::Int32 | Primitive::UInt32)) => Some(32),
        Some(Ty::Primitive(Primitive::Int64 | Primitive::UInt64)) => Some(64),
        _ => None,
    }
}

fn is_iterator_ty(ty: &Ty) -> bool {
    match ty {
        Ty::Ref { inner, .. } => is_iterator_ty(inner),
        Ty::Core(
            CoreType::CharsIter
            | CoreType::SplitIter
            | CoreType::VecIter
            | CoreType::KeysIter
            | CoreType::ValuesIter
            | CoreType::Iter
            | CoreType::MapIter
            | CoreType::FilterIter,
            _,
        ) => true,
        _ => false,
    }
}

fn is_float(primitive: Primitive) -> bool {
    matches!(primitive, Primitive::Float32 | Primitive::Float64)
}

/// Canonical Core v1 Float64 text (shortest round-trip digits, always with a decimal point or
/// exponent; NaN/inf/-0.0 spellings per the frozen numeric contract). `pub` so the MIR
/// interpreter's runtime surface (mir::interp) formats floats IDENTICALLY to this oracle —
/// one algorithm, no drift (WP-C4.4).
// WP-C6.3e: the canonical float renderers now live in `stark_runtime::format`, shared with the
// native backend so the two engines format floats identically by construction. These delegate.
pub fn canonical_float(value: f64) -> String {
    stark_runtime::format::canonical_float(value)
}

// Canonical display must use the shortest decimal representation that round-trips to the
// *declared* IEEE type. `Value::Float` carries its own `FloatWidth` tag, so both the top-level
// `println`/`.fmt()` paths and the generic recursive `Display for Value` impl (reached when a
// Float32 is nested inside a tuple/array/struct/collection) route through this for a Float32
// value instead of `canonical_float`'s `f64` shortest-round-trip digits (which would otherwise
// produce e.g. `0.10000000149011612` for `0.1f32` instead of the shorter, correct `0.1`).
pub fn canonical_float32(value: f32) -> String {
    stark_runtime::format::canonical_float32(value)
}

/// Correction-brief Issue 4: `NUM-FLOAT-OP-001` requires every primitive operation that produces
/// a NaN result to yield "the canonical quiet NaN with sign zero and all payload bits other than
/// the quiet bit zero" -- a specific, fixed bit pattern, not merely "some NaN." A platform's
/// native NaN-producing instructions are not guaranteed to agree on sign or payload bits (IEEE
/// 754 only mandates the exponent field and that the quiet bit distinguishes quiet from
/// signaling), so this must be forced explicitly rather than trusted to fall out of `f64`/`f32`
/// arithmetic. `f32::from_bits(0x7fc0_0000)`/`f64::from_bits(0x7ff8_0000_0000_0000)` are already
/// exactly Rust's own `f32::NAN`/`f64::NAN` constants -- spelled out as literal bit patterns here
/// so the canonicalization is explicit and self-documenting rather than relying on a constant
/// whose bit pattern isn't visible at the call site. The one normative exception is unary
/// negation ("Negation flips the sign bit, including for zero and NaN"): callers that implement
/// `-x` must NOT route through this, since it must flip whatever sign bit the operand already
/// had rather than forcing sign zero.
fn canonical_nan_bits(width: FloatWidth) -> f64 {
    match width {
        FloatWidth::F32 => f64::from(f32::from_bits(0x7fc0_0000)),
        FloatWidth::F64 => f64::from_bits(0x7ff8_0000_0000_0000),
    }
}

/// Forces `value` to the canonical quiet NaN for `width` if it is any NaN at all, leaving every
/// other value (including infinities and signed zero) untouched. See `canonical_nan_bits`.
fn canonicalize_nan(value: f64, width: FloatWidth) -> f64 {
    if value.is_nan() {
        canonical_nan_bits(width)
    } else {
        value
    }
}

/// Applies `canonicalize_nan` to a `Value::Float` produced by a primitive operation or standard
/// math builtin, leaving every other `Value` (including a propagated `RuntimeError`) untouched.
/// Every call site that constructs a `Value::Float` result from a computation that can produce
/// NaN routes through this -- *except* unary negation, per `canonical_nan_bits`'s doc comment.
fn canonicalize_float_result(result: Result<Value, RuntimeError>) -> Result<Value, RuntimeError> {
    result.map(|value| match value {
        Value::Float(inner, width) => Value::Float(canonicalize_nan(inner, width), width),
        other => other,
    })
}

fn standard_hash(
    value: &Value,
    ty: &Ty,
    entry_source: crate::source::SourceId,
) -> Result<u64, RuntimeError> {
    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fn frame(bytes: &mut Vec<u8>, component: Vec<u8>) {
        push_u64(bytes, component.len() as u64);
        bytes.extend(component);
    }
    fn encode(value: &Value, ty: &Ty) -> Option<Vec<u8>> {
        if let Ty::Ref { inner, .. } = ty {
            return encode(value, inner);
        }
        match (value, ty) {
            (Value::Unit, Ty::Primitive(Primitive::Unit)) => Some(Vec::new()),
            (Value::Bool(value), Ty::Primitive(Primitive::Bool)) => Some(vec![u8::from(*value)]),
            (Value::Char(value), Ty::Primitive(Primitive::Char)) => {
                Some((*value as u32).to_le_bytes().to_vec())
            }
            (
                Value::String(value) | Value::Str(value),
                Ty::Primitive(Primitive::String | Primitive::Str),
            ) => Some(value.as_bytes().to_vec()),
            (Value::Int(value), Ty::Primitive(primitive)) => {
                let bytes = match primitive {
                    Primitive::Int8 | Primitive::UInt8 => vec![*value as u8],
                    Primitive::Int16 | Primitive::UInt16 => (*value as u16).to_le_bytes().to_vec(),
                    Primitive::Int32 | Primitive::UInt32 => (*value as u32).to_le_bytes().to_vec(),
                    Primitive::Int64 | Primitive::UInt64 => (*value as u64).to_le_bytes().to_vec(),
                    _ => return None,
                };
                Some(bytes)
            }
            (Value::Tuple(values), Ty::Tuple(types)) if values.len() == types.len() => {
                let mut bytes = vec![0x02];
                push_u64(&mut bytes, values.len() as u64);
                for (slot, ty) in values.iter().zip(types) {
                    frame(&mut bytes, encode(slot.as_ref()?, ty)?);
                }
                Some(bytes)
            }
            (Value::Array(values), Ty::Array(element, _)) => {
                let mut bytes = vec![0x01];
                push_u64(&mut bytes, values.len() as u64);
                for slot in values {
                    frame(&mut bytes, encode(slot.as_ref()?, element)?);
                }
                Some(bytes)
            }
            (Value::Vec(values), Ty::Core(hir::CoreType::Vec, types)) if types.len() == 1 => {
                let mut bytes = vec![0x03];
                push_u64(&mut bytes, values.len() as u64);
                for slot in values {
                    frame(&mut bytes, encode(slot.as_ref()?, &types[0])?);
                }
                Some(bytes)
            }
            (Value::Option(value), Ty::Core(hir::CoreType::Option, types)) if types.len() == 1 => {
                let mut bytes = vec![0x04, u8::from(value.is_some())];
                if let Some(value) = value.as_deref().and_then(Option::as_ref) {
                    frame(&mut bytes, encode(value, &types[0])?);
                }
                Some(bytes)
            }
            (Value::Result(value), Ty::Core(hir::CoreType::Result, types)) if types.len() == 2 => {
                let mut bytes = vec![0x05, u8::from(value.is_err())];
                match value
                    .as_ref()
                    .map(|v| (**v).as_ref())
                    .map_err(|v| (**v).as_ref())
                {
                    Ok(Some(value)) => frame(&mut bytes, encode(value, &types[0])?),
                    Err(Some(value)) => frame(&mut bytes, encode(value, &types[1])?),
                    Ok(None) | Err(None) => return None,
                }
                Some(bytes)
            }
            _ => None,
        }
    }

    let bytes = encode(value, ty).ok_or_else(|| {
        RuntimeError::new(
            "type has no standard Hash implementation",
            Span::synthetic(entry_source),
        )
    })?;
    let mut hash = 14_695_981_039_346_656_037u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(1_099_511_628_211);
    }
    Ok(hash)
}

fn usize_arg(value: Option<Value>, span: Span) -> Result<usize, RuntimeError> {
    match value {
        Some(Value::Int(value)) => usize::try_from(value)
            .map_err(|_| RuntimeError::new("integer does not fit usize", span)),
        _ => Err(RuntimeError::new("expected integer argument", span)),
    }
}

fn u64_arg(value: Option<Value>, span: Span) -> Result<u64, RuntimeError> {
    match value {
        Some(Value::Int(value)) => {
            u64::try_from(value).map_err(|_| RuntimeError::new("integer does not fit u64", span))
        }
        _ => Err(RuntimeError::new("expected integer argument", span)),
    }
}

fn u32_arg(value: Option<Value>, span: Span) -> Result<u32, RuntimeError> {
    match value {
        Some(Value::Int(value)) => {
            u32::try_from(value).map_err(|_| RuntimeError::new("integer does not fit u32", span))
        }
        _ => Err(RuntimeError::new("expected integer argument", span)),
    }
}

fn float_arg(value: Option<Value>, span: Span) -> Result<f64, RuntimeError> {
    match value {
        Some(Value::Float(value, _)) => Ok(value),
        _ => Err(RuntimeError::new("expected Float64 argument", span)),
    }
}

/// Numeric comparison for `math::min`/`math::max`/`clamp` (`T: Ord`, Int or
/// Float only — a narrower runtime scope than the unconstrained type
/// variable these builtins get in `typecheck/body.rs`; see
/// `docs/PHASE8_GRAMMAR_GAPS.md`'s note on `assert_eq`/`Eq` for the same
/// pattern elsewhere).
fn numeric_cmp(
    a: &Option<Value>,
    b: &Option<Value>,
    span: Span,
) -> Result<std::cmp::Ordering, RuntimeError> {
    match (a, b) {
        (Some(Value::Int(a)), Some(Value::Int(b))) => Ok(a.cmp(b)),
        (Some(Value::Float(a, _)), Some(Value::Float(b, _))) => Ok(a.total_cmp(b)),
        _ => Err(RuntimeError::new(
            "expected two Int or two Float arguments",
            span,
        )),
    }
}

/// The text of a string-ish value, whichever representation carries it.
///
/// **DEV-129.** `Value::Str` and `Value::String` are the same text in two wrappers, and `==` on
/// `Value` distinguishes them. A `&str` scrutinee matched against a string literal must compare
/// CONTENT — `match s.as_str() { "beta" => .. }` is normative — so the comparison cannot be
/// variant-sensitive. It got away with being so only while `as_str` returned a `Value::Str` clone;
/// once DEV-126 made it a reference to the owning `Value::String`, every arm missed.
fn string_text(value: &Value) -> Option<&str> {
    match value {
        Value::Str(text) | Value::String(text) => Some(text.as_str()),
        _ => None,
    }
}

/// **DEV-130: structural equality with the one representation-insensitivity STARK requires.**
///
/// `Value` derives `PartialEq`, so `Str("a") != String("a")`. That is a representation difference
/// the language does not have: both are the text `a`, and `06-Standard-Library.md` gives `&str` and
/// `String` content equality.
///
/// The rule had been written inline at ONE site — the `==` operator — and omitted at the other
/// three, which is why `s.as_str() == "beta"` was true while `assert_eq(s.as_str(), "beta")` failed
/// with `left: beta, right: beta`: two values that print identically, declared unequal. Rather than
/// patch each site, this is now the single comparison every structural equality routes through, the
/// same correction DEV-128 made for `is_copy`.
///
/// Recursion into containers is deliberate: `Some(s.as_str())` against `Some("x")` compares
/// payloads, and a flat rule would report them unequal for the same reason.
///
/// `Ref` is NOT followed here — callers deref first, because following a place needs `&self` and
/// because a caller that has not deref'd has a bug this function should not hide.
fn values_equal(left: &Value, right: &Value) -> bool {
    if let (Some(left), Some(right)) = (string_text(left), string_text(right)) {
        return left == right;
    }
    let slots_equal = |left: &[Option<Value>], right: &[Option<Value>]| {
        left.len() == right.len()
            && left.iter().zip(right).all(|pair| match pair {
                (Some(left), Some(right)) => values_equal(left, right),
                (None, None) => true,
                _ => false,
            })
    };
    match (left, right) {
        (Value::Tuple(left), Value::Tuple(right))
        | (Value::Array(left), Value::Array(right))
        | (Value::Vec(left), Value::Vec(right)) => slots_equal(left, right),
        (Value::Option(left), Value::Option(right)) => match (left, right) {
            (Some(left), Some(right)) => match (left.as_ref(), right.as_ref()) {
                (Some(left), Some(right)) => values_equal(left, right),
                (None, None) => true,
                _ => false,
            },
            (None, None) => true,
            _ => false,
        },
        (Value::Result(left), Value::Result(right)) => match (left, right) {
            (Ok(left), Ok(right)) | (Err(left), Err(right)) => {
                match (left.as_ref(), right.as_ref()) {
                    (Some(left), Some(right)) => values_equal(left, right),
                    (None, None) => true,
                    _ => false,
                }
            }
            _ => false,
        },
        (Value::Boxed(left), Value::Boxed(right)) => match (left.as_ref(), right.as_ref()) {
            (Some(left), Some(right)) => values_equal(left, right),
            (None, None) => true,
            _ => false,
        },
        (
            Value::Struct {
                item: left_item,
                fields: left_fields,
            },
            Value::Struct {
                item: right_item,
                fields: right_fields,
            },
        ) => {
            left_item == right_item
                && left_fields.len() == right_fields.len()
                && left_fields.iter().all(|(name, left)| {
                    right_fields
                        .get(name)
                        .is_some_and(|right| match (left, right) {
                            (Some(left), Some(right)) => values_equal(left, right),
                            (None, None) => true,
                            _ => false,
                        })
                })
        }
        (
            Value::Enum {
                item: left_item,
                variant: left_variant,
                fields: left_fields,
                named: left_named,
            },
            Value::Enum {
                item: right_item,
                variant: right_variant,
                fields: right_fields,
                named: right_named,
            },
        ) => {
            left_item == right_item
                && left_variant == right_variant
                && slots_equal(left_fields, right_fields)
                && left_named.len() == right_named.len()
                && left_named.iter().all(|(name, left)| {
                    right_named
                        .get(name)
                        .is_some_and(|right| match (left, right) {
                            (Some(left), Some(right)) => values_equal(left, right),
                            (None, None) => true,
                            _ => false,
                        })
                })
        }
        _ => left == right,
    }
}

fn string_arg(value: Option<Value>, span: Span) -> Result<String, RuntimeError> {
    match value {
        Some(Value::Str(value) | Value::String(value)) => Ok(value),
        _ => Err(RuntimeError::new("expected string argument", span)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{parse, ParseMode};
    use crate::resolve::resolve;
    use crate::source::SourceFile;
    use crate::typecheck;
    use std::sync::Arc;

    /// Type-check only, returning the diagnostics — for tests that assert a REJECTION rather
    /// than an execution result.
    fn type_diagnostics(source: &str) -> Vec<crate::diag::Diagnostic> {
        let file = Arc::new(SourceFile::new("test.stark", source));
        let (ast, _) = parse(&file, ParseMode::Program);
        let (hir, _) = resolve(&ast, file.clone());
        typecheck::analyze(&hir).diagnostics
    }

    fn stark_string_literal_contents(value: &str) -> String {
        let mut escaped = String::with_capacity(value.len());
        for character in value.chars() {
            match character {
                '\\' => escaped.push_str("\\\\"),
                '"' => escaped.push_str("\\\""),
                '\n' => escaped.push_str("\\n"),
                '\r' => escaped.push_str("\\r"),
                '\t' => escaped.push_str("\\t"),
                _ => escaped.push(character),
            }
        }
        escaped
    }

    fn portable_filename_component(value: &str) -> String {
        value
            .chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                    character
                } else {
                    '_'
                }
            })
            .collect()
    }

    // ------------------------------------------ WP-VALUE-REP-TOTAL A3: propagation ownership --
    /// The first function item in a probe program.
    fn first_fn(hir: &Hir) -> ItemId {
        (0..hir.items.len())
            .map(|index| ItemId(index as u32))
            .find(|item| matches!(&hir.item(*item).kind, hir::ItemKind::Fn(_)))
            .expect("the probe declares a function")
    }

    /// **A pending propagation may never cross a callable boundary.**
    ///
    /// `pending_propagation` is interpreter state rather than frame state, and `expect_value` parks
    /// a value there while handing the caller a dummy `Value::Unit` to consume immediately. It is
    /// therefore an intra-expression adapter: a value still parked when a call begins would be
    /// attributed to the wrong function — read against the callee's return type on the way in, or
    /// the caller's on the way out.
    ///
    /// This is what makes A4's "validate `Flow::Propagate` against `callable.ret`" sound rather
    /// than assumed, which is why it lands before the wiring. The state is injected because no
    /// correct program can produce it — the invariant's whole claim is that it does not happen.
    #[test]
    fn a_pending_propagation_may_not_enter_a_callable_boundary() {
        let (hir, file, tables) = relation_probe("fn helper() -> Int32 { 1 }\nfn main() {}");
        let mut interp = Interpreter::new(&hir, file, &tables);
        interp.frames.push(Frame::default());

        let callable = interp
            .item_callable(first_fn(&hir))
            .expect("the probe declares a callable function");

        interp.pending_propagation = Some(Value::Int(7));
        let error = interp
            .invoke_callable(
                ResolvedInvocation {
                    callable,
                    environment: InvocationEnv::Empty,
                },
                ReceiverSource::None,
                Vec::new(),
                interp.file.synthetic_span(),
            )
            .err()
            .expect("a parked propagation must not cross into a call");

        assert_eq!(error.class, FailureClass::InternalInvariant);
        assert!(
            error.message.contains("pending propagation entered"),
            "{}",
            error.message
        );
    }

    /// The invariant holds for ORDINARY calls, which is the half that keeps it from being satisfied
    /// by refusing everything: a call with nothing parked must still run.
    #[test]
    fn an_ordinary_call_crosses_the_boundary_with_nothing_pending() {
        let (hir, file, tables) = relation_probe("fn main() {}");
        let mut interp = Interpreter::new(&hir, file, &tables);
        interp.frames.push(Frame::default());

        let callable = interp
            .item_callable(first_fn(&hir))
            .expect("the probe declares a callable function");

        assert!(interp.pending_propagation.is_none());
        assert!(interp
            .invoke_callable(
                ResolvedInvocation {
                    callable,
                    environment: InvocationEnv::Empty,
                },
                ReceiverSource::None,
                Vec::new(),
                interp.file.synthetic_span(),
            )
            .is_ok());
        assert!(
            interp.pending_propagation.is_none(),
            "nothing may be left parked after a call returns"
        );
    }

    // ------------------------------------------------ WP-VALUE-REP-TOTAL A2: the relation --

    /// An interpreter over a trivial program, for exercising the relation directly.
    ///
    /// The relation is a pure function of a type and a value, so it does not need a running
    /// program — but it DOES need the `Copy` set, which is computed from the HIR, so a real
    /// interpreter is cheaper than faking one.
    fn relation_probe(source: &str) -> (Hir, crate::source::RegisteredSource, TypeTables) {
        let file = Arc::new(SourceFile::new("test.stark", source));
        let (ast, _) = parse(&file, ParseMode::Program);
        let (hir, _) = resolve(&ast, file.clone());
        let tables = typecheck::analyze(&hir).tables;
        // AS1b-ii: the identity the parse registered, not a fresh one.
        let registered = hir
            .source_named(&file.name)
            .expect("the parse registered this file");
        (hir, registered, tables)
    }

    /// A place standing for "somewhere in the frame". The relation only ever asks a value's SHAPE,
    /// so where the place points is irrelevant here — and constructing one this way keeps the tests
    /// from depending on frame layout.
    fn probe_place() -> Place {
        Place {
            frame: 0,
            local: LocalId(0),
            projections: Vec::new(),
        }
    }

    fn str_ref() -> Ty {
        Ty::Ref {
            mutable: false,
            inner: Box::new(Ty::Primitive(crate::ast::Primitive::Str)),
        }
    }

    fn byte_slice_ref() -> Ty {
        Ty::Ref {
            mutable: false,
            inner: Box::new(Ty::Slice(Box::new(Ty::Primitive(
                crate::ast::Primitive::UInt8,
            )))),
        }
    }

    /// **The DEV-121 pairing, both directions.** `&str` may be a detached view or a reference to
    /// text; it may NOT be owned storage. That last row is the whole defect class: the static type
    /// says borrowed while runtime move behaviour sees owned storage, so passing it consumes what
    /// it only borrows.
    #[test]
    fn a_borrowed_text_type_permits_a_view_and_refuses_owned_storage() {
        let (hir, file, tables) = relation_probe("fn main() {}");
        let probe_span = file.synthetic_span();
        let interp = Interpreter::new(&hir, file, &tables);
        let _ = probe_span;

        assert!(interp.value_matches_ty(&str_ref(), &Value::Str(String::from("x"))));
        assert!(interp.value_matches_ty(&str_ref(), &Value::Ref(probe_place())));
        assert!(
            !interp.value_matches_ty(&str_ref(), &Value::String(String::from("x"))),
            "an owned String behind `&str` is the DEV-121 ownership error"
        );
    }

    /// The same shape for slices, which is the pairing INV-VALUE-REP-001 already rejected at `let`.
    #[test]
    fn a_borrowed_slice_permits_a_view_and_refuses_owned_storage() {
        let (hir, file, tables) = relation_probe("fn main() {}");
        let probe_span = file.synthetic_span();
        let interp = Interpreter::new(&hir, file, &tables);
        let _ = probe_span;

        assert!(interp.value_matches_ty(&byte_slice_ref(), &Value::Slice(probe_place(), 0, 0)));
        assert!(!interp.value_matches_ty(&byte_slice_ref(), &Value::Vec(Vec::new())));
        assert!(!interp.value_matches_ty(&byte_slice_ref(), &Value::Array(Vec::new())));
    }

    /// **The Copy-pointee rule, stated as a predicate rather than an exception list.**
    ///
    /// `&Int32` may be the bare scalar, because copying a Copy pointee cannot consume, invalidate
    /// or destroy the referent — the two representations are indistinguishable to anything the
    /// oracle can observe. `&String` may not, because flattening it would copy a non-Copy value,
    /// which is precisely a move-semantics violation.
    #[test]
    fn a_shared_reference_flattens_only_when_the_pointee_is_copy() {
        let (hir, file, tables) = relation_probe("fn main() {}");
        let probe_span = file.synthetic_span();
        let interp = Interpreter::new(&hir, file, &tables);
        let _ = probe_span;

        let int_ref = Ty::Ref {
            mutable: false,
            inner: Box::new(Ty::Primitive(crate::ast::Primitive::Int32)),
        };
        assert!(interp.value_matches_ty(&int_ref, &Value::Int(1)));
        assert!(interp.value_matches_ty(&int_ref, &Value::Ref(probe_place())));

        let string_ref = Ty::Ref {
            mutable: false,
            inner: Box::new(Ty::Primitive(crate::ast::Primitive::String)),
        };
        assert!(interp.value_matches_ty(&string_ref, &Value::Ref(probe_place())));
        assert!(
            !interp.value_matches_ty(&string_ref, &Value::String(String::new())),
            "flattening a non-Copy pointee would copy a value that must move"
        );
    }

    /// A mutable reference is never flattened, whatever the pointee. A `&mut Int32` that is a bare
    /// scalar cannot write through, and `take(&mut v)` needs the place itself.
    #[test]
    fn a_mutable_reference_is_never_flattened_even_for_a_copy_pointee() {
        let (hir, file, tables) = relation_probe("fn main() {}");
        let probe_span = file.synthetic_span();
        let interp = Interpreter::new(&hir, file, &tables);
        let _ = probe_span;

        let mut_int = Ty::Ref {
            mutable: true,
            inner: Box::new(Ty::Primitive(crate::ast::Primitive::Int32)),
        };
        assert!(interp.value_matches_ty(&mut_int, &Value::Ref(probe_place())));
        assert!(!interp.value_matches_ty(&mut_int, &Value::Int(1)));
    }

    /// **Owned `String` has two spellings**, and A2's exhaustive match is what surfaced it. Both
    /// denote the same type and both must permit exactly `Value::String`; covering only one would
    /// have left every binding written the other way unvalidated.
    #[test]
    fn both_spellings_of_owned_string_permit_the_same_representation() {
        let (hir, file, tables) = relation_probe("fn main() {}");
        let probe_span = file.synthetic_span();
        let interp = Interpreter::new(&hir, file, &tables);
        let _ = probe_span;

        let primitive = Ty::Primitive(crate::ast::Primitive::String);
        let core = Ty::Core(hir::CoreType::String, Vec::new());
        for ty in [primitive, core] {
            assert!(interp.value_matches_ty(&ty, &Value::String(String::new())));
            assert!(
                !interp.value_matches_ty(&ty, &Value::Str(String::new())),
                "an owned String represented as a view would not move when moved"
            );
        }
    }

    /// Nominal identity, not shape: two structs with identical fields are different types, so the
    /// `ItemId` is what the relation compares. Arity is checked for tuples and arrays for the same
    /// reason — it is what catches a truncated aggregate.
    #[test]
    fn nominals_are_matched_by_identity_and_aggregates_by_arity() {
        let (hir, file, tables) = relation_probe("fn main() {}");
        let probe_span = file.synthetic_span();
        let interp = Interpreter::new(&hir, file, &tables);
        let _ = probe_span;

        let one = Ty::Struct(ItemId(1), Vec::new());
        assert!(interp.value_matches_ty(
            &one,
            &Value::Struct {
                item: ItemId(1),
                fields: BTreeMap::new(),
            }
        ));
        assert!(
            !interp.value_matches_ty(
                &one,
                &Value::Struct {
                    item: ItemId(2),
                    fields: BTreeMap::new(),
                }
            ),
            "a different struct is a different type however similar its fields"
        );

        let pair = Ty::Tuple(vec![
            Ty::Primitive(crate::ast::Primitive::Int32),
            Ty::Primitive(crate::ast::Primitive::Int32),
        ]);
        assert!(interp.value_matches_ty(&pair, &Value::Tuple(vec![None, None])));
        assert!(!interp.value_matches_ty(&pair, &Value::Tuple(vec![None])));

        let three = Ty::Array(Box::new(Ty::Primitive(crate::ast::Primitive::Int32)), 3);
        assert!(interp.value_matches_ty(&three, &Value::Array(vec![None, None, None])));
        assert!(!interp.value_matches_ty(&three, &Value::Array(vec![None, None])));
    }

    /// Unsized and non-runtime types have NO permitted representation. A standalone `Ty::Slice` is
    /// distinct from `Value::Slice`, which is a perfectly valid view — same word, opposite
    /// meanings, which is why §6.6 is a separate table.
    #[test]
    fn unsized_and_non_runtime_types_permit_nothing() {
        let (hir, file, tables) = relation_probe("fn main() {}");
        let probe_span = file.synthetic_span();
        let interp = Interpreter::new(&hir, file, &tables);
        let _ = probe_span;

        let standalone_slice = Ty::Slice(Box::new(Ty::Primitive(crate::ast::Primitive::UInt8)));
        assert!(!interp.value_matches_ty(&standalone_slice, &Value::Slice(probe_place(), 0, 0)));
        assert!(!interp.value_matches_ty(&standalone_slice, &Value::Vec(Vec::new())));

        assert!(!interp.value_matches_ty(
            &Ty::Primitive(crate::ast::Primitive::Str),
            &Value::Str(String::new())
        ));
        assert!(!interp.value_matches_ty(&Ty::Never, &Value::Unit));
        assert!(!interp.value_matches_ty(&Ty::Error, &Value::Unit));
        assert!(!interp.value_matches_ty(&Ty::Param(String::from("T")), &Value::Int(1)));
    }

    /// A surviving `Ty::Param` is refused by NORMALISATION rather than by the relation, and the
    /// refusal is an internal invariant: `T` permits every representation, so a relation asked
    /// about it could only ever answer "yes".
    #[test]
    fn an_unsubstituted_parameter_is_refused_before_the_relation_sees_it() {
        let (hir, file, tables) = relation_probe("fn main() {}");
        let probe_span = file.synthetic_span();
        let interp = Interpreter::new(&hir, file, &tables);
        let _ = probe_span;

        let error = interp
            .check_value_for_ty(
                &Ty::Param(String::from("T")),
                &Value::Int(1),
                interp.file.synthetic_span(),
                RepBoundary::Parameter,
            )
            .expect_err("an unsubstituted parameter cannot be validated against");
        assert_eq!(error.class, FailureClass::InternalInvariant);
        assert!(
            error.message.contains("unsubstituted generic parameter"),
            "{}",
            error.message
        );
    }

    /// The diagnostic names the boundary and both shapes, and never the value's contents — printing
    /// them would leak program data, and describing them would mean cloning or borrowing the value,
    /// which is the behaviour this check exists to police.
    #[test]
    fn the_diagnostic_names_the_boundary_and_the_shapes_but_not_the_contents() {
        let (hir, file, tables) = relation_probe("fn main() {}");
        let probe_span = file.synthetic_span();
        let interp = Interpreter::new(&hir, file, &tables);
        let _ = probe_span;

        let error = interp
            .check_value_for_ty(
                &byte_slice_ref(),
                &Value::Vec(Vec::new()),
                interp.file.synthetic_span(),
                RepBoundary::Parameter,
            )
            .expect_err("an owned Vec behind `&[UInt8]` is a mismatch");

        assert_eq!(error.class, FailureClass::InternalInvariant);
        assert_eq!(error.trap_category, None);
        assert!(error.message.contains("DEV-121"), "{}", error.message);
        assert!(
            error.message.contains("a function parameter"),
            "the boundary is what makes the report actionable: {}",
            error.message
        );
        assert!(error.message.contains("Vec"), "{}", error.message);
    }

    /// A value that DOES match produces no error, at every boundary. The relation must not be
    /// merely strict — a check that refused correct programs would be withdrawn, which is how an
    /// invariant becomes advisory.
    #[test]
    fn a_matching_value_is_accepted_at_every_boundary() {
        let (hir, file, tables) = relation_probe("fn main() {}");
        let probe_span = file.synthetic_span();
        let interp = Interpreter::new(&hir, file, &tables);
        let _ = probe_span;

        for boundary in [
            RepBoundary::LetBinding,
            RepBoundary::Parameter,
            RepBoundary::Receiver,
            RepBoundary::Return,
            RepBoundary::Propagation,
            RepBoundary::MatchBinding,
            RepBoundary::LoopBinding,
            RepBoundary::Assignment,
            RepBoundary::FieldWrite,
            RepBoundary::ElementWrite,
            RepBoundary::AggregateField,
        ] {
            assert!(
                interp
                    .check_value_for_ty(
                        &str_ref(),
                        &Value::Str(String::from("x")),
                        interp.file.synthetic_span(),
                        boundary,
                    )
                    .is_ok(),
                "{boundary:?} rejected a correct representation"
            );
        }
    }

    /// **A1: every `Value` variant is named in the representation table, exactly once.**
    ///
    /// Three mechanisms, each catching a different way of forgetting:
    ///
    /// * `Value::kind()` has no wildcard, so a new `Value` variant is a COMPILE error there. That
    ///   is the strong guarantee and it needs no test.
    /// * `ValueKind` and `ALL` are generated from one list by `define_value_kinds!`, so `ALL`
    ///   cannot drift from the enum at all.
    /// * `WP-VALUE-REP-TOTAL.md`'s matrix is prose the compiler cannot check. The count is asserted
    ///   against the number that document states, so adding a kind without adding its row fails.
    ///
    /// **Uniqueness is asserted, not just length.** A hand-maintained list that duplicates one
    /// entry and omits another keeps its length, so a count alone passes on a list that is wrong in
    /// two places at once. Generation makes that unreachable today; the assertion stays because it
    /// is what would catch a future hand-written `ALL` if the macro were ever unwound.
    #[test]
    fn every_value_variant_is_named_in_the_representation_matrix() {
        assert_eq!(
            ValueKind::ALL.len(),
            34,
            "WP-VALUE-REP-TOTAL.md §6 documents 34 representations; update the matrix and this \
             count together, or the table stops describing the interpreter"
        );

        let unique: std::collections::HashSet<ValueKind> = ValueKind::ALL.iter().copied().collect();
        assert_eq!(
            unique.len(),
            ValueKind::ALL.len(),
            "ValueKind::ALL duplicates a kind or omits one"
        );

        let names: std::collections::HashSet<&str> =
            ValueKind::ALL.iter().map(|kind| kind.as_str()).collect();
        assert_eq!(
            names.len(),
            ValueKind::ALL.len(),
            "two kinds render to the same diagnostic name"
        );
    }

    /// `kind()` reports the representation and never the contents.
    ///
    /// A diagnostic that printed the value would leak program data into compiler output and, worse,
    /// would need to clone or borrow the value to do it — which is the very behaviour this class of
    /// check exists to police.
    #[test]
    fn a_value_kind_names_the_shape_and_not_the_contents() {
        assert_eq!(
            Value::String(String::from("secret")).kind(),
            ValueKind::String
        );
        assert_eq!(Value::Str(String::from("secret")).kind(), ValueKind::Str);
        assert_eq!(
            Value::String(String::from("secret")).kind().as_str(),
            "String"
        );

        // The pairing DEV-121 is about: identical payloads, different representations, and the
        // difference is exactly what decides whether passing the value moves it.
        assert_ne!(
            Value::Str(String::from("x")).kind(),
            Value::String(String::from("x")).kind(),
            "Str and String are structurally identical and semantically opposite"
        );
    }

    /// **A0 (DEV-121): a representation mismatch is a COMPILER DEFECT, not a language trap.**
    ///
    /// The check has always called itself "internal" in its prose while constructing its error with
    /// `RuntimeError::new`, which classifies as `FailureClass::Trap`. That combination is the
    /// dangerous one: the HIR interpreter is the behavioural oracle, so an oracle representation
    /// bug presented as a trap is something the differential harness can accept as a legitimate
    /// program outcome — and then pressure MIR and native into agreeing with it. Classification is
    /// what decides whether the harness fails loudly or quietly propagates the defect.
    ///
    /// `trap_category` is asserted `None` alongside the class because the two together are what the
    /// comparator reads; a class change with a lingering category would still look trap-shaped.
    ///
    /// The mismatch is INJECTED rather than produced by a program on purpose. Every producer this
    /// check knows about is currently correct, so a test that waited for a real firing would assert
    /// nothing — and would start passing again for the wrong reason the day a producer regressed.
    #[test]
    fn a_representation_mismatch_is_classified_as_an_internal_invariant() {
        let file = Arc::new(SourceFile::new("test.stark", "fn main() {}"));
        let (ast, _) = parse(&file, ParseMode::Program);
        let (hir, _) = resolve(&ast, file.clone());
        let mut tables = typecheck::analyze(&hir).tables;

        // A local the tables declare as `&[UInt8]` — a borrowed view.
        let local = LocalId(0);
        tables.local_types.insert(
            local,
            Ty::Ref {
                mutable: false,
                inner: Box::new(Ty::Slice(Box::new(Ty::Primitive(
                    crate::ast::Primitive::UInt8,
                )))),
            },
        );

        let registered = hir
            .source_named(&file.name)
            .expect("the parse registered this file");
        let probe_span = registered.synthetic_span();
        let interpreter = Interpreter::new(&hir, registered, &tables);
        let expected = tables
            .local_types
            .get(&local)
            .cloned()
            .expect("the probe just declared this local");
        let error = interpreter
            .check_value_for_ty(
                &expected,
                &Value::Vec(Vec::new()),
                probe_span,
                RepBoundary::LetBinding,
            )
            .expect_err("an owned Vec behind a `&[UInt8]` binding is a representation mismatch");

        assert_eq!(
            error.class,
            FailureClass::InternalInvariant,
            "a DEV-121 firing is a compiler defect: {}",
            error.message
        );
        assert_eq!(
            error.trap_category, None,
            "an internal invariant has no trap category"
        );
        assert!(!error.is_trap(), "it must not read as a language trap");
    }

    #[test]
    fn generated_source_paths_escape_stark_string_metacharacters() {
        assert_eq!(
            stark_string_literal_contents(r"C:\Users\runner\file.txt"),
            r"C:\\Users\\runner\\file.txt"
        );
        assert_eq!(
            stark_string_literal_contents("quoted\"line\ncarriage\r tab\t"),
            "quoted\\\"line\\ncarriage\\r tab\\t"
        );
        assert_eq!(
            portable_filename_component("interp::tests::file/resource"),
            "interp__tests__file_resource"
        );
    }

    /// The front end's diagnostics for `source`, for cases whose subject is a REFUSAL rather than
    /// an execution (WP-C7.9 Packet E).
    fn check_program(source: &str) -> Vec<crate::diag::Diagnostic> {
        let file = Arc::new(SourceFile::new("test.stark", source));
        let (ast, parse_diags) = parse(&file, ParseMode::Program);
        assert!(parse_diags.is_empty(), "parse diagnostics: {parse_diags:?}");
        let (hir, resolve_diags) = resolve(&ast, file.clone());
        assert!(
            resolve_diags.is_empty(),
            "resolve diagnostics: {resolve_diags:?}"
        );
        typecheck::analyze(&hir)
            .diagnostics
            .into_iter()
            .filter(|d| d.severity == crate::diag::Severity::Error)
            .collect()
    }

    fn execute(source: &str) -> Result<Execution, RuntimeError> {
        execute_with(source, None)
    }

    /// `execute`, with one producer mutation armed for this run only.
    fn execute_with(
        source: &str,
        mutation: Option<ProducerMutation>,
    ) -> Result<Execution, RuntimeError> {
        execute_mutated(
            source,
            Mutations {
                producer: mutation,
                env: None,
            },
        )
        .result
    }

    /// `execute`, with any combination of mutations armed for this run only.
    fn execute_mutated(source: &str, mutations: Mutations) -> MutatedRun {
        let file = Arc::new(SourceFile::new("test.stark", source));
        let (ast, parse_diags) = parse(&file, ParseMode::Program);
        assert!(parse_diags.is_empty(), "parse diagnostics: {parse_diags:?}");
        let (hir, resolve_diags) = resolve(&ast, file.clone());
        assert!(
            resolve_diags.is_empty(),
            "resolve diagnostics: {resolve_diags:?}"
        );
        let checked = typecheck::analyze(&hir);
        assert!(
            checked
                .diagnostics
                .iter()
                .all(|diag| diag.severity != crate::diag::Severity::Error),
            "type diagnostics: {:?}",
            checked.diagnostics
        );
        let registered = hir
            .source_named(&file.name)
            .expect("the parse registered this file");
        run_mutated(&hir, registered, &checked.tables, mutations)
    }

    // ---------------------------------------------------------------------------------------
    // DEV-121 CLASS EVIDENCE — four producer mutations, four forcing boundaries
    // ---------------------------------------------------------------------------------------
    //
    // Exit criterion 5 asks for a CLASS-level statement, not one regression case. Twelve wired
    // boundaries are not that statement on their own: a boundary that never fires is
    // indistinguishable from a boundary that is not running, and Packet 6 in particular found no
    // defect while firing on every expression the interpreter evaluates.
    //
    // Each test below follows the same three-step shape, which is what makes it evidence rather
    // than decoration:
    //
    //   1. the witness program runs CLEAN unmutated — a "detection" on an already-broken program
    //      proves nothing;
    //   2. one PRODUCER is mutated (never `check_value_for_ty`, which would only show the
    //      predicate detects an artificial mismatch);
    //   3. the real funnel refuses it, classified `InternalInvariant`, NAMING the intended
    //      boundary — so a mutation caught by the wrong wire is a failure, not a pass.
    //
    // The four classes and their forcing sites are the owner's, recorded 2026-08-08:
    //
    //   owned/view          -> ExpressionResult
    //   reference           -> Receiver
    //   function value      -> ExpressionResult
    //   aggregate/container -> AggregateField

    /// Runs `source` twice: once clean, once with `mutation` armed. Returns the mutated run's
    /// error after asserting the clean run succeeded.
    fn mutation_must_be_caught(
        source: &str,
        mutation: ProducerMutation,
        boundary: RepBoundary,
    ) -> RuntimeError {
        // Step 1 — the witness must genuinely pass first.
        execute(source).unwrap_or_else(|error| {
            panic!("witness must run clean before it is mutated: {error:?}")
        });

        // Step 2 — arm one producer mutation, on THIS execution only. Nothing global is touched,
        // so a test running in parallel beside this one is unaffected.
        let error = execute_with(source, Some(mutation))
            .err()
            .unwrap_or_else(|| {
                panic!(
                    "{mutation:?} produced a mis-represented value and NOTHING \
                                       refused it — the boundary is inert"
                )
            });

        // Step 3 — refused as a compiler defect, at the intended wire.
        assert_eq!(
            error.class,
            FailureClass::InternalInvariant,
            "a representation defect is a compiler defect, never a language trap: {}",
            error.message
        );
        assert!(
            error.message.contains(boundary.as_str()),
            "{mutation:?} must be caught at {} — a mutation caught by a different wire is not \
             evidence for this class. Got: {}",
            boundary.as_str(),
            error.message
        );
        error
    }

    /// **Class 1 — owned/view.** The original DEV-121 pairing: a producer of `&str`/`&[T]` emits
    /// OWNED storage instead of a view, so passing the value MOVES what it only borrows.
    #[test]
    fn class_1_an_owned_value_behind_a_view_type_is_refused() {
        let error = mutation_must_be_caught(
            "fn main() { let s = String::from(\"abc\"); let v = s.as_str(); println(v); }",
            ProducerMutation::OwnedForView,
            RepBoundary::ExpressionResult,
        );
        assert!(
            error.message.contains("String"),
            "the diagnostic should name the representation actually found: {}",
            error.message
        );
    }

    /// Class 1, the sequence half — `Vec::as_slice` emitting an owned `Vec` where `&[T]` is
    /// declared. Both halves of the class are exercised because the two have separate producers.
    #[test]
    fn class_1_an_owned_vec_behind_a_slice_type_is_refused() {
        mutation_must_be_caught(
            "fn main() { let v: Vec<Int32> = Vec::new(); let s = v.as_slice(); println(s.len()); }",
            ProducerMutation::OwnedForView,
            RepBoundary::ExpressionResult,
        );
    }

    /// **Class 2 — reference.** A `&self` receiver binds the pointee BY VALUE instead of a
    /// `Value::Ref` into the caller's place. This is the destructor materialization defect
    /// deliberately reintroduced, and it is why the receiver boundary had to pass for `Drop`
    /// without a `Drop`-shaped exemption: an exemption there would have made this mutation
    /// undetectable for every destructor.
    ///
    /// **The pointee must be NON-`Copy`, and the first witness got that wrong.** §6.4 licenses the
    /// bare-value form for a `Copy` pointee — copying it cannot consume, invalidate or destroy the
    /// referent, so the two representations are indistinguishable to any observation the oracle can
    /// make. A `struct Holder { n: Int32 }` is `Copy`-eligible, so the mutation was not a violation
    /// there and the relation was right to accept it. Only a non-`Copy` pointee makes the owned
    /// form observably wrong, which is exactly what the class is about.
    #[test]
    fn class_2_an_owned_value_behind_a_reference_receiver_is_refused() {
        mutation_must_be_caught(
            "struct Holder { name: String } \
             impl Holder { fn peek(&self) -> Int32 { 1 } } \
             fn main() { let h = Holder { name: String::from(\"x\") }; println(h.peek()); }",
            ProducerMutation::OwnedForReference,
            RepBoundary::Receiver,
        );
    }

    /// **Class 3 — function value.** A function item coerces to something that is not a function.
    /// The declared type is `fn(Int32) -> Int32`; §6 permits exactly one representation for it.
    #[test]
    fn class_3_a_non_function_behind_a_function_type_is_refused() {
        mutation_must_be_caught(
            "fn identity(x: Int32) -> Int32 { x } \
             fn main() { let f: fn(Int32) -> Int32 = identity; println(f(41)); }",
            ProducerMutation::NonFunctionValue,
            RepBoundary::ExpressionResult,
        );
    }

    /// **Class 4 — aggregate.** A declared field receives a mis-represented value.
    ///
    /// The mutation is injected AFTER the producer-side boundary has already accepted the value,
    /// which is the point: with `ExpressionResult` live it would otherwise catch nearly everything
    /// upstream, and this class needs to show that the AGGREGATE wire independently works.
    #[test]
    fn class_4_a_mis_represented_aggregate_field_is_refused() {
        mutation_must_be_caught(
            "struct Pair { a: Int32, b: Int32 } \
             fn main() { let p = Pair { a: 1, b: 2 }; println(p.a + p.b); }",
            ProducerMutation::WrongAggregateField,
            RepBoundary::AggregateField,
        );
    }

    /// **Audit 10-D — an independent function-value challenge.**
    ///
    /// Class 3 corrupts the function value's REPRESENTATION. This corrupts its captured generic
    /// context while leaving a perfectly valid `Value::Function` in place, which is DEV-178's
    /// defect rather than DEV-121's: the bindings are fixed at the coercion and `Ty::Fn` cannot
    /// say which instantiation produced them, so nothing downstream can reconstruct them.
    ///
    /// The witness answers `size_of::<T>()`, so losing the bindings cannot pass unnoticed the way
    /// DEV-197's identity-shaped witnesses did.
    #[test]
    fn audit_10d_a_function_value_stripped_of_its_bindings_is_refused() {
        let source = "fn width<T>(x: T) -> Int32 { size_of::<T>() as Int32 } \
                      fn main() { let f: fn(Float64) -> Int32 = width; println(f(1.5)); }";
        assert_eq!(execute(source).expect("witness runs").output, "8\n");
        let error = execute_with(source, Some(ProducerMutation::StripFunctionValueBindings))
            .expect_err("a function value with no instantiation must not execute its body");
        assert_eq!(
            error.class,
            FailureClass::InternalInvariant,
            "losing a captured instantiation is a compiler defect: {}",
            error.message
        );
    }

    /// **Audit 10-E — the other route into typed storage.**
    ///
    /// The aggregate class already has a control at CONSTRUCTION. This one writes into storage that
    /// already exists, which reaches the boundary through `write_place` rather than through
    /// `eval_struct_lit` — a different funnel with a different expected-type source.
    #[test]
    fn audit_10e_a_mis_represented_write_into_existing_storage_is_refused() {
        mutation_must_be_caught(
            "fn main() { let mut n: Int32 = 1; n = 2; println(n); }",
            ProducerMutation::WrongElementWrite,
            RepBoundary::Assignment,
        );
    }

    /// **Audit 10-C, independent of class 2.** Class 2 mutates a `&self` receiver; this is the
    /// EXCLUSIVE form, where losing place identity also loses the caller's mutation.
    #[test]
    fn audit_10c_a_mut_self_receiver_must_keep_place_identity() {
        mutation_must_be_caught(
            "struct Holder { name: String } \
             impl Holder { fn touch(&mut self) -> Int32 { 1 } } \
             fn main() { let mut h = Holder { name: String::from(\"x\") }; println(h.touch()); }",
            ProducerMutation::OwnedForReference,
            RepBoundary::Receiver,
        );
    }

    /// **DEV-203 adversary.** An interpolated field is an inline value entering a runtime
    /// operation: it never binds to a local, so no destination boundary sees it. It reached the
    /// renderer through a direct `eval_expr`, so the producer boundary did not see it either.
    ///
    /// Written as the mutation that would have caught it: `s.as_str()` inside `f"{...}"`, with the
    /// view producer emitting owned storage. Before the repair this rendered happily.
    #[test]
    fn an_interpolated_field_is_a_checked_expression_result() {
        mutation_must_be_caught(
            "fn main() { let s = String::from(\"abc\"); println(f\"{s.as_str()}\"); }",
            ProducerMutation::OwnedForView,
            RepBoundary::ExpressionResult,
        );
    }

    /// **The control on the controls.** The mutation must be OFF unless a test arms it — otherwise
    /// the four tests above would be reporting on a permanently broken interpreter rather than on
    /// an injected defect, and every other test in this suite would be failing too.
    ///
    /// Asserted through the real entry point rather than by reading a flag, because "the default
    /// is off" is a claim about what `run` does, not about a field's initialiser.
    #[test]
    fn no_producer_mutation_is_armed_by_default() {
        let source = "struct Pair { a: Int32, b: Int32 } \
                      fn main() { let p = Pair { a: 1, b: 2 }; println(p.a + p.b); }";
        assert_eq!(execute(source).expect("clean by default").output, "3\n");
        assert_eq!(
            execute_with(source, None)
                .expect("explicitly unmutated")
                .output,
            "3\n",
            "passing `None` must be identical to not arming at all"
        );
    }

    // ---------------------------------------------------------------------------------------
    // AS3 #2 REQUALIFICATION — environment installation, proved by omission
    // ---------------------------------------------------------------------------------------
    //
    // The criterion was recorded PASS once before on tests that asserted a table had an entry.
    // DEV-197 is what that missed: nine dispatch sites installed no environment at all and every
    // test passed, because the bodies involved never mentioned their own parameters. An
    // environment that is never consulted cannot be observed to be absent.
    //
    // So this requalification proves the claim by OMISSION. Each of the seven dispatch classes gets
    // a witness whose answer genuinely depends on its instantiation — `size_of::<T>()`, which is
    // 8 for `Float64` and 4 for `Int32` — and the environment is then removed at the single
    // installation point. Three things must hold for each class:
    //
    //   1. the witness passes unmutated, with the instantiation-dependent answer;
    //   2. the mutation is REACHED — a control that never reaches the installer would "detect"
    //      nothing and look like a pass, which is precisely how DEV-197 hid;
    //   3. the run fails as `InternalInvariant` — never Empty, never a skip, never a default.

    /// The three-step check every dispatch-class control runs. Returns the mutated run's error.
    fn environment_omission_must_be_observable(
        class: &str,
        source: &str,
        expected_output: &str,
    ) -> RuntimeError {
        // 1. The witness genuinely passes, with the answer its instantiation determines.
        let clean = execute(source)
            .unwrap_or_else(|error| panic!("{class}: witness must run clean: {error:?}"));
        assert_eq!(
            clean.output, expected_output,
            "{class}: the witness must depend on its instantiation, or removing the environment \
             cannot be observed — that is exactly how DEV-197 stayed hidden"
        );

        // 2 and 3. Remove the environment at the installation point.
        let run = execute_mutated(
            source,
            Mutations {
                producer: None,
                env: Some(EnvMutation::DropEnvironment),
            },
        );
        assert!(
            run.env_mutations_applied > 0,
            "{class}: the installation point was never reached, so this control tests nothing \
             about this dispatch class"
        );
        let error = run.result.err().unwrap_or_else(|| {
            panic!(
                "{class}: the environment was removed and the program still succeeded — \
                    omission is unobservable for this dispatch class"
            )
        });
        assert_eq!(
            error.class,
            FailureClass::InternalInvariant,
            "{class}: a missing environment is a compiler defect, never a language trap: {}",
            error.message
        );
        error
    }

    /// **D1 — free generic function.** `Static` selection, `Static` environment.
    #[test]
    fn d1_a_free_generic_function_needs_its_environment() {
        environment_omission_must_be_observable(
            "D1",
            "fn width<T>(x: T) -> Int32 { size_of::<T>() as Int32 } \
             fn main() { println(width(1.5)); }",
            "8\n",
        );
    }

    /// **D2 — generic associated function.** One of the two paths DEV-197 was opened for.
    #[test]
    fn d2_a_generic_associated_function_needs_its_environment() {
        environment_omission_must_be_observable(
            "D2",
            "struct S { v: Int32 } \
             impl S { fn width<T>(x: T) -> Int32 { size_of::<T>() as Int32 } } \
             fn main() { println(S::width(1.5)); }",
            "8\n",
        );
    }

    /// **D3 — generic inherent method.**
    #[test]
    fn d3_a_generic_inherent_method_needs_its_environment() {
        environment_omission_must_be_observable(
            "D3",
            "struct S { v: Int32 } \
             impl S { fn width<T>(&self, x: T) -> Int32 { size_of::<T>() as Int32 } } \
             fn main() { let s = S { v: 0 }; println(s.width(1.5)); }",
            "8\n",
        );
    }

    /// **D4 — operator dispatch into a generic impl.**
    ///
    /// This class already has real-world evidence: DEV-201 was exactly this defect, shipped, and
    /// caught by the receiver boundary as a MISSED TRAP against MIR. The control is added anyway
    /// so the requalification is reproducible rather than resting on history.
    #[test]
    fn d4_operator_dispatch_into_a_generic_impl_needs_its_environment() {
        environment_omission_must_be_observable(
            "D4",
            "struct W<T> { v: T } \
             impl<T> Eq for W<T> { fn eq(&self, other: &W<T>) -> Bool { size_of::<T>() == 8 } } \
             fn main() { let a = W { v: 1.5 }; let b = W { v: 2.5 }; \
                         if a == b { println(1); } else { println(0); } }",
            "1\n",
        );
    }

    /// **D5 — bound trait dispatch.** The body and environment come from the shared specialiser
    /// atomically; removing the environment must not leave the body running.
    #[test]
    fn d5_bound_trait_dispatch_needs_its_environment() {
        environment_omission_must_be_observable(
            "D5",
            "trait Sz { fn sz(&self) -> Int32; } \
             struct P<T> { v: T } \
             impl<T> Sz for P<T> { fn sz(&self) -> Int32 { size_of::<T>() as Int32 } } \
             fn use_it<S: Sz>(s: S) -> Int32 { s.sz() } \
             fn main() { println(use_it(P { v: 1.5 })); }",
            "8\n",
        );
    }

    /// **D6 — function value.** DEV-178 put the bindings on the VALUE precisely because
    /// `Ty::Fn` cannot say which instantiation produced it, and DEV-197 found the call site
    /// discarding them.
    ///
    /// The witness is deliberately NOT identity-shaped. DEV-197's original two defects were
    /// invisible because both bodies returned their argument unchanged, so an unbound `T` changed
    /// no answer; a control with that property would reproduce the blindness it is testing for.
    #[test]
    fn d6_a_function_value_needs_its_captured_bindings() {
        environment_omission_must_be_observable(
            "D6",
            "fn width<T>(x: T) -> Int32 { size_of::<T>() as Int32 } \
             fn main() { let f: fn(Float64) -> Int32 = width; println(f(1.5)); }",
            "8\n",
        );
    }

    /// **D7 — nested generic calls, which tests RESTORATION as well as installation.**
    ///
    /// `outer<T = Float64>` calls `inner<U = Int32>` and then reads `size_of::<T>()` again. The
    /// witness answers `848`: 8 before, 4 inside, 8 after. A stale frame would answer `844`, and
    /// the two instantiations are deliberately different so that a restoration bug cannot pass by
    /// coincidence.
    #[test]
    fn d7_nested_generic_calls_install_and_restore() {
        environment_omission_must_be_observable(
            "D7",
            "fn inner<U>(x: U) -> Int32 { size_of::<U>() as Int32 } \
             fn outer<T>(x: T) -> Int32 { \
                 let a = size_of::<T>() as Int32; \
                 let m = inner(1); \
                 let b = size_of::<T>() as Int32; \
                 a * 100 + m * 10 + b \
             } \
             fn main() { println(outer(1.5)); }",
            "848\n",
        );
    }

    /// **P8, stated as its own assertion.** D7's witness passing is the restoration proof, but it
    /// is worth failing loudly on its own: `848` means the caller's `T` survived a callee that
    /// installed a different one, and `844` means it did not.
    #[test]
    fn p8_a_callees_environment_does_not_outlive_it() {
        let execution = execute(
            "fn inner<U>(x: U) -> Int32 { size_of::<U>() as Int32 } \
             fn outer<T>(x: T) -> Int32 { \
                 let a = size_of::<T>() as Int32; \
                 let m = inner(1); \
                 let b = size_of::<T>() as Int32; \
                 a * 100 + m * 10 + b \
             } \
             fn main() { println(outer(1.5)); }",
        )
        .expect("must run");
        assert_eq!(
            execution.output, "848\n",
            "the caller's instantiation must be restored after a callee installed a different \
             one: 8 before, 4 inside, 8 after"
        );
    }

    /// **P2 — an invocation's environment is an explicit state, never an absent one.**
    ///
    /// Exhaustive on purpose: adding an `InvocationEnv` variant fails to compile here until
    /// someone states what it means and confirms the installer handles it. That is the forcing
    /// function, not the assertion below.
    #[test]
    fn p2_every_invocation_environment_variant_is_explicit() {
        fn describe(env: &InvocationEnv) -> &'static str {
            match env {
                InvocationEnv::Empty => "explicitly no generics — not absent metadata",
                InvocationEnv::Published(_) => "the environment published at this call expression",
                InvocationEnv::Concrete(_) => {
                    "bindings the checker or specialiser already resolved"
                }
                InvocationEnv::Captured(_) => "bindings the function value carries (DEV-178)",
            }
        }
        let variants = [InvocationEnv::Empty, InvocationEnv::Concrete(Vec::new())];
        for env in &variants {
            assert!(!describe(env).is_empty());
        }
    }

    /// **P6 — the environment is installed BEFORE the typed call boundaries read anything.**
    ///
    /// Not a source-order assertion: D4's receiver type is `&W<T>`, so if the receiver boundary ran
    /// before installation it would see an unsubstituted `T` on a correct program. It does not —
    /// the unmutated witness passes — and when the environment is removed, that is exactly the
    /// boundary that catches it.
    #[test]
    fn p6_typed_boundaries_run_while_the_environment_is_active() {
        let error = environment_omission_must_be_observable(
            "P6",
            "struct W<T> { v: T } \
             impl<T> Eq for W<T> { fn eq(&self, other: &W<T>) -> Bool { size_of::<T>() == 8 } } \
             fn main() { let a = W { v: 1.5 }; let b = W { v: 2.5 }; \
                         if a == b { println(1); } else { println(0); } }",
            "1\n",
        );
        assert!(
            error.message.contains(RepBoundary::Receiver.as_str()),
            "a receiver typed `&W<T>` must be read against the callee's own instantiation, so \
             removing it fails at the receiver boundary. Got: {}",
            error.message
        );
    }

    #[test]
    fn executes_functions_control_flow_and_aggregates() {
        let execution = execute("struct Pair { a: Int32, b: Int32 } fn sum(p: Pair) -> Int32 { p.a + p.b } fn main() { let p = Pair { a: 2, b: 3 }; if sum(p) == 5 { println(\"ok\"); } }").unwrap();
        assert_eq!(execution.output, "ok\n");
    }

    #[test]
    fn runtime_errors_abort() {
        let error =
            execute("fn main() { let values = [1, 2]; let index = 3; println(values[index]); }")
                .unwrap_err();
        assert!(error.message.contains("bounds") || error.message.contains("field"));
    }

    #[test]
    fn executes_core_collections_strings_and_try() {
        let execution = execute(
            "fn number(ok: Bool) -> Result<Int32, String> { if ok { Ok(7) } else { Err(String::from(\"bad\")) } } fn doubled() -> Result<Int32, String> { let value = number(true)?; Ok(value * 2) } fn main() { let mut text = String::from(\"hi\"); text.push('!'); let mut values: Vec<Int32> = Vec::new(); values.push(doubled().unwrap()); values.push(9); println(text.as_str()); println(values[0]); println(values.get(8u64).is_none()); }",
        )
        .unwrap();
        assert_eq!(execution.output, "hi!\n14\ntrue\n");
    }

    /// WP-C1.3 regression test for DEV-008: `==`/`!=` used to be pure structural equality on
    /// the interpreter's `Value` enum regardless of any user-defined `impl Eq for T`. This
    /// struct's `eq` deliberately does NOT implement structural comparison (it ignores its
    /// fields and always returns `true`), so a passing test here proves real dispatch, not a
    /// coincidental match with structural equality.
    #[test]
    fn custom_eq_impl_is_dispatched_not_structural() {
        let execution = execute(
            "struct Always { tag: Int32 } \
             impl Eq for Always { fn eq(&self, other: &Always) -> Bool { true } } \
             fn main() { \
                 let a = Always { tag: 1 }; \
                 let b = Always { tag: 2 }; \
                 println(a == b); \
             }",
        )
        .unwrap();
        assert_eq!(
            execution.output, "true\n",
            "expected the custom (always-true) eq() to be dispatched despite differing fields"
        );
    }

    /// WP-C1.3: `!=` must negate the *dispatched* result, not fall back to structural
    /// inequality when a custom `eq` exists.
    #[test]
    fn custom_eq_impl_is_dispatched_for_ne_too() {
        let execution = execute(
            "struct Never { tag: Int32 } \
             impl Eq for Never { fn eq(&self, other: &Never) -> Bool { false } } \
             fn main() { \
                 let a = Never { tag: 1 }; \
                 let b = Never { tag: 1 }; \
                 println(a != b); \
             }",
        )
        .unwrap();
        assert_eq!(
            execution.output, "true\n",
            "expected != to negate the custom (always-false) eq(), even though fields are equal"
        );
    }

    /// WP-C1.3 regression test for the companion checker finding made while investigating
    /// DEV-008: `Ty::Core` container types (Option/Result/Vec) had no arm in
    /// `require_operator_bound` at all, so `==` on `Option<Int32>` was unconditionally rejected
    /// by the type checker even though Int32 is obviously Eq. Confirms both that it now
    /// type-checks AND that comparison remains ordinary structural equality (no dispatch,
    /// consistent with Core v1 having no user-overridable Eq for compiler container types).
    #[test]
    fn option_and_vec_equality_are_structural() {
        let execution = execute(
            "fn main() { \
                 let a: Option<Int32> = Some(1); \
                 let b: Option<Int32> = Some(1); \
                 let c: Option<Int32> = Some(2); \
                 println(a == b); \
                 println(a == c); \
                 let mut v1: Vec<Int32> = Vec::new(); \
                 v1.push(1); \
                 let mut v2: Vec<Int32> = Vec::new(); \
                 v2.push(1); \
                 println(v1 == v2); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "true\nfalse\ntrue\n");
    }

    /// WP-C1.3 regression test for DEV-013: `.clone()` was not a recognized method for ANY
    /// compiler-builtin type at all (String, Vec, Option, Result, HashMap, HashSet, ...) --
    /// confirmed empirically as "method call on non-struct/enum type" before this fix, even
    /// though `Clone` as a *bound* was already correctly recognized for these types. Covers the
    /// most commonly used builtin types; the fix is generic (matches on `Value` variant), not
    /// per-type, so this is representative rather than exhaustive.
    #[test]
    fn clone_works_for_builtin_core_types() {
        let execution = execute(
            "fn main() { \
                 let s = String::from(\"hi\"); \
                 let s2 = s.clone(); \
                 println(s2.as_str()); \
                 let mut v: Vec<Int32> = Vec::new(); \
                 v.push(1); \
                 let v2 = v.clone(); \
                 println(v2.len()); \
                 let o: Option<Int32> = Some(5); \
                 let o2 = o.clone(); \
                 println(o2.is_some()); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "hi\n1\ntrue\n");
    }

    /// WP-C1.3 regression test for DEV-013: a trait method declared with a real default body
    /// (03-Type-System.md trait defaults) was never used as a fallback when an implementing
    /// type didn't override it -- confirmed empirically as "method not found" before this fix,
    /// despite the HIR already carrying the default body (`TraitItem::Method.body: Some(_)`).
    #[test]
    fn default_trait_method_runs_when_not_overridden() {
        let execution = execute(
            "trait Greet { \
                 fn name(&self) -> String; \
                 fn greet(&self) -> String { String::from(\"Hello\") } \
             } \
             struct Bob {} \
             impl Greet for Bob { fn name(&self) -> String { String::from(\"Bob\") } } \
             fn main() { let b = Bob {}; println(b.greet().as_str()); }",
        )
        .unwrap();
        assert_eq!(execution.output, "Hello\n");
    }

    /// WP-C1.3: companion test -- an implementing type that DOES override the default must use
    /// its own body, not the trait's default.
    #[test]
    fn overriding_impl_takes_precedence_over_trait_default() {
        let execution = execute(
            "trait Greet { \
                 fn name(&self) -> String; \
                 fn greet(&self) -> String { String::from(\"Hello\") } \
             } \
             struct Bob {} \
             impl Greet for Bob { \
                 fn name(&self) -> String { String::from(\"Bob\") } \
                 fn greet(&self) -> String { String::from(\"Yo\") } \
             } \
             fn main() { let b = Bob {}; println(b.greet().as_str()); }",
        )
        .unwrap();
        assert_eq!(execution.output, "Yo\n");
    }

    #[test]
    fn runs_drop_in_reverse_declaration_order() {
        let execution = execute(
            "struct Marker { name: String } impl Drop for Marker { fn drop(&mut self) { println(self.name.as_str()); } } fn main() { let first = Marker { name: String::from(\"first\") }; let second = Marker { name: String::from(\"second\") }; }",
        )
        .unwrap();
        assert_eq!(execution.output, "second\nfirst\n");
    }

    /// WP-C2.2 (DEV-034): a by-value (`self`) method call on a non-place receiver expression
    /// must evaluate the receiver expression exactly once. Previously `call_user_method`'s
    /// `Receiver::Value` arm re-evaluated the original expression after `call_method` had
    /// already evaluated it once for dispatch — "making" printed twice for one call.
    #[test]
    fn by_value_receiver_expression_evaluates_exactly_once() {
        let execution = execute(
            "struct Counter { n: Int32 } \
             impl Counter { fn consume(self) -> Int32 { self.n } } \
             fn make_counter() -> Counter { println(\"making\"); Counter { n: 1 } } \
             fn main() { println(make_counter().consume()); }",
        )
        .unwrap();
        assert_eq!(execution.output, "making\n1\n");
    }

    /// WP-C2.2 (DEV-035): a reference returned from a `&self` method (`&self.field`) must stay
    /// valid in the caller. Previously the returned `Value::Ref` pointed into the method's own
    /// popped call frame and every later dereference trapped with "dangling reference".
    #[test]
    fn reference_returned_from_ref_self_method_is_valid_in_the_caller() {
        let execution = execute(
            "struct BoxedValue { value: Int32 } \
             impl BoxedValue { fn value_ref(&self) -> &Int32 { &self.value } } \
             fn main() { let b = BoxedValue { value: 42 }; let r = b.value_ref(); println(*r); }",
        )
        .unwrap();
        assert_eq!(execution.output, "42\n");
    }

    /// WP-C2.2 (DEV-035, `&mut` variant): writing through a `&mut` returned from a
    /// `&mut self` method must be observable in the original value after the borrow ends.
    #[test]
    fn mut_reference_returned_from_mut_self_method_writes_through() {
        let execution = execute(
            "struct Holder { value: Int32 } \
             impl Holder { fn value_mut(&mut self) -> &mut Int32 { &mut self.value } } \
             fn main() { \
                 let mut h = Holder { value: 5 }; \
                 { let m = h.value_mut(); *m = 99; } \
                 println(h.value); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "99\n");
    }

    /// WP-C2.2 (DEV-035, nested): a method that calls another `&self` method on `self` and
    /// projects a field through the returned reference must rebase correctly through both
    /// popped frames in sequence.
    #[test]
    fn nested_self_method_reference_chain_rebases_through_both_frames() {
        let execution = execute(
            "struct Inner { v: Int32 } \
             struct Outer { inner: Inner } \
             impl Outer { \
                 fn inner_ref(&self) -> &Inner { &self.inner } \
                 fn v_ref(&self) -> &Int32 { &self.inner_ref().v } \
             } \
             fn main() { let o = Outer { inner: Inner { v: 7 } }; println(*o.v_ref()); }",
        )
        .unwrap();
        assert_eq!(execution.output, "7\n");
    }

    /// WP-C2.2 (DEV-037): field access through a reference (`r.v` for `r: &Inner`), in both
    /// value and place (`&r.v`) contexts, must auto-dereference at runtime. Previously the
    /// place machinery tried to project a field directly on the stored `Value::Ref` and
    /// trapped with "use of moved or invalid field" — pre-existing at Gate C1 close, found
    /// while fixing DEV-035's nested case.
    #[test]
    fn field_access_through_reference_auto_derefs() {
        let execution = execute(
            "struct Inner { v: Int32 } \
             fn main() { \
                 let i = Inner { v: 3 }; \
                 let r = &i; \
                 println(r.v); \
                 let p = &r.v; \
                 println(*p); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "3\n3\n");
    }

    /// WP-C2.2 (DEV-030): a `_`-matched element of an owned scrutinee must still be dropped
    /// exactly once. Previously the unbound portion's destructor never ran at all, for the
    /// rest of the program. Order: the bound element (`a`) drops with its binding's scope
    /// cleanup first, then the unbound remainder drops as the consumed scrutinee's cleanup.
    #[test]
    fn unbound_match_elements_of_an_owned_scrutinee_are_dropped_exactly_once() {
        let execution = execute(
            "struct Loud { label: String } \
             impl Drop for Loud { fn drop(&mut self) { println(self.label.as_str()); } } \
             fn main() { \
                 let pair = (Loud { label: String::from(\"first\") }, Loud { label: String::from(\"second\") }); \
                 match pair { (a, _) => { println(\"matched\"); } } \
                 println(\"after match\"); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "matched\nfirst\nsecond\nafter match\n");
    }

    /// WP-C2.2 (DEV-030): a fully-unbound enum-variant payload drops whole (its own subtree
    /// included) when the match consumes the scrutinee.
    #[test]
    fn fully_unbound_variant_payload_is_dropped() {
        let execution = execute(
            "struct Loud { label: String } \
             impl Drop for Loud { fn drop(&mut self) { println(self.label.as_str()); } } \
             enum Wrap { Has(Loud), Empty } \
             fn main() { \
                 let w = Wrap::Has(Loud { label: String::from(\"wrapped\") }); \
                 match w { Wrap::Has(_) => println(\"has\"), Wrap::Empty => println(\"empty\") } \
                 println(\"done\"); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "has\nwrapped\ndone\n");
    }

    /// WP-C2.2 (DEV-030): unmentioned/wildcarded struct-pattern fields drop; bound fields drop
    /// via their bindings — each exactly once.
    #[test]
    fn wildcarded_struct_pattern_fields_are_dropped() {
        let execution = execute(
            "struct Loud { label: String } \
             impl Drop for Loud { fn drop(&mut self) { println(self.label.as_str()); } } \
             struct Pair { kept: Loud, thrown: Loud } \
             fn main() { \
                 let p = Pair { \
                     kept: Loud { label: String::from(\"kept\") }, \
                     thrown: Loud { label: String::from(\"thrown\") }, \
                 }; \
                 match p { Pair { kept, thrown: _ } => { println(\"matched\"); } } \
                 println(\"done\"); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "matched\nkept\nthrown\ndone\n");
    }

    #[test]
    fn unbound_struct_pattern_fields_use_reverse_declaration_order() {
        let execution = execute(
            "struct Loud { label: String } \
             impl Drop for Loud { fn drop(&mut self) { println(self.label.as_str()); } } \
             struct Trio { zed: Loud, alpha: Loud, middle: Loud } \
             fn main() { \
                 let trio = Trio { \
                     zed: Loud { label: String::from(\"zed\") }, \
                     alpha: Loud { label: String::from(\"alpha\") }, \
                     middle: Loud { label: String::from(\"middle\") }, \
                 }; \
                 match trio { Trio { middle, zed: _, alpha: _ } => println(\"matched\") } \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "matched\nmiddle\nalpha\nzed\n");
    }

    /// WP-C2.2 (DEV-030): a by-reference scrutinee is a borrow, not an owner — matching it must
    /// not drop the referent early; the original drops at its own scope exit.
    #[test]
    fn matching_a_reference_scrutinee_does_not_drop_the_referent() {
        let execution = execute(
            "struct Loud { label: String } \
             impl Drop for Loud { fn drop(&mut self) { println(self.label.as_str()); } } \
             fn main() { \
                 let l = Loud { label: String::from(\"owned\") }; \
                 let r = &l; \
                 match r { _ => println(\"matched ref\") } \
                 println(\"still alive\"); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "matched ref\nstill alive\nowned\n");
    }

    /// WP-C2.2 (DEV-026): an inherent method shadows a same-named trait method unconditionally,
    /// even when the trait impl block is declared FIRST in the source file. Previously
    /// `find_method`'s single source-order scan returned whichever impl appeared first.
    #[test]
    fn inherent_method_shadows_trait_method_regardless_of_declaration_order() {
        let execution = execute(
            "struct Thing { } \
             trait Speak { fn say(&self) -> String { String::from(\"trait-default\") } } \
             impl Speak for Thing { } \
             impl Thing { fn say(&self) -> String { String::from(\"inherent\") } } \
             fn main() { let t = Thing { }; println(t.say().as_str()); }",
        )
        .unwrap();
        assert_eq!(execution.output, "inherent\n");
    }

    /// WP-C2.2 (DEV-027): `Ordering` is a real prelude type and nominal comparison operators
    /// dispatch to the user's `Ord::cmp` implementation.
    #[test]
    fn nominal_comparison_dispatches_through_ord_cmp() {
        let execution = execute(
            "struct Point { x: Int32 } \
             impl Ord for Point { \
                 fn cmp(&self, other: &Point) -> Ordering { \
                     if self.x < other.x { Ordering::Less } \
                     else if self.x > other.x { Ordering::Greater } \
                     else { Ordering::Equal } \
                 } \
             } \
             fn main() { \
                 println(Point { x: 1 } < Point { x: 9 }); \
                 println(Point { x: 9 } >= Point { x: 1 }); \
                 println(Point { x: 1 } > Point { x: 9 }); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "true\ntrue\nfalse\n");
    }

    /// WP-C2.2 (DEV-031): `for` consumes general Iterator-typed expressions, both standard
    /// adapters and nominal user implementations, rather than only Range/Array/Vec values.
    #[test]
    fn for_loop_accepts_standard_and_user_iterators() {
        let execution = execute(
            "struct Counter { n: Int32 } \
             impl Iterator for Counter { \
                 type Item = Int32; \
                 fn next(&mut self) -> Option<Int32> { \
                     println(\"next\"); \
                     if self.n < 3 { self.n += 1; Some(self.n) } else { None } \
                 } \
             } \
             fn main() { \
                 let mut values: Vec<Int32> = Vec::new(); \
                 values.push(4); values.push(5); \
                 for value in values.iter() { println(*value); } \
                 let counter = Counter { n: 0 }; \
                 for value in counter { println(value); } \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "4\n5\nnext\n1\nnext\n2\nnext\n3\nnext\n");
    }

    #[test]
    fn language_protocols_ignore_same_named_inherent_methods() {
        let execution = execute(
            "struct Always { tag: Int32 } \
             impl Eq for Always { fn eq(&self, other: &Always) -> Bool { true } } \
             impl Always { fn eq(&self, other: &Always) -> Bool { false } } \
             struct Point { x: Int32 } \
             impl Ord for Point { \
                 fn cmp(&self, other: &Point) -> Ordering { \
                     if self.x < other.x { Ordering::Less } \
                     else if self.x > other.x { Ordering::Greater } \
                     else { Ordering::Equal } \
                 } \
             } \
             impl Point { fn cmp(&self, other: &Point) -> Ordering { Ordering::Greater } } \
             struct Counter { n: Int32 } \
             impl Iterator for Counter { \
                 type Item = Int32; \
                 fn next(&mut self) -> Option<Int32> { \
                     if self.n < 2 { self.n += 1; Some(self.n) } else { None } \
                 } \
             } \
             impl Counter { fn next(&mut self) -> Option<Int32> { None } } \
             fn main() { \
                 println(Always { tag: 1 } == Always { tag: 2 }); \
                 println(Point { x: 1 } < Point { x: 2 }); \
                 for value in (Counter { n: 0 }) { println(value); } \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "true\ntrue\n1\n2\n");
    }

    /// **By-value `Vec` iteration is refused by the front end (WP-C7.9 Packet E, `E0105`).**
    ///
    /// This test used to assert the oracle's drop schedule for `for value in values` over an owned
    /// `Vec` — each binding dropped at the end of its iteration, and on an early `break` the
    /// unconsumed tail dropped afterwards. That behaviour was real and is still implemented here,
    /// but it is no longer reachable from source: the form type-checked and ran in this engine
    /// while no compiler could lower it, which is exactly the accepted-but-unexecutable split
    /// Packet E closed.
    ///
    /// So the assertion is inverted rather than deleted, and the drop schedule it documented is
    /// recorded above for whoever implements the lowering: `body`, `first`, `body`, `second` for
    /// the full loop; `one`, `three`, `two` for the early `break` — the broken-from binding first,
    /// then the tail in order.
    #[test]
    fn by_value_vec_iteration_is_refused_before_execution() {
        let diagnostics = check_program(
            "struct Loud { label: String, stop: Bool } \
             impl Drop for Loud { fn drop(&mut self) { println(self.label.as_str()); } } \
             fn main() { \
                 let mut values: Vec<Loud> = Vec::new(); \
                 values.push(Loud { label: String::from(\"first\"), stop: false }); \
                 for value in values { println(\"body\"); } \
             }",
        );
        assert!(
            diagnostics
                .iter()
                .any(|d| d.code.as_deref() == Some("E0105")),
            "expected an E0105 refusal, got {diagnostics:?}"
        );
    }

    #[test]
    fn collection_discard_paths_run_stark_destructors() {
        let execution = execute(
            "struct Loud { label: String } \
             impl Drop for Loud { fn drop(&mut self) { println(self.label.as_str()); } } \
             impl Eq for Loud { fn eq(&self, other: &Loud) -> Bool { false } } \
             impl Hash for Loud { fn hash(&self) -> UInt64 { 0u64 } } \
             fn main() { \
                 { \
                     let mut values: Vec<Loud> = Vec::new(); \
                     values.push(Loud { label: String::from(\"vec-clear\") }); \
                     values.clear(); \
                 } \
                 { \
                     let mut map: HashMap<Int32, Loud> = HashMap::new(); \
                     map.insert(1, Loud { label: String::from(\"map-clear\") }); \
                     map.clear(); \
                 } \
                 { \
                     let mut map: HashMap<Int32, Loud> = HashMap::new(); \
                     map.insert(1, Loud { label: String::from(\"map-scope\") }); \
                 } \
                 { \
                     let mut set: HashSet<Loud> = HashSet::new(); \
                     set.insert(Loud { label: String::from(\"set-scope\") }); \
                 } \
             }",
        )
        .unwrap();
        assert_eq!(
            execution.output,
            "vec-clear\nmap-clear\nmap-scope\nset-scope\n"
        );
    }

    #[test]
    fn collection_replacement_and_removal_drop_consumed_keys() {
        let execution = execute(
            "struct Key { label: String } \
             impl Eq for Key { fn eq(&self, other: &Key) -> Bool { true } } \
             impl Hash for Key { fn hash(&self) -> UInt64 { 0u64 } } \
             impl Drop for Key { fn drop(&mut self) { println(self.label.as_str()); } } \
             fn main() { \
                 { \
                     let mut map: HashMap<Key, Int32> = HashMap::new(); \
                     map.insert(Key { label: String::from(\"map-stored\") }, 1); \
                     map.insert(Key { label: String::from(\"map-duplicate\") }, 2); \
                     map.remove(&Key { label: String::from(\"map-probe\") }); \
                 } \
                 { \
                     let mut set: HashSet<Key> = HashSet::new(); \
                     set.insert(Key { label: String::from(\"set-stored\") }); \
                     set.insert(Key { label: String::from(\"set-duplicate\") }); \
                     set.remove(&Key { label: String::from(\"set-probe\") }); \
                 } \
             }",
        )
        .unwrap();
        assert_eq!(
            execution.output,
            "map-duplicate\nmap-stored\nset-duplicate\nset-stored\n"
        );
    }

    #[test]
    fn returned_range_and_vec_as_slice_are_borrowed_views() {
        let execution = execute(
            "struct Buffer { values: Vec<Int32> } \
             impl Buffer { fn tail(&self) -> &[Int32] { &self.values[1..3] } } \
             struct Loud { label: String } \
             impl Drop for Loud { fn drop(&mut self) { println(self.label.as_str()); } } \
             fn main() { \
                 let mut numbers: Vec<Int32> = Vec::new(); \
                 numbers.push(10); numbers.push(20); numbers.push(30); \
                 let buffer = Buffer { values: numbers }; \
                 let tail = buffer.tail(); \
                 println(tail[0]); \
                 let mut values: Vec<Loud> = Vec::new(); \
                 values.push(Loud { label: String::from(\"once\") }); \
                 let slice = values.as_slice(); \
                 println(slice.len()); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "20\n1\nonce\n");
    }

    #[test]
    fn hash_collections_use_language_eq_for_keys() {
        let execution = execute(
            "struct Key { id: Int32 } \
             impl Eq for Key { fn eq(&self, other: &Key) -> Bool { true } } \
             impl Hash for Key { fn hash(&self) -> UInt64 { 0u64 } } \
             fn main() { \
                 let mut map: HashMap<Key, Int32> = HashMap::new(); \
                 map.insert(Key { id: 1 }, 10); \
                 map.insert(Key { id: 2 }, 20); \
                 println(map.len()); \
                 println(map.contains_key(&Key { id: 99 })); \
                 match map.get_mut(&Key { id: 75 }) { \
                     Some(value) => { *value = 30; } \
                     None => {} \
                 } \
                 println(*map.get(&Key { id: 50 }).unwrap()); \
                 let mut set: HashSet<Key> = HashSet::new(); \
                 println(set.insert(Key { id: 1 })); \
                 println(set.insert(Key { id: 2 })); \
                 println(set.contains(&Key { id: 99 })); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "1\ntrue\n30\ntrue\nfalse\ntrue\n");
    }

    /// WP-C2.2 (DEV-028): range indexing in a place context creates a slice view. Reads and
    /// writes through `&[T]`/`&mut [T]` use the original aggregate rather than a copied Array.
    #[test]
    fn range_index_references_are_slice_views() {
        let execution = execute(
            "fn main() { \
                 let mut values = [10, 20, 30, 40]; \
                 { \
                     let slice: &mut [Int32] = &mut values[1..3]; \
                     slice[0] = 99; \
                 } \
                 println(values[1]); \
                 let shared: &[Int32] = &values[0..=1]; \
                 println(shared[1]); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "99\n99\n");
    }

    /// WP-C2.2 (DEV-029): struct fields drop in reverse DECLARATION order (05-Memory-Model.md,
    /// CD-011), not reverse-alphabetical order. Two structs with the same fields declared in
    /// opposite orders must produce opposite drop orders — previously both produced the same
    /// (alphabetical) order.
    #[test]
    fn struct_fields_drop_in_reverse_declaration_order() {
        let alpha_first = execute(
            "struct Loud { label: String } \
             impl Drop for Loud { fn drop(&mut self) { println(self.label.as_str()); } } \
             struct Pair { alpha: Loud, beta: Loud } \
             fn main() { let p = Pair { \
                 alpha: Loud { label: String::from(\"alpha\") }, \
                 beta: Loud { label: String::from(\"beta\") } }; }",
        )
        .unwrap();
        assert_eq!(alpha_first.output, "beta\nalpha\n");

        let beta_first = execute(
            "struct Loud { label: String } \
             impl Drop for Loud { fn drop(&mut self) { println(self.label.as_str()); } } \
             struct Pair { beta: Loud, alpha: Loud } \
             fn main() { let p = Pair { \
                 beta: Loud { label: String::from(\"beta\") }, \
                 alpha: Loud { label: String::from(\"alpha\") } }; }",
        )
        .unwrap();
        assert_eq!(beta_first.output, "alpha\nbeta\n");
    }

    /// WP-C2.2 (DEV-029, enum variant): struct-like enum variant fields follow the same
    /// reverse-declaration drop order.
    #[test]
    fn enum_variant_named_fields_drop_in_reverse_declaration_order() {
        let execution = execute(
            "struct Loud { label: String } \
             impl Drop for Loud { fn drop(&mut self) { println(self.label.as_str()); } } \
             enum E { Named { zed: Loud, ack: Loud } } \
             fn main() { let e = E::Named { \
                 zed: Loud { label: String::from(\"zed\") }, \
                 ack: Loud { label: String::from(\"ack\") } }; }",
        )
        .unwrap();
        assert_eq!(execution.output, "ack\nzed\n");
    }

    /// WP-C2.2 (DEV-033): a core/builtin-type method call resolves its receiver before
    /// evaluating arguments (03-Type-System.md "Evaluation Order", CD-007/CD-010), and a
    /// side-effecting index subexpression inside the receiver runs exactly once — previously
    /// arguments evaluated first and the receiver place was re-resolved per branch.
    #[test]
    fn core_method_receiver_resolves_before_arguments_and_only_once() {
        let execution = execute(
            "fn idx() -> Int32 { println(\"idx\"); 0 } \
             fn arg() -> Int32 { println(\"arg\"); 5 } \
             fn main() { \
                 let mut vs: Vec<Vec<Int32>> = Vec::new(); \
                 let mut inner: Vec<Int32> = Vec::new(); \
                 inner.push(1); \
                 vs.push(inner); \
                 vs[idx()].push(arg()); \
                 println(*vs.get(0u64).unwrap().get(1u64).unwrap()); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "idx\narg\n5\n");
    }

    /// WP-C2.2 (DEV-032): `HashMap`/`HashSet` iterate in first-insertion order per
    /// `06-Standard-Library.md` "Iteration Order" (CD-009): re-inserting an existing key keeps
    /// its position; remove-then-reinsert moves it to the end. Previously the `BTreeMap`-backed
    /// representation iterated in structural-`Ord` sorted order.
    #[test]
    fn hashmap_iterates_in_first_insertion_order() {
        let execution = execute(
            "fn print_keys(m: &HashMap<Int32, String>) { \
                 let mut keys = m.keys(); \
                 while true { match keys.next() { Some(k) => println(*k), None => { break; } } } \
             } \
             fn main() { \
                 let mut m: HashMap<Int32, String> = HashMap::new(); \
                 m.insert(30, String::from(\"a\")); \
                 m.insert(10, String::from(\"b\")); \
                 m.insert(20, String::from(\"c\")); \
                 print_keys(&m); \
                 m.insert(10, String::from(\"updated\")); \
                 print_keys(&m); \
                 m.remove(&30); \
                 m.insert(30, String::from(\"again\")); \
                 print_keys(&m); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "30\n10\n20\n30\n10\n20\n10\n20\n30\n");
    }

    /// WP-C2.2 (DEV-032, set variant): `HashSet` iteration follows insertion order too.
    #[test]
    fn hashset_iterates_in_first_insertion_order() {
        let execution = execute(
            "fn main() { \
                 let mut s: HashSet<Int32> = HashSet::new(); \
                 s.insert(5); \
                 s.insert(1); \
                 s.insert(3); \
                 let mut it = s.iter(); \
                 while true { match it.next() { Some(v) => println(*v), None => { break; } } } \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "5\n1\n3\n");
    }

    /// WP-C2.2 (DEV-034 companion): `&mut self` and by-value receiver semantics preserved by
    /// the receiver-handling restructure — mutation writes back, and a by-value consume still
    /// moves the receiver.
    #[test]
    fn receiver_restructure_preserves_mutation_and_move_semantics() {
        let execution = execute(
            "struct Counter { n: Int32 } \
             impl Counter { \
                 fn bump(&mut self) { self.n += 1; } \
                 fn get(&self) -> Int32 { self.n } \
                 fn consume(self) -> Int32 { self.n } \
             } \
             fn main() { \
                 let mut c = Counter { n: 10 }; \
                 c.bump(); \
                 c.bump(); \
                 println(c.get()); \
                 println(c.consume()); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "12\n12\n");
    }

    #[test]
    fn pattern_bindings_keep_payload_and_destructured_types() {
        let execution = execute(
            "enum Message { Number(Int32), Named { amount: Int32 }, Empty } struct Point { x: Int32 } fn main() { let pair = (2, 3); match pair { (a, b) => println(a + b), } let value: Option<Int32> = Some(7); match value { Some(number) => println(number * 2), None => println(0), } let message = Message::Number(9); match message { Message::Number(number) => println(number + 1), Message::Named { amount } => println(amount), Message::Empty => println(0), } let named = Message::Named { amount: 13 }; match named { Message::Named { amount } => println(amount + 1), Message::Number(number) => println(number), Message::Empty => println(0), } let point = Point { x: 11 }; match point { Point { x } => println(x + 1), } }",
        )
        .unwrap();
        assert_eq!(execution.output, "5\n14\n10\n14\n12\n");
    }

    #[test]
    fn references_write_through_and_core_methods_auto_deref() {
        let execution = execute(
            "struct Counter { value: Int32 } impl Counter { fn bump(&mut self) { self.value += 1; } } fn add_value(values: &mut Vec<Int32>) { values.push(8); println(values.len()); } fn bump_counter(counter: &mut Counter) { counter.bump(); } fn main() { let mut values: Vec<Int32> = Vec::new(); values.push(3); add_value(&mut values); println(values.len()); println(*values.get(1u64).unwrap()); let mut counter = Counter { value: 4 }; bump_counter(&mut counter); println(counter.value); }",
        )
        .unwrap();
        assert_eq!(execution.output, "2\n2\n8\n5\n");
    }

    #[test]
    fn compares_borrowed_and_owned_strings() {
        let execution = execute(
            "fn is_alice(name: &str) -> Bool { name == \"alice\" } fn main() { let owned = String::from(\"alice\"); println(is_alice(owned.as_str())); println(owned == \"alice\"); println(\"alice\" < \"bob\"); }",
        )
        .unwrap();
        assert_eq!(execution.output, "true\ntrue\ntrue\n");
    }

    #[test]
    fn executes_custom_associated_functions() {
        let execution = execute(
            "struct Stack { size: Int32 } struct Holder<T> { value: T } impl Stack { fn new() -> Stack { Stack { size: 0 } } fn with_size(size: Int32) -> Stack { Stack { size: size } } fn identity<T>(value: T) -> T { value } } impl<T> Holder<T> { fn new(value: T) -> Self { Holder { value: value } } } fn main() { let empty = Stack::new(); let filled = Stack::with_size(4); println(empty.size + filled.size); println(Stack::identity(6)); let held: Holder<Int32> = Holder::new(7); println(held.value); }",
        )
        .unwrap();
        assert_eq!(execution.output, "4\n6\n7\n");
    }

    #[test]
    fn executes_trait_associated_conversion_functions() {
        let execution = execute(
            "struct Celsius { value: Int32 } \
             struct Fahrenheit { value: Int32 } \
             impl From<Celsius> for Fahrenheit { \
                 fn from(value: Celsius) -> Fahrenheit { \
                     Fahrenheit { value: value.value * 2 } \
                 } \
             } \
             fn main() { \
                 let c = Celsius { value: 10 }; \
                 let f: Fahrenheit = Fahrenheit::from(c); \
                 println(f.value); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "20\n");
    }

    #[test]
    fn builtin_display_and_hash_are_directly_callable() {
        let execution = execute(
            "fn main() { \
                 println((12i32).fmt()); \
                 println((-0.0f64).fmt()); \
                 println((1.0f64 / 0.0f64).fmt()); \
                 println(\"a\".hash()); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "12\n-0.0\ninf\n12638187200555641996\n");
    }

    #[test]
    fn file_is_a_first_class_noncopy_resource() {
        let test_name =
            portable_filename_component(std::thread::current().name().unwrap_or("test"));
        let path = std::env::temp_dir().join(format!(
            "stark-c2-11-file-{}-{}.txt",
            std::process::id(),
            test_name
        ));
        let source_path = stark_string_literal_contents(&path.to_string_lossy());
        let source = format!(
            "fn main() {{ \
                 let mut output: File = File::create(\"{}\").unwrap(); \
                 println(output.write_str(\"hello\").unwrap()); \
                 output.close().unwrap(); \
                 let mut input: File = File::open(\"{}\").unwrap(); \
                 println(input.read_to_string().unwrap()); \
                 input.close().unwrap(); \
             }}",
            source_path, source_path
        );
        let execution = execute(&source).unwrap();
        assert_eq!(execution.output, "5\nhello\n");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn primitive_constant_patterns_compare_by_compiler_known_value() {
        let execution = execute(
            "const ONE: Int32 = 1; \
             fn classify(value: Int32) -> Int32 { \
                 match value { ONE => 10, _ => 20 } \
             } \
             fn main() { println(classify(1)); println(classify(2)); }",
        )
        .unwrap();
        assert_eq!(execution.output, "10\n20\n");
    }

    #[test]
    fn fixed_width_numeric_boundaries_and_float_rounding_are_observable() {
        let execution = execute(
            "fn main() { \
                 println(127i8); println(-127i8 - 1i8); \
                 println(32767i16); println(-32767i16 - 1i16); \
                 println(2147483647i32); println(-2147483647i32 - 1i32); \
                 println(9223372036854775807i64); \
                 println(255u8); println(65535u16); println(4294967295u32); \
                 println(18446744073709551615u64); \
                 println(~0u8); println(7i32 / -3i32); println(7i32 % -3i32); \
                 println(2i32 ** 10i32); \
                 println(16777216.0f32 + 1.0f32 == 16777216.0f32); \
                 println((-0.0f64).fmt()); \
                 let inf = 1.0f64 / 0.0f64; let nan = 0.0f64 / 0.0f64; \
                 println(inf.fmt()); println(nan.fmt()); \
                 println(nan == nan); println(nan < 1.0f64); \
                 println((0.0001f64).fmt()); println((0.00001f64).fmt()); \
                 println((1000000000000000.0f64).fmt()); \
                 println((10000000000000000.0f64).fmt()); \
             }",
        )
        .unwrap();
        assert_eq!(
            execution.output,
            "127\n-128\n32767\n-32768\n2147483647\n-2147483648\n\
             9223372036854775807\n255\n65535\n4294967295\n18446744073709551615\n\
             255\n-2\n1\n1024\ntrue\n-0.0\ninf\nNaN\nfalse\nfalse\n\
             0.0001\n1e-5\n1000000000000000.0\n1e16\n"
        );
    }

    #[test]
    fn every_integer_width_traps_on_overflow_and_invalid_operations() {
        let cases = [
            "fn main() { println(127i8 + 1i8); }",
            "fn main() { println(32767i16 + 1i16); }",
            "fn main() { println(2147483647i32 + 1i32); }",
            "fn main() { println(9223372036854775807i64 + 1i64); }",
            "fn main() { println(255u8 + 1u8); }",
            "fn main() { println(65535u16 + 1u16); }",
            "fn main() { println(4294967295u32 + 1u32); }",
            "fn main() { println(18446744073709551615u64 + 1u64); }",
            "fn main() { let min = -127i8 - 1i8; println(-min); }",
            "fn main() { println(1i32 / 0i32); }",
            "fn main() { println(1i32 % 0i32); }",
            "fn main() { println(1u8 << 8u8); }",
            "fn main() { println(2i8 ** 7i8); }",
            "fn main() { println(256i32 as UInt8); }",
        ];
        for source in cases {
            let error = execute(source).unwrap_err();
            assert!(
                error.message.contains("overflow")
                    || error.message.contains("zero")
                    || error.message.contains("shift")
                    || error.message.contains("range"),
                "{source}: {error:?}"
            );
        }
    }

    #[test]
    fn unicode_boundaries_split_replace_trim_and_case_expansion_follow_core_contract() {
        let execution = execute(
            "fn main() { \
                 let text = String::from(\"Aé中😀\"); \
                 println(text.len()); println(text.find(\"中\").unwrap()); \
                 println(text.substring(1u64, 3u64)); \
                 let mut scalars: Int32 = 0; \
                 for part in text.split(\"\") { scalars = scalars + 1; } \
                 println(scalars); \
                 let mut empty_parts: Int32 = 0; \
                 for part in String::from(\"\").split(\",\") { empty_parts = empty_parts + 1; } \
                 let mut trailing_parts: Int32 = 0; \
                 for part in String::from(\"a,\").split(\",\") { trailing_parts = trailing_parts + 1; } \
                 println(empty_parts); println(trailing_parts); \
                 println(String::from(\"ab\").replace(\"\", \"-\")); \
                 println(String::from(\"\\u{2003}ok\\u{3000}\").trim()); \
                 println(String::from(\"ß\").to_uppercase()); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "10\n3\né\n4\n0\n2\n-a-b-\nok\nSS\n");

        let error = execute("fn main() { println(String::from(\"é\").substring(1u64, 2u64)); }")
            .unwrap_err();
        assert!(error.message.contains("UTF-8 boundaries"), "{error:?}");
    }

    #[test]
    fn vec_append_drains_the_source_and_preserves_order() {
        let execution = execute(
            "fn main() { \
                 let mut left: Vec<Int32> = Vec::new(); left.push(1); \
                 let mut right: Vec<Int32> = Vec::new(); right.push(2); right.push(3); \
                 left.append(&mut right); \
                 println(left.len()); println(right.len()); \
                 println(left[0]); println(left[1]); println(left[2]); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "3\n0\n1\n2\n3\n");
    }

    /// Post-WP-C2.11 correction (external review, independently reproduced before fixing):
    /// `==`/`Ord::cmp` desugar to borrowed trait calls (`Eq::eq(&self, other: &Self)`), so
    /// comparing two non-`Copy` operands must not move them. Before the fix, evaluating a bare
    /// local as a comparison operand went through ordinary move-or-copy evaluation
    /// (`eval_path`'s `Res::Local` arm unconditionally calls `take_place`), so `a == b` for two
    /// `String`s moved both operands out of their storage; using `a` afterward failed with
    /// "use of unavailable value" despite the comparison never taking ownership.
    #[test]
    fn comparison_operands_remain_usable_afterward() {
        let execution = execute(
            "fn main() { \
                 let a = String::from(\"a\"); \
                 let b = String::from(\"b\"); \
                 let _same = a == b; \
                 let _ne = a != b; \
                 let _lt = a < b; \
                 let _le = a <= b; \
                 let _gt = a > b; \
                 let _ge = a >= b; \
                 println(a.as_str()); \
                 println(b.as_str()); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "a\nb\n");
    }

    /// Companion to `comparison_operands_remain_usable_afterward`: generic `T: Eq`/`T: Ord`
    /// bounds dispatch through the same borrowed-operand path (`eval_binary`'s nominal-lookup
    /// branch), so a generic comparison function must not move its non-`Copy` arguments either.
    #[test]
    fn generic_eq_and_ord_bounds_do_not_move_their_operands() {
        let execution = execute(
            "fn compare<T: Ord>(x: T, y: T) -> Bool { x < y } \
             fn main() { \
                 let a = String::from(\"a\"); \
                 let b = String::from(\"b\"); \
                 println(compare(a.clone(), b.clone())); \
                 println(a.as_str()); \
                 println(b.as_str()); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "true\na\nb\n");
    }

    /// Post-WP-C2.11 correction: `?` inside an aggregate initializer (tuple/array/struct/enum
    /// literal) must stop construction immediately on early transfer, not evaluate later
    /// elements for their side effects. Before the fix, `expect_value` swallowed
    /// `Flow::Propagate` into `pending_propagation` and returned a dummy `Value::Unit`, so the
    /// `.map(expect_value).collect()` pattern used to build tuples/arrays kept going -- a later
    /// side-effecting element ran even though an earlier element had already propagated an
    /// error.
    #[test]
    fn early_transfer_inside_a_tuple_stops_later_elements_from_running() {
        let execution = execute(
            "fn fail() -> Result<Int32, String> { Err(String::from(\"boom\")) } \
             fn side_effect() -> Int32 { println(\"ran\"); 0 } \
             fn helper() -> Result<(Int32, Int32, Int32), String> { \
                 let value = (1, fail()?, side_effect()); \
                 Ok(value) \
             } \
             fn main() { \
                 let _ = helper(); \
                 println(\"done\"); \
             }",
        )
        .unwrap();
        assert_eq!(
            execution.output, "done\n",
            "side_effect() must not run once fail()? has already propagated"
        );
    }

    /// Companion to the tuple case: positional enum-variant construction (`Pair::Two(a, b)`) is
    /// aggregate construction via call syntax and shares the exact same underlying bug/fix
    /// (`eval_call`'s `Res::Variant` arm used the same unchecked `.map().collect()` pattern).
    #[test]
    fn early_transfer_inside_an_enum_variant_stops_later_elements_from_running() {
        let execution = execute(
            "enum Pair { Two(Int32, Int32) } \
             fn fail() -> Result<Int32, String> { Err(String::from(\"boom\")) } \
             fn side_effect() -> Int32 { println(\"ran\"); 0 } \
             fn helper() -> Result<Pair, String> { \
                 let value = Pair::Two(fail()?, side_effect()); \
                 Ok(value) \
             } \
             fn main() { \
                 let _ = helper(); \
                 println(\"done\"); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "done\n");
    }

    /// Post-WP-C2.11 correction: already-completed aggregate elements must be destroyed in
    /// reverse completion order when a later element's evaluation triggers early transfer
    /// (matching ordinary failed-aggregate-construction cleanup), not silently leaked.
    #[test]
    fn early_transfer_inside_a_tuple_drops_completed_elements_in_reverse_order() {
        let execution = execute(
            "struct Loud { label: String } \
             impl Drop for Loud { fn drop(&mut self) { println(self.label.as_str()); } } \
             fn fail() -> Result<Int32, String> { Err(String::from(\"boom\")) } \
             fn helper() -> Result<(Loud, Loud, Int32), String> { \
                 let value = ( \
                     Loud { label: String::from(\"first\") }, \
                     Loud { label: String::from(\"second\") }, \
                     fail()?, \
                 ); \
                 Ok(value) \
             } \
             fn main() { \
                 let _ = helper(); \
                 println(\"done\"); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "second\nfirst\ndone\n");
    }

    /// Post-WP-C2.11 correction: a finite float-to-integer cast truncates toward zero and traps
    /// only when the truncated result doesn't fit the target width -- it must not reject every
    /// value with a nonzero fractional part. Before the fix, `eval_cast` rejected any
    /// `value.fract() != 0.0`, so `3.9f64 as Int32` trapped instead of producing `3`.
    #[test]
    fn float_to_int_cast_truncates_toward_zero_instead_of_trapping_on_fractions() {
        let execution = execute(
            "fn main() { \
                 println(3.9f64 as Int32); \
                 println((-3.9f64) as Int32); \
                 println(0.5f64 as Int32); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "3\n-3\n0\n");
    }

    /// Companion negative case: NaN and infinities still trap (only the fractional-truncation
    /// behavior changed, not the NaN/infinite rejection).
    #[test]
    fn float_to_int_cast_still_traps_on_nan_and_infinity() {
        let error = execute("fn main() { println((0.0f64 / 0.0f64) as Int32); }").unwrap_err();
        assert!(
            error.message.contains("out of range"),
            "NaN must still trap: {error:?}"
        );
        let error = execute("fn main() { println((1.0f64 / 0.0f64) as Int32); }").unwrap_err();
        assert!(
            error.message.contains("out of range"),
            "infinity must still trap: {error:?}"
        );
    }

    /// Post-WP-C2.11 correction: signed `MIN % -1` traps even though its mathematical result
    /// (0) is representable, matching `MIN / -1` (already trapped: the wider `i128` carrier's
    /// `checked_div`/`checked_rem` succeed where the declared width would overflow, but for
    /// `Rem` the mathematical result always happens to fit back into the declared width, so the
    /// post-hoc range check alone never catches it). Scoped to `Rem` only -- `Div` already
    /// traps correctly and needed no change.
    #[test]
    fn signed_min_rem_negative_one_traps() {
        let error = execute(
            "fn main() { \
                 let base: Int8 = -127i8; \
                 let m: Int8 = base - 1i8; \
                 println(m % -1i8); \
             }",
        )
        .unwrap_err();
        assert!(error.message.contains("overflow"), "{error:?}");
    }

    /// Companion: `MIN / -1` already trapped before this fix and must continue to.
    #[test]
    fn signed_min_div_negative_one_still_traps() {
        let error = execute(
            "fn main() { \
                 let base: Int8 = -127i8; \
                 let m: Int8 = base - 1i8; \
                 println(m / -1i8); \
             }",
        )
        .unwrap_err();
        assert!(error.message.contains("overflow"), "{error:?}");
    }

    /// Companion: ordinary `Rem`/`Div` by values other than `-1`, and `MIN % -1`/`MIN / -1` for
    /// unsigned types (which have no negative MIN and so cannot trigger this trap), are
    /// unaffected.
    #[test]
    fn rem_and_div_by_values_other_than_negative_one_are_unaffected() {
        let execution = execute(
            "fn main() { \
                 let base: Int8 = -127i8; \
                 let m: Int8 = base - 1i8; \
                 println(m % 3i8); \
                 println(m / 3i8); \
                 println(7i8 % -1i8); \
                 let u: UInt8 = 200u8; \
                 println(u % 255u8); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "-2\n-42\n0\n200\n");
    }

    /// Post-WP-C2.11 correction: `Drop::drop(&mut self)` must operate on the destructor's real
    /// storage, not a clone. Before the fix, `drop_value` bound a *clone* of the value as
    /// `self`, so any mutation performed inside `drop()` (e.g. `replace(&mut self.field, ..)`)
    /// only affected the throwaway clone: the recursive field destruction that follows always
    /// saw the pristine, never-mutated original, so the pre-destructor field value was dropped
    /// a second time and the replacement value installed during `drop()` was never dropped at
    /// all.
    #[test]
    fn drop_mutation_through_mut_self_affects_real_storage() {
        let execution = execute(
            "struct Loud { label: String } \
             impl Drop for Loud { fn drop(&mut self) { println(self.label.as_str()); } } \
             struct Container { field: Loud } \
             impl Drop for Container { \
                 fn drop(&mut self) { \
                     let old = replace(&mut self.field, Loud { label: String::from(\"replacement\") }); \
                     println(\"dropping old explicitly:\"); \
                     drop(old); \
                 } \
             } \
             fn main() { \
                 let _c = Container { field: Loud { label: String::from(\"original\") } }; \
             }",
        )
        .unwrap();
        assert_eq!(
            execution.output, "dropping old explicitly:\noriginal\nreplacement\n",
            "the explicit drop(old) must print \"original\" exactly once, and the container's \
             own end-of-scope field destruction must see and drop the replacement value, not a \
             second copy of the original"
        );
    }

    /// Companion: an ordinary `Drop` impl that does not mutate `self` is unaffected by the
    /// move-instead-of-clone receiver change (already covered indirectly by
    /// `runs_drop_in_reverse_declaration_order` above; this pins down the single-value case
    /// specifically as a regression guard for the receiver-handling rewrite itself).
    #[test]
    fn drop_without_self_mutation_still_runs_exactly_once() {
        let execution = execute(
            "struct Loud { label: String } \
             impl Drop for Loud { fn drop(&mut self) { println(self.label.as_str()); } } \
             fn main() { \
                 let _value = Loud { label: String::from(\"once\") }; \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "once\n");
    }

    /// Post-WP-C2.11 correction: canonical display must use the shortest decimal
    /// representation that round-trips to the *declared* IEEE type. `Value::Float` stores every
    /// float as `f64` (Float32 results are rounded to `f32` precision but kept in the same
    /// `f64`-carrying representation), so `println`/`.fmt()` previously always formatted via
    /// `f64`'s shortest-round-trip algorithm even for a checked-Float32 value, producing digits
    /// like `0.10000000149011612` for `0.1f32` instead of the shorter, correct `0.1`.
    #[test]
    fn float32_println_and_fmt_use_float32_round_trip_digits_not_float64() {
        let execution = execute(
            "fn main() { \
                 let x: Float32 = 0.1f32; \
                 println(x); \
                 println(x.fmt()); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "0.1\n0.1\n");
    }

    /// Companion regression guard: `Float64` formatting must be completely unaffected by the
    /// Float32-awareness added to `format_runtime_value`/`.fmt()`.
    #[test]
    fn float64_println_and_fmt_are_unaffected_by_the_float32_fix() {
        let execution = execute(
            "fn main() { \
                 let x: Float64 = 0.1f64; \
                 println(x); \
                 println(x.fmt()); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "0.1\n0.1\n");
    }

    /// Correction-brief Issue 3: `Value::Float` now carries its own `FloatWidth` tag, so the
    /// *generic* recursive `Display for Value` impl (used whenever a Float32 is nested inside
    /// a printed tuple/array/struct/collection, with no static-type context available at that
    /// point) formats correctly too -- not just the top-level `println`/`.fmt()` paths the prior
    /// WP-C2.11 pass fixed via an external type-table lookup. Before this fix, a Float32 nested
    /// in a tuple printed via `f64`'s shortest-round-trip digits (`0.10000000149011612`).
    #[test]
    fn float32_nested_in_tuple_uses_float32_round_trip_digits() {
        let execution = execute(
            "fn main() { \
                 let pair: (Float32, Int32) = (0.1f32, 7); \
                 println(pair); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "(0.1, 7)\n");
    }

    #[test]
    fn float32_nested_in_array_uses_float32_round_trip_digits() {
        let execution = execute(
            "fn main() { \
                 let values: [Float32; 2] = [0.1f32, 2.5f32]; \
                 println(values); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "[0.1, 2.5]\n");
    }

    #[test]
    fn float32_nested_in_option_and_result_use_float32_round_trip_digits() {
        let execution = execute(
            "fn main() { \
                 let some_value: Option<Float32> = Some(0.1f32); \
                 let ok_value: Result<Float32, String> = Ok(0.1f32); \
                 println(some_value); \
                 println(ok_value); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "Some(0.1)\nOk(0.1)\n");
    }

    /// WP-C4.7-9 audit: this test used to print a bare `struct` and assert the debug-ish form
    /// `{x: 0.1}`. That relied on an OVER-ACCEPTANCE — 06-Standard-Library says `Display` is not
    /// a syntax hook and user types must implement it, so `println(p)` on a `Display`-less struct
    /// is now a compile-time error, and the reference interpreter no longer invents a format for
    /// one. Its original subject (a `Float32` nested in an aggregate keeps `f32` round-trip
    /// digits) is unchanged and covered by the `Option`/`Result` and tuple siblings above and
    /// below, which exercise the same `Display for Value` width-selection path.
    #[test]
    fn printing_a_struct_without_a_display_impl_is_rejected() {
        let diagnostics = type_diagnostics(
            "struct Point { x: Float32 } \
             fn main() { \
                 let p = Point { x: 0.1f32 }; \
                 println(p); \
             }",
        );
        assert!(
            diagnostics
                .iter()
                .any(|d| d.message.contains("does not implement 'Display'")),
            "expected a Display rejection, got {diagnostics:?}"
        );
    }

    /// A Float32 arithmetic result must keep its width tag through the operation (not just at
    /// literal construction), so it still formats with `f32` round-trip digits once nested.
    #[test]
    fn float32_arithmetic_result_nested_in_tuple_uses_float32_round_trip_digits() {
        let execution = execute(
            "fn main() { \
                 let a: Float32 = 0.1f32; \
                 let b: Float32 = 0.2f32; \
                 let sum = a + b; \
                 let pair = (sum, true); \
                 println(pair); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "(0.3, true)\n");
    }

    /// Distinguishes a genuine Float32 value from an equal-valued Float64: an explicit `as
    /// Float64` cast must widen to the full `f64` shortest-round-trip digits, proving the
    /// formatting difference tracks the value's own declared width rather than always rounding
    /// to `f32`.
    #[test]
    fn float32_cast_to_float64_uses_float64_round_trip_digits_not_float32() {
        let execution = execute(
            "fn main() { \
                 let x: Float32 = 0.1f32; \
                 let y: Float64 = x as Float64; \
                 println(y); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "0.10000000149011612\n");
    }

    /// Post-WP-C2.11 correction: the standard-library math contract classifies transcendental
    /// domain errors (e.g. `sqrt` of a negative number) as producing NaN, not a language trap --
    /// distinct from the numeric-trap rules governing integer overflow/division and
    /// float-to-int casts. Before the fix, `Builtin::Sqrt` returned a `RuntimeError` ("sqrt
    /// domain error") for any negative finite input.
    #[test]
    fn negative_sqrt_returns_nan_instead_of_trapping() {
        let execution = execute(
            "fn main() { \
                 let x = sqrt(-4.0f64); \
                 println(x != x); \
                 println(x); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "true\nNaN\n");
    }

    /// Companion: ordinary non-negative `sqrt` is unaffected.
    #[test]
    fn nonnegative_sqrt_is_unaffected() {
        let execution = execute("fn main() { println(sqrt(4.0f64)); }").unwrap();
        assert_eq!(execution.output, "2.0\n");
    }

    /// Runs `source` and returns the `Value` a zero-argument function named `function_name`
    /// evaluates to -- used by the Issue-4 NaN-canonicalization tests below to inspect a
    /// `Value::Float`'s exact bit pattern via `f64::to_bits`/`f32::to_bits`, which no STARK-level
    /// program can observe on its own (there is no bit-reinterpretation primitive in Core v1;
    /// `println`'s `NaN` text is bit-pattern-insensitive, since every NaN prints identically).
    fn eval_function_result(source: &str, function_name: &str) -> Value {
        let file = Arc::new(SourceFile::new("test.stark", source));
        let (ast, parse_diags) = parse(&file, ParseMode::Program);
        assert!(parse_diags.is_empty(), "parse diagnostics: {parse_diags:?}");
        let (hir, resolve_diags) = resolve(&ast, file.clone());
        assert!(
            resolve_diags.is_empty(),
            "resolve diagnostics: {resolve_diags:?}"
        );
        let checked = typecheck::analyze(&hir);
        assert!(
            checked
                .diagnostics
                .iter()
                .all(|diag| diag.severity != crate::diag::Severity::Error),
            "type diagnostics: {:?}",
            checked.diagnostics
        );
        let registered = hir
            .source_named(&file.name)
            .expect("the parse registered this file");
        let mut interpreter = Interpreter::new(&hir, registered, &checked.tables);
        let item_id = (0..hir.items.len())
            .map(|index| ItemId(index as u32))
            .find(|item| {
                matches!(&hir.item(*item).kind, hir::ItemKind::Fn(def) if interpreter.text(def.sig.name) == function_name)
            })
            .unwrap_or_else(|| panic!("function '{function_name}' not found"));
        let span = interpreter.hir.item(item_id).span;
        let callable = interpreter
            .item_callable(item_id)
            .unwrap_or_else(|| panic!("'{function_name}' is not callable"));
        interpreter
            .invoke_callable(
                ResolvedInvocation {
                    callable,
                    environment: InvocationEnv::Empty,
                },
                ReceiverSource::None,
                Vec::new(),
                span,
            )
            .unwrap_or_else(|error| panic!("evaluating '{function_name}' failed: {error:?}"))
    }

    /// Correction-brief Issue 4 (`NUM-FLOAT-OP-001`): "operations that create a NaN produce the
    /// canonical quiet NaN with sign zero and all payload bits other than the quiet bit zero" --
    /// a specific, fixed bit pattern, not merely "some NaN." `f64::to_bits` inspection proves the
    /// exact pattern, which printed `NaN` text alone cannot (every NaN prints identically
    /// regardless of sign or payload).
    #[test]
    fn division_by_zero_produces_the_canonical_quiet_nan_bit_pattern_for_float64() {
        let result = eval_function_result(
            "fn make() -> Float64 { 0.0f64 / 0.0f64 } fn main() { }",
            "make",
        );
        let Value::Float(value, FloatWidth::F64) = result else {
            panic!("expected a tagged Float64, got {result}");
        };
        assert!(value.is_nan());
        assert_eq!(value.to_bits(), 0x7ff8_0000_0000_0000);
    }

    #[test]
    fn division_by_zero_produces_the_canonical_quiet_nan_bit_pattern_for_float32() {
        let result = eval_function_result(
            "fn make() -> Float32 { 0.0f32 / 0.0f32 } fn main() { }",
            "make",
        );
        let Value::Float(value, FloatWidth::F32) = result else {
            panic!("expected a tagged Float32, got {result}");
        };
        assert!(value.is_nan());
        assert_eq!((value as f32).to_bits(), 0x7fc0_0000);
    }

    #[test]
    fn sqrt_of_negative_produces_the_canonical_quiet_nan_bit_pattern() {
        let result = eval_function_result(
            "fn make() -> Float64 { sqrt(-1.0f64) } fn main() { }",
            "make",
        );
        let Value::Float(value, FloatWidth::F64) = result else {
            panic!("expected a tagged Float64, got {result}");
        };
        assert!(value.is_nan());
        assert_eq!(value.to_bits(), 0x7ff8_0000_0000_0000);
    }

    /// `inf - inf` is a NaN *created* by the operation itself (not a NaN propagated from an
    /// already-NaN operand) -- both are required to canonicalize identically.
    #[test]
    fn infinity_minus_infinity_produces_the_canonical_quiet_nan_bit_pattern() {
        let result = eval_function_result(
            "fn make() -> Float64 { \
                 let inf = 1.0f64 / 0.0f64; \
                 inf - inf \
             } \
             fn main() { }",
            "make",
        );
        let Value::Float(value, FloatWidth::F64) = result else {
            panic!("expected a tagged Float64, got {result}");
        };
        assert!(value.is_nan());
        assert_eq!(value.to_bits(), 0x7ff8_0000_0000_0000);
    }

    /// A NaN *propagated* from an already-NaN operand into a further arithmetic operation must
    /// also canonicalize to the same bit pattern -- not merely a freshly-created NaN.
    #[test]
    fn arithmetic_on_an_already_nan_operand_produces_the_canonical_quiet_nan_bit_pattern() {
        let result = eval_function_result(
            "fn make() -> Float64 { \
                 let n = 0.0f64 / 0.0f64; \
                 n + 1.0f64 \
             } \
             fn main() { }",
            "make",
        );
        let Value::Float(value, FloatWidth::F64) = result else {
            panic!("expected a tagged Float64, got {result}");
        };
        assert!(value.is_nan());
        assert_eq!(value.to_bits(), 0x7ff8_0000_0000_0000);
    }

    /// Cross-operation assertion required by the correction brief: every distinct NaN-producing
    /// path (zero-divided-by-zero, negative `sqrt`, infinity subtraction, and a propagated-input
    /// operation) must yield bit-for-bit the same canonical pattern for a given width -- not just
    /// each individually matching the spec's literal bit pattern.
    #[test]
    fn every_nan_producing_path_yields_the_same_canonical_bits_for_float64() {
        let paths: &[&str] = &[
            "0.0f64 / 0.0f64",
            "sqrt(-1.0f64)",
            "(1.0f64 / 0.0f64) - (1.0f64 / 0.0f64)",
            "(0.0f64 / 0.0f64) + 1.0f64",
        ];
        let bits: Vec<u64> = paths
            .iter()
            .map(|expr| {
                let source = format!("fn make() -> Float64 {{ {expr} }} fn main() {{ }}");
                match eval_function_result(&source, "make") {
                    Value::Float(value, FloatWidth::F64) => value.to_bits(),
                    other => panic!("expected a tagged Float64, got {other}"),
                }
            })
            .collect();
        assert!(
            bits.iter().all(|&b| b == bits[0]),
            "expected every path to canonicalize to the same bits, got {bits:x?}"
        );
        assert_eq!(bits[0], 0x7ff8_0000_0000_0000);
    }

    /// Companion carve-out required by `NUM-FLOAT-OP-001`: unary negation flips whatever sign bit
    /// a NaN already has -- it must NOT be routed through canonicalization, since that's a bit
    /// operation on an existing value, not an operation that "creates" a NaN result.
    #[test]
    fn negating_a_canonical_nan_flips_its_sign_bit_instead_of_forcing_sign_zero() {
        let result = eval_function_result(
            "fn make() -> Float64 { -(0.0f64 / 0.0f64) } fn main() { }",
            "make",
        );
        let Value::Float(value, FloatWidth::F64) = result else {
            panic!("expected a tagged Float64, got {result}");
        };
        assert!(value.is_nan());
        assert_eq!(value.to_bits(), 0xfff8_0000_0000_0000);
    }

    /// DEV-055: a bare, glob-imported unit enum variant did not resolve at all as an
    /// expression -- `resolve_use_tree`'s `Glob` arm only ever consulted `submodule_map` (real
    /// modules), and an enum's variants are resolved dynamically through `item_details`, never
    /// pre-populated into a module's `items` map the way a real submodule's contents are. See
    /// `resolve.rs`'s `glob_imported_enum_variant_resolves_as_bare_expression` for the
    /// resolve-stage half of this regression.
    #[test]
    fn glob_imported_enum_variant_resolves_and_executes_as_bare_expression() {
        let execution = execute(
            "enum Color { Red, Green, Blue } \
             use Color::*; \
             fn main() { \
                 let c: Color = Red; \
                 println(\"ok\"); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "ok\n");
    }

    /// DEV-055's more severe half: in *pattern* position, a bare glob-imported variant used to
    /// exhibit DEV-053's exact wildcard-collapse symptom (the first arm matched unconditionally,
    /// later arms flagged unreachable) rather than genuinely discriminating on variant identity.
    /// Confirms `match Color::Blue { Red => 1, Green => 2, Blue => 3 }` now prints `3`, not `1`.
    #[test]
    fn glob_imported_enum_variant_discriminates_in_pattern_position_not_wildcard_collapsed() {
        let execution = execute(
            "enum Color { Red, Green, Blue } \
             use Color::*; \
             fn main() { \
                 let c = Color::Blue; \
                 let n = match c { Red => 1, Green => 2, Blue => 3 }; \
                 println(n); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "3\n");
    }

    /// Companion for the group-import form (`use Color::{Red, Green, Blue};`), which hit the
    /// identical `submodule_map`-only gap in `resolve_use_tree`'s `Group` arm.
    #[test]
    fn group_imported_enum_variants_discriminate_in_pattern_position() {
        let execution = execute(
            "enum Color { Red, Green, Blue } \
             use Color::{Red, Green, Blue}; \
             fn main() { \
                 let c = Color::Blue; \
                 let n = match c { Red => 1, Green => 2, Blue => 3 }; \
                 println(n); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "3\n");
    }

    /// DEV-060 [CLOSED]: end-to-end confirmation that the fixed program (see `typecheck/mod.rs`'s
    /// `repeated_call_to_unoverridden_default_trait_method_is_no_longer_flagged_as_move` for the
    /// decisive diagnostic-level regression) both type-checks *and* executes correctly -- two
    /// calls to an un-overridden trait default method on the same receiver now produce the
    /// correct output twice, not just "no diagnostic".
    #[test]
    fn repeated_call_to_unoverridden_default_trait_method_executes_correctly() {
        let execution = execute(
            "trait Greet { \
                 fn name(&self) -> String; \
                 fn greeting(&self) -> String { self.name() } \
             } \
             struct Person { label: String } \
             impl Greet for Person { \
                 fn name(&self) -> String { self.label.clone() } \
             } \
             fn main() { \
                 let p = Person { label: String::from(\"Ada\") }; \
                 println(p.greeting()); \
                 println(p.greeting()); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "Ada\nAda\n");
    }

    /// DEV-061 [CLOSED]: indirect calls through function-value locals and parameters execute.
    /// Covers CD-021 workload items 16 (typed fn-value local), 17 (fn value passed and invoked
    /// indirectly), and 22 (`f(f(v))` — repeated indirect invocation through one `Copy` local,
    /// which also exercises the DEV-062 borrowck fix end to end).
    #[test]
    fn indirect_calls_through_fn_value_locals_and_params_execute() {
        let execution = execute(
            "fn double(x: Int32) -> Int32 { x * 2 } \
             fn apply(f: fn(Int32) -> Int32, v: Int32) -> Int32 { f(v) } \
             fn main() { \
                 let f: fn(Int32) -> Int32 = double; \
                 println(f(21)); \
                 println(apply(double, 5)); \
                 println(apply(f, 7)); \
                 println(f(f(10))); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "42\n10\n14\n40\n");
    }

    /// TYPE-FN-002 (CD-027): a generic function coerced to a concrete fn type is the
    /// monomorphised instance and executes correctly through the fn value.
    #[test]
    fn generic_fn_coerced_to_fn_value_executes() {
        let execution = execute(
            "fn identity<T>(x: T) -> T { x } \
             fn main() { \
                 let f: fn(Int32) -> Int32 = identity; \
                 println(f(41) + 1); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "42\n");
    }

    /// DEV-063 [CLOSED]: the fn-value-consuming `Option`/`Result` combinators from
    /// 06-Standard-Library.md execute, including the pass-through sides (`None`, `Err`, `Ok`
    /// for `map_err`). Covers CD-021 workload item 18.
    #[test]
    fn option_result_combinators_execute_with_fn_values() {
        let execution = execute(
            "fn double(x: Int32) -> Int32 { x * 2 } \
             fn half(n: Int32) -> Option<Int32> { \
                 if n % 2 == 0 { Some(n / 2) } else { None } \
             } \
             fn describe(code: Int32) -> String { String::from(\"error\") } \
             fn main() { \
                 println(Some(21).map(double).unwrap()); \
                 match Some(10).and_then(half) { \
                     Some(v) => println(v), \
                     None => println(\"none\"), \
                 } \
                 match Some(7).and_then(half) { \
                     Some(v) => println(v), \
                     None => println(\"none\"), \
                 } \
                 let r: Result<Int32, Int32> = Ok(4); \
                 println(r.map(double).unwrap()); \
                 let e: Result<Int32, Int32> = Err(7); \
                 match e.map(double) { \
                     Ok(v) => println(v), \
                     Err(code) => println(code), \
                 } \
                 match e.map_err(describe) { \
                     Ok(v) => println(v), \
                     Err(msg) => println(msg.as_str()), \
                 } \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "42\n5\nnone\n8\n7\nerror\n");
    }

    /// Companion regression for DEV-060 (see `typecheck/mod.rs`'s
    /// `repeated_call_to_unoverridden_default_trait_method_is_no_longer_flagged_as_move` for the
    /// decisive diagnostic-level regression): two calls to an *overridden* trait method (not a
    /// default fallback) are unaffected by DEV-060.
    #[test]
    fn repeated_call_to_overridden_trait_method_is_unaffected_by_dev060() {
        let execution = execute(
            "trait Greet { fn name(&self) -> String; } \
             struct Person { label: String } \
             impl Greet for Person { \
                 fn name(&self) -> String { self.label.clone() } \
             } \
             fn main() { \
                 let p = Person { label: String::from(\"Ada\") }; \
                 println(p.name()); \
                 println(p.name()); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "Ada\nAda\n");
    }

    /// Companion: two calls to an ordinary inherent (non-trait) method are unaffected by
    /// DEV-060.
    #[test]
    fn repeated_call_to_inherent_method_is_unaffected_by_dev060() {
        let execution = execute(
            "struct Person { label: String } \
             impl Person { \
                 fn greeting(&self) -> String { self.label.clone() } \
             } \
             fn main() { \
                 let p = Person { label: String::from(\"Ada\") }; \
                 println(p.greeting()); \
                 println(p.greeting()); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "Ada\nAda\n");
    }

    /// DEV-051 end-to-end: a trait default method calling a sibling trait method through `self`
    /// (both directly, and transitively through a chain of two default methods) now type-checks
    /// *and* executes correctly. See `typecheck/mod.rs`'s `trait_default_method_calling_sibling_
    /// trait_method_through_self_type_checks` for the type-checking half of this regression.
    #[test]
    fn trait_default_method_calling_sibling_trait_method_through_self_executes() {
        let execution = execute(
            "trait Greet { \
                 fn name(&self) -> String; \
                 fn shout(&self) -> String { self.greeting() } \
                 fn greeting(&self) -> String { self.name() } \
             } \
             struct Person { label: String } \
             impl Greet for Person { \
                 fn name(&self) -> String { self.label.clone() } \
             } \
             fn main() { \
                 let p = Person { label: String::from(\"Ada\") }; \
                 println(p.shout()); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "Ada\n");
    }

    /// DEV-053 (found building the WP-C2.12 differential corpus, fixed as a follow-up
    /// investigation): a bare `None` pattern never matched by value -- `resolve.rs`'s
    /// `lower_pattern` only recognized `Res::Variant`/`Res::Item` as "known value" resolutions
    /// for a bare identifier, never `Res::Builtin` (which is how `None` is classified), so it
    /// unconditionally fell through to "fresh local binding." A `None` arm therefore silently
    /// matched *any* value with no diagnostic -- confirmed to produce **wrong runtime output**,
    /// not merely a spurious rejection: `match Some(5) { None => 999, Some(a) => a }` printed
    /// `999`. This is the decisive end-to-end regression for that fix; `resolve.rs`/
    /// `typecheck/` carries the resolution/type-checking half.
    #[test]
    fn bare_none_pattern_matches_by_value_not_as_a_wildcard() {
        let execution = execute(
            "fn main() { \
                 let value: Option<Int32> = Some(5); \
                 let r = match value { \
                     None => 999, \
                     Some(a) => a, \
                 }; \
                 println(r); \
             }",
        )
        .unwrap();
        assert_eq!(
            execution.output, "5\n",
            "None must not silently match Some(5); expected the Some(a) arm to apply"
        );
    }

    /// Companion: the same bug, nested inside a tuple pattern (the shape that originally
    /// surfaced it while building the WP-C2.12 corpus). `(None, x)` must only match when the
    /// first component is genuinely `None`, not unconditionally like `(_, x)`.
    #[test]
    fn nested_none_pattern_in_a_tuple_matches_by_value_not_as_a_wildcard() {
        let execution = execute(
            "fn main() { \
                 let pair: (Option<Int32>, Int32) = (Some(5), 10); \
                 let r = match pair { \
                     (None, x) => x, \
                     (Some(a), _) => a, \
                     _ => -1, \
                 }; \
                 println(r); \
             }",
        )
        .unwrap();
        assert_eq!(
            execution.output, "5\n",
            "(None, x) must not match (Some(5), 10); expected the (Some(a), _) arm to apply"
        );
    }

    /// DEV-054, closed by the same fix: two `None`s within one tuple pattern used to collide as
    /// duplicate bindings (`E0204`) because each was independently misclassified as introducing
    /// a fresh local named "None". Now that `None` correctly resolves to a value pattern (which
    /// introduces no binding at all), both occurrences coexist without conflict.
    #[test]
    fn repeated_none_within_one_tuple_pattern_no_longer_collides() {
        let execution = execute(
            "fn main() { \
                 let pair: (Option<Int32>, Option<Int32>) = (None, None); \
                 let r = match pair { \
                     (None, None) => 0, \
                     _ => 1, \
                 }; \
                 println(r); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "0\n");
    }

    /// Companion regression guard: ordinary `Some(x)`/`Ok(x)`/`Err(x)` payload patterns, and
    /// plain fresh-variable bindings, are unaffected by the `None`/builtin value-pattern fix.
    #[test]
    fn ordinary_binding_and_payload_patterns_are_unaffected_by_the_none_fix() {
        let execution = execute(
            "fn classify(value: Option<Int32>) -> Int32 { \
                 match value { \
                     Some(inner) => inner, \
                     None => -1, \
                 } \
             } \
             fn main() { \
                 println(classify(Some(7))); \
                 println(classify(None)); \
                 let x = 42; \
                 println(x); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "7\n-1\n42\n");
    }

    /// Correction brief Issue 1 (found post-WP-C2.12): `?` propagation was swallowed by
    /// `expect_value` into `pending_propagation` + a dummy `Value::Unit`, and only
    /// aggregate-construction call sites (tuple/array/struct/enum literals, fixed as DEV-045)
    /// checked the flag before continuing. Every other sequential-evaluation context --
    /// ordinary/associated/builtin function calls, method calls, binary operands, `&&`/`||`,
    /// assignment, ranges, repeat expressions, `if`/`while` conditions, match scrutinees, and
    /// `break` values -- kept evaluating later sub-expressions (and their side effects) after an
    /// earlier one had already propagated. Confirmed to produce real side effects that should
    /// never have run, not just a spurious diagnostic.
    #[test]
    fn try_in_call_argument_stops_later_arguments_and_callee() {
        let execution = execute(
            "fn fail() -> Result<Int32, String> { Err(String::from(\"boom\")) } \
             fn side_effect() -> Int32 { println(\"SIDE EFFECT\"); 2 } \
             fn sink(a: Int32, b: Int32) -> Int32 { println(\"CALLED\"); b } \
             fn helper() -> Result<Int32, String> { \
                 let value = sink(fail()?, side_effect()); \
                 Ok(value) \
             } \
             fn main() { \
                 let _ = helper(); \
                 println(\"done\"); \
             }",
        )
        .unwrap();
        assert_eq!(
            execution.output, "done\n",
            "side_effect() and sink()'s own body must not run once fail()? has propagated"
        );
    }

    /// Companion: the same bug for a user-method call's argument list, and for the qualified
    /// (`is_core_value`-gated `call_core_method`) dispatch path, which required a return-type
    /// adapter rather than a direct `Flow`-returning signature since it is a large dispatcher
    /// with a single caller (`call_method`) that checks `pending_propagation` immediately.
    #[test]
    fn try_in_method_argument_stops_later_arguments_and_method_body() {
        let execution = execute(
            "fn fail() -> Result<Int32, String> { Err(String::from(\"boom\")) } \
             fn side_effect() -> Int32 { println(\"SIDE EFFECT METHOD ARG\"); 2 } \
             struct Adder { total: Int32 } \
             impl Adder { \
                 fn add(&self, a: Int32, b: Int32) -> Int32 { println(\"METHOD CALLED\"); a + b } \
             } \
             fn helper() -> Result<Int32, String> { \
                 let adder = Adder { total: 0 }; \
                 let value = adder.add(fail()?, side_effect()); \
                 Ok(value) \
             } \
             fn main() { \
                 let _ = helper(); \
                 println(\"done\"); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "done\n");
    }

    /// Companion: `?` in the left operand of a binary expression (both the comparison-operator
    /// borrowing path and the ordinary arithmetic path) must stop the right operand from
    /// evaluating, instead of continuing with a dummy `Value::Unit` left operand.
    #[test]
    fn try_in_binary_operand_stops_rhs_evaluation() {
        let execution = execute(
            "fn fail() -> Result<Int32, String> { Err(String::from(\"boom\")) } \
             fn side_effect() -> Int32 { println(\"SIDE EFFECT RHS\"); 2 } \
             fn helper() -> Result<Int32, String> { \
                 let value = fail()? + side_effect(); \
                 Ok(value) \
             } \
             fn main() { \
                 let _ = helper(); \
                 println(\"done\"); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "done\n");
    }

    /// Companion: `&&`/`||` short-circuit correctly on `Bool` operands, but the right operand
    /// itself can also contain `?` -- propagating from the right operand of a short-circuit
    /// operator must not be silently converted into `false`/`true`.
    #[test]
    fn try_in_and_or_right_operand_propagates_not_converted_to_bool() {
        let execution = execute(
            "fn fail() -> Result<Bool, String> { Err(String::from(\"boom\")) } \
             fn side_effect() -> Bool { println(\"SIDE EFFECT AND\"); true } \
             fn helper_and() -> Result<Bool, String> { \
                 let v = true && (fail()? && side_effect()); \
                 Ok(v) \
             } \
             fn helper_or() -> Result<Bool, String> { \
                 let v = false || (fail()? || side_effect()); \
                 Ok(v) \
             } \
             fn main() { \
                 let a = helper_and(); \
                 let b = helper_or(); \
                 println(\"done\"); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "done\n");
    }

    /// Companion: `?` in a range's low bound must stop the high bound from evaluating.
    #[test]
    fn try_in_range_low_bound_stops_high_bound_evaluation() {
        let execution = execute(
            "fn fail() -> Result<Int32, String> { Err(String::from(\"boom\")) } \
             fn side_effect() -> Int32 { println(\"SIDE EFFECT RANGE HI\"); 5 } \
             fn helper() -> Result<Int32, String> { \
                 let r = fail()?..side_effect(); \
                 Ok(1) \
             } \
             fn main() { \
                 let _ = helper(); \
                 println(\"done\"); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "done\n");
    }

    /// Companion: `?` in a repeat expression's value must stop the array from being built (the
    /// count is always a compile-time constant per `02-Syntax-Grammar.md`, so only the repeated
    /// value position is reachable at runtime, but the fix covers both positions defensively).
    #[test]
    fn try_in_repeat_value_stops_array_construction() {
        let execution = execute(
            "fn fail() -> Result<Int32, String> { Err(String::from(\"boom\")) } \
             fn helper() -> Result<Int32, String> { \
                 let arr = [fail()?; 3]; \
                 println(\"ARRAY BUILT\"); \
                 Ok(1) \
             } \
             fn main() { \
                 let _ = helper(); \
                 println(\"done\"); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "done\n");
    }

    /// Companion: `break` did not check `pending_propagation` after evaluating its value
    /// expression, unlike `return` (which already did) -- `break fail()?;` wrapped the dummy
    /// `Value::Unit` into `Flow::Break(Value::Unit)` instead of propagating out of the
    /// enclosing function entirely.
    #[test]
    fn try_in_break_value_propagates_out_of_the_enclosing_function() {
        let execution = execute(
            "fn fail() -> Result<Int32, String> { Err(String::from(\"boom\")) } \
             fn helper() -> Result<Int32, String> { \
                 loop { \
                     break fail()?; \
                 } \
                 println(\"UNREACHABLE\"); \
                 Ok(1) \
             } \
             fn main() { \
                 let _ = helper(); \
                 println(\"done\"); \
             }",
        )
        .unwrap();
        assert_eq!(
            execution.output, "done\n",
            "break fail()? must propagate out of helper(), not print UNREACHABLE"
        );
    }

    /// Confirms already-completed, Drop-bearing call-argument temporaries are destroyed in
    /// reverse completion order when a later argument's evaluation triggers early transfer --
    /// matching the abstract machine's failed-aggregate-construction cleanup rule, now extended
    /// to ordinary call arguments via `eval_call_arguments`.
    #[test]
    fn try_drops_completed_call_argument_temporaries_in_reverse_order() {
        let execution = execute(
            "struct Loud { label: String } \
             impl Drop for Loud { fn drop(&mut self) { println(self.label.as_str()); } } \
             fn fail() -> Result<Int32, String> { Err(String::from(\"boom\")) } \
             fn sink(a: Loud, b: Loud, c: Int32) -> Int32 { c } \
             fn helper() -> Result<Int32, String> { \
                 let value = sink( \
                     Loud { label: String::from(\"first\") }, \
                     Loud { label: String::from(\"second\") }, \
                     fail()?, \
                 ); \
                 Ok(value) \
             } \
             fn main() { \
                 let _ = helper(); \
                 println(\"done\"); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "second\nfirst\ndone\n");
    }

    /// Regression guard: `return`'s own pre-existing propagation handling (the one sequential-
    /// evaluation context that was already correct before this fix) is unaffected.
    #[test]
    fn try_in_return_expression_still_propagates_without_dummy_unit() {
        let execution = execute(
            "fn fail() -> Result<Int32, String> { Err(String::from(\"boom\")) } \
             fn helper() -> Result<Int32, String> { \
                 if true { \
                     return Ok(fail()?); \
                 } \
                 Ok(0) \
             } \
             fn main() { \
                 let _ = helper(); \
                 println(\"done\"); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "done\n");
    }

    /// Correction brief Issue 2 (found post-WP-C2.12): `Eq::eq(&self, &other)`/`Ord::cmp(&self,
    /// &other)` borrow both operands -- they never take ownership. Before this fix,
    /// `eval_binary`'s nominal-dispatch path passed owned clones for both: the receiver's clone
    /// silently vanished via ordinary Rust-level drop with *no* STARK-level `Drop::drop` call at
    /// all (data loss), while the argument's clone got a *real*, extra `Drop::drop` call fired
    /// by the callee's own normal per-parameter cleanup, before the comparison's caller-visible
    /// side effects (`println("after")`) even ran. Fixed by resolving each place operand's real
    /// `Place` (`resolve_comparison_operand`) and passing `Value::Ref(place)` into the dispatch
    /// instead of a clone.
    #[test]
    fn eq_on_drop_type_does_not_create_or_drop_clones() {
        let execution = execute(
            "struct Key { label: String } \
             impl Eq for Key { fn eq(&self, other: &Key) -> Bool { true } } \
             impl Drop for Key { fn drop(&mut self) { println(self.label.as_str()); } } \
             fn main() { \
                 let a = Key { label: String::from(\"a\") }; \
                 let b = Key { label: String::from(\"b\") }; \
                 println(a == b); \
                 println(\"after\"); \
             }",
        )
        .unwrap();
        assert_eq!(
            execution.output, "true\nafter\nb\na\n",
            "no destructor may run before \"after\"; a and b must each drop exactly once, at \
             their own normal (reverse-declaration-order) scope end"
        );
    }

    /// Companion: the same borrow contract for `Ord::cmp`.
    #[test]
    fn ord_on_drop_type_does_not_create_or_drop_clones() {
        let execution = execute(
            "struct Key { label: String, rank: Int32 } \
             impl Ord for Key { fn cmp(&self, other: &Key) -> Ordering { \
                 if self.rank < other.rank { Ordering::Less } \
                 else if self.rank > other.rank { Ordering::Greater } \
                 else { Ordering::Equal } \
             } } \
             impl Drop for Key { fn drop(&mut self) { println(self.label.as_str()); } } \
             fn main() { \
                 let a = Key { label: String::from(\"a\"), rank: 1 }; \
                 let b = Key { label: String::from(\"b\"), rank: 2 }; \
                 println(a < b); \
                 println(\"after\"); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "true\nafter\nb\na\n");
    }

    /// Companion: a field/index place operand (not a bare local) must also borrow its original
    /// storage rather than a clone-then-new-temporary.
    #[test]
    fn comparison_of_field_and_index_places_borrows_original_storage() {
        let execution = execute(
            "struct Key { label: String } \
             impl Eq for Key { fn eq(&self, other: &Key) -> Bool { true } } \
             impl Drop for Key { fn drop(&mut self) { println(self.label.as_str()); } } \
             struct Holder { key: Key } \
             fn main() { \
                 let holder = Holder { key: Key { label: String::from(\"held\") } }; \
                 let mut values: Vec<Key> = Vec::new(); \
                 values.push(Key { label: String::from(\"indexed\") }); \
                 println(holder.key == values[0]); \
                 println(\"after\"); \
             }",
        )
        .unwrap();
        assert_eq!(
            execution.output, "true\nafter\nindexed\nheld\n",
            "no destructor may run before \"after\"; holder/values are real named locals, so \
             their fields correctly drop at main's own scope end (reverse declaration order), \
             not before or during the comparison"
        );
    }

    /// Confirms a non-place (temporary, no-other-owner) comparison operand is still evaluated
    /// exactly once and destroyed exactly once, after the comparison completes -- using
    /// `promote_to_owned_temp_place` rather than the plain `promote_to_temp_place` helper, which
    /// (found while fixing this issue) does not register its temporary in `Frame::order` at all,
    /// so a value placed there is silently discarded via ordinary Rust-level deallocation with
    /// no `Drop::drop` call ever firing.
    #[test]
    fn comparison_of_temporary_operands_evaluates_each_once_and_drops_after_call() {
        let execution = execute(
            "struct Key { label: String } \
             impl Eq for Key { fn eq(&self, other: &Key) -> Bool { true } } \
             impl Drop for Key { fn drop(&mut self) { println(self.label.as_str()); } } \
             fn make(label: String) -> Key { Key { label: label } } \
             fn main() { \
                 println(make(String::from(\"temp_left\")) == make(String::from(\"temp_right\"))); \
                 println(\"after\"); \
             }",
        )
        .unwrap();
        assert_eq!(
            execution.output, "true\nafter\ntemp_right\ntemp_left\n",
            "each temporary must drop exactly once, after the comparison, in reverse creation \
             order -- not silently leaked and not dropped before \"after\""
        );
    }

    /// Companion regression guard: an ordinary `&self` method call (not a comparison) already
    /// worked correctly before this fix (`call_user_method`'s own receiver-extraction-before-
    /// cleanup handling) and must remain unaffected by the `promote_to_owned_temp_place`
    /// addition or the `eval_binary` signature change.
    #[test]
    fn shared_receiver_method_observes_original_place_without_owned_clone_cleanup() {
        let execution = execute(
            "struct Key { label: String } \
             impl Key { fn describe(&self) -> String { self.label.clone() } } \
             impl Drop for Key { fn drop(&mut self) { println(self.label.as_str()); } } \
             fn main() { \
                 let a = Key { label: String::from(\"a\") }; \
                 println(a.describe().as_str()); \
                 println(\"after\"); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "a\nafter\na\n");
    }

    /// DEV-052: `Trait::method(...)` fully-qualified call syntax (`03-Type-System.md`:670)
    /// resolved and executed for a user-declared trait but failed at resolve time
    /// (`E0200 undefined variable 'Eq::eq'`) for a compiler-known `CoreTrait` (`Eq`, `Ord`,
    /// `Hash`, `Display`, `Clone`, `Default`). Root cause: `resolve_path_relative`'s
    /// multi-segment loop only ever continued past a first segment resolving to
    /// `Res::Item(item_id)` (a real trait *declaration* item, indexed by member position) --
    /// never past `Res::CoreTrait(core_trait)`, since a `CoreTrait` has no such declaration item
    /// at all. Fixed by adding a new `Res::CoreTraitMember(CoreTrait, Span)`, resolved when the
    /// second segment names that `CoreTrait`'s one fixed callable method
    /// (`core_trait_method_name`, shared between `resolve.rs` and `interp.rs`), and dispatched
    /// through the *same* `find_method(..., Some(Res::CoreTrait(core_trait)))` lookup the
    /// `==`/`<`/etc. operator sugar already uses -- a qualified call is just an explicit
    /// spelling of the same dispatch, not a separate mechanism.
    #[test]
    fn qualified_call_to_core_trait_eq_method_resolves_and_executes() {
        let execution = execute(
            "struct Point { x: Int32 } \
             impl Eq for Point { fn eq(&self, other: &Point) -> Bool { self.x == other.x } } \
             fn main() { \
                 let a = Point { x: 1 }; \
                 let b = Point { x: 1 }; \
                 let c = Point { x: 2 }; \
                 println(Eq::eq(&a, &b)); \
                 println(Eq::eq(&a, &c)); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "true\nfalse\n");
    }

    /// Companion: `Ord::cmp` (a different `CoreTrait`, a different fixed method name) resolves
    /// and executes too, confirming the fix isn't accidentally specific to `Eq`.
    #[test]
    fn qualified_call_to_core_trait_ord_method_resolves_and_executes() {
        let execution = execute(
            "struct Point { x: Int32 } \
             impl Ord for Point { \
                 fn cmp(&self, other: &Point) -> Ordering { \
                     if self.x < other.x { Ordering::Less } \
                     else if self.x > other.x { Ordering::Greater } \
                     else { Ordering::Equal } \
                 } \
             } \
             fn main() { \
                 let a = Point { x: 1 }; \
                 let b = Point { x: 2 }; \
                 println(Ord::cmp(&a, &b)); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "Less\n");
    }

    /// Companion regression guard: the qualified-call syntax for a user-*declared* trait
    /// (`Res::TraitMember`, `call_qualified_trait`) is a separate code path from the new
    /// `CoreTrait` handling and must remain unaffected by it.
    #[test]
    fn qualified_call_to_user_declared_trait_is_unaffected_by_the_core_trait_fix() {
        let execution = execute(
            "trait Describe { fn describe(&self) -> String; } \
             struct Widget { label: String } \
             impl Describe for Widget { \
                 fn describe(&self) -> String { self.label.clone() } \
             } \
             fn main() { \
                 let w = Widget { label: String::from(\"gadget\") }; \
                 println(Describe::describe(&w)); \
             }",
        )
        .unwrap();
        assert_eq!(execution.output, "gadget\n");
    }
}
