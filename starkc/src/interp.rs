//! Gate 3 tree-walking interpreter for typed STARK HIR.

use crate::ast::{AssignOp, BinOp, Lit, Primitive, UnOp};
use crate::diag::Diagnostic;
use crate::hir::{
    self, BlockId, Builtin, CoreTrait, CoreType, ExprId, Hir, ItemId, LocalId, PatId, Res, StmtId,
};
use crate::literal::{self, LitValue};
use crate::source::{SourceFile, Span};
use crate::typecheck::{Ty, TypeTables};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::io::{Read, Write};
use std::rc::Rc;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct RuntimeError {
    pub message: String,
    pub span: Span,
    /// False for executable-target selection failures detected before the
    /// entrypoint starts. Those are compiler errors, not language traps.
    pub is_trap: bool,
}

impl RuntimeError {
    fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
            is_trap: true,
        }
    }

    fn entry(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
            is_trap: false,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Execution {
    pub output: String,
    /// Core process status produced by normal entrypoint completion.
    pub status: u8,
    /// Bytes destined for the Core stderr stream on normal `Err` completion.
    pub stderr: String,
}

/// Evaluate every declared constant before execution. This uses the same
/// abstract-machine operations as the interpreter, but only after a closed
/// syntactic-subset check has excluded runtime state and side effects.
pub fn check_constants(hir: &Hir, file: Arc<SourceFile>, tables: &TypeTables) -> Vec<Diagnostic> {
    let mut interpreter = Interpreter::new(hir, file.clone(), tables);
    interpreter.frames.push(Frame::default());
    let mut diagnostics = Vec::new();
    for (index, item) in hir.items.iter().enumerate() {
        let item_id = ItemId(index as u32);
        let hir::ItemKind::Const { value, .. } = &item.kind else {
            continue;
        };
        let item_file = hir
            .item_files
            .get(&item_id)
            .cloned()
            .unwrap_or_else(|| file.clone());
        if let Err((span, message)) = constant_expr_allowed(hir, *value) {
            diagnostics.push(
                Diagnostic::error(message, span)
                    .with_code("E0215")
                    .with_file(item_file),
            );
            continue;
        }
        if let Err(error) = interpreter.eval_const_item(item_id) {
            diagnostics.push(
                Diagnostic::error(
                    format!("constant evaluation failed: {}", error.message),
                    error.span,
                )
                .with_code("E0215")
                .with_file(item_file),
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
/// typed `Float64 -> Float64` only (`typecheck.rs`'s builtin signatures), so they always
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
    Option(Option<Box<Value>>),
    Result(Result<Box<Value>, Box<Value>>),
    Range {
        start: i128,
        end: i128,
        inclusive: bool,
    },
    /// WP-C2.2 (DEV-028): an unsized slice is a view into an existing aggregate, not a copied
    /// `Array`. The bounds are half-open indices into `place`.
    Slice(Place, usize, usize),
    Ref(Place),
    Function(ItemId),
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
            Value::Option(Some(value)) => write!(f, "Some({value})"),
            Value::Option(None) => write!(f, "None"),
            Value::Result(Ok(value)) => write!(f, "Ok({value})"),
            Value::Result(Err(value)) => write!(f, "Err({value})"),
            Value::Range {
                start,
                end,
                inclusive,
            } => write!(f, "{start}..{}{end}", if *inclusive { "=" } else { "" }),
            Value::Slice(_, start, end) => write!(f, "<slice {start}..{end}>"),
            Value::Ref(_) => write!(f, "<reference>"),
            Value::Function(item) => write!(f, "fn#{}", item.0),
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
            (Value::Function(a), Value::Function(b)) => a.cmp(b),
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
}

/// DEV-065: an out-of-range `Index` projection is the language's index-out-of-bounds TRAP
/// (CORE-V1-ABSTRACT-MACHINE), not a moved-field condition — the generic message was
/// misleading for the most common trap a user can hit.
fn projection_failure_message(projection: &Projection) -> &'static str {
    match projection {
        Projection::Index(_) | Projection::MapIndex(_) => "index out of bounds",
        Projection::Field(_) => "use of moved or invalid field",
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
    /// DEV-069: the file that DECLARES this body. Spans are file-relative, so every span read
    /// while executing the body (literals, field names, path segments) must resolve against
    /// this file — not the entry file, and not the caller's file.
    file: Arc<SourceFile>,
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
                u8::try_from(value).map_err(|_| RuntimeError::new("invalid-exit-status", span))
            }
            _ => Err(RuntimeError::new(
                "entrypoint returned a value inconsistent with its checked signature",
                span,
            )),
        }
    }

    match value {
        Value::Result(Ok(value)) => Ok((checked_status(*value, span)?, String::new())),
        Value::Result(Err(message)) => match *message {
            Value::String(message) | Value::Str(message) => Ok((1, format!("{message}\n"))),
            _ => Err(RuntimeError::new("entrypoint error is not a String", span)),
        },
        value => Ok((checked_status(value, span)?, String::new())),
    }
}

pub fn run(
    hir: &Hir,
    file: Arc<SourceFile>,
    tables: &TypeTables,
) -> Result<Execution, RuntimeError> {
    let mut interpreter = Interpreter::new(hir, file, tables);
    let (status, stderr) = interpreter.run_main()?;
    Ok(Execution {
        output: interpreter.output,
        status,
        stderr,
    })
}

/// Like [`run`], but a failure also carries the stdout accumulated before it. The MIR
/// differential comparator (C4.5e-0) needs output equality on trap paths too — two programs
/// printing different prefixes before the same trap are observably different.
pub fn run_with_partial_output(
    hir: &Hir,
    file: Arc<SourceFile>,
    tables: &TypeTables,
) -> Result<Execution, (RuntimeError, String)> {
    let mut interpreter = Interpreter::new(hir, file, tables);
    match interpreter.run_main() {
        Ok((status, stderr)) => Ok(Execution {
            output: interpreter.output,
            status,
            stderr,
        }),
        Err(error) => Err((error, interpreter.output)),
    }
}

/// Execute a specific zero-argument, receiverless function `item` as the
/// program entry point instead of `main` — used by the test runner
/// (`test_runner::run_test`) to invoke each discovered `test_*` function.
pub fn run_item(
    hir: &Hir,
    file: Arc<SourceFile>,
    tables: &TypeTables,
    item: ItemId,
) -> Result<Execution, RuntimeError> {
    let mut interpreter = Interpreter::new(hir, file, tables);
    let span = interpreter.hir.item(item).span;
    let callable = interpreter
        .item_callable(item)
        .ok_or_else(|| RuntimeError::new("item is not executable", span))?;
    interpreter.call_callable(callable, None, Vec::new(), span)?;
    Ok(Execution {
        output: interpreter.output,
        status: 0,
        stderr: String::new(),
    })
}

struct Interpreter<'a> {
    hir: &'a Hir,
    file: Arc<SourceFile>,
    tables: &'a TypeTables,
    frames: Vec<Frame>,
    output: String,
    copy_items: HashSet<ItemId>,
    pending_propagation: Option<Value>,
    const_cache: HashMap<ItemId, Value>,
    const_stack: Vec<ItemId>,
}

impl<'a> Interpreter<'a> {
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
            Value::Result(res) => match res {
                Ok(val) => Value::Result(Ok(Box::new(self.default_value_for(val)))),
                Err(err) => Value::Result(Err(Box::new(self.default_value_for(err)))),
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
            Value::Function(item) => Value::Function(*item),
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

    fn new(hir: &'a Hir, file: Arc<SourceFile>, tables: &'a TypeTables) -> Self {
        let copy_items = hir
            .items
            .iter()
            .filter_map(|item| match &item.kind {
                hir::ItemKind::Impl {
                    trait_:
                        Some(hir::TraitRef {
                            res: Res::CoreTrait(hir::CoreTrait::Copy),
                            ..
                        }),
                    self_ty,
                    ..
                } => match &hir.ty(*self_ty).kind {
                    hir::TypeKind::Path {
                        res: Res::Item(item),
                        ..
                    } => Some(*item),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        Self {
            hir,
            file,
            tables,
            frames: Vec::new(),
            output: String::new(),
            copy_items,
            pending_propagation: None,
            const_cache: HashMap::new(),
            const_stack: Vec::new(),
        }
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

    /// Read a span belonging to the body currently EXECUTING. `call_callable` keeps `self.file`
    /// on the executing body's declaring file, so this is correct for literals, path segments,
    /// and other spans inside that body — and wrong for spans of any other item, which must go
    /// through `item_text` (DEV-069).
    ///
    /// Non-panicking since WP-C4.7-4: reading a foreign span used to abort the interpreter with
    /// "byte index N out of bounds" whenever the other file was longer (DEV-069 shape (a)).
    fn text(&self, span: Span) -> &str {
        self.file
            .src
            .get(span.lo as usize..span.hi as usize)
            .unwrap_or("?")
    }

    /// The file that declares `item`.
    fn item_file(&self, item: ItemId) -> Arc<SourceFile> {
        match self.hir.item_files.get(&item) {
            Some(file) => file.clone(),
            None => self.file.clone(),
        }
    }

    /// Read a span belonging to `item`, against the file that declares it (DEV-069). Used for
    /// every cross-item read: another struct's field names, another impl's method names, a
    /// trait's method names.
    fn item_text(&self, item: ItemId, span: Span) -> &str {
        let src = match self.hir.item_files.get(&item) {
            Some(file) => &file.src,
            None => &self.file.src,
        };
        src.get(span.lo as usize..span.hi as usize).unwrap_or("?")
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
                Span::point(0),
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
        let result = self.call_callable(callable, None, Vec::new(), self.hir.item(main).span)?;
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
            file: self.item_file(item),
        })
    }

    fn call_callable(
        &mut self,
        callable: Callable,
        receiver: Option<Value>,
        args: Vec<Value>,
        span: Span,
    ) -> Result<Value, RuntimeError> {
        if args.len() != callable.params.len() {
            return Err(RuntimeError::new("runtime argument count mismatch", span));
        }
        let mut frame = Frame::default();
        if let (Some((_, local)), Some(value)) = (callable.receiver, receiver) {
            frame.insert(local, Some(value));
        }
        for (local, value) in callable.params.iter().copied().zip(args) {
            frame.insert(local, Some(value));
        }
        self.frames.push(frame);
        // DEV-069: a body executes against ITS OWN file. Saved and restored around the call so a
        // cross-file call returns the caller's file — this is the interpreter's analogue of
        // typecheck's per-item file swap, and it must be restored on the error path too.
        let caller_file = std::mem::replace(&mut self.file, callable.file);
        let result = self.eval_block(callable.body);
        if result.is_err() {
            self.file = caller_file;
            self.frames.pop();
            return result.map(|_| Value::Unit);
        }
        let flow = result?;
        self.cleanup_current_frame()?;
        self.file = caller_file;
        self.frames.pop();
        match flow {
            Flow::Value(value) | Flow::Return(value) => Ok(value),
            Flow::Propagate(value) => Ok(value),
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
                self.frame_mut().insert(*local, value);
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

    fn eval_expr(&mut self, expr_id: ExprId) -> Result<Flow, RuntimeError> {
        let expr = self.hir.expr(expr_id);
        match &expr.kind {
            hir::ExprKind::Lit(lit) => {
                let value = self.eval_lit(*lit, expr.span)?;
                Ok(Flow::Value(
                    self.normalize_numeric(value, expr_id, expr.span)?,
                ))
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
                    self.eval_binary(
                        assign_binop(*op),
                        (current, None),
                        (right, None),
                        expr_id,
                        expr.span,
                    )?
                };
                self.write_place(&place, value, expr.span)?;
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
                    Value::Option(Some(value)) => Ok(Flow::Value(*value)),
                    Value::Option(None) => Ok(Flow::Propagate(Value::Option(None))),
                    Value::Result(Ok(value)) => Ok(Flow::Value(*value)),
                    Value::Result(Err(value)) => Ok(Flow::Propagate(Value::Result(Err(value)))),
                    _ => Err(RuntimeError::new(
                        "'?' requires Option or Result",
                        expr.span,
                    )),
                }
            }
            hir::ExprKind::Tuple(values) => match self.eval_aggregate_elements(values)? {
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
                self.eval_struct_lit(*res, fields, expr.span)
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
                let value = self.expect_value(*scrutinee)?;
                if let Some(propagated) = self.pending_propagation.take() {
                    return Ok(Flow::Propagate(propagated));
                }
                for arm in arms {
                    let mut bindings = Vec::new();
                    if self.match_pattern(arm.pat, &value, &mut bindings)? {
                        for (local, value) in &bindings {
                            self.frame_mut().insert(*local, Some(value.clone()));
                        }
                        let flow = self.eval_expr(arm.body)?;
                        let locals: Vec<_> = bindings.iter().map(|(local, _)| *local).collect();
                        self.cleanup_locals(&locals)?;
                        // WP-C2.2 (DEV-030): the match consumed the scrutinee; anything the
                        // matched pattern did not bind must still be dropped exactly once.
                        // Runs after the arm body and after the bindings' own cleanup,
                        // mirroring "the scrutinee temporary outlives the arm" scoping.
                        self.drop_unbound(arm.pat, value)?;
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
                match iterable {
                    Value::Range { .. } | Value::Array(_) | Value::Vec(_) | Value::Slice(..) => {
                        let mut remaining = self.iter_values(iterable, expr.span)?.into_iter();
                        while let Some(value) = remaining.next() {
                            self.frame_mut().insert(*local, Some(value));
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
                            self.next_for_iterator(&iterator_place, expr.span)?
                        {
                            self.frame_mut().insert(*local, Some(value));
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
            Flow::Value(value) => Ok(value),
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
        let value = literal::eval_lit_value(lit, text)
            .ok_or_else(|| RuntimeError::new("invalid literal", span))?;
        // WP-C1.5 (DEV-015): defense-in-depth mirror of typecheck.rs's suffixed-literal
        // magnitude check (`check_expr`'s `Lit::Int` arm) -- re-verified here in case a literal
        // ever reaches evaluation without having gone through that check (e.g. a future
        // alternate entry point). Unsuffixed-literal-vs-inferred-type magnitude is not
        // re-checked here since that requires the type table typecheck.rs already consulted;
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
                hir::ItemKind::Fn(_) => Ok(Value::Function(item)),
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
                .ok_or_else(|| RuntimeError::new("integer overflow", span))
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
            // with custom comparison logic (typecheck.rs's `require_operator_bound` already
            // requires such an impl to exist for any struct/enum `==`, so this dispatch cannot
            // find a program that type-checks but has no matching impl). Primitives and
            // Ty::Core container types (Option/Result/Vec/Box/String) have no user-overridable
            // Eq impl in Core v1 (operator overloading is a future extension per the spec), so
            // structural comparison remains exactly correct for them -- only struct/enum values
            // are looked up here. See COMPILER-STATE.md DEV-008.
            if let Some(nominal) = nominal_item(&left) {
                if let Some(method) = self.find_method(
                    Some(nominal),
                    "eq",
                    Some(Res::CoreTrait(hir::CoreTrait::Eq)),
                ) {
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
                        Value::Ref(receiver_place),
                        vec![Value::Ref(argument_place)],
                        span,
                    )?;
                    let equal = matches!(result, Value::Bool(true));
                    return Ok(Value::Bool(if op == BinOp::Eq { equal } else { !equal }));
                }
            }
            let equal = match (&left, &right) {
                (
                    Value::String(left) | Value::Str(left),
                    Value::String(right) | Value::Str(right),
                ) => left == right,
                _ => left == right,
            };
            return Ok(Value::Bool(if op == BinOp::Eq { equal } else { !equal }));
        }
        if matches!(op, BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge) {
            // WP-C2.2 (DEV-027): comparison operators on nominal user types dispatch through
            // `Ord::cmp`, just as equality above dispatches through `Eq::eq`. The type checker
            // has already required a matching `Ord` implementation.
            if let Some(nominal) = nominal_item(&left) {
                if let Some(method) = self.find_method(
                    Some(nominal),
                    "cmp",
                    Some(Res::CoreTrait(hir::CoreTrait::Ord)),
                ) {
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
                        Value::Ref(receiver_place),
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
                        return Err(RuntimeError::new("invalid shift count", span));
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
                            return Err(RuntimeError::new("integer overflow", span));
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
                    RuntimeError::new(
                        if right == 0 && matches!(op, BinOp::Div | BinOp::Rem) {
                            "division by zero"
                        } else {
                            "integer overflow"
                        },
                        span,
                    )
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
                self.check_integer_range(value, expr, span).map(Value::Int)
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
                // trap. `.trunc() as i128` truncates finite f64 values toward zero exactly; the
                // subsequent `check_integer_range` call performs the actual representability
                // check against the target's declared width.
                if !value.is_finite() {
                    return Err(RuntimeError::new("numeric cast out of range", span));
                }
                let truncated = value.trunc() as i128;
                self.check_integer_range(truncated, expr, span)
                    .map(Value::Int)
            }
            _ => Err(RuntimeError::new("invalid numeric cast", span)),
        }
    }

    fn check_integer_range(
        &self,
        value: i128,
        expr: ExprId,
        span: Span,
    ) -> Result<i128, RuntimeError> {
        let ty = self.tables.expr_types.get(&expr);
        let valid = match ty {
            Some(Ty::Primitive(Primitive::Int8)) => i8::try_from(value).is_ok(),
            Some(Ty::Primitive(Primitive::Int16)) => i16::try_from(value).is_ok(),
            Some(Ty::Primitive(Primitive::Int32)) => i32::try_from(value).is_ok(),
            Some(Ty::Primitive(Primitive::Int64)) => i64::try_from(value).is_ok(),
            Some(Ty::Primitive(Primitive::UInt8)) => u8::try_from(value).is_ok(),
            Some(Ty::Primitive(Primitive::UInt16)) => u16::try_from(value).is_ok(),
            Some(Ty::Primitive(Primitive::UInt32)) => u32::try_from(value).is_ok(),
            Some(Ty::Primitive(Primitive::UInt64)) => u64::try_from(value).is_ok(),
            _ => true,
        };
        if valid {
            Ok(value)
        } else {
            Err(RuntimeError::new("integer overflow", span))
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
                Res::Builtin(builtin) => match self.eval_call_arguments(args)? {
                    Ok(values) => self.call_builtin(*builtin, values, span).map(Flow::Value),
                    Err(propagated) => Ok(Flow::Propagate(propagated)),
                },
                Res::Item(item) => match self.eval_call_arguments(args)? {
                    Ok(values) => {
                        let callable = self.item_callable(*item).ok_or_else(|| {
                            RuntimeError::new("item is not callable", self.hir.expr(callee).span)
                        })?;
                        self.call_callable(callable, None, values, span)
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
                    self.call_qualified_trait(*trait_id, *member, args, span)
                }
                Res::CoreTraitMember(core_trait, _) => {
                    self.call_qualified_core_trait(*core_trait, args, span)
                }
                Res::AssociatedFn(item, name) => match self.eval_call_arguments(args)? {
                    Ok(values) => {
                        let callable = self
                            .find_associated_fn(*item, self.text(*name))
                            .ok_or_else(|| {
                                RuntimeError::new("associated function not found", span)
                            })?;
                        self.call_callable(callable, None, values, span)
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
                    let Value::Function(item) = function else {
                        return Err(RuntimeError::new("expression is not callable", span));
                    };
                    match self.eval_call_arguments(args)? {
                        Ok(values) => {
                            let callable = self.item_callable(item).ok_or_else(|| {
                                RuntimeError::new("expression is not callable", span)
                            })?;
                            self.call_callable(callable, None, values, span)
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
                let Value::Function(item) = function else {
                    return Err(RuntimeError::new("expression is not callable", span));
                };
                match self.eval_call_arguments(args)? {
                    Ok(values) => {
                        let callable = self
                            .item_callable(item)
                            .ok_or_else(|| RuntimeError::new("expression is not callable", span))?;
                        self.call_callable(callable, None, values, span)
                            .map(Flow::Value)
                    }
                    Err(propagated) => Ok(Flow::Propagate(propagated)),
                }
            }
        }
    }

    fn call_builtin(
        &mut self,
        builtin: Builtin,
        mut args: Vec<Value>,
        span: Span,
    ) -> Result<Value, RuntimeError> {
        match builtin {
            Builtin::Print | Builtin::Println => {
                let value = args.pop().unwrap_or(Value::Unit);
                let deref = self.deref_value(value, span)?;
                self.output
                    .push_str(&self.format_runtime_value(&deref, span)?);
                if builtin == Builtin::Println {
                    self.output.push('\n');
                }
                Ok(Value::Unit)
            }
            Builtin::Panic => {
                let value = args.pop().unwrap_or(Value::Unit);
                let deref = self.deref_value(value, span)?;
                Err(RuntimeError::new(
                    self.format_runtime_value(&deref, span)?,
                    span,
                ))
            }
            Builtin::Assert => match args.pop() {
                Some(Value::Bool(true)) => Ok(Value::Unit),
                Some(Value::Bool(false)) => Err(RuntimeError::new("assertion failed", span)),
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
                let equal = left == right;
                let want_eq = builtin == Builtin::AssertEq;
                if equal == want_eq {
                    Ok(Value::Unit)
                } else if want_eq {
                    Err(RuntimeError::new(
                        format!("assertion failed: `(left == right)`\n  left: `{left}`\n right: `{right}`"),
                        span,
                    ))
                } else {
                    Err(RuntimeError::new(
                        format!("assertion failed: `(left != right)`\n  left: `{left}`\n right: `{right}`"),
                        span,
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
            Builtin::SizeOf | Builtin::AlignOf => Ok(Value::Int(8)),
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
            Builtin::StringFrom => Ok(Value::String(string_arg(args.pop(), span)?)),
            Builtin::StringNew => Ok(Value::String(String::new())),
            Builtin::StringWithCapacity => {
                let capacity = usize_arg(args.pop(), span)?;
                Ok(Value::String(String::with_capacity(capacity)))
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
            Builtin::Some => Ok(Value::Option(args.pop().map(Box::new))),
            Builtin::None => Ok(Value::Option(None)),
            Builtin::Ok => Ok(Value::Result(Ok(Box::new(
                args.pop().unwrap_or(Value::Unit),
            )))),
            Builtin::Err => Ok(Value::Result(Err(Box::new(
                args.pop().unwrap_or(Value::Unit),
            )))),
            Builtin::ReadFile => {
                let path = string_arg(args.pop(), span)?;
                Ok(match std::fs::read_to_string(path) {
                    Ok(value) => Value::Result(Ok(Box::new(Value::String(value)))),
                    Err(error) => Value::Result(Err(Box::new(Value::IOError(
                        IOErrorKind::from_io_error(&error),
                    )))),
                })
            }
            Builtin::WriteFile => {
                if args.len() != 2 {
                    return Err(RuntimeError::new("write_file expects two arguments", span));
                }
                let content = string_arg(args.pop(), span)?;
                let path = string_arg(args.pop(), span)?;
                Ok(match std::fs::write(path, content) {
                    Ok(()) => Value::Result(Ok(Box::new(Value::Unit))),
                    Err(error) => Value::Result(Err(Box::new(Value::IOError(
                        IOErrorKind::from_io_error(&error),
                    )))),
                })
            }
            Builtin::FileOpen | Builtin::FileCreate => {
                let path = string_arg(args.pop(), span)?;
                let result = if builtin == Builtin::FileOpen {
                    std::fs::File::open(path)
                } else {
                    std::fs::File::create(path)
                };
                Ok(match result {
                    Ok(file) => Value::Result(Ok(Box::new(Value::File(FileResource::new(file))))),
                    Err(error) => Value::Result(Err(Box::new(Value::IOError(
                        IOErrorKind::from_io_error(&error),
                    )))),
                })
            }
            // -- Phase 4E: Math constants and functions --
            Builtin::MathAbs => match args.pop() {
                Some(Value::Int(value)) => value
                    .checked_abs()
                    .map(Value::Int)
                    .ok_or_else(|| RuntimeError::new("integer overflow", span)),
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
            // -- Phase 4E: stderr --
            Builtin::Eprint => {
                eprint!("{}", string_arg(args.pop(), span)?);
                Ok(Value::Unit)
            }
            Builtin::Eprintln => {
                eprintln!("{}", string_arg(args.pop(), span)?);
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
            Builtin::TensorZeros
            | Builtin::TensorOnes
            | Builtin::TensorFull
            | Builtin::TensorFromVec
            | Builtin::TensorAdd
            | Builtin::TensorSub
            | Builtin::TensorMul
            | Builtin::TensorDiv
            | Builtin::TensorMin
            | Builtin::TensorMax
            | Builtin::TensorEq
            | Builtin::TensorNe
            | Builtin::TensorLt
            | Builtin::TensorLe
            | Builtin::TensorGt
            | Builtin::TensorGe
            | Builtin::TensorBroadcastTo
            | Builtin::TensorMatMul
            | Builtin::TensorBatchMatMul
            | Builtin::TensorConcat
            | Builtin::TensorPermute
            | Builtin::TensorReshape
            | Builtin::TensorSliceAxis
            | Builtin::TensorTranspose
            | Builtin::TensorSumAxis
            | Builtin::TensorMeanAxis
            | Builtin::TensorArgMax
            | Builtin::TensorSum
            | Builtin::TensorSoftmax
            | Builtin::TensorCast
            | Builtin::TensorScale255
            | Builtin::TensorNormalize
            | Builtin::TensorToDevice => Err(RuntimeError::new(
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
        let method = self.find_method(nominal, &name, None).ok_or_else(|| {
            RuntimeError::new(format!("method '{name}' not found at runtime"), span)
        })?;
        match self.eval_call_arguments(args)? {
            Ok(values) => self
                .call_user_method(method, receiver_place, receiver_value, values, span)
                .map(Flow::Value),
            Err(propagated) => Ok(Flow::Propagate(propagated)),
        }
    }

    fn call_qualified_trait(
        &mut self,
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
        let receiver = self.clone_place_value(&receiver_place, span)?;
        let method = self
            .find_method(
                nominal_item(&receiver),
                &method_name,
                Some(Res::Item(trait_id)),
            )
            .ok_or_else(|| RuntimeError::new("trait implementation not found", span))?;
        match self.eval_call_arguments(rest)? {
            Ok(values) => self
                .call_user_method(method, receiver_place, receiver, values, span)
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
        let receiver = self.clone_place_value(&receiver_place, span)?;
        let method = self
            .find_method(
                nominal_item(&receiver),
                method_name,
                Some(Res::CoreTrait(core_trait)),
            )
            .ok_or_else(|| RuntimeError::new("trait implementation not found", span))?;
        match self.eval_call_arguments(rest)? {
            Ok(values) => self
                .call_user_method(method, receiver_place, receiver, values, span)
                .map(Flow::Value),
            Err(propagated) => Ok(Flow::Propagate(propagated)),
        }
    }

    fn find_method(
        &self,
        nominal: Option<ItemId>,
        name: &str,
        trait_filter: Option<Res>,
    ) -> Option<Callable> {
        // WP-C2.2 (DEV-026): inherent methods shadow trait methods of the same name
        // unconditionally (03-Type-System.md "Method Calls and Auto-Borrowing", rule 1).
        // Previously a single source-order scan returned whichever matching impl block
        // appeared first in the file — a trait impl declared above an inherent impl won,
        // observably flipping which body ran based on item order alone. Two passes: inherent
        // impls first, then trait impls. A trait-qualified call (`trait_filter` set) skips the
        // inherent pass entirely, since it names the trait explicitly.
        if trait_filter.is_none() {
            if let Some(callable) = self.find_method_pass(nominal, name, None, true) {
                return Some(callable);
            }
        }
        self.find_method_pass(nominal, name, trait_filter, false)
    }

    fn find_method_pass(
        &self,
        nominal: Option<ItemId>,
        name: &str,
        trait_filter: Option<Res>,
        inherent_only: bool,
    ) -> Option<Callable> {
        let nominal = nominal?;
        // DEV-069: this scans EVERY impl in the program, including impls from other files, so
        // both the method names and the resulting body's file come from the impl's own item.
        self.hir.items.iter().enumerate().find_map(|(idx, item)| {
            let impl_id = ItemId(idx as u32);
            let hir::ItemKind::Impl {
                trait_,
                self_ty,
                items,
                ..
            } = &item.kind
            else {
                return None;
            };
            if inherent_only != trait_.is_none() {
                return None;
            }
            if trait_filter.is_some_and(
                |expected| !matches!(trait_, Some(reference) if reference.res == expected),
            ) {
                return None;
            }
            if !matches!(
                &self.hir.ty(*self_ty).kind,
                hir::TypeKind::Path { res: Res::Item(item), .. } if *item == nominal
            ) {
                return None;
            }
            let overridden = items.iter().find_map(|item| match item {
                hir::ImplItem::Fn { def, .. } if self.item_text(impl_id, def.sig.name) == name => {
                    Some(Callable {
                        receiver: def.sig.receiver.zip(def.sig.receiver_local),
                        params: def.sig.params.iter().map(|param| param.local).collect(),
                        body: def.body,
                        file: self.item_file(impl_id),
                    })
                }
                _ => None,
            });
            // WP-C1.3 (2026-07-17): fall back to the trait's own default method body
            // (`TraitItem::Method { body: Some(_), .. }`, 03-Type-System.md trait defaults) when
            // this impl block doesn't override the method. The HIR already carries default
            // bodies (they were simply never consulted here) -- confirmed empirically that a
            // trait method with a real body, left un-overridden by an implementing struct,
            // failed with "method not found" before this fix. See COMPILER-STATE.md DEV-013.
            overridden.or_else(|| {
                let trait_id = match trait_.as_ref()?.res {
                    Res::Item(id) => id,
                    _ => return None,
                };
                let hir::ItemKind::Trait {
                    items: trait_items, ..
                } = &self.hir.item(trait_id).kind
                else {
                    return None;
                };
                trait_items.iter().find_map(|item| match item {
                    hir::TraitItem::Method {
                        sig,
                        body: Some(body),
                    } if self.item_text(trait_id, sig.name) == name => Some(Callable {
                        receiver: sig.receiver.zip(sig.receiver_local),
                        params: sig.params.iter().map(|param| param.local).collect(),
                        body: *body,
                        // The default body lives in the TRAIT's file, not the impl's.
                        file: self.item_file(trait_id),
                    }),
                    _ => None,
                })
            })
        })
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
                        file: self.item_file(impl_id),
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

    fn call_user_method(
        &mut self,
        callable: Callable,
        receiver_place: Place,
        // Kept for call-site symmetry (nominal lookup resolves from it); the `&self` binding
        // itself is a genuine reference since DEV-070's fix, so the clone is no longer bound.
        _receiver_value: Value,
        args: Vec<Value>,
        span: Span,
    ) -> Result<Value, RuntimeError> {
        let Some((receiver_kind, receiver_local)) = callable.receiver else {
            return Err(RuntimeError::new("method has no receiver", span));
        };
        let receiver = match receiver_kind {
            // WP-C2.2 (DEV-034): consume the already-resolved place (proper move semantics for
            // non-Copy receivers, including partial moves out of fields) instead of re-evaluating
            // the receiver expression — the source of the confirmed double-evaluation bug.
            hir::Receiver::Value => self.take_place(&receiver_place, span)?,
            // DEV-070 (A2): a `&self` receiver binds a genuine REFERENCE to the caller's place,
            // not a value clone — the same fix the correction brief applied to `Eq::eq`/
            // `Ord::cmp` dispatch (Issue 2). With the clone, `*self` failed "cannot dereference
            // non-reference"; field reads worked only via the deref-normalizing place walk.
            // Observationally equivalent otherwise: the referent cannot be mutated while the
            // method runs (shared borrow, single-threaded), and the old clone was discarded
            // without STARK drop effects. (`&mut self` keeps its take/write-back model.)
            hir::Receiver::Ref => Value::Ref(receiver_place.clone()),
            hir::Receiver::RefMut => self
                .place_slot_mut(&receiver_place, span)?
                .take()
                .ok_or_else(|| RuntimeError::new("mutable receiver is unavailable", span))?,
        };
        let mut frame = Frame::default();
        frame.insert(receiver_local, Some(receiver));
        for (local, value) in callable.params.iter().copied().zip(args) {
            frame.insert(local, Some(value));
        }
        self.frames.push(frame);
        let method_frame = self.frames.len() - 1;
        // DEV-069: a method body executes against the file that declares its impl (or, for an
        // un-overridden trait default, the trait's file) — this is the second body-execution
        // funnel alongside `call_callable`, and it needs the same swap. Restored on BOTH exits.
        let caller_file = std::mem::replace(&mut self.file, callable.file);
        let result = self.eval_block(callable.body);
        if let Err(error) = result {
            self.file = caller_file;
            let restored = self
                .frame_mut()
                .values
                .get_mut(&receiver_local)
                .and_then(Option::take);
            self.frames.pop();
            if let (hir::Receiver::RefMut, Some(restored)) = (receiver_kind, restored) {
                self.place_slot_mut(&receiver_place, span)?
                    .replace(restored);
            }
            return Err(error);
        }
        let flow = result?;
        let restored = if receiver_kind == hir::Receiver::Value {
            None
        } else {
            self.frame_mut()
                .values
                .get_mut(&receiver_local)
                .and_then(Option::take)
        };
        // Destructors for the method's own locals still belong to the method's file, so the
        // restore happens after cleanup (matching `call_callable`).
        self.cleanup_current_frame()?;
        self.file = caller_file;
        self.frames.pop();
        if receiver_kind == hir::Receiver::RefMut {
            let restored = restored.ok_or_else(|| {
                RuntimeError::new("mutable receiver was moved by its method", span)
            })?;
            self.write_place(&receiver_place, restored, span)?;
        }
        let mut value = match flow {
            Flow::Value(value) | Flow::Return(value) => value,
            Flow::Propagate(value) => value,
            Flow::Break(_) | Flow::Continue => {
                return Err(RuntimeError::new("loop control escaped a method", span))
            }
        };
        // WP-C2.2 (DEV-035): a reference returned from a `&self`/`&mut self` method that was
        // derived from `self` (e.g. `&self.field`, or `self.items.iter()`) carries a `Place`
        // pointing into the method's own — just popped — call frame, so any later dereference
        // failed with "dangling reference" (confirmed empirically: an ordinary getter returning
        // `&self.value` crashed unconditionally). Rebase such places onto the caller-side
        // receiver place, preserving any field/index projections taken inside the method.
        // References into other locals of the popped frame are left untouched: the borrow
        // checker's return-escape check (E0103, WP-C1.4) rejects those at compile time, and if
        // one ever slips through, the existing "dangling reference" trap is the correct
        // backstop, not a silent rebase.
        rebase_frame_refs(&mut value, method_frame, receiver_local, &receiver_place);
        Ok(value)
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
                    self.language_position(&keys, key, span)?
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
                        Ok(Some(Value::Option(Some(Box::new(Value::Ref(place))))))
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
                            Ok(Some(Value::Option(old.map(Box::new))))
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
                        Ok(Some(Value::Option(value.map(Box::new))))
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
                    self.language_position(&set.0, value, span)?
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
        values: &[Value],
        needle: &Value,
        span: Span,
    ) -> Result<Option<usize>, RuntimeError> {
        for (index, value) in values.iter().enumerate() {
            if self.language_equal(value.clone(), needle.clone(), span)? {
                return Ok(Some(index));
            }
        }
        Ok(None)
    }

    fn language_equal(
        &mut self,
        left: Value,
        right: Value,
        span: Span,
    ) -> Result<bool, RuntimeError> {
        let left = self.deref_value(left, span)?;
        let right = self.deref_value(right, span)?;
        if let Some(nominal) = nominal_item(&left) {
            if let Some(method) = self.find_method(
                Some(nominal),
                "eq",
                Some(Res::CoreTrait(hir::CoreTrait::Eq)),
            ) {
                let receiver_place = self.promote_to_temp_place(left.clone(), span)?;
                let argument_place = self.promote_to_temp_place(right, span)?;
                return match self.call_user_method(
                    method,
                    receiver_place,
                    left,
                    vec![Value::Ref(argument_place)],
                    span,
                )? {
                    Value::Bool(value) => Ok(value),
                    _ => Err(RuntimeError::new("Eq::eq must return Bool", span)),
                };
            }
        }
        Ok(left == right)
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
        if let Some(result) =
            self.call_collection_ownership_method(&receiver_place, name, &mut arguments, span)?
        {
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
            return Ok(Value::Result(Ok(Box::new(Value::Unit))));
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
                    Ok(*payload)
                }
                Value::Option(None) => Ok(default),
                Value::Result(Ok(payload)) => {
                    self.drop_value(default)?;
                    Ok(*payload)
                }
                Value::Result(Err(error)) => {
                    self.drop_value(*error)?;
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
            let Value::Function(func_item) = func else {
                return Err(RuntimeError::new(
                    format!("{name} expects a function value"),
                    span,
                ));
            };
            let callable = self
                .item_callable(func_item)
                .ok_or_else(|| RuntimeError::new("expression is not callable", span))?;
            let receiver = self.take_place(&receiver_place, span)?;
            return match (receiver, name) {
                (Value::Option(option), "map") => match option {
                    Some(value) => {
                        let mapped = self.call_callable(callable, None, vec![*value], span)?;
                        Ok(Value::Option(Some(Box::new(mapped))))
                    }
                    None => Ok(Value::Option(None)),
                },
                (Value::Option(option), "and_then") => match option {
                    Some(value) => self.call_callable(callable, None, vec![*value], span),
                    None => Ok(Value::Option(None)),
                },
                (Value::Result(result), "map") => match result {
                    Ok(value) => {
                        let mapped = self.call_callable(callable, None, vec![*value], span)?;
                        Ok(Value::Result(Ok(Box::new(mapped))))
                    }
                    Err(error) => Ok(Value::Result(Err(error))),
                },
                (Value::Result(result), "map_err") => match result {
                    Ok(value) => Ok(Value::Result(Ok(value))),
                    Err(error) => {
                        let mapped = self.call_callable(callable, None, vec![*error], span)?;
                        Ok(Value::Result(Err(Box::new(mapped))))
                    }
                },
                (Value::Result(result), "and_then") => match result {
                    Ok(value) => self.call_callable(callable, None, vec![*value], span),
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
            return Ok(Value::Int(standard_hash(receiver, &receiver_ty)? as i128));
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
                return Ok(Value::Result(Err(Box::new(Value::IOError(
                    IOErrorKind::InvalidInput,
                )))));
            };
            let io_result: Result<(), std::io::Error> = match name {
                "read_to_string" => {
                    let mut bytes = Vec::new();
                    match file.read_to_end(&mut bytes) {
                        Ok(_) => match String::from_utf8(bytes) {
                            Ok(text) => {
                                return Ok(Value::Result(Ok(Box::new(Value::String(text)))))
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
                            return Ok(Value::Result(Ok(Box::new(Value::Int(count as i128)))))
                        }
                        Err(error) => Err(error),
                    }
                }
                "write" => {
                    let bytes = self.file_bytes_arg(values.next(), span)?;
                    match file.write(&bytes) {
                        Ok(count) => {
                            return Ok(Value::Result(Ok(Box::new(Value::Int(count as i128)))))
                        }
                        Err(error) => Err(error),
                    }
                }
                _ => unreachable!(),
            };
            return Ok(Value::Result(Err(Box::new(Value::IOError(
                IOErrorKind::from_io_error(&io_result.unwrap_err()),
            )))));
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
                    let position = self.language_position(&keys, &key, span)?;
                    if let Some(index) = position {
                        place.projections.push(Projection::MapIndex(index));
                    }
                    return Ok(Value::Option(position.map(|_| Box::new(Value::Ref(place)))));
                }
                _ => {
                    let index = usize_arg(values.next(), span)?;
                    let mut place = receiver_place;
                    place.projections.push(Projection::Index(index));
                    return Ok(Value::Option(
                        self.place_value(&place, span)
                            .ok()
                            .map(|_| Box::new(Value::Ref(place))),
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
                    let position = self.language_position(&keys, &key, span)?;
                    if let Some(index) = position {
                        place.projections.push(Projection::MapIndex(index));
                    }
                    return Ok(Value::Option(position.map(|_| Box::new(Value::Ref(place)))));
                }
                _ => {
                    let index = usize_arg(values.next(), span)?;
                    let mut place = receiver_place;
                    place.projections.push(Projection::Index(index));
                    return Ok(Value::Option(
                        self.place_value(&place, span)
                            .ok()
                            .map(|_| Box::new(Value::Ref(place))),
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
            return Ok(Value::Option(next_val.map(Box::new)));
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
                    if self.language_position(&set.0, &value, span)?.is_some() {
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
                        if let Some(index) = self.language_position(&keys, &k, span)? {
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
                return Ok(Value::Option(Some(Box::new(acc))));
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
            return Ok(Value::Option(found.map(Box::new)));
        }
        if name == "map" {
            let place = receiver_place.clone();
            let f = values
                .next()
                .ok_or_else(|| RuntimeError::new("map expects function argument", span))?;
            let Value::Function(func_item) = f else {
                return Err(RuntimeError::new("expected function pointer for map", span));
            };
            let iter_val = self.place_value(&place, span)?.clone();
            let iter_mut = self.place_value_mut(&place, span)?;
            *iter_mut = Value::MapIter(Box::new(iter_val), func_item);
            return Ok(self.place_value(&place, span)?.clone());
        }
        if name == "filter" {
            let place = receiver_place.clone();
            let f = values
                .next()
                .ok_or_else(|| RuntimeError::new("filter expects function argument", span))?;
            let Value::Function(pred_item) = f else {
                return Err(RuntimeError::new(
                    "expected function pointer for filter",
                    span,
                ));
            };
            let iter_val = self.place_value(&place, span)?.clone();
            let iter_mut = self.place_value_mut(&place, span)?;
            *iter_mut = Value::FilterIter(Box::new(iter_val), pred_item);
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
                    string.pop().map(|ch| Box::new(Value::Char(ch))),
                )),
                "clear" => {
                    string.clear();
                    Ok(Value::Unit)
                }
                "as_str" => Ok(Value::Str(string.clone())),
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
                        .map(|index| Box::new(Value::Int(index as i128))),
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
                "bytes" | "into_bytes" => {
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
                "pop" => Ok(Value::Option(vector.pop().flatten().map(Box::new))),
                "insert" => {
                    let index = usize_arg(values.next(), span)?;
                    if index > vector.len() {
                        return Err(RuntimeError::new("Vec insertion index out of bounds", span));
                    }
                    vector.insert(index, values.next());
                    Ok(Value::Unit)
                }
                "remove" => {
                    let index = usize_arg(values.next(), span)?;
                    if index >= vector.len() {
                        return Err(RuntimeError::new("Vec removal index out of bounds", span));
                    }
                    Ok(vector.remove(index).unwrap_or(Value::Unit))
                }
                "clear" => {
                    vector.clear();
                    Ok(Value::Unit)
                }
                "as_slice" => Ok(Value::Slice(receiver_place.clone(), 0, vector.len())),
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
                "unwrap" => option
                    .take()
                    .map(|value| *value)
                    .ok_or_else(|| RuntimeError::new("called unwrap on None", span)),
                "unwrap_or" => Ok(option
                    .take()
                    .map_or_else(|| values.next().unwrap_or(Value::Unit), |value| *value)),
                _ => Err(RuntimeError::new(
                    format!("unsupported Option method '{name}'"),
                    span,
                )),
            },
            Value::Result(result) => match name {
                "is_ok" => Ok(Value::Bool(result.is_ok())),
                "is_err" => Ok(Value::Bool(result.is_err())),
                "unwrap" => match std::mem::replace(result, Ok(Box::new(Value::Unit))) {
                    Ok(value) => Ok(*value),
                    Err(error) => Err(RuntimeError::new(
                        format!("called unwrap on Err({error})"),
                        span,
                    )),
                },
                "unwrap_or" => match std::mem::replace(result, Ok(Box::new(Value::Unit))) {
                    Ok(value) => Ok(*value),
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
                        map.insert(k, Some(v)).flatten().map(Box::new),
                    ))
                }
                "remove" => {
                    let k = values.next().ok_or_else(|| {
                        RuntimeError::new("HashMap::remove expects key ref", span)
                    })?;
                    Ok(Value::Option(map.remove(&k).flatten().map(Box::new)))
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
                        let range = values
                            .get(start..end)
                            .ok_or_else(|| RuntimeError::new("slice out of bounds", span))?;
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

    fn eval_struct_lit(
        &mut self,
        res: Res,
        fields: &[hir::FieldInit],
        span: Span,
    ) -> Result<Flow, RuntimeError> {
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
            hir::PatKind::Lit(lit) => Ok(self.eval_lit(*lit, pattern.span)? == *value),
            hir::PatKind::Path { res, .. } => match (res, value) {
                (Res::Item(item), actual) => match &self.hir.item(*item).kind {
                    hir::ItemKind::Const {
                        value: initializer, ..
                    } => Ok(self.expect_value(*initializer)? == *actual),
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
                        vec![Some((**value).clone())]
                    }
                    (Res::Builtin(Builtin::Ok), Value::Result(Ok(value))) => {
                        vec![Some((**value).clone())]
                    }
                    (Res::Builtin(Builtin::Err), Value::Result(Err(value))) => {
                        vec![Some((**value).clone())]
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
        let kind = &self.hir.pat(pat).kind;
        match kind {
            hir::PatKind::Binding { .. } => Ok(()),
            hir::PatKind::TupleVariant { pats, .. } => {
                let pats = pats.clone();
                let payloads: Vec<Option<Value>> = match value {
                    Value::Enum { fields, .. } => fields,
                    Value::Option(Some(inner)) => vec![Some(*inner)],
                    Value::Result(Ok(inner)) | Value::Result(Err(inner)) => vec![Some(*inner)],
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
        span: Span,
    ) -> Result<Option<Value>, RuntimeError> {
        let current = self.clone_place_value(iterator_place, span)?;
        let next = if nominal_item(&current).is_some() {
            let method = self
                .find_method(
                    nominal_item(&current),
                    "next",
                    Some(Res::CoreTrait(hir::CoreTrait::Iterator)),
                )
                .ok_or_else(|| RuntimeError::new("value is not iterable", span))?;
            self.call_user_method(method, iterator_place.clone(), current, Vec::new(), span)?
        } else {
            let (next, updated) = self.iterator_step(current, Some(iterator_place), span)?;
            *self.place_value_mut(iterator_place, span)? = updated;
            Value::Option(next.map(Box::new))
        };
        match next {
            Value::Option(Some(value)) => Ok(Some(*value)),
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
        let start = usize::try_from(start)
            .map_err(|_| RuntimeError::new("slice range out of bounds (negative start)", span))?;
        let mut end = usize::try_from(end)
            .map_err(|_| RuntimeError::new("slice range out of bounds (negative end)", span))?;
        if inclusive {
            end = end.checked_add(1).ok_or_else(|| {
                RuntimeError::new("slice range out of bounds (inclusive end overflow)", span)
            })?;
        }
        if start > end || end > length {
            return Err(RuntimeError::new("slice range out of bounds", span));
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
                let index = usize::try_from(index_value)
                    .map_err(|_| RuntimeError::new("negative index", self.hir.expr(*index).span))?;
                if let Value::Slice(base_place, start, end) =
                    self.place_value(&place, node.span)?.clone()
                {
                    if index >= end - start {
                        return Err(RuntimeError::new("index out of bounds", node.span));
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
        let local_id = LocalId(1000000 + self.frame().values.len() as u32);
        self.frame_mut().values.insert(local_id, Some(value));
        Ok(Place {
            frame: self.frames.len() - 1,
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
        let Value::Function(item) = func else {
            return Err(RuntimeError::new("expected a function pointer", span));
        };
        let callable = self
            .item_callable(item)
            .ok_or_else(|| RuntimeError::new("expression is not callable", span))?;
        self.call_callable(callable, None, values, span)
    }

    fn iterator_step(
        &mut self,
        mut iter: Value,
        iter_place: Option<&Place>,
        span: Span,
    ) -> Result<(Option<Value>, Value), RuntimeError> {
        match &mut iter {
            Value::CharsIter(s, ref mut idx) => {
                let opt = if *idx < s.len() {
                    let ch = s.chars().nth(*idx).unwrap();
                    *idx += 1;
                    Some(Value::Char(ch))
                } else {
                    None
                };
                Ok((opt, iter))
            }
            Value::SplitIter(parts, ref mut idx) => {
                let opt = if *idx < parts.len() {
                    let s = parts[*idx].clone();
                    *idx += 1;
                    Some(Value::String(s))
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
                        self.call_function_pointer(Value::Function(*func), vec![x], span)?;
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
                            self.call_function_pointer(Value::Function(*pred), vec![x_ref], span)?;
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
            return Err(RuntimeError::new("slice range out of bounds", span));
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
                .ok_or_else(|| RuntimeError::new(projection_failure_message(projection), span))?;
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
                .ok_or_else(|| RuntimeError::new(projection_failure_message(projection), span))?;
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

    fn write_place(&mut self, place: &Place, value: Value, span: Span) -> Result<(), RuntimeError> {
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
            Value::Struct { item, .. } | Value::Enum { item, .. } => self.copy_items.contains(item),
            Value::Option(value) => value
                .as_deref()
                .is_none_or(|value| self.value_is_copy(value)),
            Value::Result(value) => match value {
                Ok(value) | Err(value) => self.value_is_copy(value),
            },
            Value::String(_)
            | Value::Vec(_)
            | Value::Boxed(_)
            | Value::Range { .. }
            | Value::Slice(..)
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
            if let Some(callable) = self.find_drop(item) {
                let mut frame = Frame::default();
                // Move the real value into the destructor's `self` binding rather than a clone:
                // `Drop::drop(&mut self)` may legally mutate or replace fields (e.g. via
                // `replace(&mut self.field, ..)`), and those mutations must be visible to the
                // recursive field destruction below. A cloned receiver would let the destructor
                // observe and mutate a throwaway copy while `value` itself stayed pristine,
                // causing the pre-destructor field state to be dropped a second time and any
                // replacement value installed during `drop()` to never be dropped at all.
                let receiver_local = callable.receiver.map(|(_, local)| local);
                if let Some(local) = receiver_local {
                    frame.insert(local, Some(value));
                    value = Value::Unit;
                }
                self.frames.push(frame);
                // DEV-069: the THIRD body-execution funnel (alongside `call_callable` and
                // `call_user_method`) — a destructor body belongs to the file declaring its
                // `Drop` impl, which need not be the file of the code going out of scope.
                let caller_file = std::mem::replace(&mut self.file, callable.file);
                let result = self.eval_block(callable.body);
                self.file = caller_file;
                if let Some(local) = receiver_local {
                    if let Some(restored) = self
                        .frame_mut()
                        .values
                        .get_mut(&local)
                        .and_then(Option::take)
                    {
                        value = restored;
                    }
                }
                self.frames.pop();
                result?;
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
                if let Some(child) = child.take() {
                    self.drop_value(*child)?;
                }
            }
            Value::Result(result) => match std::mem::replace(result, Ok(Box::new(Value::Unit))) {
                Ok(child) | Err(child) => self.drop_value(*child)?,
            },
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

    fn find_drop(&self, item: ItemId) -> Option<Callable> {
        // DEV-069: a `Drop` impl may live in a different file from the type's user; the
        // destructor's method name and body both belong to the impl's own file.
        self.hir.items.iter().enumerate().find_map(|(idx, candidate)| {
            let impl_id = ItemId(idx as u32);
            let hir::ItemKind::Impl { trait_: Some(reference), self_ty, items, .. } = &candidate.kind else { return None; };
            if reference.res != Res::CoreTrait(hir::CoreTrait::Drop) || !matches!(&self.hir.ty(*self_ty).kind, hir::TypeKind::Path { res: Res::Item(actual), .. } if *actual == item) { return None; }
            items.iter().find_map(|item| match item {
                hir::ImplItem::Fn { def, .. } if self.item_text(impl_id, def.sig.name) == "drop" => Some(Callable { receiver: def.sig.receiver.zip(def.sig.receiver_local), params: def.sig.params.iter().map(|param| param.local).collect(), body: def.body, file: self.item_file(impl_id) }),
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
            rebase_frame_refs(inner, popped_frame, receiver_local, receiver_place);
        }
        Value::Result(Ok(inner)) | Value::Result(Err(inner)) => {
            rebase_frame_refs(inner, popped_frame, receiver_local, receiver_place);
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
        (Value::Struct { fields, .. }, Projection::Field(name))
        | (Value::Enum { named: fields, .. }, Projection::Field(name)) => fields.get(name),
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
        (Value::Struct { fields, .. }, Projection::Field(name))
        | (Value::Enum { named: fields, .. }, Projection::Field(name)) => fields.get_mut(name),
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
pub fn canonical_float(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value == f64::INFINITY {
        return "inf".to_string();
    }
    if value == f64::NEG_INFINITY {
        return "-inf".to_string();
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0".to_string()
        } else {
            "0.0".to_string()
        };
    }
    canonical_float_digits(&value.to_string())
}

// Canonical display must use the shortest decimal representation that round-trips to the
// *declared* IEEE type. `Value::Float` carries its own `FloatWidth` tag, so both the top-level
// `println`/`.fmt()` paths and the generic recursive `Display for Value` impl (reached when a
// Float32 is nested inside a tuple/array/struct/collection) route through this for a Float32
// value instead of `canonical_float`'s `f64` shortest-round-trip digits (which would otherwise
// produce e.g. `0.10000000149011612` for `0.1f32` instead of the shorter, correct `0.1`).
fn canonical_float32(value: f32) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value == f32::INFINITY {
        return "inf".to_string();
    }
    if value == f32::NEG_INFINITY {
        return "-inf".to_string();
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0".to_string()
        } else {
            "0.0".to_string()
        };
    }
    canonical_float_digits(&value.to_string())
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

fn canonical_float_digits(shortest: &str) -> String {
    let (sign, unsigned) = shortest
        .strip_prefix('-')
        .map_or(("", shortest), |rest| ("-", rest));
    let (mantissa, explicit_exponent) = unsigned
        .split_once(['e', 'E'])
        .map_or((unsigned, 0_i32), |(mantissa, exponent)| {
            (mantissa, exponent.parse::<i32>().unwrap())
        });
    let decimal_position = mantissa
        .find('.')
        .map_or(mantissa.len() as i32, |position| position as i32)
        + explicit_exponent;
    let raw_digits: String = mantissa
        .chars()
        .filter(|character| *character != '.')
        .collect();
    let leading_zeroes = raw_digits
        .bytes()
        .take_while(|digit| *digit == b'0')
        .count() as i32;
    let scientific_exponent = decimal_position - leading_zeroes - 1;
    let significant = raw_digits.trim_start_matches('0').trim_end_matches('0');
    let significant = if significant.is_empty() {
        "0"
    } else {
        significant
    };

    if (-4..=15).contains(&scientific_exponent) {
        let point = scientific_exponent + 1;
        let mut rendered = String::from(sign);
        if point <= 0 {
            rendered.push_str("0.");
            rendered.extend(std::iter::repeat_n('0', (-point) as usize));
            rendered.push_str(significant);
        } else if point as usize >= significant.len() {
            rendered.push_str(significant);
            rendered.extend(std::iter::repeat_n('0', point as usize - significant.len()));
            rendered.push_str(".0");
        } else {
            rendered.push_str(&significant[..point as usize]);
            rendered.push('.');
            rendered.push_str(&significant[point as usize..]);
        }
        rendered
    } else {
        let mut rendered = String::from(sign);
        rendered.push_str(&significant[..1]);
        if significant.len() > 1 {
            rendered.push('.');
            rendered.push_str(&significant[1..]);
        }
        rendered.push('e');
        rendered.push_str(&scientific_exponent.to_string());
        rendered
    }
}

fn standard_hash(value: &Value, ty: &Ty) -> Result<u64, RuntimeError> {
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
                if let Some(value) = value {
                    frame(&mut bytes, encode(value, &types[0])?);
                }
                Some(bytes)
            }
            (Value::Result(value), Ty::Core(hir::CoreType::Result, types)) if types.len() == 2 => {
                let mut bytes = vec![0x05, u8::from(value.is_err())];
                match value {
                    Ok(value) => frame(&mut bytes, encode(value, &types[0])?),
                    Err(value) => frame(&mut bytes, encode(value, &types[1])?),
                }
                Some(bytes)
            }
            _ => None,
        }
    }

    let bytes = encode(value, ty).ok_or_else(|| {
        RuntimeError::new("type has no standard Hash implementation", Span::new(0, 0))
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

fn float_arg(value: Option<Value>, span: Span) -> Result<f64, RuntimeError> {
    match value {
        Some(Value::Float(value, _)) => Ok(value),
        _ => Err(RuntimeError::new("expected Float64 argument", span)),
    }
}

/// Numeric comparison for `math::min`/`math::max`/`clamp` (`T: Ord`, Int or
/// Float only — a narrower runtime scope than the unconstrained type
/// variable these builtins get in `typecheck.rs`; see
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
    use crate::typecheck;

    /// Type-check only, returning the diagnostics — for tests that assert a REJECTION rather
    /// than an execution result.
    fn type_diagnostics(source: &str) -> Vec<crate::diag::Diagnostic> {
        let file = Arc::new(SourceFile::new("test.stark", source));
        let (ast, _) = parse(&file, ParseMode::Program);
        let (hir, _) = resolve(&ast, file.clone());
        typecheck::analyze(&hir, file).diagnostics
    }

    fn execute(source: &str) -> Result<Execution, RuntimeError> {
        let file = Arc::new(SourceFile::new("test.stark", source));
        let (ast, parse_diags) = parse(&file, ParseMode::Program);
        assert!(parse_diags.is_empty(), "parse diagnostics: {parse_diags:?}");
        let (hir, resolve_diags) = resolve(&ast, file.clone());
        assert!(
            resolve_diags.is_empty(),
            "resolve diagnostics: {resolve_diags:?}"
        );
        let checked = typecheck::analyze(&hir, file.clone());
        assert!(
            checked
                .diagnostics
                .iter()
                .all(|diag| diag.severity != crate::diag::Severity::Error),
            "type diagnostics: {:?}",
            checked.diagnostics
        );
        run(&hir, file, &checked.tables)
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

    /// WP-C1.3 regression test for the companion typecheck.rs finding made while investigating
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

    #[test]
    fn for_loop_drops_each_binding_and_unconsumed_tail() {
        let execution = execute(
            "struct Loud { label: String, stop: Bool } \
             impl Drop for Loud { fn drop(&mut self) { println(self.label.as_str()); } } \
             fn all() { \
                 let mut values: Vec<Loud> = Vec::new(); \
                 values.push(Loud { label: String::from(\"first\"), stop: false }); \
                 values.push(Loud { label: String::from(\"second\"), stop: false }); \
                 for value in values { println(\"body\"); } \
             } \
             fn early() { \
                 let mut values: Vec<Loud> = Vec::new(); \
                 values.push(Loud { label: String::from(\"one\"), stop: true }); \
                 values.push(Loud { label: String::from(\"two\"), stop: false }); \
                 values.push(Loud { label: String::from(\"three\"), stop: false }); \
                 for value in values { if value.stop { break; } } \
             } \
             fn main() { all(); early(); }",
        )
        .unwrap();
        assert_eq!(
            execution.output,
            "body\nfirst\nbody\nsecond\none\nthree\ntwo\n"
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
        let path = std::env::temp_dir().join(format!(
            "stark-c2-11-file-{}-{}.txt",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let source = format!(
            "fn main() {{ \
                 let mut output: File = File::create(\"{}\").unwrap(); \
                 println(output.write_str(\"hello\").unwrap()); \
                 output.close().unwrap(); \
                 let mut input: File = File::open(\"{}\").unwrap(); \
                 println(input.read_to_string().unwrap()); \
                 input.close().unwrap(); \
             }}",
            path.display(),
            path.display()
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
                 let mut scalars = text.split(\"\"); \
                 println(scalars.count()); \
                 let mut empty_parts = String::from(\"\").split(\",\"); \
                 let mut trailing_parts = String::from(\"a,\").split(\",\"); \
                 println(empty_parts.count()); println(trailing_parts.count()); \
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
        let checked = typecheck::analyze(&hir, file.clone());
        assert!(
            checked
                .diagnostics
                .iter()
                .all(|diag| diag.severity != crate::diag::Severity::Error),
            "type diagnostics: {:?}",
            checked.diagnostics
        );
        let mut interpreter = Interpreter::new(&hir, file.clone(), &checked.tables);
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
            .call_callable(callable, None, Vec::new(), span)
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

    /// DEV-060 [CLOSED]: end-to-end confirmation that the fixed program (see `typecheck.rs`'s
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

    /// Companion regression for DEV-060 (see `typecheck.rs`'s
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
    /// *and* executes correctly. See `typecheck.rs`'s `trait_default_method_calling_sibling_
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
    /// `typecheck.rs` carry the resolution/type-checking half.
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
