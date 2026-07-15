//! Gate 3 tree-walking interpreter for typed STARK HIR.

use crate::ast::{AssignOp, BinOp, Lit, Primitive, UnOp};
use crate::hir::{self, BlockId, Builtin, ExprId, Hir, ItemId, LocalId, PatId, Res, StmtId};
use crate::lexer::Base;
use crate::source::{SourceFile, Span};
use crate::typecheck::{Ty, TypeTables};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct RuntimeError {
    pub message: String,
    pub span: Span,
}

impl RuntimeError {
    fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Execution {
    pub output: String,
}

#[derive(Clone, PartialEq)]
enum Value {
    Unit,
    Bool(bool),
    Int(i128),
    Float(f64),
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
    Function(ItemId),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Unit => write!(f, "()"),
            Value::Bool(value) => write!(f, "{value}"),
            Value::Int(value) => write!(f, "{value}"),
            Value::Float(value) => write!(f, "{value}"),
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
            Value::Function(item) => write!(f, "fn#{}", item.0),
        }
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

#[derive(Clone)]
enum Projection {
    Field(String),
    Index(usize),
}

#[derive(Clone)]
struct Place {
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
}

pub fn run(
    hir: &Hir,
    file: Arc<SourceFile>,
    tables: &TypeTables,
) -> Result<Execution, RuntimeError> {
    let mut interpreter = Interpreter::new(hir, file, tables);
    interpreter.run_main()?;
    Ok(Execution {
        output: interpreter.output,
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
}

impl<'a> Interpreter<'a> {
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
        }
    }

    fn text(&self, span: Span) -> &str {
        &self.file.src[span.lo as usize..span.hi as usize]
    }

    fn run_main(&mut self) -> Result<(), RuntimeError> {
        let main = self.hir.items.iter().enumerate().find_map(|(index, item)| {
            let hir::ItemKind::Fn(def) = &item.kind else {
                return None;
            };
            (self.text(def.sig.name) == "main").then_some(ItemId(index as u32))
        });
        let Some(main) = main else {
            return Err(RuntimeError::new(
                "program has no 'main' function",
                Span::point(0),
            ));
        };
        let callable = self.item_callable(main).ok_or_else(|| {
            RuntimeError::new("'main' is not executable", self.hir.item(main).span)
        })?;
        self.call_callable(callable, None, Vec::new(), self.hir.item(main).span)?;
        Ok(())
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
        let result = self.eval_block(callable.body);
        if result.is_err() {
            self.frames.pop();
            return result.map(|_| Value::Unit);
        }
        let flow = result?;
        self.cleanup_current_frame()?;
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
            hir::ExprKind::Lit(lit) => Ok(Flow::Value(self.eval_lit(*lit, expr.span)?)),
            hir::ExprKind::Path { res, .. } => Ok(Flow::Value(self.eval_path(*res, expr_id)?)),
            hir::ExprKind::Unary { op, operand } => {
                let value = match op {
                    UnOp::Ref { .. } => self.clone_expr_place(*operand)?,
                    UnOp::Deref => self.expect_value(*operand)?,
                    _ => self.expect_value(*operand)?,
                };
                Ok(Flow::Value(self.eval_unary(*op, value, expr.span)?))
            }
            hir::ExprKind::Binary { op, lhs, rhs } => {
                if *op == BinOp::And {
                    let left = self.expect_bool(*lhs)?;
                    return Ok(Flow::Value(Value::Bool(left && self.expect_bool(*rhs)?)));
                }
                if *op == BinOp::Or {
                    let left = self.expect_bool(*lhs)?;
                    return Ok(Flow::Value(Value::Bool(left || self.expect_bool(*rhs)?)));
                }
                let left = self.expect_value(*lhs)?;
                let right = self.expect_value(*rhs)?;
                Ok(Flow::Value(
                    self.eval_binary(*op, left, right, expr_id, expr.span)?,
                ))
            }
            hir::ExprKind::Assign { op, lhs, rhs } => {
                let right = self.expect_value(*rhs)?;
                let place = self.expr_place(*lhs)?;
                let value = if *op == AssignOp::Assign {
                    right
                } else {
                    let current = self.take_place(&place, expr.span)?;
                    self.eval_binary(assign_binop(*op), current, right, expr_id, expr.span)?
                };
                self.write_place(&place, value, expr.span)?;
                Ok(Flow::Value(Value::Unit))
            }
            hir::ExprKind::Range { lo, hi, inclusive } => Ok(Flow::Value(Value::Range {
                start: self.expect_int(*lo)?,
                end: self.expect_int(*hi)?,
                inclusive: *inclusive,
            })),
            hir::ExprKind::Cast { expr: value, .. } => {
                let value = self.expect_value(*value)?;
                Ok(Flow::Value(self.eval_cast(value, expr_id, expr.span)?))
            }
            hir::ExprKind::Call { callee, args } => self.eval_call(*callee, args, expr.span),
            hir::ExprKind::Field { .. } | hir::ExprKind::TupleField { .. } => {
                let place = self.expr_place(expr_id)?;
                Ok(Flow::Value(self.take_place(&place, expr.span)?))
            }
            hir::ExprKind::Index { base, index } => {
                if matches!(self.tables.expr_types.get(index), Some(Ty::Range(_))) {
                    let range = self.expect_value(*index)?;
                    let base = self.clone_expr_place(*base)?;
                    return Ok(Flow::Value(self.slice_value(base, range, expr.span)?));
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
            hir::ExprKind::Tuple(values) => Ok(Flow::Value(Value::Tuple(
                values
                    .iter()
                    .map(|value| self.expect_value(*value).map(Some))
                    .collect::<Result<_, _>>()?,
            ))),
            hir::ExprKind::Array(values) => Ok(Flow::Value(Value::Array(
                values
                    .iter()
                    .map(|value| self.expect_value(*value).map(Some))
                    .collect::<Result<_, _>>()?,
            ))),
            hir::ExprKind::Repeat { value, count } => {
                let value = self.expect_value(*value)?;
                let count = usize::try_from(self.expect_int(*count)?).map_err(|_| {
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
                if self.expect_bool(*cond)? {
                    self.eval_block(*then_block)
                } else if let Some(else_expr) = else_ {
                    self.eval_expr(*else_expr)
                } else {
                    Ok(Flow::Value(Value::Unit))
                }
            }
            hir::ExprKind::Match { scrutinee, arms } => {
                let value = self.expect_value(*scrutinee)?;
                for arm in arms {
                    let mut bindings = Vec::new();
                    if self.match_pattern(arm.pat, &value, &mut bindings)? {
                        for (local, value) in &bindings {
                            self.frame_mut().insert(*local, Some(value.clone()));
                        }
                        let flow = self.eval_expr(arm.body)?;
                        let locals: Vec<_> = bindings.iter().map(|(local, _)| *local).collect();
                        self.cleanup_locals(&locals)?;
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
                while self.expect_bool(*cond)? {
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
                let values = self.iter_values(iterable, expr.span)?;
                for value in values {
                    self.frame_mut().insert(*local, Some(value));
                    match self.eval_block(*body)? {
                        Flow::Value(_) | Flow::Continue => {}
                        Flow::Break(_) => break,
                        flow => return Ok(flow),
                    }
                }
                self.cleanup_locals(&[*local])?;
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

    fn expect_bool(&mut self, expr: ExprId) -> Result<bool, RuntimeError> {
        match self.expect_value(expr)? {
            Value::Bool(value) => Ok(value),
            _ => Err(RuntimeError::new("expected Bool", self.hir.expr(expr).span)),
        }
    }

    fn expect_int(&mut self, expr: ExprId) -> Result<i128, RuntimeError> {
        match self.expect_value(expr)? {
            Value::Int(value) => Ok(value),
            _ => Err(RuntimeError::new(
                "expected integer",
                self.hir.expr(expr).span,
            )),
        }
    }

    fn eval_lit(&self, lit: Lit, span: Span) -> Result<Value, RuntimeError> {
        let text = self.text(span);
        match lit {
            Lit::Bool(value) => Ok(Value::Bool(value)),
            Lit::Char => parse_char(text)
                .map(Value::Char)
                .ok_or_else(|| RuntimeError::new("invalid character literal", span)),
            Lit::Str { raw } => Ok(Value::Str(parse_string(text, raw))),
            Lit::Int { base, suffix } => {
                let mut digits = text.replace('_', "");
                if let Some(suffix) = suffix {
                    let suffix = format!("{suffix:?}").to_ascii_lowercase();
                    digits.truncate(digits.len().saturating_sub(suffix.len()));
                }
                let (digits, radix) = match base {
                    Base::Dec => (digits.as_str(), 10),
                    Base::Bin => (digits.trim_start_matches("0b"), 2),
                    Base::Oct => (digits.trim_start_matches("0o"), 8),
                    Base::Hex => (digits.trim_start_matches("0x"), 16),
                };
                i128::from_str_radix(digits, radix)
                    .map(Value::Int)
                    .map_err(|_| RuntimeError::new("integer literal out of range", span))
            }
            Lit::Float { suffix } => {
                let mut number = text.replace('_', "");
                if let Some(suffix) = suffix {
                    let suffix = format!("{suffix:?}").to_ascii_lowercase();
                    number.truncate(number.len().saturating_sub(suffix.len()));
                }
                number
                    .parse::<f64>()
                    .map(Value::Float)
                    .map_err(|_| RuntimeError::new("invalid float literal", span))
            }
        }
    }

    fn eval_path(&mut self, res: Res, expr: ExprId) -> Result<Value, RuntimeError> {
        match res {
            Res::Local(local) | Res::SelfValue(local) => {
                let place = Place {
                    local,
                    projections: Vec::new(),
                };
                self.take_place(&place, self.hir.expr(expr).span)
            }
            Res::Item(item) => match &self.hir.item(item).kind {
                hir::ItemKind::Fn(_) => Ok(Value::Function(item)),
                hir::ItemKind::Const { value, .. } => self.expect_value(*value),
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
            _ => Err(RuntimeError::new(
                "path is not a runtime value",
                self.hir.expr(expr).span,
            )),
        }
    }

    fn eval_unary(&self, op: UnOp, value: Value, span: Span) -> Result<Value, RuntimeError> {
        match (op, value) {
            (UnOp::Neg, Value::Int(value)) => value
                .checked_neg()
                .map(Value::Int)
                .ok_or_else(|| RuntimeError::new("integer overflow", span)),
            (UnOp::Neg, Value::Float(value)) => Ok(Value::Float(-value)),
            (UnOp::Not, Value::Bool(value)) => Ok(Value::Bool(!value)),
            (UnOp::BitNot, Value::Int(value)) => Ok(Value::Int(!value)),
            (UnOp::Ref { .. } | UnOp::Deref, value) => Ok(value),
            _ => Err(RuntimeError::new("invalid unary operation", span)),
        }
    }

    fn eval_binary(
        &self,
        op: BinOp,
        left: Value,
        right: Value,
        expr: ExprId,
        span: Span,
    ) -> Result<Value, RuntimeError> {
        if matches!(op, BinOp::Eq | BinOp::Ne) {
            let equal = left == right;
            return Ok(Value::Bool(if op == BinOp::Eq { equal } else { !equal }));
        }
        match (left, right) {
            (Value::Int(left), Value::Int(right)) => {
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
            (Value::Float(left), Value::Float(right)) => match op {
                BinOp::Add => Ok(Value::Float(left + right)),
                BinOp::Sub => Ok(Value::Float(left - right)),
                BinOp::Mul => Ok(Value::Float(left * right)),
                BinOp::Div if right != 0.0 => Ok(Value::Float(left / right)),
                BinOp::Rem if right != 0.0 => Ok(Value::Float(left % right)),
                BinOp::Pow => Ok(Value::Float(left.powf(right))),
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
            (Value::Str(left), Value::Str(right)) => match op {
                BinOp::Lt => Ok(Value::Bool(left < right)),
                BinOp::Le => Ok(Value::Bool(left <= right)),
                BinOp::Gt => Ok(Value::Bool(left > right)),
                BinOp::Ge => Ok(Value::Bool(left >= right)),
                _ => Err(RuntimeError::new("invalid string operation", span)),
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
                Ok(Value::Float(value as f64))
            }
            Value::Float(value) if matches!(target, Ty::Primitive(p) if is_float(p)) => {
                Ok(Value::Float(value))
            }
            Value::Float(value) if matches!(target, Ty::Primitive(p) if is_integer(p)) => {
                if !value.is_finite() || value.fract() != 0.0 {
                    return Err(RuntimeError::new("numeric cast out of range", span));
                }
                let value = value as i128;
                self.check_integer_range(value, expr, span).map(Value::Int)
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

    fn eval_call(
        &mut self,
        callee: ExprId,
        args: &[ExprId],
        span: Span,
    ) -> Result<Flow, RuntimeError> {
        match &self.hir.expr(callee).kind {
            hir::ExprKind::Path { res, .. } => match res {
                Res::Builtin(builtin) => {
                    let values = args
                        .iter()
                        .map(|arg| self.expect_value(*arg))
                        .collect::<Result<Vec<_>, _>>()?;
                    self.call_builtin(*builtin, values, span).map(Flow::Value)
                }
                Res::Item(item) => {
                    let values = args
                        .iter()
                        .map(|arg| self.expect_value(*arg))
                        .collect::<Result<Vec<_>, _>>()?;
                    let callable = self.item_callable(*item).ok_or_else(|| {
                        RuntimeError::new("item is not callable", self.hir.expr(callee).span)
                    })?;
                    self.call_callable(callable, None, values, span)
                        .map(Flow::Value)
                }
                Res::Variant(item, variant) => {
                    let values = args
                        .iter()
                        .map(|arg| self.expect_value(*arg).map(Some))
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(Flow::Value(Value::Enum {
                        item: *item,
                        variant: *variant,
                        fields: values,
                        named: BTreeMap::new(),
                    }))
                }
                Res::TraitMember(trait_id, member) => {
                    self.call_qualified_trait(*trait_id, *member, args, span)
                }
                _ => Err(RuntimeError::new("expression is not callable", span)),
            },
            hir::ExprKind::Field { base, name, .. } => {
                self.call_method(*base, self.text(*name).to_string(), args, span)
            }
            _ => {
                let function = self.expect_value(callee)?;
                let Value::Function(item) = function else {
                    return Err(RuntimeError::new("expression is not callable", span));
                };
                let values = args
                    .iter()
                    .map(|arg| self.expect_value(*arg))
                    .collect::<Result<Vec<_>, _>>()?;
                let callable = self
                    .item_callable(item)
                    .ok_or_else(|| RuntimeError::new("expression is not callable", span))?;
                self.call_callable(callable, None, values, span)
                    .map(Flow::Value)
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
                self.output.push_str(&value.to_string());
                if builtin == Builtin::Println {
                    self.output.push('\n');
                }
                Ok(Value::Unit)
            }
            Builtin::Panic => Err(RuntimeError::new(
                args.pop()
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "panic".to_string()),
                span,
            )),
            Builtin::Assert => match args.pop() {
                Some(Value::Bool(true)) => Ok(Value::Unit),
                Some(Value::Bool(false)) => Err(RuntimeError::new("assertion failed", span)),
                _ => Err(RuntimeError::new("assert expects Bool", span)),
            },
            Builtin::Sqrt => match args.pop() {
                Some(Value::Float(value)) if value >= 0.0 => Ok(Value::Float(value.sqrt())),
                Some(Value::Float(_)) => Err(RuntimeError::new("sqrt domain error", span)),
                _ => Err(RuntimeError::new("sqrt expects Float64", span)),
            },
            Builtin::Drop => {
                if let Some(value) = args.pop() {
                    self.drop_value(value)?;
                }
                Ok(Value::Unit)
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
                    Err(error) => Value::Result(Err(Box::new(Value::String(error.to_string())))),
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
                    Err(error) => Value::Result(Err(Box::new(Value::String(error.to_string())))),
                })
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
            | Builtin::TensorToDevice => Err(RuntimeError::new(
                "tensor operations are not supported in the Core interpreter",
                span,
            )),
        }
    }

    fn call_method(
        &mut self,
        base: ExprId,
        name: String,
        args: &[ExprId],
        span: Span,
    ) -> Result<Flow, RuntimeError> {
        if self.is_core_value(base) {
            return self
                .call_core_method(base, &name, args, span)
                .map(Flow::Value);
        }
        let receiver_value = self.clone_expr_place(base)?;
        let nominal = nominal_item(&receiver_value);
        let method = self.find_method(nominal, &name, None).ok_or_else(|| {
            RuntimeError::new(format!("method '{name}' not found at runtime"), span)
        })?;
        let values = args
            .iter()
            .map(|arg| self.expect_value(*arg))
            .collect::<Result<Vec<_>, _>>()?;
        self.call_user_method(method, base, receiver_value, values, span)
            .map(Flow::Value)
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
        let receiver = self.clone_expr_place(*first)?;
        let method = self
            .find_method(nominal_item(&receiver), &method_name, Some(trait_id))
            .ok_or_else(|| RuntimeError::new("trait implementation not found", span))?;
        let values = rest
            .iter()
            .map(|arg| self.expect_value(*arg))
            .collect::<Result<Vec<_>, _>>()?;
        self.call_user_method(method, *first, receiver, values, span)
            .map(Flow::Value)
    }

    fn find_method(
        &self,
        nominal: Option<ItemId>,
        name: &str,
        trait_filter: Option<ItemId>,
    ) -> Option<Callable> {
        let nominal = nominal?;
        self.hir.items.iter().find_map(|item| {
            let hir::ItemKind::Impl {
                trait_,
                self_ty,
                items,
                ..
            } = &item.kind
            else {
                return None;
            };
            if trait_filter.is_some_and(|expected| {
                !matches!(trait_, Some(reference) if reference.res == Res::Item(expected))
            }) {
                return None;
            }
            if !matches!(
                &self.hir.ty(*self_ty).kind,
                hir::TypeKind::Path { res: Res::Item(item), .. } if *item == nominal
            ) {
                return None;
            }
            items.iter().find_map(|item| match item {
                hir::ImplItem::Fn { def, .. } if self.text(def.sig.name) == name => {
                    Some(Callable {
                        receiver: def.sig.receiver.zip(def.sig.receiver_local),
                        params: def.sig.params.iter().map(|param| param.local).collect(),
                        body: def.body,
                    })
                }
                _ => None,
            })
        })
    }

    fn call_user_method(
        &mut self,
        callable: Callable,
        base: ExprId,
        borrowed_receiver: Value,
        args: Vec<Value>,
        span: Span,
    ) -> Result<Value, RuntimeError> {
        let Some((receiver_kind, receiver_local)) = callable.receiver else {
            return Err(RuntimeError::new("method has no receiver", span));
        };
        let receiver = match receiver_kind {
            hir::Receiver::Value => self.expect_value(base)?,
            hir::Receiver::Ref => borrowed_receiver,
            hir::Receiver::RefMut => {
                let place = self.expr_place(base)?;
                self.place_slot_mut(&place, span)?
                    .take()
                    .ok_or_else(|| RuntimeError::new("mutable receiver is unavailable", span))?
            }
        };
        let mut frame = Frame::default();
        frame.insert(receiver_local, Some(receiver));
        for (local, value) in callable.params.iter().copied().zip(args) {
            frame.insert(local, Some(value));
        }
        self.frames.push(frame);
        let result = self.eval_block(callable.body);
        if result.is_err() {
            self.frames.pop();
            return result.map(|_| Value::Unit);
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
        self.cleanup_current_frame()?;
        self.frames.pop();
        if receiver_kind == hir::Receiver::RefMut {
            let restored = restored.ok_or_else(|| {
                RuntimeError::new("mutable receiver was moved by its method", span)
            })?;
            let place = self.expr_place(base)?;
            self.write_place(&place, restored, span)?;
        }
        match flow {
            Flow::Value(value) | Flow::Return(value) => Ok(value),
            Flow::Propagate(value) => Ok(value),
            Flow::Break(_) | Flow::Continue => {
                Err(RuntimeError::new("loop control escaped a method", span))
            }
        }
    }

    fn is_core_value(&self, expr: ExprId) -> bool {
        matches!(
            self.tables.expr_types.get(&expr),
            Some(
                Ty::Primitive(Primitive::String | Primitive::Str)
                    | Ty::Core(..)
                    | Ty::Array(..)
                    | Ty::Slice(..)
            )
        )
    }

    fn call_core_method(
        &mut self,
        base: ExprId,
        name: &str,
        args: &[ExprId],
        span: Span,
    ) -> Result<Value, RuntimeError> {
        let values = args
            .iter()
            .map(|arg| self.expect_value(*arg))
            .collect::<Result<Vec<_>, _>>()?;
        let mut values = values.into_iter();
        let mutating = matches!(
            name,
            "push" | "push_str" | "pop" | "clear" | "insert" | "remove" | "append"
        );
        let mut owned;
        let target = if mutating {
            let place = self.expr_place(base)?;
            self.place_value_mut(&place, span)?
        } else {
            owned = self.clone_expr_place(base)?;
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
                "get" => {
                    let index = usize_arg(values.next(), span)?;
                    Ok(Value::Option(
                        vector.get(index).and_then(Clone::clone).map(Box::new),
                    ))
                }
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
                "as_slice" => Ok(Value::Array(vector.clone())),
                _ => Err(RuntimeError::new(
                    format!("unsupported Vec method '{name}'"),
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
            Value::Boxed(value) if name == "into_inner" => Ok(value.take().unwrap_or(Value::Unit)),
            _ => Err(RuntimeError::new(
                format!("method '{name}' is unavailable for this value"),
                span,
            )),
        }
    }

    fn eval_struct_lit(
        &mut self,
        res: Res,
        fields: &[hir::FieldInit],
        span: Span,
    ) -> Result<Flow, RuntimeError> {
        let mut values = BTreeMap::new();
        for field in fields {
            let name = self.text(field.name).to_string();
            let value = if let Some(expr) = field.expr {
                self.expect_value(expr)?
            } else {
                let local = self.find_local_by_name(&name).ok_or_else(|| {
                    RuntimeError::new(format!("unknown shorthand field '{name}'"), field.name)
                })?;
                self.take_place(
                    &Place {
                        local,
                        projections: Vec::new(),
                    },
                    field.name,
                )?
            };
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
        &self,
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
                (
                    Res::Variant(item, variant),
                    Value::Enum {
                        item: actual,
                        variant: actual_variant,
                        ..
                    },
                ) => Ok(item == actual && variant == actual_variant),
                (Res::Builtin(Builtin::None), Value::Option(None)) => Ok(true),
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
        &self,
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
            _ => Err(RuntimeError::new("value is not iterable", span)),
        }
    }

    fn slice_value(&self, value: Value, range: Value, span: Span) -> Result<Value, RuntimeError> {
        let Value::Range {
            start,
            end,
            inclusive,
        } = range
        else {
            return Err(RuntimeError::new("slice index must be a range", span));
        };
        let start = usize::try_from(start)
            .map_err(|_| RuntimeError::new("slice start is negative", span))?;
        let mut end =
            usize::try_from(end).map_err(|_| RuntimeError::new("slice end is negative", span))?;
        if inclusive {
            end = end
                .checked_add(1)
                .ok_or_else(|| RuntimeError::new("slice range overflow", span))?;
        }
        let values = match value {
            Value::Array(values) | Value::Vec(values) => values,
            _ => return Err(RuntimeError::new("value cannot be sliced", span)),
        };
        if start > end || end > values.len() {
            return Err(RuntimeError::new("slice range out of bounds", span));
        }
        Ok(Value::Array(values[start..end].to_vec()))
    }

    fn expr_place(&mut self, expr: ExprId) -> Result<Place, RuntimeError> {
        let node = self.hir.expr(expr);
        match &node.kind {
            hir::ExprKind::Path {
                res: Res::Local(local) | Res::SelfValue(local),
                ..
            } => Ok(Place {
                local: *local,
                projections: Vec::new(),
            }),
            hir::ExprKind::Field { base, name, .. } => {
                let mut place = self.expr_place(*base)?;
                place
                    .projections
                    .push(Projection::Field(self.text(*name).to_string()));
                Ok(place)
            }
            hir::ExprKind::TupleField { base, index } => {
                let mut place = self.expr_place(*base)?;
                let index = self
                    .text(*index)
                    .parse::<usize>()
                    .map_err(|_| RuntimeError::new("invalid tuple index", *index))?;
                place.projections.push(Projection::Index(index));
                Ok(place)
            }
            hir::ExprKind::Index { base, index } => {
                let mut place = self.expr_place(*base)?;
                let index = usize::try_from(self.expect_int(*index)?)
                    .map_err(|_| RuntimeError::new("negative index", self.hir.expr(*index).span))?;
                place.projections.push(Projection::Index(index));
                Ok(place)
            }
            hir::ExprKind::Unary {
                op: UnOp::Deref,
                operand,
            } => self.expr_place(*operand),
            _ => Err(RuntimeError::new("expression is not a place", node.span)),
        }
    }

    fn clone_expr_place(&mut self, expr: ExprId) -> Result<Value, RuntimeError> {
        if let Ok(place) = self.expr_place(expr) {
            return self.place_value(&place, self.hir.expr(expr).span).cloned();
        }
        self.expect_value(expr)
    }

    fn place_value(&self, place: &Place, span: Span) -> Result<&Value, RuntimeError> {
        let mut value = self
            .frame()
            .values
            .get(&place.local)
            .and_then(Option::as_ref)
            .ok_or_else(|| RuntimeError::new("use of unavailable value", span))?;
        for projection in &place.projections {
            value = project(value, projection)
                .and_then(Option::as_ref)
                .ok_or_else(|| RuntimeError::new("use of moved or invalid field", span))?;
        }
        Ok(value)
    }

    fn place_value_mut(&mut self, place: &Place, span: Span) -> Result<&mut Value, RuntimeError> {
        let mut value = self
            .frame_mut()
            .values
            .get_mut(&place.local)
            .and_then(Option::as_mut)
            .ok_or_else(|| RuntimeError::new("use of unavailable value", span))?;
        for projection in &place.projections {
            value = project_mut(value, projection)
                .and_then(Option::as_mut)
                .ok_or_else(|| RuntimeError::new("use of moved or invalid field", span))?;
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
            .frame_mut()
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
            | Value::Float(_)
            | Value::Char(_)
            | Value::Str(_)
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
            Value::String(_) | Value::Vec(_) | Value::Boxed(_) | Value::Range { .. } => false,
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
                if let Some((_, local)) = callable.receiver {
                    frame.insert(local, Some(value.clone()));
                }
                self.frames.push(frame);
                let result = self.eval_block(callable.body);
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
            Value::Struct { fields, .. } => {
                for child in fields.values_mut().rev().filter_map(Option::take) {
                    self.drop_value(child)?;
                }
            }
            Value::Enum { fields, named, .. } => {
                for child in fields.iter_mut().rev().filter_map(Option::take) {
                    self.drop_value(child)?;
                }
                for child in named.values_mut().rev().filter_map(Option::take) {
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
            _ => {}
        }
        Ok(())
    }

    fn find_drop(&self, item: ItemId) -> Option<Callable> {
        self.hir.items.iter().find_map(|candidate| {
            let hir::ItemKind::Impl { trait_: Some(reference), self_ty, items, .. } = &candidate.kind else { return None; };
            if reference.res != Res::CoreTrait(hir::CoreTrait::Drop) || !matches!(&self.hir.ty(*self_ty).kind, hir::TypeKind::Path { res: Res::Item(actual), .. } if *actual == item) { return None; }
            items.iter().find_map(|item| match item {
                hir::ImplItem::Fn { def, .. } if self.text(def.sig.name) == "drop" => Some(Callable { receiver: def.sig.receiver.zip(def.sig.receiver_local), params: def.sig.params.iter().map(|param| param.local).collect(), body: def.body }),
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

fn project<'a>(value: &'a Value, projection: &Projection) -> Option<&'a Option<Value>> {
    match (value, projection) {
        (Value::Struct { fields, .. }, Projection::Field(name))
        | (Value::Enum { named: fields, .. }, Projection::Field(name)) => fields.get(name),
        (
            Value::Tuple(values) | Value::Array(values) | Value::Vec(values),
            Projection::Index(index),
        ) => values.get(*index),
        _ => None,
    }
}

fn project_mut<'a>(value: &'a mut Value, projection: &Projection) -> Option<&'a mut Option<Value>> {
    match (value, projection) {
        (Value::Struct { fields, .. }, Projection::Field(name))
        | (Value::Enum { named: fields, .. }, Projection::Field(name)) => fields.get_mut(name),
        (
            Value::Tuple(values) | Value::Array(values) | Value::Vec(values),
            Projection::Index(index),
        ) => values.get_mut(*index),
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

fn is_float(primitive: Primitive) -> bool {
    matches!(primitive, Primitive::Float32 | Primitive::Float64)
}

fn usize_arg(value: Option<Value>, span: Span) -> Result<usize, RuntimeError> {
    match value {
        Some(Value::Int(value)) => usize::try_from(value)
            .map_err(|_| RuntimeError::new("integer does not fit usize", span)),
        _ => Err(RuntimeError::new("expected integer argument", span)),
    }
}

fn string_arg(value: Option<Value>, span: Span) -> Result<String, RuntimeError> {
    match value {
        Some(Value::Str(value) | Value::String(value)) => Ok(value),
        _ => Err(RuntimeError::new("expected string argument", span)),
    }
}

fn parse_string(text: &str, raw: bool) -> String {
    let content = if raw {
        text.strip_prefix('r').unwrap_or(text)
    } else {
        text
    };
    let content = content
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .unwrap_or(content);
    if raw {
        return content.to_string();
    }
    let mut result = String::new();
    let mut chars = content.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            result.push(ch);
            continue;
        }
        match chars.next() {
            Some('n') => result.push('\n'),
            Some('r') => result.push('\r'),
            Some('t') => result.push('\t'),
            Some('0') => result.push('\0'),
            Some('\\') => result.push('\\'),
            Some('"') => result.push('"'),
            Some(other) => result.push(other),
            None => {}
        }
    }
    result
}

fn parse_char(text: &str) -> Option<char> {
    let content = text.strip_prefix('\'')?.strip_suffix('\'')?;
    if let Some(escaped) = content.strip_prefix('\\') {
        match escaped {
            "n" => Some('\n'),
            "r" => Some('\r'),
            "t" => Some('\t'),
            "0" => Some('\0'),
            "\\" => Some('\\'),
            "'" => Some('\''),
            _ => None,
        }
    } else {
        let mut chars = content.chars();
        let value = chars.next()?;
        chars.next().is_none().then_some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{parse, ParseMode};
    use crate::resolve::resolve;
    use crate::typecheck;

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

    #[test]
    fn runs_drop_in_reverse_declaration_order() {
        let execution = execute(
            "struct Marker { name: String } impl Drop for Marker { fn drop(&mut self) { println(self.name.as_str()); } } fn main() { let first = Marker { name: String::from(\"first\") }; let second = Marker { name: String::from(\"second\") }; }",
        )
        .unwrap();
        assert_eq!(execution.output, "second\nfirst\n");
    }
}
