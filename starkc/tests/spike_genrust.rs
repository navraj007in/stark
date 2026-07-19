//! WP-C3.2 — Generated-Rust backend spike. **ISOLATED, DISPOSABLE (charter §2.2).**
//!
//! This is a Gate C3 architecture spike, not a production backend. It is deliberately confined
//! to an integration-test file: it is NOT wired into `stark build`, adds nothing to the library
//! surface, and must not be treated as the compiler's code generator. Its only job is to produce
//! evidence for WP-C3.4's backend selection (escalation CE5).
//!
//! What it does: takes a program that has ALREADY passed the real front end (parse, resolve,
//! type/borrow check — it does not bypass any of those, charter §2.2), walks the typed HIR, and
//! lowers a *supported subset* to Rust source. Anything outside the subset returns
//! `Unsupported(reason)` cleanly — the set of unsupported constructs is itself a measurement
//! (WP-C3.2 "record unsupported constructs"). The generated Rust is compiled with `rustc`, run,
//! and its stdout + exit status compared against the reference interpreter (the semantic oracle,
//! charter §1.6 rule 6) over the frozen `exec_snapshots` corpus (`corpus_version = 1.0.0`).
//!
//! Supported subset (enough to exercise items 1, 2, and the trap half of 8 from the frozen
//! workload in NATIVE-CORE-ARCHITECTURE.md §5): integer primitives (i8..u64) and Bool; arithmetic
//! with STARK trap-on-overflow/div-by-zero semantics; comparisons and boolean logic; `let`/`let
//! mut` and assignment; `if`/`else`, `while`, `loop`+bare `break`, `continue`, `for x in a..b`;
//! block-tail values; user function definitions and calls; `print`/`println`. Everything else
//! (String/Vec, structs/enums, generics, traits, references, Drop, function values, `?`, match,
//! the provider boundary) is reported unsupported.

use starkc::ast::{BinOp, Lit, Primitive, UnOp};
use starkc::hir::{Builtin, ExprId, ExprKind, Hir, ItemKind, Res, StmtKind};
use starkc::interp;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::{SourceFile, Span};
use starkc::typecheck::{self, Ty, TypeTables};
use std::path::PathBuf;
use std::sync::Arc;

// ------------------------------------------------------------------ lowering --

struct Unsupported(String);

fn unsup(msg: impl Into<String>) -> Unsupported {
    Unsupported(msg.into())
}

struct Lowerer<'a> {
    hir: &'a Hir,
    src: &'a str,
    tables: &'a TypeTables,
    out: String,
}

impl<'a> Lowerer<'a> {
    fn text(&self, span: Span) -> &'a str {
        &self.src[span.lo as usize..span.hi as usize]
    }

    fn expr_ty(&self, id: ExprId) -> Result<Ty, Unsupported> {
        self.tables
            .expr_types
            .get(&id)
            .cloned()
            .ok_or_else(|| unsup("expression has no recorded type"))
    }

    /// Emit the full program prelude + every lowered function.
    fn lower_program(&mut self) -> Result<(), Unsupported> {
        self.out.push_str(RUNTIME_PRELUDE);
        let items = match &self.hir.root {
            starkc::hir::Root::Program(items) => items.clone(),
            _ => return Err(unsup("non-program root (snippet)")),
        };
        for item_id in items {
            match &self.hir.item(item_id).kind {
                ItemKind::Fn(def) => self.lower_fn(def)?,
                ItemKind::Struct {
                    name,
                    fields,
                    generics,
                } => {
                    // Breadth run: struct definitions. Subset = all-primitive fields (Copy).
                    if !generics.is_empty() {
                        return Err(unsup("generic struct"));
                    }
                    self.out.push_str("#[derive(Clone, Copy)]\n");
                    self.out
                        .push_str(&format!("struct {} {{\n", self.text(*name)));
                    for f in fields {
                        let fty = self.rust_type(f.ty)?;
                        self.out
                            .push_str(&format!("    {}: {fty},\n", self.text(f.name)));
                    }
                    self.out.push_str("}\n\n");
                }
                ItemKind::Impl {
                    trait_: None,
                    self_ty,
                    items: impl_items,
                    generics,
                    ..
                } => {
                    if !generics.is_empty() {
                        return Err(unsup("generic impl"));
                    }
                    let ty_name = self.rust_type(*self_ty)?;
                    self.out.push_str(&format!("impl {ty_name} {{\n"));
                    for ii in impl_items {
                        if let starkc::hir::ImplItem::Fn { def, .. } = ii {
                            let body = self.lower_method(def)?;
                            self.out.push_str(&body);
                        } else {
                            return Err(unsup("non-fn impl item"));
                        }
                    }
                    self.out.push_str("}\n\n");
                }
                ItemKind::Use(_) | ItemKind::Mod { .. } => {}
                other => return Err(unsup(format!("top-level item {}", item_kind_name(other)))),
            }
        }
        Ok(())
    }

    /// Lower a method or associated function inside an `impl` block (indented body).
    fn lower_method(&mut self, def: &starkc::hir::FnDef) -> Result<String, Unsupported> {
        if !def.sig.generics.is_empty() {
            return Err(unsup("generic method"));
        }
        let name = self.text(def.sig.name);
        let mut params = String::new();
        match def.sig.receiver {
            Some(starkc::hir::Receiver::Ref) => params.push_str("&self"),
            Some(starkc::hir::Receiver::RefMut) => params.push_str("&mut self"),
            Some(starkc::hir::Receiver::Value) => params.push_str("self"),
            None => {}
        }
        for p in &def.sig.params {
            if !params.is_empty() {
                params.push_str(", ");
            }
            let pty = self.rust_type(p.ty)?;
            params.push_str(&format!("{}: {pty}", self.text(p.name)));
        }
        let ret = match &def.sig.ret {
            starkc::hir::RetTy::Unit => "()".to_string(),
            starkc::hir::RetTy::Ty(t) => self.rust_type(*t)?,
            starkc::hir::RetTy::Never(_) => return Err(unsup("never-returning method")),
        };
        let body = self.lower_block(def.body)?;
        Ok(format!("    fn {name}({params}) -> {ret} {body}\n"))
    }

    /// General type lowering: primitives, nominal struct/enum types, and core Option/Result/
    /// String/str.
    fn rust_type(&self, ty_id: starkc::hir::TypeId) -> Result<String, Unsupported> {
        use starkc::hir::CoreType;
        match &self.hir.ty(ty_id).kind {
            starkc::hir::TypeKind::Primitive(Primitive::String) => Ok("String".to_string()),
            starkc::hir::TypeKind::Primitive(Primitive::Str) => Ok("&str".to_string()),
            starkc::hir::TypeKind::Primitive(p) => Ok(rust_prim(*p)
                .ok_or_else(|| unsup(format!("primitive {p:?}")))?
                .to_string()),
            starkc::hir::TypeKind::Path { res, path, args } => match res {
                Res::Item(item) => match &self.hir.item(*item).kind {
                    ItemKind::Struct { name, .. } | ItemKind::Enum { name, .. } => {
                        Ok(self.text(*name).to_string())
                    }
                    _ => Err(unsup("path type is not a struct/enum")),
                },
                Res::CoreType(CoreType::String) => Ok("String".to_string()),
                Res::CoreType(ct @ (CoreType::Option | CoreType::Result)) => {
                    let inner = self.type_args(args)?;
                    let head = if matches!(ct, CoreType::Option) {
                        "Option"
                    } else {
                        "Result"
                    };
                    Ok(format!("{head}<{}>", inner.join(", ")))
                }
                // A generic type parameter (`T`) — the name is valid Rust.
                Res::TypeParam => Ok(self.text(path.span).to_string()),
                _ => Err(unsup(format!("path type: {}", self.text(path.span)))),
            },
            _ => Err(unsup("unsupported type form")),
        }
    }

    fn type_args(
        &self,
        args: &Option<starkc::hir::GenericArgs>,
    ) -> Result<Vec<String>, Unsupported> {
        let mut out = Vec::new();
        if let Some(a) = args {
            for arg in &a.args {
                match arg {
                    starkc::hir::GenericArg::Type(t) => out.push(self.rust_type(*t)?),
                    _ => return Err(unsup("non-type generic argument")),
                }
            }
        }
        Ok(out)
    }

    fn lower_pat(&self, pid: starkc::hir::PatId) -> Result<String, Unsupported> {
        use starkc::hir::PatKind;
        match &self.hir.pat(pid).kind {
            PatKind::Wild => Ok("_".to_string()),
            PatKind::Binding { name, .. } => Ok(self.text(*name).to_string()),
            PatKind::Lit(_) => Ok(self.text(self.hir.pat(pid).span).to_string()),
            // None, unit enum variants, and constants: the path text (`None`, `Enum::Variant`) is
            // valid Rust for the subset.
            PatKind::Path { path, .. } => Ok(self.text(path.span).to_string()),
            PatKind::TupleVariant { path, pats, .. } => {
                let mut inner = Vec::new();
                for p in pats {
                    inner.push(self.lower_pat(*p)?);
                }
                Ok(format!("{}({})", self.text(path.span), inner.join(", ")))
            }
            PatKind::Struct { path, fields, .. } => {
                let mut fs = Vec::new();
                for f in fields {
                    match f.pat {
                        Some(p) => {
                            fs.push(format!("{}: {}", self.text(f.name), self.lower_pat(p)?))
                        }
                        None => fs.push(self.text(f.name).to_string()), // shorthand
                    }
                }
                Ok(format!("{} {{ {} }}", self.text(path.span), fs.join(", ")))
            }
            PatKind::Tuple(pats) => {
                let mut inner = Vec::new();
                for p in pats {
                    inner.push(self.lower_pat(*p)?);
                }
                Ok(format!("({})", inner.join(", ")))
            }
            PatKind::Array(_) => Err(unsup("array pattern")),
            PatKind::Error => Err(unsup("error pattern")),
        }
    }

    fn lower_fn(&mut self, def: &starkc::hir::FnDef) -> Result<(), Unsupported> {
        if def.sig.receiver.is_some() {
            return Err(unsup("method (self receiver)"));
        }
        let name = self.text(def.sig.name);
        let generics = self.lower_generics(&def.sig.generics)?;
        let mut params = String::new();
        for (i, p) in def.sig.params.iter().enumerate() {
            if i > 0 {
                params.push_str(", ");
            }
            let pty = self.rust_type(p.ty)?;
            params.push_str(&format!("{}: {pty}", self.text(p.name)));
        }
        let ret = match &def.sig.ret {
            starkc::hir::RetTy::Unit => "()".to_string(),
            starkc::hir::RetTy::Ty(t) => self.rust_type(*t)?,
            starkc::hir::RetTy::Never(_) => return Err(unsup("never-returning function")),
        };
        // `main` stays `main`; everything else keeps its STARK name (valid Rust idents in-subset).
        self.out
            .push_str(&format!("fn {name}{generics}({params}) -> {ret} {{\n"));
        let body = self.lower_block(def.body)?;
        self.out.push_str(&body);
        self.out.push_str("\n}\n\n");
        Ok(())
    }

    /// Lower generic parameters with trait bounds: `<T: Ord, U: Clone>`. STARK's compiler-known
    /// bound names (Ord/Eq/Clone/...) match Rust's spelling for the subset.
    fn lower_generics(
        &self,
        generics: &[starkc::hir::GenericParam],
    ) -> Result<String, Unsupported> {
        if generics.is_empty() {
            return Ok(String::new());
        }
        let mut parts = Vec::new();
        for g in generics {
            let mut bounds = Vec::new();
            for bound in &g.bounds {
                bounds.push(self.text(bound.path.span).to_string());
            }
            if bounds.is_empty() {
                parts.push(self.text(g.name).to_string());
            } else {
                parts.push(format!("{}: {}", self.text(g.name), bounds.join(" + ")));
            }
        }
        Ok(format!("<{}>", parts.join(", ")))
    }

    fn convert_param_ty(&self, ty_id: starkc::hir::TypeId) -> Result<&'static str, Unsupported> {
        match &self.hir.ty(ty_id).kind {
            starkc::hir::TypeKind::Primitive(p) => {
                rust_prim(*p).ok_or_else(|| unsup(format!("primitive {p:?}")))
            }
            _ => Err(unsup("non-primitive parameter/return type")),
        }
    }

    /// Lower a block to a Rust brace-block string (statements + optional tail value).
    fn lower_block(&mut self, block_id: starkc::hir::BlockId) -> Result<String, Unsupported> {
        let block = self.hir.block(block_id);
        let mut s = String::from("{\n");
        for &stmt_id in &block.stmts {
            let line = self.lower_stmt(stmt_id)?;
            s.push_str(&line);
            s.push('\n');
        }
        if let Some(tail) = block.tail {
            s.push_str(&self.lower_expr(tail)?);
            s.push('\n');
        }
        s.push('}');
        Ok(s)
    }

    fn lower_stmt(&mut self, stmt_id: starkc::hir::StmtId) -> Result<String, Unsupported> {
        match &self.hir.stmt(stmt_id).kind {
            StmtKind::Empty => Ok(String::new()),
            StmtKind::Expr { expr, semi } => {
                let e = self.lower_expr(*expr)?;
                Ok(if *semi { format!("{e};") } else { e })
            }
            StmtKind::Let {
                mutable,
                name,
                ty,
                init,
                ..
            } => {
                let mutkw = if *mutable { "mut " } else { "" };
                let n = self.text(*name);
                let annot = match ty {
                    Some(t) => format!(": {}", self.convert_param_ty(*t)?),
                    None => String::new(),
                };
                match init {
                    Some(e) => Ok(format!("let {mutkw}{n}{annot} = {};", self.lower_expr(*e)?)),
                    None => Ok(format!("let {mutkw}{n}{annot};")),
                }
            }
            StmtKind::Return(opt) => match opt {
                Some(e) => Ok(format!("return {};", self.lower_expr(*e)?)),
                None => Ok("return;".to_string()),
            },
            StmtKind::Break(opt) => match opt {
                None => Ok("break;".to_string()),
                Some(_) => Err(unsup("break with value (loop-as-value)")),
            },
            StmtKind::Continue => Ok("continue;".to_string()),
            StmtKind::Item(_) => Err(unsup("nested item")),
            StmtKind::Error => Err(unsup("error statement")),
        }
    }

    /// Lower an expression to a Rust expression string.
    fn lower_expr(&mut self, id: ExprId) -> Result<String, Unsupported> {
        match &self.hir.expr(id).kind {
            ExprKind::Lit(lit) => self.lower_lit(id, lit),
            ExprKind::Path { path, res, .. } => match res {
                Res::Local(_) | Res::SelfValue(_) => Ok(self.text(path.span).to_string()),
                Res::Builtin(Builtin::None) => Ok("None".to_string()),
                other => Err(unsup(format!(
                    "path resolving to {other:?} in value position"
                ))),
            },
            ExprKind::Unary { op, operand } => {
                let inner = self.lower_expr(*operand)?;
                match op {
                    UnOp::Neg => Ok(format!("({inner}).s_neg()")),
                    UnOp::Not => Ok(format!("(!{inner})")),
                    UnOp::BitNot => Err(unsup("bitwise-not operator")),
                    UnOp::Ref { .. } => Err(unsup("reference-taking (&/&mut)")),
                    UnOp::Deref => Err(unsup("dereference")),
                }
            }
            ExprKind::Binary { op, lhs, rhs } => self.lower_binary(*op, *lhs, *rhs),
            ExprKind::Assign { op, lhs, rhs } => {
                let l = self.lower_expr(*lhs)?;
                let r = self.lower_expr(*rhs)?;
                use starkc::ast::AssignOp::*;
                let helper = match op {
                    Assign => return Ok(format!("{l} = {r}")),
                    AddAssign => "s_add",
                    SubAssign => "s_sub",
                    MulAssign => "s_mul",
                    DivAssign => "s_div",
                    RemAssign => "s_rem",
                    _ => return Err(unsup("compound bit/pow assignment")),
                };
                Ok(format!("{l} = ({l}).{helper}({r})"))
            }
            ExprKind::Call { callee, args } => self.lower_call(*callee, args),
            ExprKind::If {
                cond,
                then_block,
                else_,
            } => {
                let c = self.lower_expr(*cond)?;
                let t = self.lower_block(*then_block)?;
                match else_ {
                    Some(e) => Ok(format!("if {c} {t} else {}", self.lower_expr(*e)?)),
                    None => Ok(format!("if {c} {t}")),
                }
            }
            ExprKind::While { cond, body } => Ok(format!(
                "while {} {}",
                self.lower_expr(*cond)?,
                self.lower_block(*body)?
            )),
            ExprKind::Loop { body } => Ok(format!("loop {}", self.lower_block(*body)?)),
            ExprKind::For {
                var, iter, body, ..
            } => {
                let (lo, hi, incl) = match &self.hir.expr(*iter).kind {
                    ExprKind::Range { lo, hi, inclusive } => (*lo, *hi, *inclusive),
                    _ => return Err(unsup("for-loop over a non-range iterator")),
                };
                let op = if incl { "..=" } else { ".." };
                Ok(format!(
                    "for {} in {}{op}{} {}",
                    self.text(*var),
                    self.lower_expr(lo)?,
                    self.lower_expr(hi)?,
                    self.lower_block(*body)?
                ))
            }
            ExprKind::Block(b) => self.lower_block(*b),
            // Everything below is deliberately out of the spike subset.
            ExprKind::Field { base, name, .. } => Ok(format!(
                "({}).{}",
                self.lower_expr(*base)?,
                self.text(*name)
            )),
            ExprKind::TupleField { .. } => Err(unsup("tuple field")),
            ExprKind::Index { .. } => Err(unsup("indexing")),
            ExprKind::Try(_) => Err(unsup("`?` operator")),
            ExprKind::Tuple(_) => Err(unsup("tuple")),
            ExprKind::Array(_) | ExprKind::Repeat { .. } => Err(unsup("array")),
            ExprKind::StructLit { path, res, fields } => {
                let name = match res {
                    Res::Item(item) => match &self.hir.item(*item).kind {
                        ItemKind::Struct { name, .. } => self.text(*name).to_string(),
                        _ => return Err(unsup("struct literal of non-struct")),
                    },
                    _ => {
                        return Err(unsup(format!(
                            "struct literal path {}",
                            self.text(path.span)
                        )))
                    }
                };
                let mut out = format!("{name} {{ ");
                for (i, f) in fields.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    let val = match f.expr {
                        Some(e) => self.lower_expr(e)?,
                        None => self.text(f.name).to_string(), // field shorthand
                    };
                    out.push_str(&format!("{}: {val}", self.text(f.name)));
                }
                out.push_str(" }");
                Ok(out)
            }
            ExprKind::Match { scrutinee, arms } => {
                let s = self.lower_expr(*scrutinee)?;
                let mut out = format!("match ({s}) {{\n");
                for arm in arms {
                    let pat = self.lower_pat(arm.pat)?;
                    let body = self.lower_expr(arm.body)?;
                    out.push_str(&format!("    {pat} => {{ {body} }}\n"));
                }
                out.push('}');
                Ok(out)
            }
            ExprKind::Cast { .. } => Err(unsup("`as` cast")),
            ExprKind::Range { .. } => Err(unsup("range value outside a for-loop")),
            ExprKind::Error => Err(unsup("error expression")),
        }
    }

    fn lower_binary(&mut self, op: BinOp, lhs: ExprId, rhs: ExprId) -> Result<String, Unsupported> {
        let l = self.lower_expr(lhs)?;
        let r = self.lower_expr(rhs)?;
        use BinOp::*;
        // Arithmetic on integers goes through trap-checked helpers to preserve STARK's
        // always-trap-on-overflow/div-by-zero semantics regardless of build profile.
        let arith = match op {
            Add => Some("s_add"),
            Sub => Some("s_sub"),
            Mul => Some("s_mul"),
            Div => Some("s_div"),
            Rem => Some("s_rem"),
            _ => None,
        };
        if let Some(helper) = arith {
            let ty = self.expr_ty(lhs)?;
            if !matches!(&ty, Ty::Primitive(p) if is_integer(*p)) {
                return Err(unsup("arithmetic on a non-integer operand"));
            }
            return Ok(format!("({l}).{helper}({r})"));
        }
        let rust_op = match op {
            Eq => "==",
            Ne => "!=",
            Lt => "<",
            Le => "<=",
            Gt => ">",
            Ge => ">=",
            And => "&&",
            Or => "||",
            Pow => return Err(unsup("`**` power operator")),
            BitAnd | BitOr | BitXor | Shl | Shr => return Err(unsup("bitwise/shift operator")),
            Add | Sub | Mul | Div | Rem => unreachable!(),
        };
        Ok(format!("({l} {rust_op} {r})"))
    }

    fn lower_call(&mut self, callee: ExprId, args: &[ExprId]) -> Result<String, Unsupported> {
        // Method call: `receiver.method(args)` — callee is a Field expression.
        if let ExprKind::Field { base, name, .. } = &self.hir.expr(callee).kind {
            let recv = self.lower_expr(*base)?;
            let mut lowered = Vec::new();
            for &a in args {
                lowered.push(self.lower_expr(a)?);
            }
            return Ok(format!(
                "({recv}).{}({})",
                self.text(*name),
                lowered.join(", ")
            ));
        }
        let ExprKind::Path { res, path, .. } = &self.hir.expr(callee).kind else {
            return Err(unsup("indirect / non-path call"));
        };
        match res {
            Res::Builtin(Builtin::Println) => {
                let a = self.lower_one_arg(args)?;
                Ok(format!("stark_println({a})"))
            }
            Res::Builtin(Builtin::Print) => {
                let a = self.lower_one_arg(args)?;
                Ok(format!("stark_print({a})"))
            }
            Res::Builtin(ctor @ (Builtin::Some | Builtin::Ok | Builtin::Err)) => {
                let mut lowered = Vec::new();
                for &a in args {
                    lowered.push(self.lower_expr(a)?);
                }
                let head = match ctor {
                    Builtin::Some => "Some",
                    Builtin::Ok => "Ok",
                    Builtin::Err => "Err",
                    _ => unreachable!(),
                };
                Ok(format!("{head}({})", lowered.join(", ")))
            }
            Res::Builtin(Builtin::StringFrom) => {
                // String::from(literal)
                let mut lowered = Vec::new();
                for &a in args {
                    lowered.push(self.lower_expr(a)?);
                }
                Ok(format!("String::from({})", lowered.join(", ")))
            }
            Res::Builtin(b) => Err(unsup(format!("builtin {b:?}"))),
            Res::Item(item_id) => {
                let ItemKind::Fn(def) = &self.hir.item(*item_id).kind else {
                    return Err(unsup("call to a non-function item"));
                };
                let name = self.text(def.sig.name);
                let mut lowered = Vec::new();
                for &a in args {
                    lowered.push(self.lower_expr(a)?);
                }
                Ok(format!("{name}({})", lowered.join(", ")))
            }
            // Associated-function call: `Type::func(args)` (e.g. `Point::new(3, 4)`). The path
            // text is already `Type::func` and is valid Rust for the subset.
            Res::AssociatedFn(..) => {
                let mut lowered = Vec::new();
                for &a in args {
                    lowered.push(self.lower_expr(a)?);
                }
                Ok(format!("{}({})", self.text(path.span), lowered.join(", ")))
            }
            other => Err(unsup(format!("call to {other:?}"))),
        }
    }

    fn lower_one_arg(&mut self, args: &[ExprId]) -> Result<String, Unsupported> {
        use starkc::hir::CoreType;
        if args.len() != 1 {
            return Err(unsup("print/println with != 1 argument"));
        }
        // Peel references — `&str`, `&String`, `&Int32` all print via the value's Display.
        let mut ty = self.expr_ty(args[0])?;
        while let Ty::Ref { inner, .. } = ty {
            ty = *inner;
        }
        let printable = matches!(&ty, Ty::Primitive(p)
            if is_integer(*p) || *p == Primitive::Bool || *p == Primitive::Str || *p == Primitive::String)
            || matches!(&ty, Ty::Core(CoreType::String, _));
        if !printable {
            return Err(unsup("print/println of an unsupported value type"));
        }
        self.lower_expr(args[0])
    }

    fn lower_lit(&self, id: ExprId, lit: &Lit) -> Result<String, Unsupported> {
        match lit {
            Lit::Bool(b) => Ok(b.to_string()),
            Lit::Int { .. } => {
                let ty = self.expr_ty(id)?;
                let Ty::Primitive(p) = ty else {
                    return Err(unsup("integer literal with non-primitive type"));
                };
                let suffix =
                    rust_prim(p).ok_or_else(|| unsup(format!("int literal type {p:?}")))?;
                let raw = self.text(self.hir.expr(id).span);
                let base = strip_int_suffix(raw);
                Ok(format!("{base}{suffix}"))
            }
            Lit::Float { .. } => Err(unsup("float literal")),
            // STARK string-literal syntax (incl. the leading `r` on raw strings) is valid Rust.
            Lit::Str { .. } => Ok(self.text(self.hir.expr(id).span).to_string()),
            Lit::Char => Err(unsup("char literal")),
        }
    }
}

fn rust_prim(p: Primitive) -> Option<&'static str> {
    Some(match p {
        Primitive::Int8 => "i8",
        Primitive::Int16 => "i16",
        Primitive::Int32 => "i32",
        Primitive::Int64 => "i64",
        Primitive::UInt8 => "u8",
        Primitive::UInt16 => "u16",
        Primitive::UInt32 => "u32",
        Primitive::UInt64 => "u64",
        Primitive::Bool => "bool",
        Primitive::Unit => "()",
        _ => return None,
    })
}

fn is_integer(p: Primitive) -> bool {
    matches!(
        p,
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

fn strip_int_suffix(text: &str) -> &str {
    for suf in ["i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"] {
        if let Some(base) = text.strip_suffix(suf) {
            return base;
        }
    }
    text
}

fn item_kind_name(k: &ItemKind) -> &'static str {
    match k {
        ItemKind::Fn(_) => "fn",
        ItemKind::Struct { .. } => "struct",
        ItemKind::Enum { .. } => "enum",
        ItemKind::Trait { .. } => "trait",
        ItemKind::Impl { .. } => "impl",
        ItemKind::Const { .. } => "const",
        ItemKind::TypeAlias { .. } => "type-alias",
        ItemKind::Use(_) => "use",
        ItemKind::Mod { .. } => "mod",
        ItemKind::Model(_) => "model",
    }
}

/// Emitted verbatim at the top of every generated program. Integer arithmetic is routed through
/// `StarkChecked` so overflow and division-by-zero *trap* (exit 101) in every build profile,
/// matching STARK's abort-without-unwind semantics — the generated binary never relies on Rust's
/// profile-dependent overflow behaviour.
const RUNTIME_PRELUDE: &str = r#"// GENERATED by the WP-C3.2 generated-Rust spike. Disposable.
#![allow(dead_code, unused_parens, unused_mut, clippy::all)]

fn stark_trap(msg: &str) -> ! {
    eprintln!("trap: {msg}");
    std::process::exit(101);
}

fn stark_println<T: std::fmt::Display>(x: T) { println!("{x}"); }
fn stark_print<T: std::fmt::Display>(x: T) { print!("{x}"); }

trait StarkChecked: Copy {
    fn s_add(self, o: Self) -> Self;
    fn s_sub(self, o: Self) -> Self;
    fn s_mul(self, o: Self) -> Self;
    fn s_div(self, o: Self) -> Self;
    fn s_rem(self, o: Self) -> Self;
    fn s_neg(self) -> Self;
}

macro_rules! impl_checked {
    ($t:ty) => {
        impl StarkChecked for $t {
            fn s_add(self, o: Self) -> Self { self.checked_add(o).unwrap_or_else(|| stark_trap("integer overflow")) }
            fn s_sub(self, o: Self) -> Self { self.checked_sub(o).unwrap_or_else(|| stark_trap("integer overflow")) }
            fn s_mul(self, o: Self) -> Self { self.checked_mul(o).unwrap_or_else(|| stark_trap("integer overflow")) }
            fn s_div(self, o: Self) -> Self { self.checked_div(o).unwrap_or_else(|| stark_trap("divide by zero")) }
            fn s_rem(self, o: Self) -> Self { self.checked_rem(o).unwrap_or_else(|| stark_trap("divide by zero")) }
            fn s_neg(self) -> Self { (0 as $t).checked_sub(self).unwrap_or_else(|| stark_trap("integer overflow")) }
        }
    };
}
impl_checked!(i8); impl_checked!(i16); impl_checked!(i32); impl_checked!(i64);
impl_checked!(u8); impl_checked!(u16); impl_checked!(u32); impl_checked!(u64);

"#;

// ------------------------------------------------------------------ harness --

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/exec_snapshots")
}

/// Every top-level frozen corpus case (workload items 1-10). The spike attempts each; the ones
/// it cannot lower are reported, not failures.
const CORPUS_CASES: &[&str] = &[
    "expr_stmt__01_arithmetic_and_precedence",
    "expr_stmt__02_if_else_and_block_tail",
    "expr_stmt__03_loops_break_continue",
    "expr_stmt__04_match_and_patterns",
    "primitive__01_integer_widths_and_overflow_traps",
    "primitive__02_integer_overflow_traps",
    "primitive__03_float_arithmetic_and_casts",
    "struct_enum_trait__01_struct_construction_and_methods",
    "struct_enum_trait__02_enum_and_pattern_match",
    "struct_enum_trait__03_generic_function_and_trait_bound",
    "struct_enum_trait__04_trait_default_and_override",
    "ownership_drop__01_move_and_drop_order",
    "ownership_drop__02_shared_borrow_does_not_move",
    "option_result__01_option_construction_and_match",
    "option_result__02_result_and_try_propagation",
    "collection_iter__01_vec_push_index_iterate",
    "collection_iter__02_hashmap_insert_get_iteration_order",
];

struct Front {
    hir: Hir,
    file: Arc<SourceFile>,
    tables: TypeTables,
}

fn front_end(name: &str) -> Front {
    let path = corpus_dir().join(format!("{name}.stark"));
    let source = std::fs::read_to_string(&path).unwrap();
    let file = Arc::new(SourceFile::new(path.to_string_lossy().into_owned(), source));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{name}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{name}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    let errs: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .collect();
    assert!(errs.is_empty(), "{name}: typecheck: {errs:?}");
    Front {
        hir,
        file,
        tables: checked.tables,
    }
}

fn lower(front: &Front) -> Result<String, String> {
    let mut l = Lowerer {
        hir: &front.hir,
        src: &front.file.src,
        tables: &front.tables,
        out: String::new(),
    };
    match l.lower_program() {
        Ok(()) => Ok(l.out),
        Err(Unsupported(why)) => Err(why),
    }
}

fn rustc_available() -> bool {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

struct RanBinary {
    stdout: String,
    exit_code: Option<i32>,
    compile_ms: u128,
}

fn compile_and_run(rust_src: &str, tag: &str) -> RanBinary {
    let dir = std::env::temp_dir().join(format!("stark_genrust_spike_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let src_path = dir.join("main.rs");
    let bin_path = dir.join("main_bin");
    std::fs::write(&src_path, rust_src).unwrap();

    let t0 = std::time::Instant::now();
    let compile = std::process::Command::new("rustc")
        .arg(&src_path)
        .arg("-o")
        .arg(&bin_path)
        .output()
        .expect("rustc invocation failed");
    let compile_ms = t0.elapsed().as_millis();
    assert!(
        compile.status.success(),
        "generated Rust failed to compile ({tag}):\nSTDERR:\n{}\n--- SOURCE ---\n{rust_src}",
        String::from_utf8_lossy(&compile.stderr)
    );

    let run = std::process::Command::new(&bin_path)
        .output()
        .expect("running generated binary failed");
    let _ = std::fs::remove_dir_all(&dir);
    RanBinary {
        stdout: String::from_utf8_lossy(&run.stdout).into_owned(),
        exit_code: run.status.code(),
        compile_ms,
    }
}

/// The core spike result: for every frozen corpus case, either the generated binary matches the
/// interpreter oracle, or the case is cleanly reported as unsupported. A supported case that
/// mismatches is a hard failure (a semantic-parity defect).
#[test]
fn genrust_spike_matches_interpreter_on_frozen_corpus() {
    if !rustc_available() {
        eprintln!(
            "SKIP (MANUAL evidence class): rustc not available in this environment; \
             the WP-C3.2 spike's compile+run leg cannot be exercised here."
        );
        return;
    }

    let mut supported = Vec::new();
    let mut unsupported = Vec::new();
    let mut total_compile_ms = 0u128;

    for &name in CORPUS_CASES {
        let front = front_end(name);
        let oracle = interp::run(&front.hir, front.file.clone(), &front.tables);

        match lower(&front) {
            Err(why) => unsupported.push((name, why)),
            Ok(rust_src) => {
                let ran = compile_and_run(&rust_src, name);
                total_compile_ms += ran.compile_ms;
                match oracle {
                    Ok(exec) => {
                        assert_eq!(
                            ran.exit_code,
                            Some(0),
                            "{name}: interpreter completed (status {}) but generated binary \
                             exited {:?}",
                            exec.status,
                            ran.exit_code
                        );
                        assert_eq!(
                            ran.stdout, exec.output,
                            "{name}: stdout mismatch (semantic-parity defect)\n\
                             interpreter:\n{}\ngenerated:\n{}",
                            exec.output, ran.stdout
                        );
                        supported.push((name, ran.compile_ms, "match/normal"));
                    }
                    Err(ref trap) => {
                        assert!(
                            trap.is_trap,
                            "{name}: oracle errored but not a trap: {trap:?}"
                        );
                        assert_ne!(
                            ran.exit_code,
                            Some(0),
                            "{name}: interpreter trapped but generated binary exited 0 \
                             (missed trap — a soundness-relevant parity defect)"
                        );
                        supported.push((name, ran.compile_ms, "match/trap"));
                    }
                }
            }
        }
    }

    eprintln!("\n===== WP-C3.2 generated-Rust spike: frozen-corpus coverage =====");
    eprintln!(
        "supported & matched: {}/{}",
        supported.len(),
        CORPUS_CASES.len()
    );
    for (name, ms, kind) in &supported {
        eprintln!("  OK   {name}  ({kind}, rustc {ms} ms)");
    }
    eprintln!(
        "unsupported (recorded, not failures): {}",
        unsupported.len()
    );
    for (name, why) in &unsupported {
        eprintln!("  --   {name}  [{why}]");
    }
    if !supported.is_empty() {
        eprintln!(
            "mean rustc compile time over supported cases: {} ms",
            total_compile_ms / supported.len() as u128
        );
    }
    eprintln!("================================================================\n");

    // The spike is only meaningful if it actually lowered and matched a non-trivial slice of the
    // frozen workload. Lock in the constructs proven to lower cleanly against the interpreter.
    let matched: std::collections::HashSet<&str> = supported.iter().map(|(n, _, _)| *n).collect();
    for required in [
        "expr_stmt__01_arithmetic_and_precedence", // items 1 (arithmetic/precedence)
        "expr_stmt__03_loops_break_continue",      // item 2 (loops/calls/for/break/continue)
        "primitive__02_integer_overflow_traps",    // item 8 (trap -> abort parity)
    ] {
        assert!(
            matched.contains(required),
            "spike regression: expected to lower and match `{required}`, but it was not in the \
             supported-and-matched set"
        );
    }
}

/// Documents, as an executable assertion, that constructs outside the spike subset are reported
/// cleanly (returned as `Unsupported`) rather than silently mislowered — the "record unsupported
/// constructs" deliverable. Does not require rustc.
#[test]
fn genrust_spike_reports_unsupported_constructs_cleanly() {
    // Cases still outside the (breadth-extended) subset: they must report unsupported, not emit
    // possibly-wrong code. (Structs, generics, Option/match, and String became *supported* by the
    // breadth extension and are covered by the match-the-interpreter test instead.)
    let cases = [
        ("collection_iter__01_vec_push_index_iterate", "Vec"),
        (
            "collection_iter__02_hashmap_insert_get_iteration_order",
            "HashMap",
        ),
        ("primitive__03_float_arithmetic_and_casts", "Float64"),
        ("option_result__02_result_and_try_propagation", "? operator"),
    ];
    for (name, note) in cases {
        let front = front_end(name);
        let result = lower(&front);
        assert!(
            result.is_err(),
            "{name} ({note}) unexpectedly lowered — the spike must report it unsupported, not \
             emit (possibly wrong) code for an out-of-subset construct"
        );
    }
}
