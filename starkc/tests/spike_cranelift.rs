//! WP-C3.3 — Direct (Cranelift) backend spike. **ISOLATED, DISPOSABLE (charter §2.2).**
//!
//! The direct-backend counterpart to the WP-C3.2 generated-Rust spike, over the same frozen
//! workload subset, for a like-for-like comparison feeding WP-C3.4's backend selection (CE5).
//! Confined to this integration-test file: NOT wired into `stark build`, not the compiler's code
//! generator. It consumes programs already validated by the real front end (parse, resolve,
//! type/borrow check — no front-end check is bypassed) and lowers a supported subset of typed HIR
//! to Cranelift IR, emits a native object file (no JIT, so no `unsafe` — the crate forbids it),
//! links it with the host `cc` against a tiny C runtime into a standalone executable, runs it, and
//! compares stdout + exit status against the reference interpreter (the semantic oracle) over the
//! frozen `exec_snapshots` corpus (`corpus_version = 1.0.0`).
//!
//! Supported subset: signed integer primitives (Int8/16/32/64) and Bool; arithmetic with STARK
//! trap semantics (add/sub/mul overflow via `trapnz`; div/rem by zero and `INT_MIN/-1` via
//! Cranelift's trapping `sdiv`/`srem`); unary neg and bool not; comparisons; `let`/`let mut` and
//! assignment; `if`/`else`, `while`, `loop`+bare `break`, `continue`, `for x in a..b`; block-tail
//! values; non-generic receiverless functions and calls; `print`/`println` of an integer or bool.
//! Everything else is reported unsupported (recorded, not mislowered).

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{
    types, AbiParam, Block, InstBuilder, Signature, TrapCode, Type, Value,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::{isa, Context};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_module::{default_libcall_names, FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
use starkc::ast::{BinOp, Lit, Primitive, UnOp};
use starkc::hir::{Builtin, ExprId, ExprKind, FnDef, Hir, ItemId, ItemKind, Res, StmtKind};
use starkc::interp;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::{SourceFile, Span};
use starkc::typecheck::{self, Ty, TypeTables};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use target_lexicon::Triple;

struct Unsupported(String);
fn unsup(m: impl Into<String>) -> Unsupported {
    Unsupported(m.into())
}

/// STARK primitive -> (Cranelift type, is_signed_integer). `Bool` maps to I8 (0/1).
fn clif_prim(p: Primitive) -> Option<(Type, bool)> {
    Some(match p {
        Primitive::Int8 => (types::I8, true),
        Primitive::Int16 => (types::I16, true),
        Primitive::Int32 => (types::I32, true),
        Primitive::Int64 => (types::I64, true),
        Primitive::Bool => (types::I8, false),
        _ => return None,
    })
}

// ------------------------------------------------------------------- builder --

struct ClifSpike<'a> {
    hir: &'a Hir,
    src: &'a str,
    tables: &'a TypeTables,
    module: ObjectModule,
    fns: HashMap<u32, FuncId>,
    println_i64: FuncId,
    println_bool: FuncId,
}

/// Per-function lowering state.
struct FnState {
    /// STARK LocalId.0 -> Cranelift variable.
    vars: HashMap<u32, (Variable, Type)>,
    next_var: usize,
    /// (continue-target, break-target) stack for the enclosing loops.
    loops: Vec<(Block, Block)>,
}

impl<'a> ClifSpike<'a> {
    fn text(&self, span: Span) -> &'a str {
        &self.src[span.lo as usize..span.hi as usize]
    }

    fn expr_prim(&self, id: ExprId) -> Result<Primitive, Unsupported> {
        match self.tables.expr_types.get(&id) {
            Some(Ty::Primitive(p)) => Ok(*p),
            Some(_) => Err(unsup("non-primitive expression type")),
            None => Err(unsup("expression has no recorded type")),
        }
    }

    fn expr_clif_ty(&self, id: ExprId) -> Result<Type, Unsupported> {
        let p = self.expr_prim(id)?;
        clif_prim(p)
            .map(|(t, _)| t)
            .ok_or_else(|| unsup(format!("primitive {p:?}")))
    }

    /// Declare (but do not define) every top-level function so calls resolve in any order.
    fn declare_all(&mut self) -> Result<Vec<(ItemId, FuncId)>, Unsupported> {
        let items = match &self.hir.root {
            starkc::hir::Root::Program(items) => items.clone(),
            _ => return Err(unsup("non-program root")),
        };
        let mut defined = Vec::new();
        for item_id in items {
            match &self.hir.item(item_id).kind {
                ItemKind::Fn(def) => {
                    if !def.sig.generics.is_empty() {
                        return Err(unsup("generic function"));
                    }
                    if def.sig.receiver.is_some() {
                        return Err(unsup("method (self receiver)"));
                    }
                    let sig = self.build_signature(def)?;
                    let raw = self.text(def.sig.name);
                    let sym = if raw == "main" { "stark_main" } else { raw };
                    let fid = self
                        .module
                        .declare_function(sym, Linkage::Export, &sig)
                        .map_err(|e| unsup(format!("declare_function: {e}")))?;
                    self.fns.insert(item_id.0, fid);
                    defined.push((item_id, fid));
                }
                ItemKind::Use(_) | ItemKind::Mod { .. } => {}
                _ => return Err(unsup("non-function top-level item")),
            }
        }
        Ok(defined)
    }

    fn build_signature(&mut self, def: &FnDef) -> Result<Signature, Unsupported> {
        let mut sig = self.module.make_signature();
        for p in &def.sig.params {
            let (t, _) = self.clif_ty_of(p.ty)?;
            sig.params.push(AbiParam::new(t));
        }
        match &def.sig.ret {
            starkc::hir::RetTy::Unit => {}
            starkc::hir::RetTy::Ty(t) => {
                let (ct, _) = self.clif_ty_of(*t)?;
                sig.returns.push(AbiParam::new(ct));
            }
            starkc::hir::RetTy::Never(_) => return Err(unsup("never-returning function")),
        }
        Ok(sig)
    }

    fn clif_ty_of(&self, ty_id: starkc::hir::TypeId) -> Result<(Type, bool), Unsupported> {
        match &self.hir.ty(ty_id).kind {
            starkc::hir::TypeKind::Primitive(p) => {
                clif_prim(*p).ok_or_else(|| unsup(format!("primitive {p:?}")))
            }
            _ => Err(unsup("non-primitive parameter/return type")),
        }
    }

    fn define_fn(&mut self, def: &FnDef, fid: FuncId) -> Result<(), Unsupported> {
        let sig = self.build_signature(def)?;
        let mut ctx = Context::new();
        ctx.func.signature = sig;
        let mut fctx = FunctionBuilderContext::new();
        let mut st = FnState {
            vars: HashMap::new(),
            next_var: 0,
            loops: Vec::new(),
        };

        // Everything that touches the builder is scoped so the &mut ctx.func borrow ends before
        // define_function.
        {
            let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
            let entry = b.create_block();
            b.append_block_params_for_function_params(entry);
            b.switch_to_block(entry);
            for (i, p) in def.sig.params.iter().enumerate() {
                let (t, _) = self.clif_ty_of(p.ty)?;
                let var = Variable::from_u32(st.next_var as u32);
                st.next_var += 1;
                b.declare_var(var, t);
                let val = b.block_params(entry)[i];
                b.def_var(var, val);
                st.vars.insert(p.local.0, (var, t));
            }

            let (tail, falls) = self.lower_block(&mut b, &mut st, def.body)?;
            if falls {
                match &def.sig.ret {
                    starkc::hir::RetTy::Unit => {
                        b.ins().return_(&[]);
                    }
                    starkc::hir::RetTy::Ty(_) => {
                        let v =
                            tail.ok_or_else(|| unsup("function body missing a return value"))?;
                        b.ins().return_(&[v]);
                    }
                    starkc::hir::RetTy::Never(_) => unreachable!(),
                }
            }
            b.seal_all_blocks();
            b.finalize();
        }

        self.module
            .define_function(fid, &mut ctx)
            .map_err(|e| unsup(format!("define_function: {e}")))?;
        Ok(())
    }

    /// Lower a block. Returns `(tail value if any, falls_through)`.
    fn lower_block(
        &mut self,
        b: &mut FunctionBuilder,
        st: &mut FnState,
        block_id: starkc::hir::BlockId,
    ) -> Result<(Option<Value>, bool), Unsupported> {
        let block = self.hir.block(block_id);
        for &sid in &block.stmts {
            let falls = self.lower_stmt(b, st, sid)?;
            if !falls {
                return Ok((None, false));
            }
        }
        match block.tail {
            // A Unit-typed tail (if-without-else, while/loop/for, a unit-returning call, an
            // assignment) is a statement that happens to sit in tail position — lower it as one,
            // yielding no block value. Only a genuinely value-typed tail goes through lower_value.
            Some(e) if self.is_unit_expr(e) => {
                let falls = self.lower_stmt_expr(b, st, e)?;
                Ok((None, falls))
            }
            Some(e) => Ok((Some(self.lower_value(b, st, e)?), true)),
            None => Ok((None, true)),
        }
    }

    fn is_unit_expr(&self, e: ExprId) -> bool {
        matches!(
            self.tables.expr_types.get(&e),
            Some(Ty::Primitive(Primitive::Unit))
        )
    }

    /// Lower a statement. Returns `falls_through` (false = the current block was terminated).
    fn lower_stmt(
        &mut self,
        b: &mut FunctionBuilder,
        st: &mut FnState,
        sid: starkc::hir::StmtId,
    ) -> Result<bool, Unsupported> {
        match &self.hir.stmt(sid).kind {
            StmtKind::Empty => Ok(true),
            StmtKind::Expr { expr, .. } => self.lower_stmt_expr(b, st, *expr),
            StmtKind::Let {
                mutable: _,
                name,
                local,
                ty,
                init,
            } => {
                // Determine the Cranelift type from the annotation if present, else from the init.
                let clif_t = if let Some(t) = ty {
                    self.clif_ty_of(*t)?.0
                } else if let Some(e) = init {
                    self.expr_clif_ty(*e)?
                } else {
                    return Err(unsup(format!(
                        "let '{}' without a type annotation or initializer",
                        self.text(*name)
                    )));
                };
                let var = Variable::from_u32(st.next_var as u32);
                st.next_var += 1;
                b.declare_var(var, clif_t);
                st.vars.insert(local.0, (var, clif_t));
                if let Some(e) = init {
                    let v = self.lower_value(b, st, *e)?;
                    b.def_var(var, v);
                }
                Ok(true)
            }
            StmtKind::Return(opt) => {
                match opt {
                    Some(e) => {
                        let v = self.lower_value(b, st, *e)?;
                        b.ins().return_(&[v]);
                    }
                    None => {
                        b.ins().return_(&[]);
                    }
                }
                Ok(false)
            }
            StmtKind::Break(None) => {
                let (_, brk) = *st
                    .loops
                    .last()
                    .ok_or_else(|| unsup("break outside a loop"))?;
                b.ins().jump(brk, &[]);
                Ok(false)
            }
            StmtKind::Break(Some(_)) => Err(unsup("break with value")),
            StmtKind::Continue => {
                let (cont, _) = *st
                    .loops
                    .last()
                    .ok_or_else(|| unsup("continue outside a loop"))?;
                b.ins().jump(cont, &[]);
                Ok(false)
            }
            StmtKind::Item(_) => Err(unsup("nested item")),
            StmtKind::Error => Err(unsup("error statement")),
        }
    }

    /// Lower an expression used as a statement (unit result). Returns `falls_through`.
    fn lower_stmt_expr(
        &mut self,
        b: &mut FunctionBuilder,
        st: &mut FnState,
        id: ExprId,
    ) -> Result<bool, Unsupported> {
        match &self.hir.expr(id).kind {
            ExprKind::Assign { op, lhs, rhs } => {
                use starkc::ast::AssignOp::*;
                let Res::Local(local) = self.path_res(*lhs)? else {
                    return Err(unsup("assignment to a non-local place"));
                };
                let (var, _ty) = *st
                    .vars
                    .get(&local.0)
                    .ok_or_else(|| unsup("assignment to unknown local"))?;
                let rhs_v = self.lower_value(b, st, *rhs)?;
                let new = match op {
                    Assign => rhs_v,
                    AddAssign | SubAssign | MulAssign | DivAssign | RemAssign => {
                        let cur = b.use_var(var);
                        let binop = match op {
                            AddAssign => BinOp::Add,
                            SubAssign => BinOp::Sub,
                            MulAssign => BinOp::Mul,
                            DivAssign => BinOp::Div,
                            RemAssign => BinOp::Rem,
                            _ => unreachable!(),
                        };
                        let prim = self.expr_prim(*lhs)?;
                        self.emit_arith(b, binop, cur, rhs_v, prim)?
                    }
                    _ => return Err(unsup("compound bit/pow assignment")),
                };
                b.def_var(var, new);
                Ok(true)
            }
            ExprKind::If {
                cond,
                then_block,
                else_,
            } => {
                let c = self.lower_value(b, st, *cond)?;
                let then_b = b.create_block();
                let merge = b.create_block();
                let else_b = if else_.is_some() {
                    b.create_block()
                } else {
                    merge
                };
                b.ins().brif(c, then_b, &[], else_b, &[]);

                b.switch_to_block(then_b);
                let (_, tf) = self.lower_block(b, st, *then_block)?;
                if tf {
                    b.ins().jump(merge, &[]);
                }

                if let Some(else_e) = else_ {
                    b.switch_to_block(else_b);
                    let ef = self.lower_stmt_expr(b, st, *else_e)?;
                    if ef {
                        b.ins().jump(merge, &[]);
                    }
                }
                b.switch_to_block(merge);
                Ok(true)
            }
            ExprKind::While { cond, body } => {
                let header = b.create_block();
                let body_b = b.create_block();
                let exit = b.create_block();
                b.ins().jump(header, &[]);
                b.switch_to_block(header);
                let c = self.lower_value(b, st, *cond)?;
                b.ins().brif(c, body_b, &[], exit, &[]);
                b.switch_to_block(body_b);
                st.loops.push((header, exit));
                let (_, bf) = self.lower_block(b, st, *body)?;
                st.loops.pop();
                if bf {
                    b.ins().jump(header, &[]);
                }
                b.switch_to_block(exit);
                Ok(true)
            }
            ExprKind::Loop { body } => {
                let body_b = b.create_block();
                let exit = b.create_block();
                b.ins().jump(body_b, &[]);
                b.switch_to_block(body_b);
                st.loops.push((body_b, exit));
                let (_, bf) = self.lower_block(b, st, *body)?;
                st.loops.pop();
                if bf {
                    b.ins().jump(body_b, &[]);
                }
                b.switch_to_block(exit);
                Ok(true)
            }
            ExprKind::For {
                var,
                local,
                iter,
                body,
            } => {
                let (lo, hi, incl) = match &self.hir.expr(*iter).kind {
                    ExprKind::Range { lo, hi, inclusive } => (*lo, *hi, *inclusive),
                    _ => return Err(unsup("for over a non-range iterator")),
                };
                let ty = self.expr_clif_ty(lo)?;
                let lo_v = self.lower_value(b, st, lo)?;
                let hi_v = self.lower_value(b, st, hi)?;
                let iv = Variable::from_u32(st.next_var as u32);
                st.next_var += 1;
                b.declare_var(iv, ty);
                b.def_var(iv, lo_v);
                st.vars.insert(local.0, (iv, ty));
                let _ = var;

                let header = b.create_block();
                let body_b = b.create_block();
                let latch = b.create_block();
                let exit = b.create_block();
                b.ins().jump(header, &[]);
                b.switch_to_block(header);
                let cur = b.use_var(iv);
                let cc = if incl {
                    IntCC::SignedLessThanOrEqual
                } else {
                    IntCC::SignedLessThan
                };
                let c = b.ins().icmp(cc, cur, hi_v);
                b.ins().brif(c, body_b, &[], exit, &[]);
                b.switch_to_block(body_b);
                st.loops.push((latch, exit));
                let (_, bf) = self.lower_block(b, st, *body)?;
                st.loops.pop();
                if bf {
                    b.ins().jump(latch, &[]);
                }
                b.switch_to_block(latch);
                let cur2 = b.use_var(iv);
                let one = b.ins().iconst(ty, 1);
                let nxt = b.ins().iadd(cur2, one);
                b.def_var(iv, nxt);
                b.ins().jump(header, &[]);
                b.switch_to_block(exit);
                Ok(true)
            }
            ExprKind::Block(bl) => {
                let (_, f) = self.lower_block(b, st, *bl)?;
                Ok(f)
            }
            // A call in statement position (e.g. println, or a void user fn).
            ExprKind::Call { .. } => {
                self.lower_call(b, st, id)?;
                Ok(true)
            }
            _ => Err(unsup("unsupported expression in statement position")),
        }
    }

    /// Lower an expression that must produce a value.
    fn lower_value(
        &mut self,
        b: &mut FunctionBuilder,
        st: &mut FnState,
        id: ExprId,
    ) -> Result<Value, Unsupported> {
        match &self.hir.expr(id).kind {
            ExprKind::Lit(lit) => self.lower_lit(b, id, lit),
            ExprKind::Path { res, .. } => match res {
                Res::Local(local) | Res::SelfValue(local) => {
                    let (var, _) = *st
                        .vars
                        .get(&local.0)
                        .ok_or_else(|| unsup("use of unknown local"))?;
                    Ok(b.use_var(var))
                }
                _ => Err(unsup("path in value position is not a local")),
            },
            ExprKind::Unary { op, operand } => {
                let v = self.lower_value(b, st, *operand)?;
                match op {
                    UnOp::Neg => Ok(b.ins().ineg(v)),
                    UnOp::Not => {
                        // bool: I8 0/1 -> (x == 0)
                        Ok(b.ins().icmp_imm(IntCC::Equal, v, 0))
                    }
                    _ => Err(unsup("unsupported unary operator")),
                }
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let l = self.lower_value(b, st, *lhs)?;
                let r = self.lower_value(b, st, *rhs)?;
                use BinOp::*;
                match op {
                    Add | Sub | Mul | Div | Rem => {
                        let prim = self.expr_prim(*lhs)?;
                        self.emit_arith(b, *op, l, r, prim)
                    }
                    Eq | Ne | Lt | Le | Gt | Ge => {
                        let cc = match op {
                            Eq => IntCC::Equal,
                            Ne => IntCC::NotEqual,
                            Lt => IntCC::SignedLessThan,
                            Le => IntCC::SignedLessThanOrEqual,
                            Gt => IntCC::SignedGreaterThan,
                            Ge => IntCC::SignedGreaterThanOrEqual,
                            _ => unreachable!(),
                        };
                        Ok(b.ins().icmp(cc, l, r))
                    }
                    _ => Err(unsup("unsupported binary operator (logic/bitwise/pow)")),
                }
            }
            ExprKind::Call { .. } => self
                .lower_call(b, st, id)?
                .ok_or_else(|| unsup("call used as a value returns unit")),
            ExprKind::If {
                cond,
                then_block,
                else_,
            } => {
                // if-as-value: both arms must yield a value of the same clif type.
                let else_e = else_.ok_or_else(|| unsup("if-as-value without else"))?;
                let ty = self.expr_clif_ty(id)?;
                let c = self.lower_value(b, st, *cond)?;
                let then_b = b.create_block();
                let else_b = b.create_block();
                let merge = b.create_block();
                b.append_block_param(merge, ty);
                b.ins().brif(c, then_b, &[], else_b, &[]);
                b.switch_to_block(then_b);
                let (tv, tf) = self.lower_block(b, st, *then_block)?;
                if tf {
                    let v = tv.ok_or_else(|| unsup("if-then arm yielded no value"))?;
                    b.ins().jump(merge, &[v]);
                }
                b.switch_to_block(else_b);
                let ev = self.lower_value(b, st, else_e)?;
                b.ins().jump(merge, &[ev]);
                b.switch_to_block(merge);
                Ok(b.block_params(merge)[0])
            }
            ExprKind::Block(bl) => {
                let (tv, _) = self.lower_block(b, st, *bl)?;
                tv.ok_or_else(|| unsup("block in value position yielded no value"))
            }
            _ => Err(unsup("unsupported value expression")),
        }
    }

    fn emit_arith(
        &self,
        b: &mut FunctionBuilder,
        op: BinOp,
        l: Value,
        r: Value,
        prim: Primitive,
    ) -> Result<Value, Unsupported> {
        let (ty, _signed) = clif_prim(prim).ok_or_else(|| unsup("arith on non-int"))?;
        let bits = ty.bits() as i64;
        match op {
            BinOp::Add => {
                let res = b.ins().iadd(l, r);
                // signed overflow: (res^l) & (res^r) has the sign bit set
                let x1 = b.ins().bxor(res, l);
                let x2 = b.ins().bxor(res, r);
                let a = b.ins().band(x1, x2);
                let ovf = b.ins().icmp_imm(IntCC::SignedLessThan, a, 0);
                b.ins().trapnz(ovf, TrapCode::IntegerOverflow);
                Ok(res)
            }
            BinOp::Sub => {
                let res = b.ins().isub(l, r);
                // signed overflow: (l^r) & (l^res) sign bit set
                let x1 = b.ins().bxor(l, r);
                let x2 = b.ins().bxor(l, res);
                let a = b.ins().band(x1, x2);
                let ovf = b.ins().icmp_imm(IntCC::SignedLessThan, a, 0);
                b.ins().trapnz(ovf, TrapCode::IntegerOverflow);
                Ok(res)
            }
            BinOp::Mul => {
                // widen to double width, multiply, check the result fits the narrow signed range.
                let wide = match bits {
                    8 => types::I16,
                    16 => types::I32,
                    32 => types::I64,
                    64 => types::I128,
                    _ => return Err(unsup("mul width")),
                };
                let lw = b.ins().sextend(wide, l);
                let rw = b.ins().sextend(wide, r);
                let prod = b.ins().imul(lw, rw);
                let narrowed = b.ins().ireduce(ty, prod);
                let back = b.ins().sextend(wide, narrowed);
                let ne = b.ins().icmp(IntCC::NotEqual, back, prod);
                b.ins().trapnz(ne, TrapCode::IntegerOverflow);
                Ok(narrowed)
            }
            // sdiv/srem trap on divide-by-zero and INT_MIN/-1 automatically.
            BinOp::Div => Ok(b.ins().sdiv(l, r)),
            BinOp::Rem => Ok(b.ins().srem(l, r)),
            _ => Err(unsup("non-arithmetic op in emit_arith")),
        }
    }

    fn lower_lit(
        &self,
        b: &mut FunctionBuilder,
        id: ExprId,
        lit: &Lit,
    ) -> Result<Value, Unsupported> {
        match lit {
            Lit::Bool(v) => Ok(b.ins().iconst(types::I8, if *v { 1 } else { 0 })),
            Lit::Int { .. } => {
                let ty = self.expr_clif_ty(id)?;
                let raw = self.text(self.hir.expr(id).span);
                let digits: String = raw
                    .chars()
                    .take_while(|c| {
                        c.is_ascii_digit() || *c == 'x' || c.is_ascii_hexdigit() || *c == '_'
                    })
                    .filter(|c| *c != '_')
                    .collect();
                let val: i64 = if let Some(hex) = digits.strip_prefix("0x") {
                    i64::from_str_radix(hex, 16).map_err(|_| unsup("hex literal parse"))?
                } else {
                    digits.parse().map_err(|_| unsup("int literal parse"))?
                };
                Ok(b.ins().iconst(ty, val))
            }
            _ => Err(unsup("non-integer/bool literal")),
        }
    }

    fn path_res(&self, id: ExprId) -> Result<Res, Unsupported> {
        match &self.hir.expr(id).kind {
            ExprKind::Path { res, .. } => Ok(*res),
            _ => Err(unsup("expected a path")),
        }
    }

    /// Lower a call. Returns `Some(value)` for a value-returning call, `None` for unit.
    fn lower_call(
        &mut self,
        b: &mut FunctionBuilder,
        st: &mut FnState,
        id: ExprId,
    ) -> Result<Option<Value>, Unsupported> {
        let ExprKind::Call { callee, args } = &self.hir.expr(id).kind else {
            return Err(unsup("not a call"));
        };
        let callee = *callee;
        let args = args.clone();
        let ExprKind::Path { res, .. } = &self.hir.expr(callee).kind else {
            return Err(unsup("indirect/non-path call"));
        };
        match res {
            Res::Builtin(Builtin::Println | Builtin::Print) => {
                if args.len() != 1 {
                    return Err(unsup("print/println arity"));
                }
                let prim = self.expr_prim(args[0])?;
                let v = self.lower_value(b, st, args[0])?;
                if prim == Primitive::Bool {
                    let fref = self.module.declare_func_in_func(self.println_bool, b.func);
                    b.ins().call(fref, &[v]);
                } else if clif_prim(prim).map(|(_, s)| s).unwrap_or(false) {
                    // sign-extend to i64 for the runtime printer
                    let ext = if self.expr_clif_ty(args[0])? == types::I64 {
                        v
                    } else {
                        b.ins().sextend(types::I64, v)
                    };
                    let fref = self.module.declare_func_in_func(self.println_i64, b.func);
                    b.ins().call(fref, &[ext]);
                } else {
                    return Err(unsup("print/println of an unsupported type"));
                }
                Ok(None)
            }
            Res::Builtin(other) => Err(unsup(format!("builtin {other:?}"))),
            Res::Item(item_id) => {
                let fid = *self
                    .fns
                    .get(&item_id.0)
                    .ok_or_else(|| unsup("call to an undeclared function"))?;
                let is_unit = matches!(
                    self.hir.item(*item_id).kind,
                    ItemKind::Fn(ref d) if matches!(d.sig.ret, starkc::hir::RetTy::Unit)
                );
                let mut vals = Vec::new();
                for a in &args {
                    vals.push(self.lower_value(b, st, *a)?);
                }
                let fref = self.module.declare_func_in_func(fid, b.func);
                let call = b.ins().call(fref, &vals);
                if is_unit {
                    Ok(None)
                } else {
                    Ok(Some(b.inst_results(call)[0]))
                }
            }
            _ => Err(unsup("unsupported callee")),
        }
    }
}

/// Compile a whole program to object bytes, or report the first unsupported construct.
fn compile_program(front: &Front) -> Result<Vec<u8>, String> {
    let mut fb = settings::builder();
    fb.set("is_pic", "true").unwrap();
    let flags = settings::Flags::new(fb);
    let isa = isa::lookup(Triple::host())
        .map_err(|e| format!("isa lookup: {e}"))?
        .finish(flags)
        .map_err(|e| format!("isa finish: {e}"))?;
    let obj = ObjectBuilder::new(isa, "stark_spike", default_libcall_names())
        .map_err(|e| format!("object builder: {e}"))?;
    let mut module = ObjectModule::new(obj);

    // Runtime imports.
    let mut psig_i64 = module.make_signature();
    psig_i64.params.push(AbiParam::new(types::I64));
    let println_i64 = module
        .declare_function("stark_println_i64", Linkage::Import, &psig_i64)
        .unwrap();
    let mut psig_b = module.make_signature();
    psig_b.params.push(AbiParam::new(types::I8));
    let println_bool = module
        .declare_function("stark_println_bool", Linkage::Import, &psig_b)
        .unwrap();

    let mut spike = ClifSpike {
        hir: &front.hir,
        src: &front.file.src,
        tables: &front.tables,
        module,
        fns: HashMap::new(),
        println_i64,
        println_bool,
    };

    let defined = spike.declare_all().map_err(|Unsupported(w)| w)?;
    for (item_id, fid) in defined {
        // `def` borrows `front.hir` (shared); `spike` only holds a *shared* reference to the same
        // Hir, so taking `&mut spike` here does not conflict — both are shared borrows of the Hir.
        let ItemKind::Fn(def) = &front.hir.item(item_id).kind else {
            continue;
        };
        spike.define_fn(def, fid).map_err(|Unsupported(w)| w)?;
    }

    let product = spike.module.finish();
    product.emit().map_err(|e| format!("emit: {e}"))
}

// ------------------------------------------------------------------ harness --

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/exec_snapshots")
}

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

const RUNTIME_C: &str = r#"#include <stdio.h>
#include <stdlib.h>
void stark_println_i64(long long x) { printf("%lld\n", x); }
void stark_println_bool(signed char x) { printf(x ? "true\n" : "false\n"); }
extern void stark_main(void);
int main(void) { stark_main(); return 0; }
"#;

fn cc_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

struct Ran {
    stdout: String,
    exit_code: Option<i32>,
    build_ms: u128,
}

fn link_and_run(obj_bytes: &[u8], tag: &str) -> Ran {
    let dir = std::env::temp_dir().join(format!("stark_clif_spike_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let obj = dir.join("prog.o");
    let cmain = dir.join("rt.c");
    let bin = dir.join("prog_bin");
    std::fs::write(&obj, obj_bytes).unwrap();
    std::fs::write(&cmain, RUNTIME_C).unwrap();

    let t0 = std::time::Instant::now();
    let link = std::process::Command::new("cc")
        .arg(&cmain)
        .arg(&obj)
        .arg("-o")
        .arg(&bin)
        .output()
        .expect("cc");
    let build_ms = t0.elapsed().as_millis();
    assert!(
        link.status.success(),
        "link failed ({tag}): {}",
        String::from_utf8_lossy(&link.stderr)
    );
    let run = std::process::Command::new(&bin).output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    Ran {
        stdout: String::from_utf8_lossy(&run.stdout).into_owned(),
        exit_code: run.status.code(),
        build_ms,
    }
}

#[test]
fn cranelift_spike_matches_interpreter_on_frozen_corpus() {
    if !cc_available() {
        eprintln!("SKIP (MANUAL): no cc linker available.");
        return;
    }

    let mut supported = Vec::new();
    let mut unsupported = Vec::new();

    let mut total_codegen_ms = 0u128;
    let mut total_link_ms = 0u128;
    for &name in CORPUS_CASES {
        let front = front_end(name);
        let oracle = interp::run(&front.hir, front.file.clone(), &front.tables);
        let t0 = std::time::Instant::now();
        let compiled = compile_program(&front);
        let codegen_ms = t0.elapsed().as_millis();
        match compiled {
            Err(why) => unsupported.push((name, why)),
            Ok(obj) => {
                let ran = link_and_run(&obj, name);
                total_codegen_ms += codegen_ms;
                total_link_ms += ran.build_ms;
                match oracle {
                    Ok(exec) => {
                        assert_eq!(
                            ran.exit_code,
                            Some(0),
                            "{name}: interpreter ok but native exited {:?}",
                            ran.exit_code
                        );
                        assert_eq!(
                            ran.stdout, exec.output,
                            "{name}: stdout mismatch\ninterp:\n{}\nnative:\n{}",
                            exec.output, ran.stdout
                        );
                        supported.push((name, ran.build_ms, "match/normal"));
                    }
                    Err(ref trap) => {
                        assert!(trap.is_trap, "{name}: oracle errored, not a trap");
                        assert_ne!(
                            ran.exit_code,
                            Some(0),
                            "{name}: interpreter trapped but native exited 0 (missed trap)"
                        );
                        supported.push((name, ran.build_ms, "match/trap"));
                    }
                }
            }
        }
    }

    eprintln!("\n===== WP-C3.3 Cranelift spike: frozen-corpus coverage =====");
    eprintln!(
        "supported & matched: {}/{}",
        supported.len(),
        CORPUS_CASES.len()
    );
    for (n, ms, k) in &supported {
        eprintln!("  OK   {n}  ({k}, link {ms} ms)");
    }
    eprintln!("unsupported (recorded): {}", unsupported.len());
    for (n, w) in &unsupported {
        eprintln!("  --   {n}  [{w}]");
    }
    if !supported.is_empty() {
        let n = supported.len() as u128;
        eprintln!(
            "mean Cranelift codegen: {} ms/case; mean cc link: {} ms/case",
            total_codegen_ms / n,
            total_link_ms / n
        );
    }
    eprintln!("===========================================================\n");

    let matched: std::collections::HashSet<&str> = supported.iter().map(|(n, _, _)| *n).collect();
    for req in [
        "expr_stmt__01_arithmetic_and_precedence",
        "expr_stmt__03_loops_break_continue",
        "primitive__02_integer_overflow_traps",
    ] {
        assert!(
            matched.contains(req),
            "spike regression: `{req}` not matched"
        );
    }
}
