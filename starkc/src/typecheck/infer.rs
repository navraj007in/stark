//! **AS7 Packet 6 — inference: substitution, unification, grounding and defaulting.**
//!
//! One level above `state`: `infer` may depend on `state` and `types`, and on nothing else inside
//! `typecheck`. This is the first extraction that moves real *behaviour* rather than data or
//! storage, so the behaviour and diagnostic suites are load-bearing here rather than
//! confirmatory — `unify` alone decides the text and the span of a large share of the checker's
//! diagnostics.
//!
//! **The tensor bridge, and only the bridge.** `unify_tensor_types` and `emit_tensor_unify_error`
//! live here because unification is where a tensor type meets a Core one. They delegate every
//! *decision* to `extensions::tensor`'s `UnifyCtx` and own only the Core-side plumbing: what to
//! unify against what, and how the failure is rendered. AS6 froze that boundary and AS7 does not
//! reopen it.

use super::state::TypeChecker;
use super::types::is_integer_primitive;
use super::types::{substitute_ty, ExtensionTy, Ty, TypeVarId};
use crate::ast::Primitive;
use crate::diag::Diagnostic;
use crate::extensions::tensor::types::{Shape, TensorKind, TensorTy, UnifyError};
use crate::hir::{self};
use crate::literal;
use crate::source::Span;
use std::collections::HashMap;

impl TypeChecker<'_> {
    pub(super) fn resolve(&self, ty: &Ty) -> Ty {
        match ty {
            Ty::Infer(id) => {
                if let Some(target) = self.subst.get(id) {
                    self.resolve(target)
                } else {
                    ty.clone()
                }
            }
            Ty::Ref { mutable, inner } => Ty::Ref {
                mutable: *mutable,
                inner: Box::new(self.resolve(inner)),
            },
            Ty::Struct(item, args) => {
                Ty::Struct(*item, args.iter().map(|arg| self.resolve(arg)).collect())
            }
            Ty::Enum(item, args) => {
                Ty::Enum(*item, args.iter().map(|arg| self.resolve(arg)).collect())
            }
            Ty::Core(core, args) => {
                Ty::Core(*core, args.iter().map(|arg| self.resolve(arg)).collect())
            }
            Ty::Tuple(elems) => Ty::Tuple(elems.iter().map(|e| self.resolve(e)).collect()),
            Ty::Array(elem, len) => Ty::Array(Box::new(self.resolve(elem)), *len),
            Ty::Slice(elem) => Ty::Slice(Box::new(self.resolve(elem))),
            Ty::Fn { params, ret } => Ty::Fn {
                params: params.iter().map(|p| self.resolve(p)).collect(),
                ret: Box::new(self.resolve(ret)),
            },
            Ty::Range(elem) => Ty::Range(Box::new(self.resolve(elem))),
            Ty::Extension(ext) => Ty::Extension(ext.clone()),
            _ => ty.clone(),
        }
    }

    /// Deep-resolve a type for publication in [`TypeTables`], additionally
    /// grounding tensor shape dimensions through the tensor unification context
    /// (e.g. a model's fresh output dim `N` bound to `1` by a `predict` call).
    /// Unlike [`Self::resolve`] this is *not* used on the unification hot path,
    /// so backend consumers see concrete shapes wherever they are determined.
    pub(super) fn ground(&self, ty: &Ty) -> Ty {
        let ty = self.resolve(ty);
        self.ground_tensor_dims(&ty)
    }

    pub(super) fn ground_tensor_dims(&self, ty: &Ty) -> Ty {
        match ty {
            Ty::Extension(ext) => match &**ext {
                ExtensionTy::Tensor(TensorKind::Tensor(t)) => {
                    let dims: Vec<_> = t
                        .shape
                        .dims
                        .iter()
                        .map(|d| self.tensor_ctx.resolve_dim(d).unwrap_or_else(|_| d.clone()))
                        .collect();
                    // Grounding preserves rank; keep spans only if they still align.
                    let spans = if t.shape.spans.len() == dims.len() {
                        t.shape.spans.clone()
                    } else {
                        Vec::new()
                    };
                    Ty::Extension(Box::new(ExtensionTy::Tensor(TensorKind::Tensor(
                        TensorTy {
                            dtype: t.dtype,
                            shape: Shape { dims, spans },
                            device: t.device,
                            range: t.range,
                        },
                    ))))
                }
                _ => ty.clone(),
            },
            Ty::Ref { mutable, inner } => Ty::Ref {
                mutable: *mutable,
                inner: Box::new(self.ground_tensor_dims(inner)),
            },
            Ty::Struct(item, args) => Ty::Struct(
                *item,
                args.iter().map(|a| self.ground_tensor_dims(a)).collect(),
            ),
            Ty::Enum(item, args) => Ty::Enum(
                *item,
                args.iter().map(|a| self.ground_tensor_dims(a)).collect(),
            ),
            Ty::Core(core, args) => Ty::Core(
                *core,
                args.iter().map(|a| self.ground_tensor_dims(a)).collect(),
            ),
            Ty::Tuple(elems) => {
                Ty::Tuple(elems.iter().map(|e| self.ground_tensor_dims(e)).collect())
            }
            Ty::Array(elem, len) => Ty::Array(Box::new(self.ground_tensor_dims(elem)), *len),
            Ty::Slice(elem) => Ty::Slice(Box::new(self.ground_tensor_dims(elem))),
            Ty::Range(elem) => Ty::Range(Box::new(self.ground_tensor_dims(elem))),
            Ty::Fn { params, ret } => Ty::Fn {
                params: params.iter().map(|p| self.ground_tensor_dims(p)).collect(),
                ret: Box::new(self.ground_tensor_dims(ret)),
            },
            _ => ty.clone(),
        }
    }

    pub(super) fn occurs_in(&self, id: TypeVarId, ty: &Ty) -> bool {
        match ty {
            Ty::Infer(other_id) => id == *other_id,
            Ty::Ref { inner, .. } => self.occurs_in(id, inner),
            Ty::Struct(_, args) | Ty::Enum(_, args) | Ty::Core(_, args) => {
                args.iter().any(|arg| self.occurs_in(id, arg))
            }
            Ty::Tuple(elems) => elems.iter().any(|e| self.occurs_in(id, e)),
            Ty::Array(elem, _) => self.occurs_in(id, elem),
            Ty::Slice(elem) => self.occurs_in(id, elem),
            Ty::Fn { params, ret } => {
                params.iter().any(|p| self.occurs_in(id, p)) || self.occurs_in(id, ret)
            }
            Ty::Range(elem) => self.occurs_in(id, elem),
            Ty::Extension(ext) => match &**ext {
                ExtensionTy::Tensor(_) | ExtensionTy::Model(_) | ExtensionTy::ModelError => false,
            },
            _ => false,
        }
    }

    /// WP-C4.7-6.3: force an integer literal's type NOW, for the places that cannot wait for the
    /// deferred defaulting pass because they must branch on a concrete type — chiefly method
    /// resolution, where `3.cmp(&5)` needs a real receiver type to find candidates. Returns the
    /// type unchanged when it is not an open integer-literal variable.
    pub(super) fn default_int_literal_now(&mut self, ty: &Ty) -> Ty {
        let resolved = self.resolve(ty);
        let Ty::Infer(id) = resolved else {
            return resolved;
        };
        let Some(&(value, _)) = self.int_literal_vars.get(&id) else {
            return resolved;
        };
        let primitive = if i32::try_from(value).is_ok() {
            Primitive::Int32
        } else {
            Primitive::Int64
        };
        let concrete = Ty::Primitive(primitive);
        self.subst.insert(id, concrete.clone());
        concrete
    }

    /// WP-C6.2b-F2: default UNCONSTRAINED integer literals anywhere inside a type, not only at the
    /// top level. Method resolution must branch on a concrete receiver, and `let w = W { v: 7 };
    /// w.get()` gives `W<_infer>` where `_infer` is the literal `7`'s variable — so a trait/inherent
    /// impl written for the specific instance `W<Int32>` never matched `W<_infer>`. Defaulting the
    /// literal (03 solving step 5, "int literals default to Int32") makes the receiver `W<Int32>`
    /// so the concrete-instance impl matches. Only literal variables are touched (`int_literal_vars`);
    /// a genuine unbound inference variable is left alone.
    pub(super) fn default_int_literals_deep(&mut self, ty: &Ty) -> Ty {
        let ty = self.default_int_literal_now(ty);
        match ty {
            Ty::Struct(id, args) => Ty::Struct(
                id,
                args.iter()
                    .map(|a| self.default_int_literals_deep(a))
                    .collect(),
            ),
            Ty::Enum(id, args) => Ty::Enum(
                id,
                args.iter()
                    .map(|a| self.default_int_literals_deep(a))
                    .collect(),
            ),
            Ty::Core(core, args) => Ty::Core(
                core,
                args.iter()
                    .map(|a| self.default_int_literals_deep(a))
                    .collect(),
            ),
            Ty::Tuple(elems) => Ty::Tuple(
                elems
                    .iter()
                    .map(|e| self.default_int_literals_deep(e))
                    .collect(),
            ),
            Ty::Array(elem, n) => Ty::Array(Box::new(self.default_int_literals_deep(&elem)), n),
            Ty::Slice(elem) => Ty::Slice(Box::new(self.default_int_literals_deep(&elem))),
            Ty::Ref { mutable, inner } => Ty::Ref {
                mutable,
                inner: Box::new(self.default_int_literals_deep(&inner)),
            },
            Ty::Range(inner) => Ty::Range(Box::new(self.default_int_literals_deep(&inner))),
            other => other,
        }
    }

    /// WP-C4.7-6.3: 03-Type-System solving step 5 — "default an **unconstrained** integer literal
    /// to `Int32` when representable, otherwise `Int64`". Runs after all bodies are checked, so
    /// every expected type has had its chance to constrain the literal first. A literal that a
    /// later use constrained (TYPE-INFER-001 permits that for an unannotated local) is already
    /// bound and is left alone.
    pub(super) fn default_unconstrained_int_literals(&mut self) {
        // RESOLVE first, then default the END of the chain. A literal variable is frequently
        // bound to ANOTHER variable rather than to a concrete type — `MyOpt::Some2(7)` unifies
        // the literal with the enum's own element variable — and that made the literal look
        // "constrained" while the chain terminated at an unbound, non-literal variable. Such a
        // chain used to escape defaulting entirely and surface as `type Infer(N)` at MIR
        // lowering, which is precisely the failure this ordering prevents.
        let pending: Vec<(TypeVarId, i128)> = self
            .int_literal_vars
            .iter()
            .filter_map(|(&id, &(value, _))| match self.resolve(&Ty::Infer(id)) {
                Ty::Infer(open) => Some((open, value)),
                _ => None,
            })
            .collect();
        for (id, value) in pending {
            let primitive = if i32::try_from(value).is_ok() {
                Primitive::Int32
            } else {
                Primitive::Int64
            };
            self.subst.insert(id, Ty::Primitive(primitive));
        }
    }

    pub(super) fn unify(&mut self, t1: Ty, t2: Ty, span: Span) -> Result<(), ()> {
        let t1 = self.resolve(&t1);
        let t2 = self.resolve(&t2);

        match (t1, t2) {
            (Ty::Infer(id1), Ty::Infer(id2)) if id1 == id2 => Ok(()),
            (Ty::Infer(id), other) | (other, Ty::Infer(id)) => {
                if self.occurs_in(id, &other) {
                    self.diags.push(
                        Diagnostic::error("recursive type inference mismatch", span)
                            .with_code("E0001"),
                    );
                    return Err(());
                }
                if !self.bind_int_literal_var(id, &other, span)? {
                    return Err(());
                }
                self.subst.insert(id, other);
                Ok(())
            }
            (Ty::Primitive(p1), Ty::Primitive(p2)) if p1 == p2 => Ok(()),
            (Ty::Struct(s1, args1), Ty::Struct(s2, args2)) if s1 == s2 => {
                self.unify_type_lists(args1, args2, span)
            }
            (Ty::Enum(e1, args1), Ty::Enum(e2, args2)) if e1 == e2 => {
                self.unify_type_lists(args1, args2, span)
            }
            (Ty::Core(c1, args1), Ty::Core(c2, args2)) if c1 == c2 => {
                self.unify_type_lists(args1, args2, span)
            }
            (
                Ty::Ref {
                    mutable: false,
                    inner: expected,
                },
                Ty::Ref {
                    mutable: true,
                    inner: actual,
                },
            ) => self.unify(*expected, *actual, span),
            (
                Ty::Ref {
                    mutable: m1,
                    inner: i1,
                },
                Ty::Ref {
                    mutable: m2,
                    inner: i2,
                },
            ) => {
                if m1 == m2 {
                    self.unify(*i1, *i2, span)
                } else {
                    self.diags.push(
                        Diagnostic::error("reference mutability mismatch", span).with_code("E0001"),
                    );
                    Err(())
                }
            }
            (Ty::Tuple(elems1), Ty::Tuple(elems2)) => {
                if elems1.len() == elems2.len() {
                    for (e1, e2) in elems1.into_iter().zip(elems2) {
                        self.unify(e1, e2, span)?;
                    }
                    Ok(())
                } else {
                    self.diags
                        .push(Diagnostic::error("tuple size mismatch", span).with_code("E0001"));
                    Err(())
                }
            }
            (Ty::Array(e1, len1), Ty::Array(e2, len2)) => {
                if len1 == len2 {
                    self.unify(*e1, *e2, span)
                } else {
                    self.diags
                        .push(Diagnostic::error("array length mismatch", span).with_code("E0001"));
                    Err(())
                }
            }
            (Ty::Slice(e1), Ty::Slice(e2)) => self.unify(*e1, *e2, span),
            (Ty::Slice(expected), Ty::Array(actual, _)) => self.unify(*expected, *actual, span),
            (
                Ty::Fn {
                    params: p1,
                    ret: r1,
                },
                Ty::Fn {
                    params: p2,
                    ret: r2,
                },
            ) => {
                if p1.len() == p2.len() {
                    for (param1, param2) in p1.into_iter().zip(p2) {
                        self.unify(param1, param2, span)?;
                    }
                    self.unify(*r1, *r2, span)
                } else {
                    self.diags.push(
                        Diagnostic::error("function signature parameters mismatch", span)
                            .with_code("E0005"),
                    );
                    Err(())
                }
            }
            (Ty::Range(e1), Ty::Range(e2)) => self.unify(*e1, *e2, span),
            (Ty::Param(p1), Ty::Param(p2)) if p1 == p2 => Ok(()),
            (Ty::Extension(a), Ty::Extension(b)) => match (a.as_ref(), b.as_ref()) {
                (ExtensionTy::Tensor(ta), ExtensionTy::Tensor(tb)) => {
                    self.unify_tensor_types(ta, tb, span)
                }
                (ExtensionTy::Model(ma), ExtensionTy::Model(mb)) => {
                    if ma.item_id == mb.item_id {
                        Ok(())
                    } else {
                        let name_a =
                            if let hir::ItemKind::Model(def) = &self.hir.item(ma.item_id).kind {
                                self.text(def.name).to_string()
                            } else {
                                "Model".to_string()
                            };
                        let name_b =
                            if let hir::ItemKind::Model(def) = &self.hir.item(mb.item_id).kind {
                                self.text(def.name).to_string()
                            } else {
                                "Model".to_string()
                            };
                        self.diags.push(
                            Diagnostic::error(
                                format!("type mismatch: model `{name_a}` and model `{name_b}`"),
                                span,
                            )
                            .with_code("E0005"),
                        );
                        Err(())
                    }
                }
                (ExtensionTy::ModelError, ExtensionTy::ModelError) => Ok(()),
                _ => {
                    self.diags.push(
                        Diagnostic::error(
                            format!(
                                "type mismatch: `{}` and `{}`",
                                self.ty_to_string(&Ty::Extension(a.clone())),
                                self.ty_to_string(&Ty::Extension(b.clone()))
                            ),
                            span,
                        )
                        .with_code("E0005"),
                    );
                    Err(())
                }
            },
            (Ty::Never, _) | (_, Ty::Never) => Ok(()),
            (Ty::Error, _) | (_, Ty::Error) => Ok(()),
            (t1_resolved, t2_resolved) => {
                self.diags.push(
                    Diagnostic::error(
                        format!(
                            "type mismatch: expected '{}', found '{}'",
                            self.ty_to_string(&t1_resolved),
                            self.ty_to_string(&t2_resolved)
                        ),
                        span,
                    )
                    .with_code("E0001"),
                );
                Err(())
            }
        }
    }

    pub(super) fn unify_type_lists(
        &mut self,
        left: Vec<Ty>,
        right: Vec<Ty>,
        span: Span,
    ) -> Result<(), ()> {
        if left.len() != right.len() {
            self.diags.push(
                Diagnostic::error("generic argument count mismatch", span).with_code("E0001"),
            );
            return Err(());
        }
        for (left, right) in left.into_iter().zip(right) {
            self.unify(left, right, span)?;
        }
        Ok(())
    }

    /// Unify two tensor types, delegating shape/device unification to the
    /// extension and rendering a provenance-rich diagnostic on mismatch (§9).
    pub(super) fn unify_tensor_types(
        &mut self,
        a: &TensorKind,
        b: &TensorKind,
        span: Span,
    ) -> Result<(), ()> {
        match (a, b) {
            (TensorKind::Tensor(ta), TensorKind::Tensor(tb)) => {
                match self.tensor_ctx.unify_tensor(ta, tb) {
                    Ok(()) => Ok(()),
                    Err(err) => {
                        self.emit_tensor_unify_error(&err, span);
                        Err(())
                    }
                }
            }
            (TensorKind::TensorDyn(da), TensorKind::TensorDyn(db)) if da == db => Ok(()),
            (TensorKind::TensorAny, TensorKind::TensorAny) => Ok(()),
            _ => {
                self.diags.push(
                    Diagnostic::error(
                        format!(
                            "tensor type mismatch: expected `{}`, found `{}`",
                            self.tensor_ctx.display_tensor(a),
                            self.tensor_ctx.display_tensor(b)
                        ),
                        span,
                    )
                    .with_code("E0212"),
                );
                Err(())
            }
        }
    }

    pub(super) fn emit_tensor_unify_error(&mut self, err: &UnifyError, span: Span) {
        let msg = match err {
            UnifyError::DTypeMismatch { expected, found } => format!(
                "tensor element type mismatch: expected `{}`, found `{}`",
                expected.name(),
                found.name()
            ),
            UnifyError::RankMismatch { expected, found } => {
                format!("tensor rank mismatch: expected rank {expected}, found rank {found}")
            }
            UnifyError::DimMismatch {
                axis,
                expected,
                found,
                expected_origin,
                found_origin,
                ..
            } => format!(
                "tensor dimension mismatch at axis {axis}: expected `{}` from {expected_origin}, found `{}` from {found_origin}",
                self.tensor_ctx.display_dim(expected),
                self.tensor_ctx.display_dim(found)
            ),
            UnifyError::DeviceMismatch { expected, found } => {
                format!("tensor device mismatch: expected `{expected}`, found `{found}`")
            }
            UnifyError::RangeMismatch { expected, found } => {
                format!(
                    "tensor value-range mismatch: expected `{expected}`, found `{found}`"
                )
            }
            UnifyError::Arithmetic => "tensor dimension arithmetic overflowed".to_string(),
        };
        let mut diagnostic = Diagnostic::error(msg, span).with_code("E0212");
        if let UnifyError::DimMismatch {
            expected_span,
            found_span,
            ..
        } = err
        {
            if let Some(found) = found_span {
                diagnostic.span = *found;
            }
            if let Some(expected) = expected_span {
                if let Some(source) = self.hir.sources.get(expected.source) {
                    let (line, column) = source.line_col(expected.lo);
                    diagnostic = diagnostic
                        .with_note(format!("expected dimension originates at {line}:{column}"));
                }
            }
            if let Some(found) = found_span {
                if let Some(source) = self.hir.sources.get(found.source) {
                    let (line, column) = source.line_col(found.lo);
                    diagnostic = diagnostic
                        .with_note(format!("found dimension originates at {line}:{column}"));
                }
            }
        }
        self.diags.push(diagnostic);
    }

    pub(super) fn instantiate_ty(&self, ty: &Ty, map: &HashMap<String, Ty>) -> Ty {
        match ty {
            Ty::Param(name) => {
                if let Some(target) = map.get(name) {
                    return target.clone();
                }
                // WP-C6.2c: a projection `T::Item` instantiates by substituting the base type
                // parameter and resolving the associated type through the concrete impl.
                if let Some((base, assoc)) = name.split_once("::") {
                    if let Some(Ty::Struct(id, _) | Ty::Enum(id, _)) = map.get(base) {
                        if let Some(bound) = self.assoc_projections.get(&(*id, assoc.to_string())) {
                            return bound.clone();
                        }
                    }
                }
                ty.clone()
            }
            Ty::Ref { mutable, inner } => Ty::Ref {
                mutable: *mutable,
                inner: Box::new(self.instantiate_ty(inner, map)),
            },
            Ty::Struct(item, args) => Ty::Struct(
                *item,
                args.iter()
                    .map(|arg| self.instantiate_ty(arg, map))
                    .collect(),
            ),
            Ty::Enum(item, args) => Ty::Enum(
                *item,
                args.iter()
                    .map(|arg| self.instantiate_ty(arg, map))
                    .collect(),
            ),
            Ty::Core(core, args) => Ty::Core(
                *core,
                args.iter()
                    .map(|arg| self.instantiate_ty(arg, map))
                    .collect(),
            ),
            Ty::Tuple(elems) => {
                Ty::Tuple(elems.iter().map(|e| self.instantiate_ty(e, map)).collect())
            }
            Ty::Array(elem, len) => Ty::Array(Box::new(self.instantiate_ty(elem, map)), *len),
            Ty::Slice(elem) => Ty::Slice(Box::new(self.instantiate_ty(elem, map))),
            Ty::Fn { params, ret } => Ty::Fn {
                params: params.iter().map(|p| self.instantiate_ty(p, map)).collect(),
                ret: Box::new(self.instantiate_ty(ret, map)),
            },
            Ty::Range(elem) => Ty::Range(Box::new(self.instantiate_ty(elem, map))),
            Ty::Extension(ext) => match &**ext {
                ExtensionTy::Tensor(_) | ExtensionTy::Model(_) | ExtensionTy::ModelError => {
                    ty.clone()
                }
            },
            _ => ty.clone(),
        }
    }

    /// `Self` in a trait signature means the concrete receiver at this call site.
    pub(super) fn subst_self_ty(ty: Ty, self_ty: &Ty) -> Ty {
        let mut map = HashMap::new();
        map.insert("Self".to_string(), self_ty.clone());
        substitute_ty(&ty, &map)
    }

    /// WP-C6.2c: rewrite `Self` in a trait method's converted type to the concrete receiver.
    /// `Self` alone becomes `recv`; `Self::Item` becomes `recv::Item` (a projection string that a
    /// later normalisation step resolves). Applied to a method-call result before it is returned.
    pub(super) fn subst_self(ty: &Ty, recv: &str) -> Ty {
        match ty {
            Ty::Param(name) if name == "Self" => Ty::Param(recv.to_string()),
            Ty::Param(name) => match name.strip_prefix("Self::") {
                Some(assoc) => Ty::Param(format!("{recv}::{assoc}")),
                None => ty.clone(),
            },
            Ty::Ref { mutable, inner } => Ty::Ref {
                mutable: *mutable,
                inner: Box::new(Self::subst_self(inner, recv)),
            },
            Ty::Struct(item, args) => Ty::Struct(
                *item,
                args.iter().map(|a| Self::subst_self(a, recv)).collect(),
            ),
            Ty::Enum(item, args) => Ty::Enum(
                *item,
                args.iter().map(|a| Self::subst_self(a, recv)).collect(),
            ),
            Ty::Core(core, args) => Ty::Core(
                *core,
                args.iter().map(|a| Self::subst_self(a, recv)).collect(),
            ),
            Ty::Tuple(elems) => {
                Ty::Tuple(elems.iter().map(|e| Self::subst_self(e, recv)).collect())
            }
            Ty::Array(elem, len) => Ty::Array(Box::new(Self::subst_self(elem, recv)), *len),
            Ty::Slice(elem) => Ty::Slice(Box::new(Self::subst_self(elem, recv))),
            Ty::Range(elem) => Ty::Range(Box::new(Self::subst_self(elem, recv))),
            other => other.clone(),
        }
    }

    // AS7 Packet 6: integer-literal variable binding is inference-variable internals.
    /// WP-C4.7-6.3: gate binding an integer-literal inference var.
    ///
    /// Returns `Ok(true)` if the binding may proceed. An integer literal is not a wildcard: it
    /// may adopt any primitive INTEGER type whose range holds its value, and nothing else. This
    /// is expected-type propagation, not a coercion — 03's step 4 confines coercions to explicit
    /// coercion sites — so it does not open an implicit-conversion hole: only the literal itself
    /// is retyped, never a typed value.
    pub(super) fn bind_int_literal_var(
        &mut self,
        id: TypeVarId,
        other: &Ty,
        span: Span,
    ) -> Result<bool, ()> {
        let Some(&(value, lit_span)) = self.int_literal_vars.get(&id) else {
            return Ok(true);
        };
        // Binding to another variable keeps it open; the eventual concrete binding is checked.
        // `!` coerces to every type (the never-coercion rule) and `Ty::Error` is recovery — both
        // pass through untouched rather than being reported as a literal-typing failure.
        if matches!(other, Ty::Infer(_) | Ty::Never | Ty::Error) {
            return Ok(true);
        }
        let Ty::Primitive(primitive) = other else {
            self.diags.push(
                Diagnostic::error(
                    format!(
                        "type mismatch: expected '{}', found an integer literal",
                        self.ty_to_string(other)
                    ),
                    span,
                )
                .with_code("E0001"),
            );
            return Ok(false);
        };
        if !is_integer_primitive(*primitive) {
            self.diags.push(
                Diagnostic::error(
                    format!(
                        "type mismatch: expected '{}', found an integer literal",
                        self.ty_to_string(other)
                    ),
                    span,
                )
                .with_code("E0001"),
            );
            return Ok(false);
        }
        if !literal::primitive_int_range_contains(*primitive, value) {
            self.diags.push(
                Diagnostic::error(
                    format!(
                        "integer literal out of range for '{}'",
                        self.ty_to_string(other)
                    ),
                    lit_span,
                )
                .with_code("E0008"),
            );
            return Ok(false);
        }
        Ok(true)
    }
}
