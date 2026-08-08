//! **AS7 Packet 8 — pattern typing and binding.**
//!
//! Sits above `bounds` and below `body`: patterns may use `bounds`, `convert`, `traits`, `infer`,
//! `state` and `types`. Nothing below may reach back into it.
//!
//! Two related but distinct jobs live here: giving a pattern a type and binding its names
//! (`check_pat_with_mode`), and the structural relations between patterns that later analyses ask
//! about — subsumption, whether a pattern is a constructor, whether it is irrefutable.
//!
//! **The binding-mode rule is a deliberate divergence from Rust and is documented on `BindMode`
//! in `types`.** STARK copies a `Copy` component and binds a non-`Copy` one by reference, rather
//! than binding everything by reference under a reference scrutinee. Extraction does not touch it.

use super::state::TypeChecker;
use super::types::{convert_float_suffix, convert_int_suffix, BindMode, Ty, VariantFields};

use crate::ast::{Lit, Primitive};
use crate::diag::Diagnostic;
use crate::hir::{self, Builtin, CoreType, ExprId, PatId, Res};
use crate::literal;

impl TypeChecker<'_> {
    pub(super) fn pat_subsumes(&self, a: &hir::PatNode, b: &hir::PatNode) -> bool {
        match (&a.kind, &b.kind) {
            (hir::PatKind::Wild | hir::PatKind::Binding { .. }, _) => true,
            (_, hir::PatKind::Wild | hir::PatKind::Binding { .. }) => false,
            (hir::PatKind::Lit(la), hir::PatKind::Lit(lb)) => {
                // WP-C1.5: `Lit` itself carries no value for Int/Float/Str (only shape tags --
                // base/suffix/raw), so comparing it directly treats any two same-kind literal
                // patterns as equal regardless of value, e.g. `match x { 1 => .., 2 => .. }`
                // spuriously flagged the second arm as unreachable. Parse both literals' actual
                // values from their source text (the same logic `interp.rs` uses to evaluate
                // them) and compare those instead.
                match (
                    literal::eval_lit_value(*la, self.text(a.span), &self.hir.str_lits),
                    literal::eval_lit_value(*lb, self.text(b.span), &self.hir.str_lits),
                ) {
                    (Some(va), Some(vb)) => va == vb,
                    // Unparseable literal: fall back to the old shape-only comparison rather
                    // than silently treating it as never-equal (matches this function's existing
                    // "when in doubt" bias -- it also does not exist to catch parse failures).
                    _ => la == lb,
                }
            }
            (hir::PatKind::Path { res: ra, .. }, hir::PatKind::Path { res: rb, .. }) => ra == rb,
            (hir::PatKind::Tuple(pa), hir::PatKind::Tuple(pb)) => {
                pa.len() == pb.len()
                    && pa
                        .iter()
                        .zip(pb)
                        .all(|(&ia, &ib)| self.pat_subsumes(self.hir.pat(ia), self.hir.pat(ib)))
            }
            (hir::PatKind::Array(pa), hir::PatKind::Array(pb)) => {
                pa.len() == pb.len()
                    && pa
                        .iter()
                        .zip(pb)
                        .all(|(&ia, &ib)| self.pat_subsumes(self.hir.pat(ia), self.hir.pat(ib)))
            }
            (
                hir::PatKind::TupleVariant {
                    res: ra, pats: pa, ..
                },
                hir::PatKind::TupleVariant {
                    res: rb, pats: pb, ..
                },
            ) => {
                ra == rb
                    && pa.len() == pb.len()
                    && pa
                        .iter()
                        .zip(pb)
                        .all(|(&ia, &ib)| self.pat_subsumes(self.hir.pat(ia), self.hir.pat(ib)))
            }
            (
                hir::PatKind::Struct {
                    res: ra,
                    fields: fa,
                    ..
                },
                hir::PatKind::Struct {
                    res: rb,
                    fields: fb,
                    ..
                },
            ) => {
                if ra != rb {
                    return false;
                }
                for field_a in fa {
                    let name_a = self.text(field_a.name);
                    let Some(field_b) = fb.iter().find(|f| self.text(f.name) == name_a) else {
                        return false;
                    };
                    match (field_a.pat, field_b.pat) {
                        (Some(pa), Some(pb)) => {
                            if !self.pat_subsumes(self.hir.pat(pa), self.hir.pat(pb)) {
                                return false;
                            }
                        }
                        (Some(_), None) => return false,
                        _ => {}
                    }
                }
                true
            }
            _ => false,
        }
    }

    /// WP-C1.5: whether a pattern always matches, regardless of the scrutinee's value -- used
    /// alongside the top-level `Wild`/`Binding` check to decide match-arm exhaustiveness. A bare
    /// `Wild`/`Binding` is trivially irrefutable; a `Tuple`/`Array` pattern is irrefutable if
    /// every element is; a `Struct` pattern is irrefutable if every explicit field sub-pattern
    /// is (a shorthand field with no sub-pattern, e.g. `Point { x }`, is itself a binding).
    /// Without this, `match pair { (a, b) => .. }` (a fully-binding tuple pattern, matches any
    /// tuple) was flagged as non-exhaustive by the new general "requires wildcard" rule below,
    /// even though this single arm covers every possible tuple value.
    /// Does this pattern name a CONSTRUCTOR — a variant path, a struct shape, a tuple or an array?
    ///
    /// Used to decide whether a reference-typed scrutinee is an error (PAT-BIND-001: `&T` is not a
    /// nominal type, so a constructor path cannot name it). A wildcard or a plain binding names no
    /// constructor and is fine against a reference — `match r { other => .. }` binds the reference
    /// and is not what the rule forbids. Literal patterns likewise cannot apply to a reference and
    /// are rejected by ordinary unification, so they need no separate report here.
    pub(super) fn pat_is_constructor(&self, pat_id: PatId) -> bool {
        !matches!(
            &self.hir.pat(pat_id).kind,
            hir::PatKind::Wild | hir::PatKind::Binding { .. } | hir::PatKind::Lit(_)
        )
    }

    pub(super) fn is_irrefutable(&self, pat: &hir::PatNode) -> bool {
        match &pat.kind {
            hir::PatKind::Wild | hir::PatKind::Binding { .. } => true,
            hir::PatKind::Tuple(pats) | hir::PatKind::Array(pats) => pats
                .iter()
                .all(|&pat_id| self.is_irrefutable(self.hir.pat(pat_id))),
            // A `Struct { .. }` pattern matching an *enum variant* (`res: Res::Variant`) is not
            // irrefutable on its own -- other variants can still occur. Only a plain-struct
            // pattern (exactly one possible shape) can be irrefutable this way.
            hir::PatKind::Struct { res, fields, .. } if !matches!(res, Res::Variant(..)) => {
                fields.iter().all(|field| {
                    field
                        .pat
                        .is_none_or(|pat_id| self.is_irrefutable(self.hir.pat(pat_id)))
                })
            }
            _ => false,
        }
    }

    pub(super) fn scrutinee_reads_through_ref(&self, expr: ExprId) -> bool {
        match &self.hir.expr(expr).kind {
            hir::ExprKind::Unary {
                op: crate::ast::UnOp::Deref,
                ..
            } => true,
            hir::ExprKind::Field { base, .. } | hir::ExprKind::TupleField { base, .. } => {
                matches!(self.expr_types.get(base), Some(Ty::Ref { .. }))
                    || self.scrutinee_reads_through_ref(*base)
            }
            _ => false,
        }
    }

    pub(super) fn check_pat_with_mode(
        &mut self,
        pat_id: PatId,
        expected: Ty,
        bind_non_copy_by_ref: BindMode,
    ) -> Ty {
        let pat = self.hir.pat(pat_id);
        match &pat.kind {
            hir::PatKind::Lit(lit) => match lit {
                Lit::Int { suffix, .. } => {
                    if let Some(s) = suffix {
                        Ty::Primitive(convert_int_suffix(*s))
                    } else {
                        Ty::Primitive(Primitive::Int32)
                    }
                }
                Lit::Float { suffix, .. } => {
                    if let Some(s) = suffix {
                        Ty::Primitive(convert_float_suffix(*s))
                    } else {
                        Ty::Primitive(Primitive::Float64)
                    }
                }
                Lit::Str { .. } => Ty::Ref {
                    mutable: false,
                    inner: Box::new(Ty::Primitive(Primitive::Str)),
                },
                Lit::Char => Ty::Primitive(Primitive::Char),
                Lit::Bool(_) => Ty::Primitive(Primitive::Bool),
            },
            hir::PatKind::Wild => expected,
            hir::PatKind::Binding { local, .. } => {
                let binding_ty = if bind_non_copy_by_ref.binds_by_ref(self.is_copy_ty(&expected)) {
                    Ty::Ref {
                        mutable: false,
                        inner: Box::new(expected.clone()),
                    }
                } else {
                    expected.clone()
                };
                self.local_types.insert(*local, binding_ty);
                expected
            }
            hir::PatKind::Path { res, .. } => match res {
                Res::Item(item_id) => {
                    if let Some(const_ty) = self.const_types.get(item_id) {
                        let const_ty = const_ty.clone();
                        if !matches!(
                            self.resolve(&const_ty),
                            Ty::Primitive(
                                Primitive::Int8
                                    | Primitive::Int16
                                    | Primitive::Int32
                                    | Primitive::Int64
                                    | Primitive::UInt8
                                    | Primitive::UInt16
                                    | Primitive::UInt32
                                    | Primitive::UInt64
                                    | Primitive::Float32
                                    | Primitive::Float64
                                    | Primitive::Bool
                                    | Primitive::Char
                            )
                        ) {
                            self.diags.push(
                                Diagnostic::error(
                                    "constant patterns are restricted to primitive scalar values",
                                    pat.span,
                                )
                                .with_code("E0305")
                                .with_note(
                                    "aggregate and other nonprimitive constants cannot be patterns",
                                ),
                            );
                            Ty::Error
                        } else {
                            const_ty
                        }
                    } else {
                        Ty::Error
                    }
                }
                Res::Variant(enum_id, _) => {
                    let args = self.nominal_use_args(*enum_id, None, pat.span);
                    Ty::Enum(*enum_id, args)
                }
                // Companion to resolve.rs's `lower_pattern` fix: a bare `None` pattern now
                // reaches here as `PatKind::Path { res: Res::Builtin(Builtin::None), .. }`
                // (previously unreachable -- `None` always fell through to a fresh binding).
                // No payload to check; mirrors the `Res::Builtin(Builtin::Some)` no-payload-
                // present arm of the `TupleVariant` case just below, which likewise returns the
                // expected type unchecked against the specific builtin/type pairing (relying on
                // the caller's `unify(scr_ty, pat_ty, ..)` to catch a genuine mismatch).
                Res::Builtin(Builtin::None) => self.resolve(&expected),
                _ => Ty::Error,
            },
            hir::PatKind::TupleVariant { res, pats, .. } => {
                if let Res::Variant(enum_id, variant_idx) = res {
                    let args = match self.resolve(&expected) {
                        Ty::Enum(expected_id, args) if expected_id == *enum_id => args,
                        _ => self.nominal_use_args(*enum_id, None, pat.span),
                    };
                    let map = self.nominal_param_map(*enum_id, &args);
                    let tys_opt = self.enum_variants.get(enum_id).and_then(|variants| {
                        let variant = &variants[*variant_idx as usize];
                        if let VariantFields::Tuple(tys) = &variant.fields {
                            Some(tys.clone())
                        } else {
                            None
                        }
                    });
                    if let Some(tys) = tys_opt {
                        for (p, expected_t) in pats.iter().zip(tys) {
                            let expected_t = self.instantiate_ty(&expected_t, &map);
                            let p_ty = self.check_pat_with_mode(
                                *p,
                                expected_t.clone(),
                                bind_non_copy_by_ref,
                            );
                            let _ = self.unify(expected_t, p_ty, p.span(self.hir));
                        }
                    }
                    Ty::Enum(*enum_id, args)
                } else if let Res::Builtin(builtin) = res {
                    let resolved = self.resolve(&expected);
                    let payload = match (builtin, &resolved) {
                        (Builtin::Some, Ty::Core(CoreType::Option, args)) => args.first().cloned(),
                        (Builtin::Ok, Ty::Core(CoreType::Result, args)) => args.first().cloned(),
                        (Builtin::Err, Ty::Core(CoreType::Result, args)) => args.get(1).cloned(),
                        // **DEV-205: `IOError::Other(msg)` was missing here**, so its sub-pattern
                        // was never checked: the binding got no `local_types` entry and every use
                        // of it was typed `Ty::Error`. The program ran and printed correctly, which
                        // is why nothing found it for as long as nothing read the tables — the
                        // DEV-121 shape, in the checker rather than the oracle. The payload is the
                        // `String` the constructor's own signature already declares.
                        (Builtin::IOErrorOther, Ty::Core(CoreType::IOError, _)) => {
                            Some(Ty::Primitive(Primitive::String))
                        }
                        _ => None,
                    };
                    if let (Some(subpat), Some(payload)) = (pats.first(), payload) {
                        let p_ty = self.check_pat_with_mode(
                            *subpat,
                            payload.clone(),
                            bind_non_copy_by_ref,
                        );
                        let _ = self.unify(payload, p_ty, subpat.span(self.hir));
                    }
                    resolved
                } else {
                    Ty::Error
                }
            }
            hir::PatKind::Struct { res, fields, .. } => {
                if let Res::Item(struct_id) = res {
                    let args = self.nominal_use_args(*struct_id, None, pat.span);
                    let map = self.nominal_param_map(*struct_id, &args);
                    let expected_fields = self
                        .struct_fields
                        .get(struct_id)
                        .cloned()
                        .unwrap_or_default();
                    for field in fields {
                        let f_name = self.text(field.name);
                        if let Some(expected_f_ty) = expected_fields.get(f_name) {
                            if let Some(sub_pat) = field.pat {
                                let expected_f_ty = self.instantiate_ty(expected_f_ty, &map);
                                let p_ty = self.check_pat_with_mode(
                                    sub_pat,
                                    expected_f_ty.clone(),
                                    bind_non_copy_by_ref,
                                );
                                let _ = self.unify(expected_f_ty, p_ty, field.name);
                            } else if let Some(local) = field.local {
                                let expected_f_ty = self.instantiate_ty(expected_f_ty, &map);
                                let binding_ty = if bind_non_copy_by_ref
                                    .binds_by_ref(self.is_copy_ty(&expected_f_ty))
                                {
                                    Ty::Ref {
                                        mutable: false,
                                        inner: Box::new(expected_f_ty.clone()),
                                    }
                                } else {
                                    expected_f_ty.clone()
                                };
                                self.local_types.insert(local, binding_ty);
                            }
                        }
                    }
                    Ty::Struct(*struct_id, args)
                } else if let Res::Variant(enum_id, variant_idx) = res {
                    let args = match self.resolve(&expected) {
                        Ty::Enum(expected_id, args) if expected_id == *enum_id => args,
                        _ => self.nominal_use_args(*enum_id, None, pat.span),
                    };
                    let map = self.nominal_param_map(*enum_id, &args);
                    let expected_fields = self
                        .enum_variants
                        .get(enum_id)
                        .and_then(|variants| variants.get(*variant_idx as usize))
                        .and_then(|variant| match &variant.fields {
                            VariantFields::Struct(fields) => Some(fields.clone()),
                            _ => None,
                        })
                        .unwrap_or_default();
                    for field in fields {
                        let name = self.text(field.name);
                        if let Some(field_ty) = expected_fields.get(name) {
                            let field_ty = self.instantiate_ty(field_ty, &map);
                            if let Some(subpat) = field.pat {
                                let pat_ty = self.check_pat_with_mode(
                                    subpat,
                                    field_ty.clone(),
                                    bind_non_copy_by_ref,
                                );
                                let _ = self.unify(field_ty, pat_ty, field.name);
                            } else if let Some(local) = field.local {
                                let binding_ty = if bind_non_copy_by_ref
                                    .binds_by_ref(self.is_copy_ty(&field_ty))
                                {
                                    Ty::Ref {
                                        mutable: false,
                                        inner: Box::new(field_ty.clone()),
                                    }
                                } else {
                                    field_ty.clone()
                                };
                                self.local_types.insert(local, binding_ty);
                            }
                        }
                    }
                    Ty::Enum(*enum_id, args)
                } else {
                    Ty::Error
                }
            }
            hir::PatKind::Tuple(elems) => {
                let expected_elems = match self.resolve(&expected) {
                    Ty::Tuple(tys) if tys.len() == elems.len() => tys,
                    _ => (0..elems.len()).map(|_| self.new_type_var()).collect(),
                };
                let tys = elems
                    .iter()
                    .zip(expected_elems)
                    .map(|(&p, ty)| self.check_pat_with_mode(p, ty, bind_non_copy_by_ref))
                    .collect();
                Ty::Tuple(tys)
            }
            hir::PatKind::Array(elems) => {
                let elem_ty = match self.resolve(&expected) {
                    Ty::Array(elem, _) | Ty::Slice(elem) => *elem,
                    _ => self.new_type_var(),
                };
                for &e in elems {
                    let ety = self.check_pat_with_mode(e, elem_ty.clone(), bind_non_copy_by_ref);
                    let _ = self.unify(elem_ty.clone(), ety, pat.span);
                }
                Ty::Array(Box::new(elem_ty), elems.len() as u64)
            }
            hir::PatKind::Error => Ty::Error,
        }
    }
}
