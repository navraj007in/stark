//! Tensor element types, tensor/device types, and their unification
//! (`tensor` extension §4, §5, §8).
//!
//! These are the extension-owned semantic representations. They are kept out
//! of the Core `Ty` enum: the Core checker refers to a tensor type only
//! through an opaque handle, and all equality/unification/display for tensors
//! is delegated here. Dimension arithmetic lives in [`super::dim`].
//!
//! Unification binds dimension and device *variables* (like type variables):
//! a fresh variable unifies with anything by binding; two ground/symbolic
//! sides compare by normal-form equality and, if not decidable, are a
//! compile-time error (spec §5 — the checker never defers to runtime).

use super::dim::{DimVar, Poly};
use crate::source::Span;
use std::collections::HashMap;
use std::fmt;

/// Element type of a tensor (`DType` kind, §4.1). No implicit coercions exist
/// between dtypes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DType {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Float16,
    BFloat16,
    Bool,
}

impl DType {
    pub fn name(self) -> &'static str {
        match self {
            DType::Int8 => "Int8",
            DType::Int16 => "Int16",
            DType::Int32 => "Int32",
            DType::Int64 => "Int64",
            DType::UInt8 => "UInt8",
            DType::UInt16 => "UInt16",
            DType::UInt32 => "UInt32",
            DType::UInt64 => "UInt64",
            DType::Float32 => "Float32",
            DType::Float64 => "Float64",
            DType::Float16 => "Float16",
            DType::BFloat16 => "BFloat16",
            DType::Bool => "Bool",
        }
    }
}

/// A device variable identity (fresh when `device` is omitted, §8).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct DeviceVar(pub u32);

/// Device placement of a tensor (§8).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Device {
    Cpu,
    /// `Cuda<N>` — CUDA device index N.
    Cuda(u32),
    /// A device variable (device-polymorphic until unified).
    Var(DeviceVar),
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => f.write_str("Cpu"),
            Device::Cuda(n) => write!(f, "Cuda<{n}>"),
            Device::Var(v) => write!(f, "?dev{}", v.0),
        }
    }
}

/// A tensor shape: an ordered list of dimension polynomials. Rank is always
/// static (`shape.len()`); there are no unknown-rank tensor types (§3.3).
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Shape {
    pub dims: Vec<Poly>,
}

impl Shape {
    pub fn new(dims: Vec<Poly>) -> Shape {
        Shape { dims }
    }
    pub fn rank(&self) -> usize {
        self.dims.len()
    }
    /// A shape is fully static iff every dim is a constant (§3.3).
    pub fn is_static(&self) -> bool {
        self.dims.iter().all(|d| d.as_constant().is_some())
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{d}")?;
        }
        f.write_str("]")
    }
}

/// A statically-shaped tensor type `Tensor<T, S, device = D>`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TensorTy {
    pub dtype: DType,
    pub shape: Shape,
    pub device: Device,
}

/// The three tensor forms (§4.2, §4.3). `TensorDyn`/`TensorAny` are the
/// type-erased boundary forms; the only bridge to `Tensor` is `refine`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum TensorKind {
    Tensor(TensorTy),
    /// dtype known, shape dynamic.
    TensorDyn(DType),
    /// dtype and shape dynamic.
    TensorAny,
}

impl TensorKind {
    /// Tensors are owned `Move` values, never `Copy` (§4.2). This holds for
    /// every tensor form.
    pub fn is_copy(&self) -> bool {
        false
    }
}

impl fmt::Display for TensorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorKind::Tensor(t) => {
                write!(f, "Tensor<{}, {}", t.dtype.name(), t.shape)?;
                match t.device {
                    Device::Var(_) => {}
                    dev => write!(f, ", device = {dev}")?,
                }
                f.write_str(">")
            }
            TensorKind::TensorDyn(d) => write!(f, "TensorDyn<{}>", d.name()),
            TensorKind::TensorAny => f.write_str("TensorAny"),
        }
    }
}

/// Where a dimension variable came from — the provenance category tracked for
/// diagnostics (§9). Every symbolic dim can be traced to its origin.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OriginKind {
    /// A dimension generic parameter (`<B: Dim>`).
    Param,
    /// A `refine` existential binding (§4.3).
    Refine,
    /// A port of an imported model (§7).
    ImportedPort,
    /// The result of a tensor operation (§6).
    OpResult,
}

/// Provenance for one dimension variable: its source span, origin category,
/// and a human-readable label used verbatim in diagnostics.
#[derive(Clone, Debug)]
pub struct DimProvenance {
    pub span: Span,
    pub origin: OriginKind,
    pub label: String,
}

/// Why a shape/device unification failed. Carries enough to build a
/// provenance-rich diagnostic at the call site.
#[derive(Clone, Debug)]
pub enum UnifyError {
    DTypeMismatch {
        expected: DType,
        found: DType,
    },
    RankMismatch {
        expected: usize,
        found: usize,
    },
    /// Positional dimension mismatch: the two dims are not provably equal and
    /// cannot be decided (spec §5).
    DimMismatch {
        axis: usize,
        expected: Poly,
        found: Poly,
    },
    DeviceMismatch {
        expected: Device,
        found: Device,
    },
    /// A dimension arithmetic overflow surfaced during unification.
    Arithmetic,
}

/// Unification environment: substitutions for dimension and device variables,
/// plus the provenance registry. Owned by the checker for the duration of one
/// checking scope.
#[derive(Default)]
pub struct UnifyCtx {
    dim_subst: HashMap<DimVar, Poly>,
    device_subst: HashMap<DeviceVar, Device>,
    provenance: HashMap<DimVar, DimProvenance>,
    unifiable_dims: std::collections::HashSet<DimVar>,
    next_dim: u32,
    next_device: u32,
}

impl UnifyCtx {
    pub fn new() -> UnifyCtx {
        UnifyCtx::default()
    }

    /// Register a fresh dimension variable that unification may bind (a
    /// signature's `Dim` parameter, or a `refine` existential).
    pub fn fresh_dim(&mut self, prov: DimProvenance) -> DimVar {
        let v = DimVar(self.next_dim);
        self.next_dim += 1;
        self.unifiable_dims.insert(v);
        self.provenance.insert(v, prov);
        v
    }

    /// A dimension variable that is *not* unifiable — a rigid symbol (e.g. an
    /// already-bound enclosing parameter). Used to model "reuse of a bound
    /// variable asserts equality" (§4.3).
    pub fn rigid_dim(&mut self, prov: DimProvenance) -> DimVar {
        let v = DimVar(self.next_dim);
        self.next_dim += 1;
        self.provenance.insert(v, prov);
        v
    }

    pub fn fresh_device(&mut self) -> Device {
        let v = DeviceVar(self.next_device);
        self.next_device += 1;
        Device::Var(v)
    }

    pub fn provenance(&self, var: DimVar) -> Option<&DimProvenance> {
        self.provenance.get(&var)
    }

    /// Apply the current substitution to a polynomial, replacing bound
    /// unifiable variables by their values (recursively) and re-normalizing.
    pub fn resolve_dim(&self, poly: &Poly) -> Result<Poly, UnifyError> {
        // Substitute each bound variable; iterate to a fixed point with a
        // bounded number of passes to avoid pathological chains.
        let mut current = poly.clone();
        for _ in 0..64 {
            let mut changed = false;
            let mut acc = Poly::zero();
            for (var, coeff, mono_vars) in monomials(&current) {
                let mut term = Poly::constant(coeff);
                for v in &mono_vars {
                    let factor = match self.dim_subst.get(v) {
                        Some(bound) => {
                            changed = true;
                            bound.clone()
                        }
                        None => Poly::var(*v),
                    };
                    term = term.mul(&factor).map_err(|_| UnifyError::Arithmetic)?;
                }
                acc = acc.add(&term).map_err(|_| UnifyError::Arithmetic)?;
                let _ = var;
            }
            current = acc;
            if !changed {
                break;
            }
        }
        Ok(current)
    }

    fn resolve_device(&self, device: Device) -> Device {
        let mut current = device;
        for _ in 0..64 {
            match current {
                Device::Var(v) => match self.device_subst.get(&v) {
                    Some(&bound) => current = bound,
                    None => break,
                },
                _ => break,
            }
        }
        current
    }

    /// Unify two dimensions positionally (§3.3/§5). Binds a single unbound
    /// unifiable variable; otherwise requires normal-form equality.
    pub fn unify_dim(&mut self, a: &Poly, b: &Poly, axis: usize) -> Result<(), UnifyError> {
        let a = self.resolve_dim(a)?;
        let b = self.resolve_dim(b)?;
        if a.equal(&b) {
            return Ok(());
        }
        if let Some(var) = self.as_unifiable_var(&a) {
            self.dim_subst.insert(var, b);
            return Ok(());
        }
        if let Some(var) = self.as_unifiable_var(&b) {
            self.dim_subst.insert(var, a);
            return Ok(());
        }
        Err(UnifyError::DimMismatch {
            axis,
            expected: a,
            found: b,
        })
    }

    /// If `poly` is exactly a single unbound unifiable variable `1*v`, return
    /// it (so it can be bound).
    fn as_unifiable_var(&self, poly: &Poly) -> Option<DimVar> {
        let ms: Vec<_> = monomials(poly);
        match ms.as_slice() {
            [(_, 1, vars)] if vars.len() == 1 => {
                let v = vars[0];
                (self.unifiable_dims.contains(&v) && !self.dim_subst.contains_key(&v)).then_some(v)
            }
            _ => None,
        }
    }

    pub fn unify_device(&mut self, a: Device, b: Device) -> Result<(), UnifyError> {
        let a = self.resolve_device(a);
        let b = self.resolve_device(b);
        match (a, b) {
            (Device::Var(v), other) | (other, Device::Var(v)) => {
                self.device_subst.insert(v, other);
                Ok(())
            }
            (x, y) if x == y => Ok(()),
            (expected, found) => Err(UnifyError::DeviceMismatch { expected, found }),
        }
    }

    /// Unify two tensor types (§4.2): dtype equal (no coercion), rank equal,
    /// dims equal positionally, devices unify.
    pub fn unify_tensor(&mut self, a: &TensorTy, b: &TensorTy) -> Result<(), UnifyError> {
        if a.dtype != b.dtype {
            return Err(UnifyError::DTypeMismatch {
                expected: a.dtype,
                found: b.dtype,
            });
        }
        if a.shape.rank() != b.shape.rank() {
            return Err(UnifyError::RankMismatch {
                expected: a.shape.rank(),
                found: b.shape.rank(),
            });
        }
        for (axis, (da, db)) in a.shape.dims.iter().zip(&b.shape.dims).enumerate() {
            self.unify_dim(da, db, axis)?;
        }
        self.unify_device(a.device, b.device)
    }
}

/// Decompose a polynomial into `(term_index, coefficient, variables)` triples,
/// expanding exponents into repeated variables. A helper for substitution and
/// single-variable detection without exposing `Poly`'s internals.
fn monomials(poly: &Poly) -> Vec<(usize, i64, Vec<DimVar>)> {
    poly.iter_terms()
        .enumerate()
        .map(|(i, (vars, coeff))| (i, coeff, vars))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span() -> Span {
        Span { lo: 0, hi: 0 }
    }
    fn prov(label: &str) -> DimProvenance {
        DimProvenance {
            span: span(),
            origin: OriginKind::Param,
            label: label.to_string(),
        }
    }
    fn lit(n: i64) -> Poly {
        Poly::constant(n)
    }

    #[test]
    fn dtype_never_coerces() {
        let mut ctx = UnifyCtx::new();
        let a = TensorTy {
            dtype: DType::Float32,
            shape: Shape::new(vec![lit(4)]),
            device: ctx.fresh_device(),
        };
        let b = TensorTy {
            dtype: DType::Float16,
            shape: Shape::new(vec![lit(4)]),
            device: ctx.fresh_device(),
        };
        assert!(matches!(
            ctx.unify_tensor(&a, &b),
            Err(UnifyError::DTypeMismatch { .. })
        ));
    }

    #[test]
    fn rank_must_match() {
        let mut ctx = UnifyCtx::new();
        let a = TensorTy {
            dtype: DType::Float32,
            shape: Shape::new(vec![lit(4), lit(4)]),
            device: ctx.fresh_device(),
        };
        let b = TensorTy {
            dtype: DType::Float32,
            shape: Shape::new(vec![lit(4)]),
            device: ctx.fresh_device(),
        };
        assert!(matches!(
            ctx.unify_tensor(&a, &b),
            Err(UnifyError::RankMismatch {
                expected: 2,
                found: 1
            })
        ));
    }

    #[test]
    fn symbolic_variable_binds_then_asserts_equal() {
        let mut ctx = UnifyCtx::new();
        let m = ctx.fresh_dim(prov("M"));
        // [M] unifies with [768] -> M := 768.
        ctx.unify_dim(&Poly::var(m), &lit(768), 0).unwrap();
        assert_eq!(
            ctx.resolve_dim(&Poly::var(m)).unwrap().as_constant(),
            Some(768)
        );
        // Now [M] against [512] is a mismatch (already bound to 768).
        assert!(matches!(
            ctx.unify_dim(&Poly::var(m), &lit(512), 0),
            Err(UnifyError::DimMismatch { .. })
        ));
    }

    #[test]
    fn static_dim_mismatch_is_error() {
        let mut ctx = UnifyCtx::new();
        assert!(matches!(
            ctx.unify_dim(&lit(768), &lit(512), 2),
            Err(UnifyError::DimMismatch { axis: 2, .. })
        ));
    }

    #[test]
    fn two_unrelated_symbols_do_not_unify() {
        // Rigid (non-unifiable) symbols M and N: [M] vs [N] is undecidable.
        let mut ctx = UnifyCtx::new();
        let m = ctx.rigid_dim(prov("M"));
        let n = ctx.rigid_dim(prov("N"));
        assert!(matches!(
            ctx.unify_dim(&Poly::var(m), &Poly::var(n), 0),
            Err(UnifyError::DimMismatch { .. })
        ));
    }

    #[test]
    fn implicit_device_polymorphism_unifies() {
        let mut ctx = UnifyCtx::new();
        let dev = ctx.fresh_device(); // omitted device -> fresh var
                                      // fresh var unifies with a concrete device.
        ctx.unify_device(dev, Device::Cuda(0)).unwrap();
        assert_eq!(ctx.resolve_device(dev), Device::Cuda(0));
    }

    #[test]
    fn concrete_device_mismatch_is_error() {
        let mut ctx = UnifyCtx::new();
        assert!(matches!(
            ctx.unify_device(Device::Cpu, Device::Cuda(0)),
            Err(UnifyError::DeviceMismatch { .. })
        ));
    }

    #[test]
    fn distinct_refinements_are_distinct_symbols() {
        // Two separate `refine` calls introduce distinct fresh variables even
        // if spelled alike; they must not unify as equal by default.
        let mut ctx = UnifyCtx::new();
        let b1 = ctx.rigid_dim(DimProvenance {
            span: span(),
            origin: OriginKind::Refine,
            label: "B".into(),
        });
        let b2 = ctx.rigid_dim(DimProvenance {
            span: span(),
            origin: OriginKind::Refine,
            label: "B".into(),
        });
        assert!(matches!(
            ctx.unify_dim(&Poly::var(b1), &Poly::var(b2), 0),
            Err(UnifyError::DimMismatch { .. })
        ));
    }

    #[test]
    fn polynomial_shapes_unify_positionally() {
        // Tensor<F32, [B, H]> vs Tensor<F32, [4, 8]> with B, H unifiable.
        let mut ctx = UnifyCtx::new();
        let bvar = ctx.fresh_dim(prov("B"));
        let hvar = ctx.fresh_dim(prov("H"));
        let a = TensorTy {
            dtype: DType::Float32,
            shape: Shape::new(vec![Poly::var(bvar), Poly::var(hvar)]),
            device: ctx.fresh_device(),
        };
        let b = TensorTy {
            dtype: DType::Float32,
            shape: Shape::new(vec![lit(4), lit(8)]),
            device: ctx.fresh_device(),
        };
        ctx.unify_tensor(&a, &b).unwrap();
        assert_eq!(
            ctx.resolve_dim(&Poly::var(bvar)).unwrap().as_constant(),
            Some(4)
        );
        assert_eq!(
            ctx.resolve_dim(&Poly::var(hvar)).unwrap().as_constant(),
            Some(8)
        );
    }

    #[test]
    fn checker_does_not_solve_equations() {
        // `2*H` against `8` is NOT unifiable: the checker does no equation
        // solving (spec §3.3), so this is a compile-time mismatch, not H := 4.
        let mut ctx = UnifyCtx::new();
        let hvar = ctx.fresh_dim(prov("H"));
        let two_h = Poly::var(hvar).mul(&lit(2)).unwrap();
        assert!(matches!(
            ctx.unify_dim(&two_h, &lit(8), 0),
            Err(UnifyError::DimMismatch { .. })
        ));
    }

    #[test]
    fn tensors_are_move_not_copy() {
        let t = TensorKind::Tensor(TensorTy {
            dtype: DType::Float32,
            shape: Shape::new(vec![lit(1)]),
            device: Device::Cpu,
        });
        assert!(!t.is_copy());
        assert!(!TensorKind::TensorAny.is_copy());
        assert!(!TensorKind::TensorDyn(DType::Float32).is_copy());
    }

    #[test]
    fn display_includes_shape_and_dtype() {
        let t = TensorKind::Tensor(TensorTy {
            dtype: DType::Float32,
            shape: Shape::new(vec![lit(1024), lit(768)]),
            device: Device::Cpu,
        });
        assert_eq!(t.to_string(), "Tensor<Float32, [1024, 768], device = Cpu>");
    }
}
