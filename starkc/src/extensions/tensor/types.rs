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
    /// A generic parameter of kind `DType`.
    Var(u32),
}

impl DType {
    pub fn name(self) -> String {
        match self {
            DType::Int8 => "Int8".to_string(),
            DType::Int16 => "Int16".to_string(),
            DType::Int32 => "Int32".to_string(),
            DType::Int64 => "Int64".to_string(),
            DType::UInt8 => "UInt8".to_string(),
            DType::UInt16 => "UInt16".to_string(),
            DType::UInt32 => "UInt32".to_string(),
            DType::UInt64 => "UInt64".to_string(),
            DType::Float32 => "Float32".to_string(),
            DType::Float64 => "Float64".to_string(),
            DType::Float16 => "Float16".to_string(),
            DType::BFloat16 => "BFloat16".to_string(),
            DType::Bool => "Bool".to_string(),
            DType::Var(id) => format!("?dtype{id}"),
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
    /// Source origin for each dimension when the shape came from source.
    pub spans: Vec<Span>,
}

impl Shape {
    pub fn new(dims: Vec<Poly>) -> Shape {
        Shape {
            dims,
            spans: Vec::new(),
        }
    }
    pub fn with_spans(dims: Vec<Poly>, spans: Vec<Span>) -> Shape {
        debug_assert_eq!(dims.len(), spans.len());
        Shape { dims, spans }
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

/// A statically-tracked image value-range: the semantic tensor property Gate 7
/// adds on top of shape/dtype/device (`Tensor<T, S, range = R>`). It is a
/// compile-time property only — transitions are made by named operations
/// (`scale_255`, `normalize`) and the marker does not survive into generated
/// code. `Unspecified` is the default (no claim) and never widens: a ranged
/// value cannot be silently laundered into `Unspecified`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ValueRange {
    /// No value-range claim (the default for tensors without a `range =` arg).
    #[default]
    Unspecified,
    /// Integer image values conceptually in `[0, 255]`.
    ByteRange,
    /// Floating-point values conceptually in `[0, 1]`.
    UnitRange,
    /// Channel-wise mean/std normalised values.
    Normalized,
}

impl ValueRange {
    /// The source-level name, as written in a `range = ...` annotation.
    pub fn name(self) -> &'static str {
        match self {
            ValueRange::Unspecified => "Unspecified",
            ValueRange::ByteRange => "ByteRange",
            ValueRange::UnitRange => "UnitRange",
            ValueRange::Normalized => "Normalized",
        }
    }
}

impl fmt::Display for ValueRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A statically-shaped tensor type `Tensor<T, S, device = D, range = R>`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TensorTy {
    pub dtype: DType,
    pub shape: Shape,
    pub device: Device,
    pub range: ValueRange,
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
                if t.range != ValueRange::Unspecified {
                    write!(f, ", range = {}", t.range)?;
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
        expected_origin: Box<str>,
        found_origin: Box<str>,
        expected_span: Option<Span>,
        found_span: Option<Span>,
    },
    DeviceMismatch {
        expected: Device,
        found: Device,
    },
    /// Value-range (semantic) mismatch: two tensors carry incompatible
    /// image value-range states (Gate 7). A ranged value never widens to
    /// `Unspecified`, so an accidental erase surfaces here too.
    RangeMismatch {
        expected: ValueRange,
        found: ValueRange,
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
    dtype_subst: HashMap<u32, DType>,
    device_subst: HashMap<DeviceVar, Device>,
    provenance: HashMap<DimVar, DimProvenance>,
    unifiable_dims: std::collections::HashSet<DimVar>,
    unifiable_dtypes: std::collections::HashSet<u32>,
    unifiable_devices: std::collections::HashSet<DeviceVar>,
    next_dim: u32,
    next_dtype: u32,
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
        self.unifiable_devices.insert(v);
        Device::Var(v)
    }

    pub fn rigid_device(&mut self) -> Device {
        let v = DeviceVar(self.next_device);
        self.next_device += 1;
        Device::Var(v)
    }

    pub fn fresh_dtype(&mut self) -> DType {
        let id = self.next_dtype;
        self.next_dtype += 1;
        self.unifiable_dtypes.insert(id);
        DType::Var(id)
    }

    pub fn rigid_dtype(&mut self) -> DType {
        let id = self.next_dtype;
        self.next_dtype += 1;
        DType::Var(id)
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
        self.unify_dim_with_origins(a, b, axis, None, None)
    }

    fn unify_dim_with_origins(
        &mut self,
        a: &Poly,
        b: &Poly,
        axis: usize,
        expected_span: Option<Span>,
        found_span: Option<Span>,
    ) -> Result<(), UnifyError> {
        let a = self.resolve_dim(a)?;
        let b = self.resolve_dim(b)?;
        if a.equal(&b) {
            return Ok(());
        }
        if let Some(var) = self.as_unifiable_var(&a) {
            if self.poly_contains(&b, var) {
                return Err(self.dim_mismatch(axis, a, b, expected_span, found_span));
            }
            self.dim_subst.insert(var, b);
            return Ok(());
        }
        if let Some(var) = self.as_unifiable_var(&b) {
            if self.poly_contains(&a, var) {
                return Err(self.dim_mismatch(axis, a, b, expected_span, found_span));
            }
            self.dim_subst.insert(var, a);
            return Ok(());
        }
        Err(self.dim_mismatch(axis, a, b, expected_span, found_span))
    }

    fn poly_contains(&self, poly: &Poly, needle: DimVar) -> bool {
        poly.iter_terms()
            .any(|(vars, _)| vars.into_iter().any(|var| var == needle))
    }

    fn dim_origin(&self, poly: &Poly) -> Box<str> {
        let mut origins = poly
            .iter_terms()
            .flat_map(|(vars, _)| vars)
            .filter_map(|var| self.provenance.get(&var))
            .map(|p| format!("{} ({:?})", p.label, p.origin))
            .collect::<Vec<_>>();
        origins.sort();
        origins.dedup();
        if origins.is_empty() {
            "literal dimension".into()
        } else {
            origins.join(", ").into_boxed_str()
        }
    }

    fn dim_mismatch(
        &self,
        axis: usize,
        expected: Poly,
        found: Poly,
        expected_span: Option<Span>,
        found_span: Option<Span>,
    ) -> UnifyError {
        UnifyError::DimMismatch {
            expected_origin: self.dim_origin(&expected),
            found_origin: self.dim_origin(&found),
            axis,
            expected,
            found,
            expected_span,
            found_span,
        }
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
            (Device::Var(a), Device::Var(b)) if a == b => Ok(()),
            (Device::Var(v), other) if self.unifiable_devices.contains(&v) => {
                self.device_subst.insert(v, other);
                Ok(())
            }
            (other, Device::Var(v)) if self.unifiable_devices.contains(&v) => {
                self.device_subst.insert(v, other);
                Ok(())
            }
            (x, y) if x == y => Ok(()),
            (expected, found) => Err(UnifyError::DeviceMismatch { expected, found }),
        }
    }

    fn resolve_dtype(&self, dtype: DType) -> DType {
        let mut current = dtype;
        for _ in 0..64 {
            match current {
                DType::Var(v) => match self.dtype_subst.get(&v) {
                    Some(&bound) => current = bound,
                    None => break,
                },
                _ => break,
            }
        }
        current
    }

    pub fn unify_dtype(&mut self, a: DType, b: DType) -> Result<(), UnifyError> {
        let a = self.resolve_dtype(a);
        let b = self.resolve_dtype(b);
        match (a, b) {
            (DType::Var(a), DType::Var(b)) if a == b => Ok(()),
            (DType::Var(v), other) if self.unifiable_dtypes.contains(&v) => {
                self.dtype_subst.insert(v, other);
                Ok(())
            }
            (other, DType::Var(v)) if self.unifiable_dtypes.contains(&v) => {
                self.dtype_subst.insert(v, other);
                Ok(())
            }
            (x, y) if x == y => Ok(()),
            (expected, found) => Err(UnifyError::DTypeMismatch { expected, found }),
        }
    }

    /// Unify two tensor types (§4.2): dtype equal (no coercion), rank equal,
    /// dims equal positionally, devices unify.
    pub fn unify_tensor(&mut self, a: &TensorTy, b: &TensorTy) -> Result<(), UnifyError> {
        self.unify_dtype(a.dtype, b.dtype)?;
        if a.shape.rank() != b.shape.rank() {
            return Err(UnifyError::RankMismatch {
                expected: a.shape.rank(),
                found: b.shape.rank(),
            });
        }
        for (axis, (da, db)) in a.shape.dims.iter().zip(&b.shape.dims).enumerate() {
            self.unify_dim_with_origins(
                da,
                db,
                axis,
                a.shape.spans.get(axis).copied(),
                b.shape.spans.get(axis).copied(),
            )?;
        }
        self.unify_device(a.device, b.device)?;
        if a.range != b.range {
            return Err(UnifyError::RangeMismatch {
                expected: a.range,
                found: b.range,
            });
        }
        Ok(())
    }

    /// Render a tensor type for diagnostics, resolving bound variables through
    /// the substitution and printing dimension variables by their provenance
    /// label (`B`, `N`, ...) rather than internal ids.
    pub fn display_tensor(&self, kind: &TensorKind) -> String {
        match kind {
            TensorKind::Tensor(t) => {
                let device = match self.resolve_device(t.device) {
                    Device::Var(_) => String::new(),
                    d => format!(", device = {d}"),
                };
                let range = match t.range {
                    ValueRange::Unspecified => String::new(),
                    r => format!(", range = {r}"),
                };
                format!(
                    "Tensor<{}, {}{}{}>",
                    self.resolve_dtype(t.dtype).name(),
                    self.display_shape(&t.shape),
                    device,
                    range
                )
            }
            TensorKind::TensorDyn(d) => format!("TensorDyn<{}>", d.name()),
            TensorKind::TensorAny => "TensorAny".to_string(),
        }
    }

    /// Render a shape, resolving and labelling its dimensions.
    pub fn display_shape(&self, shape: &Shape) -> String {
        let dims: Vec<String> = shape.dims.iter().map(|d| self.display_dim(d)).collect();
        format!("[{}]", dims.join(", "))
    }

    /// Render a single dimension polynomial with provenance labels.
    pub fn display_dim(&self, poly: &Poly) -> String {
        let resolved = self.resolve_dim(poly).unwrap_or_else(|_| poly.clone());
        let label = |v: DimVar| {
            self.provenance
                .get(&v)
                .map_or_else(|| format!("?d{}", v.0), |p| p.label.clone())
        };
        let mut terms: Vec<String> = resolved
            .iter_terms()
            .map(|(vars, coeff)| {
                if vars.is_empty() {
                    return coeff.to_string();
                }
                let body = vars.iter().map(|&v| label(v)).collect::<Vec<_>>().join("*");
                if coeff == 1 {
                    body
                } else {
                    format!("{coeff}*{body}")
                }
            })
            .collect();
        // Variables before the constant reads more naturally.
        terms.reverse();
        if terms.is_empty() {
            "0".to_string()
        } else {
            terms.join(" + ")
        }
    }

    /// The provenance label for a dimension variable, if known.
    pub fn dim_label(&self, var: DimVar) -> Option<&str> {
        self.provenance.get(&var).map(|p| p.label.as_str())
    }

    /// Render a shape's element count as a per-axis product using source-level
    /// dimension names, e.g. `B * 125 * 13 * 13`. Unlike the normalized volume
    /// polynomial, this keeps the reshape's own factors visible and never shows
    /// internal ids. An empty (rank-0) shape has a product of `1`.
    pub fn shape_product_display(&self, shape: &Shape) -> String {
        if shape.dims.is_empty() {
            return "1".to_string();
        }
        shape
            .dims
            .iter()
            .map(|d| self.display_dim(d))
            .collect::<Vec<_>>()
            .join(" * ")
    }

    /// Describe the source origin of every distinct symbolic dimension appearing
    /// in `shapes`, in first-appearance order, for diagnostic notes such as
    /// ``dimension `B` originates from a function generic parameter``. Literal
    /// dimensions contribute nothing; internal ids are never exposed.
    pub fn dim_origin_notes(&self, shapes: &[&Shape]) -> Vec<String> {
        let mut seen: Vec<DimVar> = Vec::new();
        let mut notes = Vec::new();
        for shape in shapes {
            for dim in &shape.dims {
                let resolved = self.resolve_dim(dim).unwrap_or_else(|_| dim.clone());
                for (vars, _) in resolved.iter_terms() {
                    for var in vars {
                        if seen.contains(&var) {
                            continue;
                        }
                        seen.push(var);
                        if let Some(prov) = self.provenance.get(&var) {
                            let origin = match prov.origin {
                                OriginKind::Param => "a function generic parameter",
                                OriginKind::Refine => "a `refine` boundary",
                                OriginKind::ImportedPort => "an imported model port",
                                OriginKind::OpResult => "a tensor operation result",
                            };
                            notes.push(format!(
                                "dimension `{}` originates from {origin}",
                                prov.label
                            ));
                        }
                    }
                }
            }
        }
        notes
    }

    /// Freshen the generic variables in a stored tensor signature for one
    /// call. Shared maps preserve equality across every parameter and return
    /// type in that call while separate calls remain independent.
    pub fn freshen_tensor(
        &mut self,
        kind: &TensorKind,
        dims: &mut HashMap<DimVar, DimVar>,
        dtypes: &mut HashMap<u32, DType>,
        devices: &mut HashMap<DeviceVar, Device>,
    ) -> Result<TensorKind, UnifyError> {
        let fresh_dtype = |ctx: &mut Self, dtype: DType, map: &mut HashMap<u32, DType>| {
            if let DType::Var(id) = dtype {
                *map.entry(id).or_insert_with(|| ctx.fresh_dtype())
            } else {
                dtype
            }
        };
        match kind {
            TensorKind::Tensor(tensor) => {
                let mut fresh_dims = Vec::with_capacity(tensor.shape.dims.len());
                for poly in &tensor.shape.dims {
                    let mut rebuilt = Poly::zero();
                    for (_, coefficient, variables) in monomials(poly) {
                        let mut term = Poly::constant(coefficient);
                        for variable in variables {
                            let fresh = if let Some(fresh) = dims.get(&variable) {
                                *fresh
                            } else {
                                let provenance = self.provenance.get(&variable).cloned().unwrap_or(
                                    DimProvenance {
                                        span: Span { lo: 0, hi: 0 },
                                        origin: OriginKind::Param,
                                        label: format!("?d{}", variable.0),
                                    },
                                );
                                let fresh = self.fresh_dim(provenance);
                                dims.insert(variable, fresh);
                                fresh
                            };
                            term = term
                                .mul(&Poly::var(fresh))
                                .map_err(|_| UnifyError::Arithmetic)?;
                        }
                        rebuilt = rebuilt.add(&term).map_err(|_| UnifyError::Arithmetic)?;
                    }
                    fresh_dims.push(rebuilt);
                }
                let device = match tensor.device {
                    Device::Var(id) => *devices.entry(id).or_insert_with(|| self.fresh_device()),
                    concrete => concrete,
                };
                Ok(TensorKind::Tensor(TensorTy {
                    dtype: fresh_dtype(self, tensor.dtype, dtypes),
                    shape: Shape::with_spans(fresh_dims, tensor.shape.spans.clone()),
                    device,
                    range: tensor.range,
                }))
            }
            TensorKind::TensorDyn(dtype) => {
                Ok(TensorKind::TensorDyn(fresh_dtype(self, *dtype, dtypes)))
            }
            TensorKind::TensorAny => Ok(TensorKind::TensorAny),
        }
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
            range: ValueRange::Unspecified,
        };
        let b = TensorTy {
            dtype: DType::Float16,
            shape: Shape::new(vec![lit(4)]),
            device: ctx.fresh_device(),
            range: ValueRange::Unspecified,
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
            range: ValueRange::Unspecified,
        };
        let b = TensorTy {
            dtype: DType::Float32,
            shape: Shape::new(vec![lit(4)]),
            device: ctx.fresh_device(),
            range: ValueRange::Unspecified,
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
    fn cyclic_dimension_binding_is_rejected() {
        let mut ctx = UnifyCtx::new();
        let b = ctx.fresh_dim(prov("B"));
        let b_plus_one = Poly::var(b).add(&lit(1)).unwrap();
        assert!(matches!(
            ctx.unify_dim(&Poly::var(b), &b_plus_one, 0),
            Err(UnifyError::DimMismatch { .. })
        ));
        assert_eq!(ctx.resolve_dim(&Poly::var(b)).unwrap(), Poly::var(b));
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
            range: ValueRange::Unspecified,
        };
        let b = TensorTy {
            dtype: DType::Float32,
            shape: Shape::new(vec![lit(4), lit(8)]),
            device: ctx.fresh_device(),
            range: ValueRange::Unspecified,
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
            range: ValueRange::Unspecified,
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
            range: ValueRange::Unspecified,
        });
        assert_eq!(t.to_string(), "Tensor<Float32, [1024, 768], device = Cpu>");
    }
}
