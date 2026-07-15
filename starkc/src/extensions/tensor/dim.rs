//! Dimension algebra for the `tensor` extension (spec §3.2–§3.3).
//!
//! A dimension expression denotes a non-negative integer and forms a
//! **multivariate polynomial** over dimension variables with integer
//! coefficients. Equality is decided by polynomial normalization — two
//! expressions are equal iff their canonical normal forms are identical
//! (e.g. `B * (H + 1)` equals `B*H + B`). This is the *only* arithmetic fact
//! the checker knows; there is no range or inequality reasoning in v0.1.
//!
//! This module is the pure algebraic core: it knows nothing about spans,
//! diagnostics, or the checker. Provenance (where a variable came from) is
//! tracked per [`DimVar`] by the checker, not here, so that polynomials remain
//! cheaply comparable. All fallible operations return [`DimError`] rather than
//! panicking, so malformed or overflowing input degrades into a diagnostic.

use std::collections::BTreeMap;
use std::fmt;

/// Identity of a dimension variable. Assigned by the checker; two variables
/// with distinct ids are distinct even if spelled alike (existential sizes,
/// spec §4.3). The pure algebra only compares ids.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct DimVar(pub u32);

/// A monomial: a product of dimension variables with positive exponents.
/// The empty product is the constant monomial `1`. Kept canonical (no
/// zero exponents) so equal monomials have identical representations.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct Monomial(BTreeMap<DimVar, u32>);

impl Monomial {
    fn one() -> Monomial {
        Monomial(BTreeMap::new())
    }

    fn single(var: DimVar) -> Monomial {
        let mut m = BTreeMap::new();
        m.insert(var, 1);
        Monomial(m)
    }

    fn is_one(&self) -> bool {
        self.0.is_empty()
    }

    /// Multiply two monomials by adding exponents, checking for exponent
    /// overflow (pathological input such as `x * x * x * ...`).
    fn mul(&self, other: &Monomial) -> Result<Monomial, DimError> {
        let mut out = self.0.clone();
        for (&var, &exp) in &other.0 {
            let slot = out.entry(var).or_insert(0);
            *slot = slot.checked_add(exp).ok_or(DimError::Overflow)?;
        }
        Ok(Monomial(out))
    }
}

/// A multivariate polynomial in canonical normal form: a map from monomials to
/// non-zero integer coefficients. The empty map is the zero polynomial.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Poly {
    terms: BTreeMap<Monomial, i64>,
}

/// A failure in dimension arithmetic. Never a panic.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DimError {
    /// Coefficient or exponent arithmetic exceeded the representable range.
    Overflow,
}

impl fmt::Display for DimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DimError::Overflow => f.write_str("dimension arithmetic overflowed"),
        }
    }
}

impl Poly {
    /// The zero polynomial.
    pub fn zero() -> Poly {
        Poly::default()
    }

    /// A constant polynomial.
    pub fn constant(value: i64) -> Poly {
        let mut p = Poly::default();
        if value != 0 {
            p.terms.insert(Monomial::one(), value);
        }
        p
    }

    /// A single dimension variable (`x`).
    pub fn var(var: DimVar) -> Poly {
        let mut p = Poly::default();
        p.terms.insert(Monomial::single(var), 1);
        p
    }

    /// Insert `coeff * mono`, dropping the term if the running coefficient
    /// reaches zero, so the map stays canonical.
    fn add_term(&mut self, mono: Monomial, coeff: i64) -> Result<(), DimError> {
        if coeff == 0 {
            return Ok(());
        }
        match self.terms.get_mut(&mono) {
            Some(existing) => {
                let sum = existing.checked_add(coeff).ok_or(DimError::Overflow)?;
                if sum == 0 {
                    self.terms.remove(&mono);
                } else {
                    *existing = sum;
                }
            }
            None => {
                self.terms.insert(mono, coeff);
            }
        }
        Ok(())
    }

    pub fn add(&self, other: &Poly) -> Result<Poly, DimError> {
        let mut out = self.clone();
        for (mono, &coeff) in &other.terms {
            out.add_term(mono.clone(), coeff)?;
        }
        Ok(out)
    }

    pub fn neg(&self) -> Result<Poly, DimError> {
        let mut out = Poly::default();
        for (mono, &coeff) in &self.terms {
            let neg = coeff.checked_neg().ok_or(DimError::Overflow)?;
            out.terms.insert(mono.clone(), neg);
        }
        Ok(out)
    }

    pub fn sub(&self, other: &Poly) -> Result<Poly, DimError> {
        self.add(&other.neg()?)
    }

    pub fn mul(&self, other: &Poly) -> Result<Poly, DimError> {
        let mut out = Poly::default();
        for (lm, &lc) in &self.terms {
            for (rm, &rc) in &other.terms {
                let coeff = lc.checked_mul(rc).ok_or(DimError::Overflow)?;
                let mono = lm.mul(rm)?;
                out.add_term(mono, coeff)?;
            }
        }
        Ok(out)
    }

    /// The constant value of a constant polynomial, if it is one.
    pub fn as_constant(&self) -> Option<i64> {
        match self.terms.len() {
            0 => Some(0),
            1 => self
                .terms
                .get(&Monomial::one())
                .copied()
                .filter(|_| self.terms.keys().all(Monomial::is_one)),
            _ => None,
        }
    }

    /// Whether this polynomial is **provably** non-negative for every
    /// assignment of its (non-negative-integer) variables. True iff every
    /// coefficient is non-negative — then the value is a sum of non-negative
    /// terms. A constant is judged by its sign. This is deliberately
    /// conservative: `2*N - 3` and `B - 1` are *not* provable and must be
    /// rejected, matching spec §3.3 (non-negativity only from literal
    /// constants).
    pub fn is_provably_nonnegative(&self) -> bool {
        self.terms.values().all(|&c| c >= 0)
    }

    /// Whether the two polynomials are equal by normal form. Equivalent to
    /// `==`, provided for readability at call sites where the "normal-form
    /// equality" semantics matter.
    pub fn equal(&self, other: &Poly) -> bool {
        self == other
    }

    /// Iterate the terms as `(variables, coefficient)`, with each monomial's
    /// exponents expanded into repeated variables (`x^2` → `[x, x]`). Exposes
    /// the polynomial structure for substitution without leaking internals.
    pub fn iter_terms(&self) -> impl Iterator<Item = (Vec<DimVar>, i64)> + '_ {
        self.terms.iter().map(|(mono, &coeff)| {
            let mut vars = Vec::new();
            for (&var, &exp) in &mono.0 {
                for _ in 0..exp {
                    vars.push(var);
                }
            }
            (vars, coeff)
        })
    }
}

impl fmt::Display for Poly {
    /// A deterministic, human-readable rendering for diagnostics, e.g.
    /// `B*H + B` or `2*N + 6`. Terms are ordered by the canonical monomial
    /// order; the empty polynomial renders as `0`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return f.write_str("0");
        }
        let mut first = true;
        // Higher-degree / lexicographically-later monomials first reads more
        // naturally (variables before the constant); iterate in reverse of the
        // canonical order, which puts the constant term last.
        for (mono, &coeff) in self.terms.iter().rev() {
            if !first {
                f.write_str(" + ")?;
            }
            first = false;
            if mono.is_one() {
                write!(f, "{coeff}")?;
            } else {
                if coeff != 1 {
                    write!(f, "{coeff}*")?;
                }
                let mut fv = true;
                for (var, &exp) in &mono.0 {
                    for _ in 0..exp {
                        if !fv {
                            f.write_str("*")?;
                        }
                        fv = false;
                        write!(f, "d{}", var.0)?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(n: u32) -> DimVar {
        DimVar(n)
    }

    // Named variables for readability: B=0, H=1, N=2.
    fn b() -> Poly {
        Poly::var(v(0))
    }
    fn h() -> Poly {
        Poly::var(v(1))
    }
    fn n() -> Poly {
        Poly::var(v(2))
    }
    fn c(k: i64) -> Poly {
        Poly::constant(k)
    }

    #[test]
    fn distributivity_identity_from_spec() {
        // B * (H + 1) == B*H + B
        let lhs = b().mul(&h().add(&c(1)).unwrap()).unwrap();
        let rhs = b().mul(&h()).unwrap().add(&b()).unwrap();
        assert!(lhs.equal(&rhs), "{lhs} != {rhs}");
    }

    #[test]
    fn linear_identity_from_spec() {
        // 2 * (N + 3) == 2*N + 6
        let lhs = c(2).mul(&n().add(&c(3)).unwrap()).unwrap();
        let rhs = n().mul(&c(2)).unwrap().add(&c(6)).unwrap();
        assert!(lhs.equal(&rhs), "{lhs} != {rhs}");
    }

    #[test]
    fn inequality_is_detected() {
        // B*H + B  !=  B*H + H
        let lhs = b().mul(&h()).unwrap().add(&b()).unwrap();
        let rhs = b().mul(&h()).unwrap().add(&h()).unwrap();
        assert!(!lhs.equal(&rhs));
        // 2*N + 6 != 2*N + 5
        assert!(!c(2)
            .mul(&n().add(&c(3)).unwrap())
            .unwrap()
            .equal(&n().mul(&c(2)).unwrap().add(&c(5)).unwrap()));
    }

    #[test]
    fn subtraction_cancels_to_constant() {
        // (N + 3) - N == 3
        let d = n().add(&c(3)).unwrap().sub(&n()).unwrap();
        assert_eq!(d.as_constant(), Some(3));
        // (N + 3) - (N + 5) == -2
        let neg = n()
            .add(&c(3))
            .unwrap()
            .sub(&n().add(&c(5)).unwrap())
            .unwrap();
        assert_eq!(neg.as_constant(), Some(-2));
    }

    #[test]
    fn nonnegativity_rules() {
        assert!(c(0).is_provably_nonnegative());
        assert!(c(5).is_provably_nonnegative());
        assert!(!c(-2).is_provably_nonnegative());
        // 2*N + 6 provable; 2*N - 3 not; B - 1 not; B*H + B provable.
        assert!(n()
            .mul(&c(2))
            .unwrap()
            .add(&c(6))
            .unwrap()
            .is_provably_nonnegative());
        assert!(!n()
            .mul(&c(2))
            .unwrap()
            .sub(&c(3))
            .unwrap()
            .is_provably_nonnegative());
        assert!(!b().sub(&c(1)).unwrap().is_provably_nonnegative());
        assert!(b()
            .mul(&h())
            .unwrap()
            .add(&b())
            .unwrap()
            .is_provably_nonnegative());
    }

    #[test]
    fn zero_and_constant_forms() {
        assert_eq!(Poly::zero().as_constant(), Some(0));
        assert_eq!(b().sub(&b()).unwrap().as_constant(), Some(0));
        assert!(b().as_constant().is_none());
        assert_eq!(c(7).as_constant(), Some(7));
    }

    #[test]
    fn display_is_deterministic() {
        // B*H + B  (vars render as d0, d1, ...)
        let p = b().mul(&h()).unwrap().add(&b()).unwrap();
        assert_eq!(p.to_string(), "d0*d1 + d0");
        assert_eq!(
            c(2).mul(&n().add(&c(3)).unwrap()).unwrap().to_string(),
            "2*d2 + 6"
        );
        assert_eq!(Poly::zero().to_string(), "0");
    }

    #[test]
    fn overflow_is_an_error_not_a_panic() {
        let big = Poly::constant(i64::MAX);
        assert_eq!(big.add(&Poly::constant(1)).unwrap_err(), DimError::Overflow);
        assert_eq!(big.mul(&Poly::constant(2)).unwrap_err(), DimError::Overflow);
    }

    // ---- bounded, deterministic property tests (no external dependency) ----

    /// A tiny xorshift PRNG so the generated tests are reproducible.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn small(&mut self) -> i64 {
            (self.next() % 7) as i64 - 3 // [-3, 3]
        }
        fn var(&mut self) -> Poly {
            Poly::var(v((self.next() % 3) as u32))
        }
    }

    /// Build a random small polynomial from a bounded expression tree.
    fn gen(rng: &mut Rng, depth: u32) -> Poly {
        if depth == 0 {
            return if rng.next() % 2 == 0 {
                Poly::constant(rng.small())
            } else {
                rng.var()
            };
        }
        let a = gen(rng, depth - 1);
        let b = gen(rng, depth - 1);
        match rng.next() % 3 {
            0 => a.add(&b).unwrap_or_else(|_| Poly::zero()),
            1 => a.sub(&b).unwrap_or_else(|_| Poly::zero()),
            _ => a.mul(&b).unwrap_or_else(|_| Poly::zero()),
        }
    }

    #[test]
    fn property_equality_is_reflexive_and_symmetric_and_stable() {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        for _ in 0..2000 {
            let p = gen(&mut rng, 3);
            let q = gen(&mut rng, 3);
            // Reflexivity and normalization stability (clone is identical).
            assert!(p.equal(&p));
            assert_eq!(p.clone(), p);
            // Symmetry of equality.
            assert_eq!(p.equal(&q), q.equal(&p));
            // Commutativity of addition and multiplication under normalization.
            if let (Ok(pq), Ok(qp)) = (p.add(&q), q.add(&p)) {
                assert!(pq.equal(&qp));
            }
            if let (Ok(pq), Ok(qp)) = (p.mul(&q), q.mul(&p)) {
                assert!(pq.equal(&qp));
            }
        }
    }

    #[test]
    fn property_never_panics() {
        // Deeper trees exercise more overflow/normalization paths; the harness
        // asserts merely by not panicking.
        let mut rng = Rng(0xDEADBEEFCAFEBABE);
        for _ in 0..1000 {
            let p = gen(&mut rng, 5);
            let _ = p.is_provably_nonnegative();
            let _ = p.as_constant();
            let _ = p.to_string();
        }
    }
}
