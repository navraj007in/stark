# Package capability derivation

**Status:** normative package-tooling contract, implemented as WP-P1.6 (2026-08-09).

For every compiled package, each reference to a host-backed provider function or resource
contributes that interface's vocabulary-v1 capability. Derivation is conservative and
reference-level: a reference contributes whether or not it is reachable, called, or eliminated by
optimization. It does not vary with dead-code analysis.

The root application's derived authority is the sorted union across the complete resolved
dependency graph. The root manifest supplies an upper-bound envelope:

```text
derived transitive closure ⊆ root manifest capabilities
```

If the relation does not hold, `stark check` and `stark build` fail before execution or backend
emission. Each diagnostic names the missing capability, the package that contributed it, and the
specific `provider_api.functions.<item>` or `provider_api.resources.<nominal>` reference. A declared
capability that is not derived is accepted; it may cover another target or optional configuration
and is an audit signal rather than an error.

Provider admission follows derived interface references, not spare entries in the envelope. Thus
declaring unused authority neither selects nor links an unrelated native provider. Native provider
metadata remains trusted input: derivation proves which declared interface the STARK graph uses, not
what the provider implementation does internally.
