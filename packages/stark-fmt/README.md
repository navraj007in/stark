# stark-fmt

`stark-fmt` v0.1 is a pure-STARK formatting package. It builds a line of text by alternating
literal fragments with values rendered through their `Display` implementation.

It exists as the proof workload for **DEV-DISPLAY-DISPATCH**: before that work package, a generic
`T: Display` bound could be *written* and the method it exists to provide could not be *called*, so
no formatting package was writable at all. Every method here that touches a value is generic over
`Display` and does nothing but call `Display::fmt`.

## Public API

```stark
pub struct Line;

impl Line {
    pub fn new() -> Line;
    pub fn text(self, value: &str) -> Line;
    pub fn value<T: Display>(self, value: &T) -> Line;
    pub fn done(self) -> String;
}

pub fn to_string<T: Display>(value: &T) -> String;
```

## Behaviour

`Line` is built by chaining: each method takes `self` by value and returns a new `Line`, so a line
reads in the order its parts appear.

```stark
let msg = Line::new()
    .text("pkg=")
    .value(&name)
    .text(" n=")
    .value(&count)
    .text(" r=")
    .value(&ratio)
    .text(" ok=")
    .value(&ok)
    .done();

println(msg.as_str());
```

```text
pkg=stark n=42 r=0.75 ok=true
```

`value` **borrows** what it renders. `Display::fmt` is `&self` (06-Standard-Library.md,
STD-FORMAT-001), so a value stays usable after being formatted — including a non-`Copy` one, and
including an affine one such as a type with its own `Drop`.

The rendering is STARK's canonical `Display` output and is byte-identical to what `println` of the
same value produces: integers in base 10, `Bool` as `true`/`false`, `Char`/`String`/`str` as their
UTF-8 content, floats as the shortest round-tripping decimal at their declared width, and a user
type as whatever its `impl Display` returns.

## `to_string`

`to_string` is a free function, not a method, and that is deliberate. The method form
`x.to_string()` would need `impl<T: Display> ToString for T`, and Core v1 has neither blanket
implementations nor extension traits. Adding a resolver branch keyed on the name `to_string`
instead would rebuild exactly the two-tier trait model DEV-DISPLAY-DISPATCH removed. Recorded as
DEV-167; the free function is the supported form until a blanket-implementation decision is made.

## Exclusions

Not here, and not planned here: format strings, interpolation, variadic arguments, padding, width,
precision, alignment, grouping, locale awareness, or colour. This package answers one question —
can a package author write `fn value<T: Display>(..)` and have it work — and adding formatting
ergonomics on top would obscure the answer.

`Display` is also **not a serialisation format**: it performs no escaping. Do not build JSON or any
other quoted format by concatenating `Display` renderings; use `stark-json`.

## Capabilities

None. `stark-fmt` reaches nothing outside the process and needs no native provider, so it runs
under `stark run` as well as `stark build`.
