# stark-semver

`stark-semver` provides strict Semantic Versioning 2.0.0 parsing, formatting, precedence
comparison, and a deliberately small requirement matcher.

Implemented:

- `MAJOR.MINOR.PATCH` parsing;
- prerelease identifiers;
- build metadata identifiers;
- canonical formatting;
- SemVer precedence comparison, ignoring build metadata;
- exact requirements;
- caret requirements;
- tilde requirements;
- comma-separated comparison sets such as `>=1.2.0,<2.0.0`;
- malformed-input errors for empty input, invalid core versions, invalid numbers, invalid
  identifiers, and invalid requirements.

Excluded from v0.1:

- npm/Cargo-complete range grammar;
- whitespace-tolerant requirements;
- wildcard requirements;
- disjunctions such as `||`;
- coercing partial versions.
