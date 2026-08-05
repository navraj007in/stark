# stark-uuid

`stark-uuid` provides a pure-STARK UUID v0.1 surface.

Implemented:

- UUID value type backed by 16 bytes;
- canonical hyphenated parsing;
- uppercase input acceptance;
- canonical lowercase formatting;
- nil UUID;
- byte construction and byte extraction;
- UUID v4 construction from caller-provided random bytes;
- UUID v7 construction from caller-provided Unix milliseconds and random fields, which **rejects**
  a timestamp wider than 48 bits or a `rand_a` wider than 12 bits rather than masking them down —
  a truncated field yields a well-formed UUID standing for a different instant or different
  entropy, which nothing downstream can detect;
- version inspection;
- variant inspection;
- nil, equality, and ordering helpers.

Excluded from v0.1:

- provider-backed UUID generation;
- compact 32-character parsing;
- braced parsing;
- URN parsing;
- namespace UUIDs.
