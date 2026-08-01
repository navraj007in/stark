# stark-form Specification

## Purpose
Provide `application/x-www-form-urlencoded` form parsing and serialization, explicitly mapping `+` to space and space to `+`.

## Public Types & API
- `FormPair`: `name`, `value`
- `FormLimits`: `max_total_bytes`, `max_pair_count`, `max_name_bytes`, `max_value_bytes`
- `FormError`: Resource limits and wrapped `PercentError`.
- `parse(input: &String, limits: FormLimits) -> Result<Vec<FormPair>, FormError>`
- `serialize(pairs: &[FormPair], limits: FormLimits) -> Result<String, FormError>`

## Semantics
- Exclusively owns `+`-as-space logic (decodes `+` to `0x20`, encodes `0x20` to `+`).
- Preserves key ordering and duplicate keys.
- Uses `stark-percent` for other non-unreserved byte encoding.
