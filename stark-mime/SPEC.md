# stark-mime Specification

## Purpose
Provide a bounded HTTP Media Type (MIME) parser and formatter supporting `type/subtype`, `type/subtype; parameter=value`, and quoted string parameter values.

## Public Types & API
- `MediaTypeParameter`: `name`, `value`
- `MediaType`: `type_name`, `subtype`, `parameters`
- `MediaTypeLimits`: `max_total_bytes`, `max_parameter_count`, `max_parameter_name_bytes`, `max_parameter_value_bytes`
- `parse_media_type`, `format_media_type`, `media_type_is`, `media_type_parameter`

## Semantics
- Validates type, subtype, and parameter names using `stark-ascii::is_tchar`.
- Comparison of type, subtype, and parameter names is ASCII case-insensitive.
- Preserves duplicate parameter names in appearance order.
- Quotes formatted values if they contain spaces or non-token bytes.
