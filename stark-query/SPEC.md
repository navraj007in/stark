# stark-query Specification

## Purpose
Provide strict RFC-style query string parsing and serialization for URL queries without form semantics (`+` remains literal character `+`).

## Public Types & API
- `QueryPair`: `name`, `value`
- `QueryLimits`: `max_total_bytes`, `max_pair_count`, `max_name_bytes`, `max_value_bytes`
- `QueryError`: Limit errors and wrapped `PercentError`.
- `parse(input: &String, limits: QueryLimits) -> Result<Vec<QueryPair>, QueryError>`
- `serialize(pairs: &[QueryPair], limits: QueryLimits) -> Result<String, QueryError>`

## Semantics
- Preserves key order and duplicate keys.
- Percent-decodes names and values using `stark-percent`.
- Treats `+` strictly as literal character `+`.
- Percent-encodes names and values using `PercentEncodeSet::QueryComponent` during serialization.
