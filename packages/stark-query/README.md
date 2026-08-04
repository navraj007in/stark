# stark-query

RFC-style query string parsing and serialization for STARK.

## API Summary

```stark
pub struct QueryPair {
    pub name: String,
    pub value: String,
}

pub struct QueryLimits {
    pub max_total_bytes: UInt64,
    pub max_pair_count: UInt64,
    pub max_name_bytes: UInt64,
    pub max_value_bytes: UInt64,
}

pub enum QueryError {
    ExceededTotalBytesLimit,
    ExceededPairCountLimit,
    ExceededNameLimit,
    ExceededValueLimit,
    PercentDecodeError(PercentError),
}

pub fn parse(input: &String, limits: QueryLimits) -> Result<Vec<QueryPair>, QueryError>;
pub fn serialize(pairs: &[QueryPair], limits: QueryLimits) -> Result<String, QueryError>;
```
