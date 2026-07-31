# stark-form

`application/x-www-form-urlencoded` parsing and serialization for STARK.

## API Summary

```stark
pub struct FormPair {
    pub name: String,
    pub value: String,
}

pub struct FormLimits {
    pub max_total_bytes: UInt64,
    pub max_pair_count: UInt64,
    pub max_name_bytes: UInt64,
    pub max_value_bytes: UInt64,
}

pub enum FormError {
    ExceededTotalBytesLimit,
    ExceededPairCountLimit,
    ExceededNameLimit,
    ExceededValueLimit,
    PercentDecodeError(PercentError),
}

pub fn parse(input: &String, limits: FormLimits) -> Result<Vec<FormPair>, FormError>;
pub fn serialize(pairs: &[FormPair], limits: FormLimits) -> Result<String, FormError>;
```
