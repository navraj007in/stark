# stark-percent

Strict RFC percent encoding and decoding for STARK URLs and query components.

## API Summary

```stark
pub enum PercentEncodeSet {
    PathSegment,
    Path,
    QueryComponent,
}

pub enum PercentError {
    IncompleteEscape(UInt64),
    InvalidHexDigit(UInt64, UInt8),
    OutputTooLarge,
}

pub fn encode(input: &[UInt8], set: PercentEncodeSet) -> String;
pub fn decode(input: &String) -> Result<Vec<UInt8>, PercentError>;
```
