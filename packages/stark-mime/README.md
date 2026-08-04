# stark-mime

Bounded HTTP Media Type (MIME) parser and formatter for STARK.

## API Summary

```stark
pub struct MediaTypeParameter {
    pub name: String,
    pub value: String,
}

pub struct MediaType {
    pub type_name: String,
    pub subtype: String,
    pub parameters: Vec<MediaTypeParameter>,
}

pub struct MediaTypeLimits {
    pub max_total_bytes: UInt64,
    pub max_parameter_count: UInt64,
    pub max_parameter_name_bytes: UInt64,
    pub max_parameter_value_bytes: UInt64,
}

pub enum MediaTypeError {
    EmptyType,
    EmptySubtype,
    MissingSlash,
    InvalidTokenByte(UInt64, UInt8),
    MalformedQuote(UInt64),
    ExceededTotalBytesLimit,
    ExceededParameterCountLimit,
    ExceededParameterNameLimit,
    ExceededParameterValueLimit,
}

pub fn parse_media_type(input: &String, limits: MediaTypeLimits) -> Result<MediaType, MediaTypeError>;
pub fn format_media_type(media_type: &MediaType) -> String;
pub fn media_type_is(media_type: &MediaType, type_name: &String, subtype: &String) -> Bool;
pub fn media_type_parameter<'a>(media_type: &'a MediaType, name: &String) -> Option<&'a String>;
```
