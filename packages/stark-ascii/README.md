# stark-ascii

Byte-first ASCII primitives for STARK HTTP, URL, and parser packages.

## API Summary

```stark
pub fn is_ascii(byte: UInt8) -> Bool;

pub fn is_ascii_alpha(byte: UInt8) -> Bool;
pub fn is_ascii_uppercase(byte: UInt8) -> Bool;
pub fn is_ascii_lowercase(byte: UInt8) -> Bool;
pub fn is_ascii_digit(byte: UInt8) -> Bool;
pub fn is_ascii_hex_digit(byte: UInt8) -> Bool;
pub fn is_ascii_whitespace(byte: UInt8) -> Bool;
pub fn is_ascii_control(byte: UInt8) -> Bool;

pub fn is_tchar(byte: UInt8) -> Bool;

pub fn to_ascii_lowercase(byte: UInt8) -> UInt8;
pub fn to_ascii_uppercase(byte: UInt8) -> UInt8;

pub fn eq_ignore_ascii_case(left: &[UInt8], right: &[UInt8]) -> Bool;
pub fn string_eq_ignore_ascii_case(left: &String, right: &String) -> Bool;
pub fn char_from_ascii(byte: UInt8) -> Option<Char>;
```
