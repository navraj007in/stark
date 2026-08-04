# stark-checksum

`stark-checksum` provides pure-STARK non-cryptographic checksum utilities.

Implemented:

- CRC32 using the reflected IEEE polynomial `0xEDB88320`;
- Adler-32;
- one-shot byte-slice APIs;
- incremental state APIs;
- lowercase fixed-width `UInt32` hex formatting.

This package makes no cryptographic integrity or authenticity claims.
