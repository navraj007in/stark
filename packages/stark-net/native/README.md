# stark-net native provider

Standalone blocking TCP provider for Native Provider ABI v0.1.

The first implementation accepts explicit socket address strings. Loopback IPv4
(`127.0.0.1:port`) is mandatory and covered by tests. The provider does not choose hidden bind
addresses, does not default to `0.0.0.0`, does not read addresses from the environment, and does
not automatically select public interfaces. Hostname resolution is left to `std::net` only when
the explicit address string requires it.

Reads and writes use caller-owned buffers. Writes return bytes accepted and do not implement
package-level `write_all` retry loops. Resource handles expose only ABI resource IDs, never raw OS
socket handles.
