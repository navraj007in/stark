#!/usr/bin/env python3
"""The one mapping from `c78/capabilities.toml` to a qualification record's capability rows.

This exists because it was written twice. `make-record.py` built the rows and `compare-records.py`
rebuilt the *expected* rows from the same manifest, so the comparison only held while two
hand-maintained dicts stayed identical. CD-365 added TLS to the first and not the second, and the
qualification job failed with `linux-x64 capabilities differ from manifest` — a mismatch between
the compiler's own two copies, saying nothing about the platform it named.

Adding a capability row is now one edit here, and the producer and the checker cannot disagree.
"""


def capability_summary(manifest):
    return {
        "time": manifest["stark_time"]["provider_metadata"],
        "args_env": manifest["stark_env"]["provider_metadata"],
        "file": manifest["stark_file"]["provider_metadata"],
        "tcp": manifest["stark_net"]["loopback_provider"],
        "stark_time_e2e": manifest["stark_time"]["native_e2e"],
        "args_env_e2e": manifest["stark_env"]["native_e2e"],
        "file_e2e": manifest["stark_file"]["native_e2e"],
        "tcp_e2e": manifest["stark_net"]["native_e2e"],
        # HC9 (CD-365). `tls_transfer` is a SEPARATE row from `tls_e2e` deliberately: "TLS works"
        # and "a resource crossed from one provider to another and was released exactly once" are
        # different claims, and CD-360 left the second one open for HC9 to close.
        "tls": manifest["stark_tls"]["provider_metadata"],
        "tls_e2e": manifest["stark_tls"]["native_e2e"],
        "tls_transfer": manifest["stark_tls"]["cross_provider_transfer"],
    }
