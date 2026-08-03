#!/usr/bin/env bash
# HC9 — the controlled certificate fixtures every TLS negative test needs.
#
# **These are CHECKED IN, and regenerating them is deliberate work.** A qualification run must not
# depend on `openssl` being installed, on its version, or on the wall clock at generation time: an
# expired certificate generated relative to "now" stops being a useful fixture the moment someone
# regenerates it on a machine whose clock is wrong, and a not-yet-valid one silently becomes valid
# when its date arrives. Every validity window below is therefore an ABSOLUTE date, chosen so the
# intended verdict holds for two decades.
#
# CD-361: HC9's fixture uses `ExplicitRoots` containing `ca.cert.pem`. Nothing here touches the
# machine's trust store, and nothing here is trusted by anything outside this directory.
#
# Regenerate with:
#     ./generate.sh
#
# and commit the result. The private keys are test-only, generated here, and secret from nobody.
set -euo pipefail
cd "$(dirname "$0")"

# ECDSA P-256: supported by every rustls profile including aws-lc-rs FIPS, and two orders of
# magnitude faster to generate than RSA-2048, which matters when a CI run does it from scratch.
key() { openssl ecparam -name prime256v1 -genkey -noout -out "$1"; }

# A CA, self-signed over an absolute window.
#   $1 basename   $2 CN   $3 notBefore   $4 notAfter
ca() {
  key "$1.key.pem"
  openssl req -new -x509 -key "$1.key.pem" -out "$1.cert.pem" \
    -subj "/CN=$2" -sha256 \
    -not_before "$3" -not_after "$4" \
    -addext "basicConstraints=critical,CA:TRUE" \
    -addext "keyUsage=critical,keyCertSign,cRLSign"
}

# A leaf, signed by a CA, over an absolute window.
#   $1 basename  $2 CN  $3 SAN dns  $4 issuer basename  $5 notBefore  $6 notAfter
leaf() {
  key "$1.key.pem"
  openssl req -new -key "$1.key.pem" -out "$1.csr.pem" -subj "/CN=$2"
  openssl x509 -req -in "$1.csr.pem" -out "$1.cert.pem" \
    -CA "$4.cert.pem" -CAkey "$4.key.pem" -set_serial "0x$(openssl rand -hex 8)" -sha256 \
    -not_before "$5" -not_after "$6" \
    -extfile <(printf 'basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\nsubjectAltName=DNS:%s\n' "$3")
  rm -f "$1.csr.pem"
}

# An intermediate CA, signed by a root.
#   $1 basename  $2 CN  $3 issuer basename  $4 notBefore  $5 notAfter
intermediate() {
  key "$1.key.pem"
  openssl req -new -key "$1.key.pem" -out "$1.csr.pem" -subj "/CN=$2"
  openssl x509 -req -in "$1.csr.pem" -out "$1.cert.pem" \
    -CA "$3.cert.pem" -CAkey "$3.key.pem" -set_serial "0x$(openssl rand -hex 8)" -sha256 \
    -not_before "$4" -not_after "$5" \
    -extfile <(printf 'basicConstraints=critical,CA:TRUE,pathlen:0\nkeyUsage=critical,keyCertSign,cRLSign\n')
  rm -f "$1.csr.pem"
}

VALID_FROM="20260101000000Z"
VALID_TO="20460101000000Z"

# The trust anchor HC9's `ExplicitRoots` policy is given.
ca ca "STARK HC9 Test Root CA" "$VALID_FROM" "$VALID_TO"

# A root nothing trusts: the "untrusted certificate" case is a well-formed chain to the WRONG
# anchor, which is the realistic failure. A malformed certificate would test the parser instead.
ca rogue-ca "STARK HC9 Rogue Root CA" "$VALID_FROM" "$VALID_TO"

# The happy path. `stark.test` is a reserved-for-testing name that resolves nowhere: the consumer
# connects to 127.0.0.1 over TCP and presents this name for SNI and verification, so hostname
# checking is exercised without depending on a resolver.
leaf server           "stark.test" "stark.test" ca "$VALID_FROM" "$VALID_TO"

# The four negative leaves. Each differs from `server` in EXACTLY ONE property, so a test that
# fails names one cause.
leaf expired          "stark.test" "stark.test" ca       "20200101000000Z" "20210101000000Z"
leaf not-yet-valid    "stark.test" "stark.test" ca       "20400101000000Z" "20450101000000Z"
leaf wrong-host       "other.test" "other.test" ca       "$VALID_FROM" "$VALID_TO"
leaf untrusted        "stark.test" "stark.test" rogue-ca "$VALID_FROM" "$VALID_TO"

# HC10. `localhost` leaves, for the HTTP client.
#
# `stark-tls`'s own consumer dials 127.0.0.1 and presents `stark.test` separately, so it needs no
# resolvable name. The HTTP CLIENT resolves the URL's host, so its fixture must use a name that
# actually resolves — otherwise DNS fails before TLS is ever reached and the test proves nothing
# about certificates.
leaf localhost           "localhost" "localhost" ca       "$VALID_FROM" "$VALID_TO"
leaf localhost-untrusted "localhost" "localhost" rogue-ca "$VALID_FROM" "$VALID_TO"

# A chain the server can serve INCOMPLETE: root -> intermediate -> leaf. Presenting only the leaf
# must fail, because the verifier cannot build a path to the root without the intermediate.
intermediate intermediate-ca "STARK HC9 Test Intermediate CA" ca "$VALID_FROM" "$VALID_TO"
leaf chained "stark.test" "stark.test" intermediate-ca "$VALID_FROM" "$VALID_TO"

# The full chain, for the control case: the same leaf VERIFIES when the intermediate is sent.
cat chained.cert.pem intermediate-ca.cert.pem > chained-fullchain.cert.pem

echo "fixtures regenerated:"
for f in *.cert.pem; do
  printf '  %-32s %s\n' "$f" "$(openssl x509 -in "$f" -noout -subject -dates | tr '\n' ' ')"
done
