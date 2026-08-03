#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# **STARK's output contract is UTF-8 bytes; this script's stdout must be too.**
#
# The subprocess is already read as UTF-8, so `completed.stdout` is correct text. Writing it out is
# where it broke: Python encodes stdout using the CONSOLE's encoding, which on Windows is cp1252,
# and a STARK program that prints an emoji then dies here with UnicodeEncodeError -- in the script
# reporting the result, not in the program under test. Linux and macOS never showed it because
# their default is already UTF-8.
#
# `errors="replace"` rather than strict: this is a reporting path, and a byte it cannot render must
# not fail a qualification run that otherwise passed. The comparison against `expected_stdout`
# happens on the decoded text above, so substitution here cannot mask a real mismatch.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class PackageCase:
    package: str
    consumer: str
    expected_stdout: str | None
    # CD-347 — EXECUTED SURFACE. Names the resource types the package exposes, each of which must
    # have its acquire / use / close observed by a NATIVE consumer run.
    #
    # This exists because CD-345 found `stark-net` passing all seven steps while `connect`, `read`,
    # `write` and `close` had never been called by anything. The consumer only formatted addresses,
    # so a build-breaking defect (DEV-146) sat in the package undetected: nothing had ever lowered
    # a call into the raw bindings. A gate that a resource package can pass without exercising its
    # resources is not a gate for resource packages.
    #
    # Why a SEPARATE native consumer rather than folding it into the ordinary one: step 5 runs
    # `stark run`, and the interpreter has no provider layer — any consumer touching a bound
    # resource dies with "provider binding not lowered". So the resource exercise is native-only by
    # construction, and the split is forced by the toolchain, not chosen.
    #
    # Empty means "this package holds no host resources", which is the honest state for the ten
    # pure packages and must stay easy to declare.
    #
    # CD-348 — EXECUTED SURFACE, BY PACKAGE CATEGORY. The bar differs, and stating it per category
    # is what stops a future package satisfying "executed surface" through a path that only ever
    # runs an expected error:
    #
    #   pure package              the ordinary consumer executes each principal public behaviour
    #   function-shaped provider  the native consumer SUCCESSFULLY invokes each capability family
    #   resource-shaped provider  the native consumer SUCCESSFULLY acquires, uses and releases
    #                             every resource type -- both release paths, explicit and by drop
    #   failure-only environment  a deterministic negative path is allowed, but must be LABELLED
    #                             lowering/linking evidence, never lifecycle evidence
    #
    # `needs_echo_peer` / `needs_http_peer` are what make the resource category reachable: without
    # a live peer the consumer can only prove the failure path, which is the weaker claim.
    resources: tuple[str, ...] = ()
    resource_consumer: str | None = None
    resource_expected_stdout: str | None = None
    needs_echo_peer: bool = False
    needs_http_peer: bool = False
    # HC9. Three TLS peers with a controlled certificate chain, so `stark-tls`'s lifecycle evidence
    # covers a VERIFIED session and a rejected one. Certificates come from `stark-tls/fixtures`,
    # generated with absolute validity windows -- see that directory's `generate.sh`.
    needs_tls_peer: bool = False
    # A package whose ENTIRE surface requires a provider has no consumer that can run under step 5:
    # the interpreter has no provider layer, so `stark run` cannot reach any of it. Such a case
    # sets this, and its execution evidence comes from the resource block instead.
    #
    # This must not become a way to skip execution. It is only accepted alongside `resources` and a
    # `resource_consumer`, so an exempt package is executed MORE than an ordinary one -- natively,
    # against a live peer -- never less. Validated below rather than left to reviewer discipline,
    # because CD-345 is the record of what an unexecuted gate step costs.
    interpreter_exempt: bool = False
    # CD-355: public callables that CANNOT be called, mapped to the open defect that blocks them.
    # Not a convenience waiver — the check REFUSES an entry whose item has become callable, so a
    # fix to the underlying defect forces the entry out rather than letting it rot. Its purpose is
    # to make the cost of an open defect countable instead of invisible.
    surface_blocked: tuple[tuple[str, str], ...] = ()


CASES = [
    PackageCase(
        package="stark-json",
        consumer="stark-json-consumer",
        expected_stdout='{"name":"stark","items":[1,true,null],"unicode":"\U0001f600"}\n',
    ),
    PackageCase(
        package="stark-url",
        consumer="stark-url-consumer",
        expected_stdout="q=stark%20url&tag=compiler&tag=language&emoji=%F0%9F%98%80\n",
    ),
    PackageCase(
        package="stark-base64",
        consumer="stark-base64-consumer",
        expected_stdout="Zm9vYmFy\n",
    ),
    PackageCase(
        package="stark-hex",
        consumer="stark-hex-consumer",
        expected_stdout="48656c6c6f\n",
    ),
    PackageCase(
        package="stark-uuid",
        consumer="stark-uuid-consumer",
        expected_stdout="f81d4fae-7dec-11d0-a765-00a0c91e6bf6\n",
    ),
    # The HTTP substrate (CD-304), added CD-326. Nothing in CI had ever run these five — not
    # their tests, not `fmt --check`, not a native build — which is how three of them stayed
    # unformatted from the day they landed until CD-325, and how `stark-mime`, `stark-query`
    # and `stark-form` shipped with ZERO tests until CD-320.
    #
    # Their `expected_stdout` is a marker line rather than a computed result, because these
    # consumers are smoke tests for the package graph: what they prove is that the package
    # checks, tests, formats, and that its consumer runs identically through the interpreter
    # and as a native binary. The per-function behaviour is asserted by each package's own
    # `test_*` suite, which the `stark test` step above runs.
    PackageCase(
        package="stark-ascii",
        consumer="stark-ascii-consumer",
        expected_stdout="ASCII_CONSUMER_OK\n",
    ),
    PackageCase(
        package="stark-percent",
        consumer="stark-percent-consumer",
        expected_stdout="PERCENT_CONSUMER_OK\n",
    ),
    PackageCase(
        package="stark-mime",
        consumer="stark-mime-consumer",
        expected_stdout="MIME_CONSUMER_OK\n",
    ),
    PackageCase(
        package="stark-query",
        consumer="stark-query-consumer",
        expected_stdout="QUERY_CONSUMER_OK\n",
    ),
    PackageCase(
        package="stark-form",
        consumer="stark-form-consumer",
        expected_stdout="FORM_CONSUMER_OK\n",
    ),
    # HC5/HC6 — pure packages, so the ordinary consumer bar applies: each principal public
    # behaviour executed, no resources to acquire or release.
    PackageCase(
        package="stark-http-core",
        consumer="stark-http-core-consumer",
        expected_stdout="STARK_HTTP_CORE_CONSUMER_OK\n",
    ),
    PackageCase(
        package="stark-http-serialize",
        consumer="stark-http-serialize-consumer",
        expected_stdout="STARK_HTTP_SERIALIZE_CONSUMER_OK\n",
    ),
    PackageCase(
        package="stark-net",
        consumer="stark-net-consumer",
        expected_stdout="STARK_NET_CONSUMER_OK\n",
        resources=("TcpStream",),
        resource_consumer="stark-net-resource-consumer",
        resource_expected_stdout="STARK_NET_RESOURCE_OK\n",
        needs_echo_peer=True,
    ),
    PackageCase(
        package="stark-http-parser",
        consumer="stark-http-parser-consumer",
        expected_stdout=(
            "stark-http-parser consumer\n"
            "  identical result at every split: 98\n"
            '  status 200, body "hello world", Content-Type text/plain\n'
            "  rejected as expected: conflicting Content-Length headers\n"
        ),
    ),
    PackageCase(
        package="stark-http-client",
        consumer="stark-http-client-consumer",
        # The ordinary consumer IS the resource consumer here: every useful thing this package does
        # requires a socket, so there is no pure surface worth running under `stark run`. Step 5 is
        # therefore expected to be unreachable for it -- see `interpreter_exempt`.
        expected_stdout=None,
        interpreter_exempt=True,
        resource_consumer="stark-http-client-consumer",
        resource_expected_stdout=(
            "  fixed: 200, Content-Length framing, body and headers intact\n"
            "  chunked: 200, chunks decoded and joined\n"
            "  fragmented: 200, reassembled across several socket reads\n"
            "  close-early: reported as expected: the peer closed before the response completed\n"
            "  refused port: connect failure reported, no stream acquired\n"
            # HC10. Both the success and the two REFUSALS are named: a gate that only observed the
            # happy path would pass against a client that skipped verification entirely.
            "  https: 200 over a verified TLS 1.3 session, headers and body intact\n"
            "  https: chunked body decoded inside the TLS session\n"
            "  https: untrusted chain refused, reported as a TLS failure\n"
            "  https: hostname mismatch refused, though the chain itself is trusted\n"
            "  https: a cleartext peer on the secure path is refused\n"
            "  https: POST with a JSON body and a bearer token arrived intact\n"
            "STARK_HTTP_CLIENT_RESOURCE_OK\n"
        ),
        resources=("TcpStream", "TlsStream"),
        needs_http_peer=True,
    ),
    # HC9 — the first package built on a CROSS-PROVIDER TRANSFER (CD-360). Its consumer is also its
    # resource consumer, for the same reason `stark-http-client`'s is: every useful operation needs
    # a socket and a handshake, so there is no pure surface worth running under `stark run`.
    #
    # The expected output is the evidence, line by line. Both protocol versions are named because a
    # client that cannot tell 1.2 from 1.3 cannot claim either; the rejection line is there because
    # "it failed" is not the claim -- "it failed with the specific, actionable error" is.
    PackageCase(
        package="stark-tls",
        consumer="stark-tls-consumer",
        expected_stdout=None,
        interpreter_exempt=True,
        resources=("TlsStream",),
        resource_consumer="stark-tls-consumer",
        resource_expected_stdout=(
            "  tls: TLS 1.3 session verified, used and closed explicitly\n"
            "  tls: TLS 1.2 session verified, used and closed explicitly\n"
            "  tls: untrusted root rejected as certificate issuer is not trusted\n"
            "  tls: drop released the session and the socket under it\n"
            "STARK_TLS_RESOURCE_OK\n"
        ),
        needs_tls_peer=True,
    ),
]


ECHO_PORT = 39187


@contextlib.contextmanager
def echo_peer():
    """A loopback echo listener, so a resource consumer can complete a real lifecycle.

    CD-348. Without a peer, a TCP consumer can only prove that its surface lowers, links and
    starts -- `connect` fails, so nothing is acquired and `write`/`read`/`close`/drop-release are
    compiled but never executed. That is lowering evidence, not lifecycle evidence, and the
    distinction is exactly what CD-345 was about.

    Binding is asserted rather than attempted: if the port is taken, qualification FAILS loudly.
    Silently skipping would restore the weaker claim while the gate still reported success.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind(("127.0.0.1", ECHO_PORT))
    except OSError as exc:
        server.close()
        raise SystemExit(
            f"qualification cannot bind the echo peer on 127.0.0.1:{ECHO_PORT}: {exc}. "
            f"A resource package's lifecycle evidence needs a live peer; refusing to fall back "
            f"to a failure-only path, which would be lowering evidence reported as lifecycle "
            f"evidence (CD-348)."
        ) from exc
    server.listen(8)
    server.settimeout(30)
    stop = threading.Event()

    def serve():
        while not stop.is_set():
            try:
                conn, _addr = server.accept()
            except (TimeoutError, OSError):
                return
            with conn:
                try:
                    data = conn.recv(64)
                    if data:
                        conn.sendall(data)
                except OSError:
                    pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        server.close()
        thread.join(timeout=5)



# HC10. Lifted out of `http_peer` so the HTTPS peer serves the SAME four routes. Sharing the
# responder is what makes the cleartext and TLS cases comparable: any difference in what the client
# observes is the transport, not the fixture.
#
# `conn` is any object with `sendall` -- a plain socket or an `ssl.SSLSocket`.
def read_full_request(conn) -> bytes:
    """Head plus, if `Content-Length` says so, the body.

    Reading only to the blank line was enough while every route ignored the request. `/echo`
    reflects the body, so a peer that stopped at the head would reflect nothing and the POST case
    would assert against an empty string — passing for a client that never sent one.
    """
    request = b""
    while b"\r\n\r\n" not in request:
        chunk = conn.recv(4096)
        if not chunk:
            return request
        request += chunk
    head, _, body = request.partition(b"\r\n\r\n")
    length = 0
    for line in head.split(b"\r\n")[1:]:
        name, _, value = line.partition(b":")
        if name.lower() == b"content-length":
            try:
                length = int(value.strip())
            except ValueError:
                return request
    while len(body) < length:
        chunk = conn.recv(4096)
        if not chunk:
            break
        body += chunk
    return head + b"\r\n\r\n" + body


def respond_http_route(conn, target, request=b""):
    if target == "/fixed":
        body = b"fixed-body-ok"
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: "
            + str(len(body)).encode()
            + b"\r\n\r\n"
            + body
        )
    elif target == "/chunked":
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
            b"7\r\nchunked\r\n8\r\n-body-ok\r\n0\r\n\r\n"
        )
    elif target == "/fragmented":
        body = b"fragmented-body-reassembled"
        head = (
            b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: "
            + str(len(body)).encode()
            + b"\r\n\r\n"
        )
        # Split mid-header as well as mid-body: the head must survive fragmentation too.
        for piece in (head[:12], head[12:], body[:5], body[5:14], body[14:]):
            conn.sendall(piece)
            time.sleep(0.02)
    elif target == "/close-early":
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nonly-a-few-bytes"
        )
    elif target == "/echo":
        # HC10. Reflects what the peer actually RECEIVED, so the consumer can assert that its
        # method, its custom header and its body survived the transport rather than merely that a
        # response came back. A route returning a constant would pass for a client that sent
        # nothing at all.
        head, _, body = request.partition(b"\r\n\r\n")
        lines = head.split(b"\r\n")
        method = lines[0].split(b" ")[0] if lines else b"?"
        authorization = b"-"
        content_type = b"-"
        for line in lines[1:]:
            name, _, value = line.partition(b":")
            if name.lower() == b"authorization":
                authorization = value.strip()
            elif name.lower() == b"content-type":
                content_type = value.strip()
        reply = b"|".join([method, authorization, content_type, body])
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: "
            + str(len(reply)).encode()
            + b"\r\n\r\n"
            + reply
        )
    else:
        conn.sendall(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")


HTTP_PORT = 39188


@contextlib.contextmanager
def http_peer():
    """A loopback HTTP/1.1 peer, so `stark-http-client` can complete real request lifecycles.

    Four routes, each pinning a response shape the client must handle differently -- fixed,
    chunked, fragmented across several writes, and truncated by an early close. The fragmented and
    truncated routes are the ones that matter: they are what a client which assumes one recv() per
    response, or which treats a short body as complete, gets wrong.

    As with `echo_peer`, binding is asserted rather than attempted. A skipped peer would silently
    downgrade lifecycle evidence to lowering evidence while the gate still reported success.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind(("127.0.0.1", HTTP_PORT))
    except OSError as exc:
        server.close()
        raise SystemExit(
            f"qualification cannot bind the HTTP peer on 127.0.0.1:{HTTP_PORT}: {exc}. "
            f"A resource package's lifecycle evidence needs a live peer; refusing to fall back "
            f"to a failure-only path (CD-348)."
        ) from exc
    server.listen(8)
    server.settimeout(30)
    stop = threading.Event()

    def serve():
        while not stop.is_set():
            try:
                conn, _addr = server.accept()
            except (TimeoutError, OSError):
                return
            with conn:
                try:
                    conn.settimeout(10)
                    request = read_full_request(conn)
                    if not request:
                        continue
                    target = request.split(b" ")[1].decode("ascii", "replace")
                    respond_http_route(conn, target, request)
                except OSError:
                    pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        server.close()
        thread.join(timeout=5)


# HC10. The same four routes, behind TLS. `https_peers` serves the fixture chain for `stark.test`
# on one port and a chain to a root the client is NOT given on another.
HTTPS_PORT = 39192
HTTPS_UNTRUSTED_PORT = 39193


@contextlib.contextmanager
def https_peers(fixtures: Path):
    """Two loopback HTTPS peers, trusted and untrusted (HC10).

    Certificate verification is the property HTTPS exists for, so the gate must observe a rejection
    as well as a success. A run that only proved the happy path would pass just as well against a
    client that skipped verification entirely — the failure mode that matters, and the one that is
    invisible from outside.

    Binding is asserted, as with every other peer: a skipped peer would silently downgrade lifecycle
    evidence to lowering evidence while the gate still reported success (CD-348).
    """
    servers = []
    threads = []
    stop = threading.Event()

    def context(cert: str, key: str) -> ssl.SSLContext:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(fixtures / cert), str(fixtures / key))
        ctx.minimum_version = ssl.TLSVersion.TLSv1_3
        return ctx

    def handle(raw, ctx: ssl.SSLContext) -> None:
        try:
            raw.settimeout(15)
            with ctx.wrap_socket(raw, server_side=True) as tls:
                request = read_full_request(tls)
                if not request:
                    return
                target = request.split(b" ")[1].decode("ascii", "replace")
                respond_http_route(tls, target, request)
        except (OSError, ssl.SSLError, IndexError):
            # The untrusted peer and the hostname-mismatch case both end with the CLIENT aborting
            # the handshake. That is the expected outcome being measured, not a harness failure.
            pass

    def serve(server, ctx: ssl.SSLContext) -> None:
        while not stop.is_set():
            try:
                raw, _addr = server.accept()
            except (TimeoutError, OSError):
                return
            threading.Thread(target=handle, args=(raw, ctx), daemon=True).start()

    # `localhost` rather than `stark.test`: the HTTP client RESOLVES the URL's host, so a name
    # that resolves nowhere fails at DNS and never reaches the certificate check the case exists
    # for. `stark-tls`'s own consumer has no such constraint — it dials 127.0.0.1 and presents the
    # name separately — which is why the two use different fixtures.
    plan = [
        (HTTPS_PORT, "localhost.cert.pem", "localhost.key.pem"),
        (HTTPS_UNTRUSTED_PORT, "localhost-untrusted.cert.pem", "localhost-untrusted.key.pem"),
    ]
    try:
        for port, cert, key in plan:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                server.bind(("127.0.0.1", port))
            except OSError as exc:
                server.close()
                raise SystemExit(
                    f"qualification cannot bind the HTTPS peer on 127.0.0.1:{port}: {exc}. "
                    f"A resource package's lifecycle evidence needs a live peer; refusing to fall "
                    f"back to a failure-only path (CD-348)."
                ) from exc
            server.listen(8)
            server.settimeout(30)
            servers.append(server)
            thread = threading.Thread(target=serve, args=(server, context(cert, key)), daemon=True)
            thread.start()
            threads.append(thread)
        yield
    finally:
        stop.set()
        for server in servers:
            server.close()
        for thread in threads:
            thread.join(timeout=5)


@contextlib.contextmanager
def http_and_https_peers(fixtures: Path):
    """Both halves at once: `stark-http-client` exercises cleartext and TLS in ONE run.

    Nested rather than merged, so each peer keeps its own binding assertion and its own failure
    message. A combined implementation would report "the peer failed to bind" without saying which.
    """
    with http_peer():
        with https_peers(fixtures):
            yield


TLS_PORT_TLS13 = 39189
TLS_PORT_TLS12 = 39190
TLS_PORT_UNTRUSTED = 39191

@contextlib.contextmanager
def tls_peer(fixtures: Path):
    """Three loopback TLS peers with a controlled certificate chain (HC9).

    HC13 forbids qualification depending on public internet services, and HC9's claims are
    unreachable without a peer whose certificate the tester controls: nobody operates an
    untrusted-root endpoint for testing, and "TLS 1.3 was negotiated" means nothing unless another
    peer could have answered with 1.2.

        39189  a trusted chain, TLS 1.3 only
        39190  the same chain, TLS 1.2 only
        39191  a chain to a root the client does not trust

    The versions are PINNED rather than left to the library's defaults, which is what makes the
    consumer's version assertions deterministic across Python and OpenSSL releases.

    The certificates are checked in with absolute validity windows (`fixtures/generate.sh`), so this
    needs no `openssl` and no `cryptography` module -- only the standard library's `ssl`.

    As with the other peers, binding is asserted rather than attempted. A skipped peer would
    silently downgrade lifecycle evidence to lowering evidence while the gate still reported
    success (CD-348).
    """
    servers = []
    threads = []
    stop = threading.Event()

    def context(cert: str, key: str, version) -> ssl.SSLContext:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(fixtures / cert), str(fixtures / key))
        ctx.minimum_version = version
        ctx.maximum_version = version
        return ctx

    def recv_exactly(sock, count: int) -> bytes | None:
        out = b""
        while len(out) < count:
            chunk = sock.recv(count - len(out))
            if not chunk:
                return None
            out += chunk
        return out

    def handle(raw, ctx: ssl.SSLContext) -> None:
        try:
            raw.settimeout(15)
            with ctx.wrap_socket(raw, server_side=True) as tls:
                while True:
                    head = recv_exactly(tls, 8)
                    if head is None:
                        return
                    body = recv_exactly(tls, int.from_bytes(head, "big"))
                    if body is None:
                        return
                    tls.sendall(head)
                    tls.sendall(body)
        except (OSError, ssl.SSLError):
            # An untrusted-root peer sees the client abort the handshake. That is the expected
            # outcome for 39191, not a harness failure.
            pass

    def serve(server, ctx: ssl.SSLContext) -> None:
        while not stop.is_set():
            try:
                raw, _addr = server.accept()
            except (TimeoutError, OSError):
                return
            threading.Thread(target=handle, args=(raw, ctx), daemon=True).start()

    plan = [
        (TLS_PORT_TLS13, "server.cert.pem", "server.key.pem", ssl.TLSVersion.TLSv1_3),
        (TLS_PORT_TLS12, "server.cert.pem", "server.key.pem", ssl.TLSVersion.TLSv1_2),
        (TLS_PORT_UNTRUSTED, "untrusted.cert.pem", "untrusted.key.pem", ssl.TLSVersion.TLSv1_3),
    ]
    try:
        for port, cert, key, version in plan:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                server.bind(("127.0.0.1", port))
            except OSError as exc:
                server.close()
                raise SystemExit(
                    f"qualification cannot bind the TLS peer on 127.0.0.1:{port}: {exc}. "
                    f"A resource package's lifecycle evidence needs a live peer; refusing to fall "
                    f"back to a failure-only path (CD-348)."
                ) from exc
            server.listen(8)
            server.settimeout(30)
            servers.append(server)
            thread = threading.Thread(
                target=serve, args=(server, context(cert, key, version)), daemon=True
            )
            thread.start()
            threads.append(thread)
        yield
    finally:
        stop.set()
        for server in servers:
            server.close()
        for thread in threads:
            thread.join(timeout=5)


def run(cmd: list[str], cwd: Path, expected_stdout: str | None = None) -> None:
    label = f"{cwd.name}: {' '.join(cmd)}"
    print(f"::group::{label}", flush=True)
    try:
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        sys.stdout.write(completed.stdout)
        if completed.returncode != 0:
            raise SystemExit(f"{label} failed with exit status {completed.returncode}")
        if expected_stdout is not None and completed.stdout != expected_stdout:
            raise SystemExit(
                f"{label} stdout mismatch\n"
                f"expected: {expected_stdout!r}\n"
                f"actual:   {completed.stdout!r}"
            )
    finally:
        print("::endgroup::", flush=True)


def strip_stark_comments(text: str) -> str:
    """Remove comments so a NAME MENTIONED IN PROSE never counts as a call site."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def declared_surface(stark: Path, package_dir: Path) -> list[dict]:
    """The package's public surface, as the COMPILER sees it.

    `stark doc` walks the AST, so this cannot drift from the source the way a regex over `pub fn`
    would. It is also why DEV-152 had to be fixed first: until it was, `impl` methods on a type
    with no page-level item were silently dropped from the doc index, so a surface check built on
    it would have certified `stark-net`'s seven `TcpStream` methods as fully covered while none of
    them had ever been called.
    """
    with tempfile.TemporaryDirectory() as tmp:
        run([str(stark), "doc", "--output", tmp], package_dir)
        with open(Path(tmp) / "search.json") as handle:
            return json.load(handle)


def check_declared_surface_is_called(stark: Path, repo_root: Path, case: "PackageCase") -> None:
    """CD-355: every public callable must be CALLED by the package's own tests or its consumers.

    The gate's other steps prove a package builds and its consumer runs. None of them proves that
    what a package DECLARES is reached by anything. That gap has now cost three separate stretches:

      * CD-345 -- `stark-net` passed all seven steps while `connect`/`read`/`write`/`close` had
        never been called, hiding a build-breaking defect (DEV-146);
      * CD-347 -- fixed that for resource LIFECYCLES, by requiring a native consumer;
      * CD-354 (DEV-151) -- the same failure one level in. `set_read_timeout` was declared under
        CD-346, qualified, documented, and unbuildable at every call site, because nothing had
        ever called it.

    Each round closed the instance and left the class open. This closes the class: a declared
    callable that nothing calls fails qualification.

    **The bar is the package's OWN evidence** -- its tests plus its own consumers -- not "called by
    something, anywhere in the tree". A package must stand on its own: a downstream caller can be
    deleted, and it proves nothing about the package in isolation anyway.

    **Matching is textual, and deliberately biased toward FALSE PASSES.** An identifier followed by
    `(` counts as a call. That cannot see through an alias or a generic dispatch, so it may credit
    a call that never happens -- but comments are stripped first, so it will not credit prose. The
    bias is chosen: a false FAILURE would force a package to add a fake call to satisfy the gate,
    which is worse than a missed one, because it teaches that gate output is noise.
    """
    entries = declared_surface(stark, repo_root / case.package)

    sources = [repo_root / case.package / "src"]
    for consumer in (case.consumer, case.resource_consumer):
        if consumer:
            sources.append(repo_root / consumer / "src")
    haystack = "\n".join(
        strip_stark_comments(path.read_text(encoding="utf-8"))
        for source in sources
        if source.is_dir()
        for path in sorted(source.rglob("*.stark"))
    )

    uncalled = []
    for entry in entries:
        name = entry["name"]
        if entry["kind"] == "fn":
            if not re.search(r"\b" + re.escape(name) + r"\s*\(", haystack):
                uncalled.append(name)
        elif entry["kind"] == "method":
            short = name.split("::")[-1]
            # `.name(` for a receiver method, `Type::name(` for an associated function.
            called = re.search(r"[.]\s*" + re.escape(short) + r"\s*\(", haystack) or re.search(
                r"\b" + re.escape(name) + r"\s*\(", haystack
            )
            if not called:
                uncalled.append(name)

    callable_count = sum(1 for e in entries if e["kind"] in ("fn", "method"))

    # A blocked entry whose item is now CALLED is a stale waiver, and stale waivers are how a gate
    # rots into decoration. Refused in the same breath as an uncalled item.
    blocked = dict(case.surface_blocked)
    stale = sorted(name for name in blocked if name not in uncalled)
    if stale:
        listing = "\n".join(f"      {name}" for name in stale)
        raise SystemExit(
            f"{case.package}: these are recorded as blocked, but are now called:\n{listing}\n\n"
            f"    The defect that blocked them is fixed. Remove the `surface_blocked` entries in "
            f"the same change, or the next uncalled item hides behind a waiver nobody rereads."
        )
    for name in sorted(set(uncalled) & set(blocked)):
        print(f"  surface: {name} UNCALLABLE — blocked by {blocked[name]}", flush=True)
    uncalled = [name for name in uncalled if name not in blocked]

    if uncalled:
        listing = "\n".join(f"      {name}" for name in uncalled)
        raise SystemExit(
            f"{case.package}: {len(uncalled)} of {callable_count} public callables are never "
            f"called by the package's own tests or its consumers:\n{listing}\n\n"
            f"    Declared-but-uncalled is where DEV-146 and DEV-151 both hid: each was accepted, "
            f"qualified and shipped, and neither could be BUILT at a call site. Either exercise "
            f"these, or remove them -- an API nothing calls is not evidence of anything."
        )
    print(f"  surface: {callable_count} public callables, all called", flush=True)



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stark", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[2], type=Path)
    parser.add_argument("--exe-suffix", default=".exe" if os.name == "nt" else "")
    args = parser.parse_args()

    stark = args.stark.resolve()
    repo_root = args.repo_root.resolve()

    for case in CASES:
        package_dir = repo_root / case.package
        consumer_dir = repo_root / case.consumer
        run([str(stark), "check"], package_dir)
        run([str(stark), "test"], package_dir)
        check_declared_surface_is_called(stark, repo_root, case)
        run([str(stark), "fmt", "--check"], package_dir)
        run([str(stark), "check"], consumer_dir)
        if case.interpreter_exempt:
            if not case.resources or case.resource_consumer is None:
                raise SystemExit(
                    f"{case.package} sets interpreter_exempt without declaring resources and a "
                    f"resource_consumer. The exemption is only sound when native execution "
                    f"replaces the skipped step; on its own it would skip execution outright, "
                    f"which is the CD-345 hole."
                )
            print(
                f"  step 5 (`stark run`) skipped for {case.consumer}: every operation in "
                f"{case.package} requires a provider, and the interpreter has no provider layer. "
                f"Execution evidence comes from the native resource run below.",
                flush=True,
            )
        else:
            run([str(stark), "run"], consumer_dir, expected_stdout=case.expected_stdout)
            run([str(stark), "build", "--no-build-cache"], consumer_dir)
            artifact = (
                consumer_dir / "target" / "stark" / "debug" / f"{case.consumer}{args.exe_suffix}"
            )
            run([str(artifact)], consumer_dir, expected_stdout=case.expected_stdout)

        # CD-347: the executed-surface requirement. A package that declares resources must ship a
        # native consumer that acquires, uses and closes each one. `stark run` is deliberately NOT
        # part of this sequence -- the interpreter has no provider layer.
        if case.resources:
            if case.resource_consumer is None:
                raise SystemExit(
                    f"{case.package} declares resources {case.resources} but names no "
                    f"resource_consumer. A resource package must exercise acquire/use/close "
                    f"natively; see CD-345 for what a happy-path-only gate concealed."
                )
            resource_dir = repo_root / case.resource_consumer
            if not resource_dir.is_dir():
                raise SystemExit(
                    f"{case.package}: resource consumer {case.resource_consumer} does not exist"
                )
            run([str(stark), "check"], resource_dir)
            run([str(stark), "fmt", "--check"], resource_dir)
            run([str(stark), "build", "--no-build-cache"], resource_dir)
            resource_artifact = (
                resource_dir
                / "target"
                / "stark"
                / "debug"
                / f"{case.resource_consumer}{args.exe_suffix}"
            )
            if case.needs_echo_peer or case.needs_http_peer or case.needs_tls_peer:
                # The peer is started AFTER the build and torn down after the run: the build takes
                # seconds, and a listener held open across it is a socket kept for no reason.
                # The fixtures directory comes from `--repo-root`, not from this file's own
                # location: every other path in this script does, and deriving one of them
                # differently is how a copy of the script run from elsewhere half-works.
                fixtures = repo_root / "stark-tls" / "fixtures"
                if case.needs_tls_peer:
                    peer = lambda: tls_peer(fixtures)
                elif case.needs_http_peer:
                    # HC10: the HTTP client now exercises cleartext AND TLS in one run.
                    peer = lambda: http_and_https_peers(fixtures)
                else:
                    peer = echo_peer
                with peer():
                    run(
                        [str(resource_artifact)],
                        resource_dir,
                        expected_stdout=case.resource_expected_stdout,
                    )
            else:
                run(
                    [str(resource_artifact)],
                    resource_dir,
                    expected_stdout=case.resource_expected_stdout,
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
