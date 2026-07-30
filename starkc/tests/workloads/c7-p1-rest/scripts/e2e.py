#!/usr/bin/env python3
"""Raw-socket functional qualification for the bounded C7 P1 REST server."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    request: bytes
    status: int
    reason: str
    body: bytes
    split: bool = False


def response(status: int, reason: str, body: bytes) -> bytes:
    return (
        f"HTTP/1.1 {status} {reason}\r\n"
        "Content-Type: application/json\r\n"
        "Connection: close\r\n"
        f"Content-Length: {len(body)}\r\n\r\n"
    ).encode("ascii") + body


def request(method: str, path: str, body: bytes = b"", headers: bytes = b"") -> bytes:
    return (
        f"{method} {path} HTTP/1.1\r\nHost: localhost\r\n".encode("ascii")
        + headers
        + b"\r\n"
        + body
    )


def post(body: bytes, *, split: bool = False) -> Case:
    raw = request(
        "POST",
        "/items",
        body,
        f"Content-Length: {len(body)}\r\n".encode("ascii"),
    )
    name_token = body[len(b'{"name":') : -1]
    return Case(raw, 201, "Created", b'{"id":3,"name":' + name_token + b"}", split)


def cases() -> list[Case]:
    bad = b'{"error":"bad_request"}'
    not_found = b'{"error":"not_found"}'
    corpus = [
        Case(request("GET", "/health", headers=b"Connection: close\r\n"), 200, "OK", b'{"status":"ok"}'),
        Case(request("GET", "/health"), 200, "OK", b'{"status":"ok"}'),
        Case(request("GET", "/items/1"), 200, "OK", b'{"id":1,"name":"alpha"}'),
        Case(request("GET", "/items/2"), 200, "OK", b'{"id":2,"name":"beta"}'),
        Case(request("GET", "/items/999"), 404, "Not Found", not_found),
        Case(request("GET", "/items/nope"), 400, "Bad Request", bad),
        Case(request("GET", "/items/"), 400, "Bad Request", bad),
        Case(request("GET", "/items/-1"), 400, "Bad Request", bad),
        Case(request("GET", "/items/18446744073709551616"), 400, "Bad Request", bad),
        post(b'{"name":"gamma"}', split=True),
        post(b'{"name":"a\\n\\"b"}'),
        post('{"name":"café"}'.encode()),
    ]
    for body in (
        b'{"name":}',
        b'{"other":1}',
        b'{"name":"a","name":"b"}',
        b'{"name":3}',
        b'{"name":""}',
    ):
        corpus.append(
            Case(
                request(
                    "POST",
                    "/items",
                    body,
                    f"Content-Length: {len(body)}\r\n".encode("ascii"),
                ),
                400,
                "Bad Request",
                bad,
            )
        )
    corpus.extend(
        [
            Case(request("POST", "/items", b'{"name":"x"}'), 400, "Bad Request", bad),
            Case(
                request(
                    "POST",
                    "/items",
                    b'{"name":"x"}',
                    b"Content-Length: 12\r\nContent-Length: 13\r\n",
                ),
                400,
                "Bad Request",
                bad,
            ),
            Case(
                request("POST", "/items", b"0\r\n\r\n", b"Transfer-Encoding: chunked\r\n"),
                400,
                "Bad Request",
                bad,
            ),
            Case(request("GET", "/unknown"), 404, "Not Found", not_found),
            Case(
                request("PUT", "/health"),
                405,
                "Method Not Allowed",
                b'{"error":"method_not_allowed"}',
            ),
            Case(
                b"GET /health HTTP/1.1\r\nConnection: close\r\n\r\n",
                400,
                "Bad Request",
                bad,
            ),
            Case(
                b"GET /health HTTP/1.1\r\nhost: localhost\r\n\r\n",
                200,
                "OK",
                b'{"status":"ok"}',
            ),
        ]
    )
    assert len(corpus) == 24
    return corpus


def reserve_address() -> tuple[str, int]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()


def connect(address: tuple[str, int], process: subprocess.Popen[bytes]) -> socket.socket:
    deadline = time.monotonic() + 15
    while True:
        if process.poll() is not None:
            _, stderr = process.communicate()
            raise RuntimeError(
                f"server exited before accept: {process.returncode}; "
                f"stderr={stderr.decode(errors='replace')}"
            )
        try:
            return socket.create_connection(address, timeout=5)
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def run(binary: Path) -> None:
    address = reserve_address()
    environment = os.environ.copy()
    environment["STARK_P1_BIND"] = f"{address[0]}:{address[1]}"
    process = subprocess.Popen(
        [str(binary)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        for index, case in enumerate(cases(), start=1):
            with connect(address, process) as client:
                if case.split:
                    at = len(case.request) - 5
                    client.sendall(case.request[:at])
                    time.sleep(0.01)
                    client.sendall(case.request[at:])
                else:
                    client.sendall(case.request)
                client.shutdown(socket.SHUT_WR)
                chunks: list[bytes] = []
                while True:
                    try:
                        chunk = client.recv(65536)
                    except ConnectionResetError:
                        break
                    if not chunk:
                        break
                    chunks.append(chunk)
            actual = b"".join(chunks)
            expected = response(case.status, case.reason, case.body)
            if actual != expected:
                process.kill()
                stdout, stderr = process.communicate()
                raise AssertionError(
                    f"case {index} mismatch\n"
                    f"request={case.request!r}\nexpected={expected!r}\nactual={actual!r}\n"
                    f"stdout={stdout!r}\nstderr={stderr!r}"
                )
        return_code = process.wait(timeout=10)
        if return_code != 0:
            _, stderr = process.communicate()
            raise RuntimeError(
                f"bounded server exit {return_code}: {stderr.decode(errors='replace')}"
            )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(__file__).parents[1] / "target/stark/debug/c7-p1-rest",
    )
    args = parser.parse_args()
    run(args.binary.resolve())
    print("c7-p1 e2e: 24/24 raw HTTP cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
