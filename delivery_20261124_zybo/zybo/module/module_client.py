#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z6 host client: forwards ONE request to the Zybo root module and prints the board's reply.

  python3 module_client.py [--host 192.168.1.2] [--port 3041] [--timeout 120] < request.json

Reads one line of JSON from stdin, sends it to the deployment service over TCP, reads one line
back and writes exactly that line to stdout. Exit status 0 after the reply was printed.
If the board cannot be reached or does not answer with one JSON line, nothing is written to
stdout, a short reason goes to stderr and the exit status is 1.

The link is Ethernet (TCP). The request (and for unlock therefore the passphrase) travels only
through stdin, the socket and the direct cable; it never appears in argv, a file or a log.
Python 3.6+ with no third-party packages; runs on the Jetson, on the PC or on the board itself.
"""
import argparse
import json
import socket
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="192.168.1.2")
    ap.add_argument("--port", type=int, default=3041)
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds to wait for the reply")
    args = ap.parse_args()
    request = sys.stdin.buffer.read(1 << 20).strip(b"\r\n")
    if not request or b"\n" in request or b"\r" in request or len(request) > 16384:
        sys.stderr.write("module_client: stdin must hold exactly one short JSON line\n")
        return 1
    try:
        json.loads(request.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        sys.stderr.write("module_client: the request line is not JSON\n")
        return 1
    buf = bytearray()
    try:
        conn = socket.create_connection((args.host, args.port), timeout=min(args.timeout, 10.0))
        try:
            conn.settimeout(args.timeout)
            conn.sendall(request + b"\n")
            while b"\n" not in buf:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf.extend(chunk)
                if len(buf) > 65536:
                    break
        finally:
            conn.close()
    except (OSError, socket.timeout) as exc:
        sys.stderr.write("module_client: could not reach %s:%d (%s)\n" % (args.host, args.port, type(exc).__name__))
        return 1
    reply = bytes(buf).split(b"\n", 1)[0].strip(b"\r")
    try:
        obj = json.loads(reply.decode("utf-8"))
        if not isinstance(obj, dict):
            raise ValueError
    except (UnicodeDecodeError, ValueError):
        sys.stderr.write("module_client: the board did not answer with one JSON line\n")
        return 1
    sys.stdout.buffer.write(reply + b"\n")
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
