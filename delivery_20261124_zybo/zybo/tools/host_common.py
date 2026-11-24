"""Shared helpers for the host-side scripts of req_1120 part Z (Python 3.10+)."""
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


class Remote(object):
    """Runs commands on the Zybo over SSH and keeps a JSONL record of every call."""

    def __init__(self, host="xilinx@192.168.1.2", log_dir=None, connect_timeout=6):
        self.host = host
        self.connect_timeout = connect_timeout
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def _ssh(self, argv, timeout):
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=%d" % self.connect_timeout,
                "-o", "ServerAliveInterval=15", self.host, argv], timeout

    def _record(self, kind, argv, proc, started, log_name):
        if not self.log_dir:
            return
        entry = {"utc": utc(), "kind": kind, "argv": argv, "returncode": proc.returncode,
                 "duration_s": round(time.monotonic() - started, 3)}
        if log_name:
            (self.log_dir / log_name).write_text(proc.stdout + proc.stderr)
            entry["output_log"] = log_name
        with (self.log_dir / "commands.jsonl").open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def run(self, argv, log_name=None, timeout=60, check=True, sudo=False):
        """argv is a list executed on the board without a shell (sudo -n when sudo=True)."""
        remote_argv = (["sudo", "-n"] if sudo else []) + [str(a) for a in argv]
        cmd, t = self._ssh(shlex.join(remote_argv), timeout)
        started = time.monotonic()
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=t)
        self._record("ssh", remote_argv, proc, started, log_name)
        if check and proc.returncode:
            raise RuntimeError("remote command failed ({}): {}\n{}".format(
                remote_argv[0], shlex.join(remote_argv), (proc.stdout + proc.stderr)[-4000:]))
        return proc

    def get(self, argv, timeout=60, sudo=False, log_name=None):
        return self.run(argv, timeout=timeout, sudo=sudo, log_name=log_name).stdout

    def shell(self, command, log_name=None, timeout=60, check=True):
        """Runs one shell string on the board (needed for redirections and background processes)."""
        cmd, t = self._ssh(command, timeout)
        started = time.monotonic()
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=t)
        self._record("ssh_shell", command, proc, started, log_name)
        if check and proc.returncode:
            raise RuntimeError("remote shell command failed ({}): {}\n{}".format(
                proc.returncode, command, (proc.stdout + proc.stderr)[-4000:]))
        return proc

    def put(self, sources, destination_dir, timeout=120):
        started = time.monotonic()
        proc = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=%d" % self.connect_timeout,
                               *map(str, sources), self.host + ":" + str(destination_dir) + "/"],
                              text=True, capture_output=True, timeout=timeout)
        self._record("scp_put", [Path(s).name for s in sources] + ["->", str(destination_dir)], proc, started, None)
        if proc.returncode:
            raise RuntimeError("scp failed: " + proc.stderr)

    def get_dir(self, remote_dir, local_dir, timeout=600):
        """Copies remote_dir into local_dir (scp -r; the board runs OpenSSH 7.6)."""
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        proc = subprocess.run(["scp", "-q", "-r", "-o", "BatchMode=yes", "-o", "ConnectTimeout=%d" % self.connect_timeout,
                               self.host + ":" + str(remote_dir), str(local_dir)],
                              text=True, capture_output=True, timeout=timeout)
        self._record("scp_get", [str(remote_dir), "->", str(local_dir)], proc, started, None)
        if proc.returncode:
            raise RuntimeError("scp failed: " + proc.stderr)
        return local_dir / Path(remote_dir).name


def ssh_ok(host="xilinx@192.168.1.2", timeout=4):
    """TCP-level reachability of the board's SSH port (no credentials needed)."""
    import socket
    try:
        with socket.create_connection((host.split("@")[-1], 22), timeout=timeout):
            return True
    except OSError:
        return False
