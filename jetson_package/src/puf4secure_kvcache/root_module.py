"""Root module: the device root never leaves it.

The module holds the PUF, the fuzzy extractor and the key derivation behind one
interface and answers three requests from the host:

* ``unlock(passphrase)`` reconstructs the root from the PUF, checks the tag that
  binds the root to the public helper data, derives the unlock key by iterating
  HMAC-SHA256 under the root, starting from the passphrase, and erases the root;
* ``derive(label)`` returns HMAC-SHA256(unlock key, label), one key per store;
* ``lock()`` erases the unlock key.

The host sees only derived keys, never the root. An adversary who holds the
device and reads everything it stores can therefore test a passphrase only by
asking this module, and every guess costs one unlock: the iteration count sets
how long. A failed reconstruction raises ``ReconstructionError``, which the
cold tier treats as a cache miss.

``SimulatedRootModule`` stands in for the hardware in the cold-tier experiments.
``CommandRootModule`` forwards the three requests to a hardware module through a
configured client command; the passphrase travels on stdin, never in argv.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import subprocess
import time
from pathlib import Path

from .puf_sim import PUFSim, ReconstructionError, _device_root

UNLOCK_DOMAIN = b"BoundKV-unlock|"
TAG_DOMAIN = b"BoundKV-tag|"
DEFAULT_ITERATIONS = 1000        # tests; a deployment calibrates its own count


class ModuleLocked(RuntimeError):
    """A key was requested while the module holds no unlock key."""


def _f(key: bytes, msg: bytes) -> bytes:
    return hmac.new(key, msg, hashlib.sha256).digest()


def unlock_key(root: bytes, passphrase: bytes, iterations: int) -> bytes:
    """z_0 = F_r(unlock ‖ passphrase), z_j = F_r(z_{j-1}); returns z_iterations.

    Every step is keyed by the root, so the chain runs only where the root is,
    and its steps are sequential, so a guess cannot be spread over cores.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    z = _f(root, UNLOCK_DOMAIN + passphrase)
    for _ in range(iterations):
        z = _f(root, z)
    return z


def binding_tag(root: bytes, helper: bytes) -> bytes:
    """F_r(tag ‖ helper): changing the helper data fails the check whatever the
    response, so a modified helper reveals nothing about the response."""
    return _f(root, TAG_DOMAIN + helper)


def iterations_for(seconds: float, probe: int = 20000) -> int:
    """The iteration count that makes one unlock take ``seconds`` here."""
    root = bytes(32)
    start = time.perf_counter()
    unlock_key(root, b"probe", probe)
    rate = probe / (time.perf_counter() - start)
    return max(1, int(rate * seconds))


class SimulatedRootModule:
    """The module interface over a simulated PUF root (``puf_sim``).

    ``helper`` and ``tag`` are the public enrollment data the device stores; the
    adversary may read and replace both. ``mode``, ``ber`` and the other keyword
    arguments select the simulated response exactly as in ``PUFSim``:
    ``noisy`` within the correction budget reconstructs the root, beyond it
    raises, and ``wrong_device`` puts this enrollment on another chip.
    """

    def __init__(self, device_id: str, *, iterations: int = DEFAULT_ITERATIONS,
                 **puf_kwargs):
        self.device_id = device_id
        self.iterations = iterations
        self._puf = PUFSim(device_id=device_id, session_nonce=b"root-module",
                           **puf_kwargs)
        self.helper = b"sim-enrollment|" + device_id.encode("utf-8")
        self.tag = binding_tag(_device_root(device_id), self.helper)
        self._k_u = None
        self.unlocks = 0
        self.unlock_seconds = None

    def unlock(self, passphrase: bytes) -> None:
        start = time.perf_counter()
        self.lock()
        self.unlocks += 1
        root = self._puf._reconstructed_root()
        if not hmac.compare_digest(binding_tag(root, self.helper), self.tag):
            raise ReconstructionError("Root does not match the enrolled tag")
        self._k_u = bytearray(unlock_key(root, passphrase, self.iterations))
        self.unlock_seconds = time.perf_counter() - start

    def derive(self, label: bytes) -> bytes:
        if self._k_u is None:
            raise ModuleLocked("Unlock the module before deriving keys")
        return _f(bytes(self._k_u), label)

    def lock(self) -> None:
        # Overwrites our buffer; Python may still hold transient copies.
        if self._k_u is not None:
            self._k_u[:] = bytes(len(self._k_u))
        self._k_u = None

    def __getstate__(self):
        raise TypeError("Root module state must not be serialized")

    def report(self):
        return {"backend": "simulator", "device_id": self.device_id,
                "iterations": self.iterations, "unlocks": self.unlocks,
                "unlock_seconds": self.unlock_seconds,
                "locked": self._k_u is None}


class CommandRootModule:
    """Forward unlock, derive and lock to a hardware module.

    Each request runs the configured client command without a shell, writes one
    JSON object to its stdin, and expects one JSON object on stdout:
    ``{"op": "unlock", "passphrase_hex": ...}``, ``{"op": "derive",
    "label_hex": ...}`` returning ``key_hex``, and ``{"op": "lock"}``. Replies
    carry ``success`` and ``device_id``; a failed reply names ``error``, where
    ``reconstruction`` is a cache miss and ``locked`` a missing unlock. The
    module, not this client, holds the unlock key between requests.
    """
    backend = "hardware"

    def __init__(self, device_id: str, command, *, timeout: float = 60):
        if not command or not all(isinstance(arg, str) for arg in command):
            raise ValueError("Module command must be a non-empty argv list")
        self.device_id = device_id
        self.command = list(command)
        self.timeout = timeout

    def _request(self, op: str, **fields) -> dict:
        request = json.dumps({"op": op, **fields}).encode()
        try:
            proc = subprocess.run(self.command, input=request, capture_output=True,
                                  timeout=self.timeout, check=False)
            reply = json.loads(proc.stdout)
            if reply.get("device_id") != self.device_id:
                raise ValueError("identity")
        except Exception:
            # Replies may carry keys; never echo them into an exception.
            raise ReconstructionError("Root module request failed validation") from None
        if reply.get("success") is not True:
            if reply.get("error") == "locked":
                raise ModuleLocked("Root module is locked")
            raise ReconstructionError("Root module could not reconstruct its root")
        return reply

    def unlock(self, passphrase: bytes) -> dict:
        """Returns the phase timings the module reports, if any."""
        reply = self._request("unlock", passphrase_hex=passphrase.hex())
        timing = reply.get("timing", {})
        return timing if isinstance(timing, dict) else {}

    def derive(self, label: bytes) -> bytes:
        key = bytes.fromhex(self._request("derive", label_hex=label.hex())["key_hex"])
        if len(key) != 32:
            raise ReconstructionError("Root module returned a malformed key")
        return key

    def lock(self) -> None:
        self._request("lock")


def module_from_config(path, **kwargs):
    """``{"backend": "simulator", "device_id": ...}`` or ``{"backend":
    "hardware", "device_id": ..., "command": [...], "timeout_seconds": 60}``."""
    config = json.loads(Path(path).read_text())
    if config["backend"] == "simulator":
        return SimulatedRootModule(config["device_id"],
                                   iterations=config.get("iterations", DEFAULT_ITERATIONS),
                                   **kwargs)
    if config["backend"] == "hardware":
        return CommandRootModule(config["device_id"], config["command"],
                                 timeout=config.get("timeout_seconds", 60))
    raise ValueError("Module config must select simulator or hardware")
