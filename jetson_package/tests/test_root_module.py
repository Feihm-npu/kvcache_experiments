"""Root-module contract: the host gets per-store keys and never the root, every
passphrase guess costs one unlock, and a modified enrollment reveals nothing.

The adversary holds the device: it reads and replaces the public enrollment
data and submits any passphrase, but it reaches the root only through the
module's three requests.
"""

from __future__ import annotations

import pickle
import sys

import pytest

from puf4secure_kvcache.puf_sim import ReconstructionError
from puf4secure_kvcache.root_module import (CommandRootModule, ModuleLocked,
                                            SimulatedRootModule, unlock_key)

PASS = b"owner passphrase"
LABEL = b"cold|aead|model=m|store=s"


def _unlocked(device="device_A", passphrase=PASS, **kw):
    m = SimulatedRootModule(device, iterations=8, **kw)
    m.unlock(passphrase)
    return m


def test_keys_depend_on_device_and_passphrase_only():
    a = _unlocked().derive(LABEL)
    assert a == _unlocked().derive(LABEL)                    # stable across unlocks
    assert a != _unlocked(passphrase=b"a guess").derive(LABEL)
    assert a != _unlocked(device="device_B").derive(LABEL)
    assert a != _unlocked().derive(LABEL + b"x")


def test_locked_module_derives_nothing():
    m = SimulatedRootModule("device_A", iterations=8)
    with pytest.raises(ModuleLocked):
        m.derive(LABEL)
    m.unlock(PASS)
    m.derive(LABEL)
    m.lock()
    with pytest.raises(ModuleLocked):
        m.derive(LABEL)


def test_modified_enrollment_fails_whatever_the_response():
    for field in ("helper", "tag"):
        m = SimulatedRootModule("device_A", iterations=8)
        setattr(m, field, getattr(m, field)[:-1] + b"\x00")
        with pytest.raises(ReconstructionError):
            m.unlock(PASS)
        with pytest.raises(ModuleLocked):                    # a failed unlock stays locked
            m.derive(LABEL)


def test_enrollment_moved_to_another_chip_fails():
    with pytest.raises(ReconstructionError):
        _unlocked(mode="wrong_device")


def test_noisy_root_within_budget_unlocks_and_beyond_is_a_miss():
    assert _unlocked(mode="noisy", ber=0.02, noise_seed=3).derive(LABEL) \
        == _unlocked().derive(LABEL)
    with pytest.raises(ReconstructionError):
        _unlocked(mode="noisy", ber=0.45, noise_seed=3)


def test_unlock_cost_grows_with_the_iteration_count():
    root = bytes(range(32))
    assert unlock_key(root, PASS, 1) != unlock_key(root, PASS, 2)
    fast = SimulatedRootModule("device_A", iterations=10)
    slow = SimulatedRootModule("device_A", iterations=20000)
    fast.unlock(PASS)
    slow.unlock(PASS)
    assert slow.unlock_seconds > 5 * fast.unlock_seconds


def test_module_state_is_not_serializable():
    with pytest.raises(TypeError):
        pickle.dumps(_unlocked())


def _client(script):
    return [sys.executable, "-c", script]


def test_command_module_forwards_requests_without_argv_secrets():
    script = ("import json,sys; r=json.load(sys.stdin); "
              "print(json.dumps({'success': True, 'device_id': 'A', "
              "'key_hex': '01'*32} if r['op']=='derive' else "
              "{'success': True, 'device_id': 'A'}))")
    m = CommandRootModule("A", _client(script))
    m.unlock(b"secret passphrase")
    assert m.derive(LABEL) == b"\x01" * 32
    assert not any("secret" in arg for arg in m.command)
    with pytest.raises(ReconstructionError):
        CommandRootModule("B", _client(script)).derive(LABEL)


def test_command_module_maps_errors_and_never_leaks_replies():
    marker = "KEY-MATERIAL-MARKER"
    for error, exc in (("locked", ModuleLocked), ("reconstruction", ReconstructionError)):
        script = (f"import json; print(json.dumps({{'success': False, 'device_id': 'A', "
                  f"'error': '{error}', 'note': '{marker}'}}))")
        with pytest.raises(exc) as info:
            CommandRootModule("A", _client(script)).unlock(PASS)
        assert marker not in str(info.value)
    with pytest.raises(ReconstructionError) as info:
        CommandRootModule("A", _client(f"print('{marker}')")).unlock(PASS)
    assert marker not in str(info.value)
