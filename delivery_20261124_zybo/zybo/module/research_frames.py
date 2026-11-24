"""Research-protocol (PUF-COLP v1) requests for the Z8 interface-closure test.

Each entry is one "research-mode read command": a list of frames a PC client would send to the
frozen research service (docs/protocol.md) to read PUF responses. The challenges come from the
Z3 set, so none of them is in K. Sent to the deployment service, every entry must be rejected.
"""
import json
import random
import struct

import numpy as np

MAGIC = b"PUF1"
HEADER = struct.Struct("<4sHI")
CMD_HELLO, CMD_ECHO, CMD_CONFIGURE, CMD_RESET = 0x01, 0x02, 0x10, 0x11
CMD_SEND_CHALLENGES, CMD_GET_RESPONSES, CMD_GET_OIT, CMD_GET_PARAMS = 0x20, 0x21, 0x30, 0x31


def frame(cmd, payload=b""):
    return HEADER.pack(MAGIC, cmd, len(payload)) + payload


def words(challenge_rows):
    pw = np.uint64(1) << np.arange(64, dtype=np.uint64)
    return (np.asarray(challenge_rows, dtype=np.uint64) * pw).sum(axis=1).astype("<u8").tobytes()


def research_commands(challenges, count=100, seed=20261124):
    """Returns a list of dicts: {"index", "description", "payload": bytes}. `challenges` is the Z3 set."""
    rng = random.Random(seed)                  # stdlib RNG: the file runs on numpy 1.13 and 2.x alike
    challenges = np.asarray(challenges)
    rows = rng.sample(range(len(challenges)), count * 4)
    out = []
    for i in range(count):
        kind = i % 10
        r = rows[4 * i:4 * i + 4]
        if kind == 0:
            frames, desc = [frame(CMD_HELLO)], "HELLO"
        elif kind == 1:
            frames, desc = [frame(CMD_CONFIGURE, json.dumps({"family": "apuf_lanes", "instance_id": 0, "seed": 1}).encode())], "CONFIGURE apuf_lanes"
        elif kind == 2:
            frames, desc = [frame(CMD_CONFIGURE, json.dumps({"family": "ff_lanes", "instance_id": 0, "seed": 1}).encode()),
                            frame(CMD_SEND_CHALLENGES, words(challenges[r[:1]])),
                            frame(CMD_GET_RESPONSES, struct.pack("<I", 1))], "CONFIGURE ff_lanes + SEND_CHALLENGES + GET_RESPONSES"
        elif kind == 3:
            frames, desc = [frame(CMD_CONFIGURE, json.dumps({"family": "apuf_lanes", "instance_id": 0, "seed": 1}).encode()),
                            frame(CMD_SEND_CHALLENGES, words(challenges[r[:2]])),
                            frame(CMD_GET_RESPONSES, struct.pack("<I", 2))], "CONFIGURE apuf_lanes + 2 challenges + GET_RESPONSES"
        elif kind == 4:
            frames, desc = [frame(CMD_SEND_CHALLENGES, words(challenges[r])), frame(CMD_GET_RESPONSES, struct.pack("<I", 4))], "SEND_CHALLENGES (4) + GET_RESPONSES without CONFIGURE"
        elif kind == 5:
            frames, desc = [frame(CMD_GET_RESPONSES, struct.pack("<I", 1))], "GET_RESPONSES alone"
        elif kind == 6:
            fam = ("oi", "flam", "ct", "oi_lanes", "tpdoa", "puf4x4")[i // 10 % 6]
            frames, desc = [frame(CMD_CONFIGURE, json.dumps({"family": fam, "instance_id": 0, "seed": 1}).encode()),
                            frame(CMD_SEND_CHALLENGES, words(challenges[r[:1]])),
                            frame(CMD_GET_RESPONSES, struct.pack("<I", 1))], "CONFIGURE %s + challenge + GET_RESPONSES" % fam
        elif kind == 7:
            frames, desc = [frame(CMD_GET_PARAMS), frame(CMD_GET_OIT)], "GET_PARAMS + GET_OIT"
        elif kind == 8:
            frames, desc = [frame(CMD_RESET), frame(CMD_ECHO, words(challenges[r[:1]]))], "RESET + ECHO"
        else:
            frames, desc = [frame(0x77, words(challenges[r])), frame(CMD_SEND_CHALLENGES, words(challenges[r[:1]]))], "unknown command 0x77 + SEND_CHALLENGES"
        payload = b"".join(frames)
        assert payload[:4] == MAGIC
        out.append({"index": i, "description": desc, "payload": payload})
    return out


def json_impostors(challenges, count=10, seed=20261124):
    """Not research-protocol frames; line-JSON requests that try to get responses another way."""
    rng = random.Random(seed + 1)
    challenges = np.asarray(challenges)
    rows = rng.sample(range(len(challenges)), count)
    out = []
    for i in range(count):
        row = [int(b) for b in challenges[rows[i]]]
        kind = i % 5
        if kind == 0:
            obj, desc = {"family": "apuf_lanes", "challenges": [row]}, "family + challenge rows (no op)"
        elif kind == 1:
            obj, desc = {"op": "read", "family": "apuf_lanes", "challenges": [row]}, "op=read"
        elif kind == 2:
            obj, desc = {"op": "unlock"}, "unlock without passphrase"
        elif kind == 3:
            obj, desc = {"op": "derive", "label_hex": "00", "challenges": [row]}, "derive with extra field"
        else:
            obj, desc = {"op": "enroll", "lanes": [0, 1, 2, 3], "t": 8}, "op=enroll"
        out.append({"index": i, "description": desc, "payload": json.dumps(obj).encode()})
    return out
