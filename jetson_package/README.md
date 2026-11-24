# jetson_package

Code for part J of `../req_1120/README.md`: the protected attention kernels, the cold tier, the host side of the root module, and the measurement scripts. The steps, commands and outputs are listed in part J.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
python scripts/triton_smoke.py && python -m pytest -q tests
```

`configs/` holds module-config templates for J6. The simulator config runs J6 without boards.
