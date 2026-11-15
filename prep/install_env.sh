#!/bin/bash
# Environment of the official SuperMimic task under $ASSETS/env: Python 3.13 (supermimic requires >=3.13,<3.14) with
# the exact versions of mjlab_research's own uv.lock, extra cu130 (torch 2.14.0+cu130, warp-lang 1.18.0, mujoco 3.16.0,
# mujoco-warp 3.16.0, triton 3.8.0), every package checked against the lock's sha256 (--require-hashes).
# supermimic/uv.lock is not used: it pins torch 2.10 / mujoco 3.10 and predates the published mjlab_research.
# Neither project is installed: every run puts its tree's supermimic/src and mjlab_research/src on PYTHONPATH.
# supermimic's own extra dependencies: viser (already in the lock) and ruff (a linter, installed without a pin check).
#   ASSETS  the task's assets directory     SNAP  the snapshot (default: <this repository>/targets/reference)
#   PYPI_INDEX  package index (default PyPI); every package is still checked against the lock's sha256.
# The torch wheel is downloaded from the lock's own URL (download-r2.pytorch.org) and checked first.
set -euo pipefail
A=${ASSETS:?set ASSETS}; SNAP=${SNAP:-$(cd "$(dirname "$0")/.." && pwd)/targets/reference}
UV=${UV:-/usr/local/bin/uv}
PYPI_INDEX=${PYPI_INDEX:-https://pypi.org/simple/}
export UV_CACHE_DIR=$A/uv-cache UV_PYTHON_INSTALL_DIR=$A/python
unset UV_INDEX_STRATEGY
mkdir -p "$A/wheels"
[ -x "$A/env/bin/python" ] || "$UV" venv --python 3.13 "$A/env"
(cd "$SNAP/mjlab_research" && "$UV" export --frozen --extra cu130 --no-dev --no-emit-project --no-annotate --no-header) \
  > "$A/requirements.lock.txt"
python3 - "$A/requirements.lock.txt" "$A/wheels" <<'PY'
import re, sys
lines = open(sys.argv[1]).read().replace('\\\n', ' ').splitlines()
torch = [l for l in lines if l.startswith('torch==') and '+cu130' in l]
assert len(torch) == 1, torch
print('torch pin:', torch[0].split(';')[0])
PY
# the torch wheel from the lock's own URL, checked against the lock's hash
read -r URL SHA < <(python3 - "$SNAP/mjlab_research/uv.lock" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
m = re.search(r'url = "(https://download-r2\.pytorch\.org/whl/cu130/torch-2\.14\.0%2Bcu130-cp313-cp313-manylinux_2_28_x86_64\.whl)", hash = "sha256:([0-9a-f]+)"', s)
print(m.group(1), m.group(2))
PY
)
F="$A/wheels/torch-2.14.0+cu130-cp313-cp313-manylinux_2_28_x86_64.whl"
[ -f "$F" ] || curl -fsSL -o "$F" "$URL"
echo "$SHA  $F" | sha256sum -c
"$UV" pip install --python "$A/env/bin/python" --no-deps --require-hashes -r "$A/requirements.lock.txt" \
  --index-url "$PYPI_INDEX" --find-links "$A/wheels" --index-strategy unsafe-best-match
"$UV" pip install --python "$A/env/bin/python" --index-url "$PYPI_INDEX" "ruff>=0.14.10"
"$UV" pip freeze --python "$A/env/bin/python" > "$A/env_freeze.txt"
PYTHONPATH=$SNAP/supermimic/src:$SNAP/mjlab_research/src "$A/env/bin/python" - <<'PY'
import torch, mujoco, warp, mujoco_warp, mjlab, supermimic
print('torch', torch.__version__, torch.version.cuda, 'gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0))
print('mujoco', mujoco.__version__, 'warp', warp.__version__, 'mjlab from', mjlab.__file__)
x = torch.randn(1024, 1024, device='cuda'); print('matmul ok', float((x @ x).abs().mean()))
PY
