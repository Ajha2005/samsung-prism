#!/usr/bin/env bash
# SessionStart hook: make the repo runnable in a fresh Claude Code (web) session.
# Installs the *core* deps only (fast, no network model downloads) so the offline
# test suite and demo work immediately. The heavy eval extras (mteb, torch) are
# left to `make install-eval` when a leaderboard run is actually needed.
set -e

cd "$(dirname "$0")/../.." || exit 0

python3 -m pip install --quiet --disable-pip-version-check -r requirements.txt >/dev/null 2>&1 || true
python3 -m pip install --quiet --disable-pip-version-check -e . >/dev/null 2>&1 || true
python3 -m pip install --quiet --disable-pip-version-check pytest >/dev/null 2>&1 || true

echo "prism ready: run 'python -m pytest -q' or 'python -m prism.cli demo --backend hashing'"
