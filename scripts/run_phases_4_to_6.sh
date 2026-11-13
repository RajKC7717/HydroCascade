#!/usr/bin/env bash
# Phases 4, 5 and 6, one OS process each.
#
# One process per phase is not tidiness: this machine has 7.2 GB of RAM and a
# single process that fits the cascade model, then runs every ablation, then
# the backtest, climbs until the OS reaps it. Exiting between phases returns
# the memory to the system, and each phase reads its predecessor's artefacts
# from disk, so a phase that dies can be restarted on its own.
set -u
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe

run_phase () {
  local name="$1" code="$2"
  echo "=== $name starting $(date +%H:%M:%S) ==="
  PYTHONUNBUFFERED=1 "$PY" -u -c "
import torch
torch.set_num_threads(2)          # the fits are small; more threads just adds RSS
from hydrocascade.logging_setup import setup; setup()
from hydrocascade.config import load_config; cfg = load_config()
$code
"
  local rc=$?
  echo "=== $name exited rc=$rc $(date +%H:%M:%S) ==="
  return $rc
}

run_phase "PHASE 4" "
from hydrocascade.cascade import stages
stages.run(cfg, n_restarts=3, steps=160)
print('PHASE 4 COMPLETE')
" || exit 1

run_phase "PHASE 5" "
from hydrocascade.experiments import run as ex
ex.run(cfg, n_restarts=3, steps=160)
print('PHASE 5 COMPLETE')
" || exit 1

run_phase "PHASE 6" "
from hydrocascade.realtime import run as rt
rt.run(cfg)
print('PHASE 6 COMPLETE')
" || exit 1

echo "ALL PHASES COMPLETE"
