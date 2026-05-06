#!/usr/bin/env bash
set -euo pipefail

cd /home/umrobotics/C-3-Framework
source .venv/bin/activate

# Wait for the current MSE run to finish, then start the Euclidean sweep.
tail --pid=352560 -f /dev/null

bash scripts/run_tenebrio_euclidean_sweep.sh
