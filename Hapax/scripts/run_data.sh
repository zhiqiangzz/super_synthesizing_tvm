#!/usr/bin/env bash
# Regenerate every data file the paper reads, at the current HEAD of this worktree.
# Refuses to run if the tool's sources differ from HEAD (the data would not be
# reproducible from a commit).
set -eo pipefail
HAPAX="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WT="$(dirname "$HAPAX")"
cd "$WT"
if ! git diff --quiet HEAD -- python/tvm/te/superopt super_opt/examples tests/python/te/superopt; then
  echo "the tool tree differs from HEAD; commit or stash it before regenerating data" >&2
  exit 1
fi
PY="$HAPAX/scripts/py"
echo "==> tests";            "$PY" -m pytest -q -p no:cacheprovider tests/python/te/superopt > "$HAPAX/data/pytest.log" 2>&1
tail -1 "$HAPAX/data/pytest.log"
echo "==> fused operators";  "$PY" "$HAPAX/scripts/collect_fused.py" "$HAPAX/data/fused_ops.json"
echo "==> unfused programs"; "$PY" "$HAPAX/scripts/collect_unfused.py" "$HAPAX/data/unfused.json"
echo "==> prior papers";     "$PY" "$HAPAX/scripts/collect_prior.py" "$HAPAX/data/prior.json"
echo "==> hazards";          "$PY" "$HAPAX/scripts/check_hazards.py" "$HAPAX/data/hazards.json"
echo "==> proofs and time";  "$PY" "$HAPAX/scripts/proofs_and_timing.py" "$HAPAX/data/proofs_timing.json"
echo "==> worked examples";  "$PY" "$HAPAX/scripts/illus.py" "$HAPAX/data/illus.json" >/dev/null 2>&1
echo "==> listings";         "$PY" "$HAPAX/scripts/listings.py" >/dev/null 2>&1
echo "done; now: python3 -I $HAPAX/scripts/gen_tex.py"
