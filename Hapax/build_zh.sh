#!/usr/bin/env bash
# Build the prose-only Chinese version without rebuilding the English PDFs.
set -euo pipefail
HAPAX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HAPAX"
ROOT="$(dirname "$(git -C "$HAPAX" rev-parse --path-format=absolute --git-common-dir)")"
TECTONIC="$ROOT/.pixi/envs/paper/bin/tectonic"
[ -x "$TECTONIC" ] || TECTONIC="$ROOT/.pixi/envs/slides/bin/tectonic"
MODE=draft
for arg in "$@"; do
  case "$arg" in
    --draft) MODE=draft ;;
    --submit) MODE=submit ;;
    *) echo "unknown option $arg" >&2; exit 2 ;;
  esac
done
if [ "$MODE" = submit ]; then
  python3 -I scripts/gen_tex.py
  printf '\\submittrue\n' > gen/mode.tex
else
  python3 -I scripts/gen_tex.py --allow-stale
  printf '\\submitfalse\n' > gen/mode.tex
fi
python3 -I scripts/check_zh.py
mkdir -p build
"$TECTONIC" -X compile --keep-logs --keep-intermediates --outdir build main-zh.tex
python3 -I scripts/check_zh.py --pdf
python3 -I scripts/check_cites.py build/main-zh.aux
