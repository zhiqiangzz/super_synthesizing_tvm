#!/usr/bin/env bash
# Build the prose-only Chinese version without rebuilding the English PDFs.
set -euo pipefail
HAPAX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HAPAX"
ROOT="$(dirname "$(git -C "$HAPAX" rev-parse --path-format=absolute --git-common-dir)")"
TECTONIC="$ROOT/.pixi/envs/paper/bin/tectonic"
[ -x "$TECTONIC" ] || TECTONIC="$ROOT/.pixi/envs/slides/bin/tectonic"
python3 -I scripts/gen_tex.py
python3 -I scripts/check_zh.py
mkdir -p build
"$TECTONIC" -X compile --keep-logs --keep-intermediates --outdir build main-zh.tex
python3 -I scripts/check_zh.py --pdf
python3 -I scripts/check_cites.py build/main-zh.aux
