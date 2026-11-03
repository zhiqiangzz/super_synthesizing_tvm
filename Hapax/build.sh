#!/usr/bin/env bash
# Build the Hapax paper.
#
#   bash build.sh            # draft: macros from data/, main.pdf + supplement.pdf, checks
#   bash build.sh --data     # first rerun every data script at the current HEAD (minutes)
#   bash build.sh --submit   # refuses stale data, placeholders and unverified citations
#
# TeX comes from the root pixi environment ("paper" if it exists, else "slides"):
# tectonic 0.17 with its TeX Live 2022 bundle; acmart.cls v2.20 and the ACM .bst are
# vendored next to main.tex and take precedence over the bundle's copies.
set -eo pipefail
HAPAX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HAPAX"
ROOT="$(dirname "$(git -C "$HAPAX" rev-parse --path-format=absolute --git-common-dir)")"
TECTONIC="$ROOT/.pixi/envs/paper/bin/tectonic"
[ -x "$TECTONIC" ] || TECTONIC="$ROOT/.pixi/envs/slides/bin/tectonic"

MODE=draft
DATA=0
for arg in "$@"; do
  case "$arg" in
    --data) DATA=1 ;;
    --submit) MODE=submit ;;
    --draft) MODE=draft ;;
    *) echo "unknown option $arg" >&2; exit 2 ;;
  esac
done

[ "$DATA" = 1 ] && bash scripts/run_data.sh

if [ "$MODE" = submit ]; then
  python3 -I scripts/gen_tex.py
  printf '\\submittrue\n' > gen/mode.tex
else
  python3 -I scripts/gen_tex.py --allow-stale
  printf '\\submitfalse\n' > gen/mode.tex
fi

mkdir -p build
for doc in main supplement pipeline; do
  echo "==> $doc.tex"
  "$TECTONIC" -X compile --keep-logs --keep-intermediates --outdir build "$doc.tex"
done

bash scripts/check_pdf.sh "$MODE"
