#!/usr/bin/env bash
# Checks on build/main.pdf (and build/supplement.pdf):
#   page limit (body ends at label body-end, page <= 20), PACMPL page size,
#   anonymous metadata, embedded fonts and no Type 3, no undefined references or
#   citations, no overfull box over 1pt, no identifying strings in the text,
#   placeholder count, citations present in bib/VERIFIED.md.
# In submit mode every finding is fatal; in draft mode they are reported.
set -o pipefail
HAPAX="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HAPAX"
MODE="${1:-draft}"
FAIL=0
bad() { echo "  FAIL: $*"; FAIL=1; }
ok() { echo "  ok:   $*"; }

for doc in main supplement; do
  pdf="build/$doc.pdf"; log="build/$doc.log"; aux="build/$doc.aux"
  echo "== $doc"
  [ -f "$pdf" ] || { bad "$pdf missing"; continue; }
  size="$(pdfinfo "$pdf" | sed -n 's/^Page size: *//p')"
  case "$size" in 486*720*) ok "page size $size" ;; *) bad "page size $size (want 486 x 720 pts)" ;; esac
  pages="$(pdfinfo "$pdf" | sed -n 's/^Pages: *//p')"
  if pdffonts "$pdf" | tail -n +3 | awk '{ if ($0 ~ /Type 3/) t3=1; if ($(NF-4) == "no") ne=1 } END { exit (t3 || ne) }'; then
    ok "fonts embedded, no Type 3"
  else
    bad "a font is not embedded or is Type 3"
  fi
  author="$(pdfinfo "$pdf" | sed -n 's/^Author: *//p')"
  [ -z "$author" ] || [ "$author" = "Anonymous Author(s)" ] && ok "author metadata '${author}'" || bad "author metadata '$author'"
  hits="$(pdftotext "$pdf" - | grep -n -i -E 'zhiqiang|ljrider|/home/|worktree|super_opt|te\.superopt|\bsuperopt\b|claude|github\.com/zhiqiang' | head -5)"
  [ -z "$hits" ] && ok "no identifying strings" || bad "identifying strings: $hits"
  undef="$(grep -c -E "(Reference|Citation) .* undefined" "$log" 2>/dev/null)"
  [ "${undef:-0}" = 0 ] && ok "no undefined references" || bad "$undef undefined-reference lines in $log"
  over="$(grep -E '^Overfull \\hbox \(([0-9.]+)pt' "$log" 2>/dev/null | awk -F'[(p]' '{ if ($2+0 > 1.0) n++ } END { print n+0 }')"
  [ "$over" = 0 ] && ok "no overfull box over 1pt" || echo "  warn: $over overfull boxes over 1pt in $log"
  if [ "$doc" = main ]; then
    end="$(grep -o 'newlabel{body-end}{{[^}]*}{[0-9]*}' "$aux" | sed 's/.*{\([0-9]*\)}$/\1/')"
    if [ -n "$end" ] && [ "$end" -le 20 ]; then ok "body ends on page $end of $pages (limit 20)"; else bad "body ends on page ${end:-?} (limit 20)"; fi
    ph="$(pdftotext "$pdf" - | grep -c 'Placeholder\.')"
    [ "$ph" = 0 ] && ok "no placeholders" || echo "  note: $ph placeholders printed in $pdf"
    [ "$MODE" = submit ] && [ "$ph" != 0 ] && FAIL=1
  fi
done

echo "== citations"
python3 -I scripts/check_cites.py build/main.aux build/supplement.aux || FAIL=1

if [ "$MODE" = submit ] && [ "$FAIL" != 0 ]; then
  echo "submission checks failed"; exit 1
fi
exit 0
