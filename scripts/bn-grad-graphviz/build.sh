#!/usr/bin/env bash
# Generate the batchnorm computational-graph figures for the part-6 post.
# Batch size 4 throughout, so the fan-in/fan-out is visible without clutter.
# Files whose first line mentions "neato -n" use hand-pinned coordinates.
set -euo pipefail

OUT="/home/joel/.openclaw/workspace/joelfernandes.org/images/bn-grad"
mkdir -p "$OUT"
HERE="$(cd "$(dirname "$0")" && pwd)"

for f in "$HERE"/*.dot; do
  base="$(basename "$f" .dot)"
  if head -1 "$f" | grep -q "neato -n"; then
    neato -n -Tsvg "$f" -o "$OUT/$base.svg"
  else
    dot -Tsvg "$f" -o "$OUT/$base.svg"
  fi
  echo "wrote $OUT/$base.svg"
done

# graphviz cannot place combining marks; draw the circumflex as a path instead
python3 "$HERE/hatfix.py" "$OUT"/*.svg
