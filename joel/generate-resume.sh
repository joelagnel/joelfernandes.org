#!/bin/bash
# Generate joel-resume.pdf from joel-resume.html
# Requires: google-chrome or chromium, fonts-crosextra-carlito (Calibri-compatible)
#
# Install font if missing:
#   sudo apt-get install -y fonts-crosextra-carlito && fc-cache -f
#
# Usage: ./generate-resume.sh [--display :99]
#   --display   X display to use (default: :99 for headless Xvfb)

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

DISPLAY_ARG="${1:-:99}"
if [[ "$1" == "--display" ]]; then
  DISPLAY_ARG="$2"
fi

CHROME="google-chrome"
if ! command -v google-chrome &>/dev/null; then
  CHROME="chromium-browser"
fi

DISPLAY="$DISPLAY_ARG" "$CHROME" \
  --headless \
  --disable-gpu \
  --no-sandbox \
  --print-to-pdf="$DIR/joel-resume.pdf" \
  --print-to-pdf-no-header \
  "$DIR/joel-resume.html"

echo "Generated: $DIR/joel-resume.pdf"
