#!/bin/bash
# Generate joel-resume.pdf from joel-resume.html
# Requires: google-chrome, fonts-crosextra-carlito (Calibri-compatible)
#
# Install font if missing:
#   sudo apt-get install -y fonts-crosextra-carlito && fc-cache -f
#
# Usage: ./generate-resume.sh

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

google-chrome --headless --disable-gpu \
  --print-to-pdf="$DIR/joel-resume.pdf" \
  --no-pdf-header-footer \
  "$DIR/joel-resume.html"

echo "Generated: $DIR/joel-resume.pdf"
