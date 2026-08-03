#!/bin/bash
# sync.sh — Live code updater. Pulls latest source from Google Drive so the
# Pi reads updated files automatically without manual SCP or SSH.
# Usage: ./sync.sh "optional commit message"

MSG=${1:-"Update"}

git add -A
git commit -m "$MSG" 2>/dev/null || echo "(nothing new to commit)"
git push origin main --force-with-lease
echo "Done."
