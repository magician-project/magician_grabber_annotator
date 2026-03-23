#!/usr/bin/env bash
# git_stats.sh — Show commit stats since a given date
# Usage: ./gitProgress.sh [SINCE_DATE]
# Example: ./gitProgress.sh "2025-01-01"
#          ./gitProgress.sh "3 weeks ago"

set -euo pipefail

SINCE="${1:-"1 month ago"}"

# Validate we're inside a git repo
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
  echo "Error: not inside a git repository." >&2
  exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Git Stats since: $SINCE"
echo "  Repo: $(basename "$(git rev-parse --show-toplevel)")"
echo "  Branch: $(git branch --show-current)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Commit count ─────────────────────────────
COMMITS=$(git log --since="$SINCE" --oneline | wc -l | tr -d ' ')
echo ""
echo "📦 Commits:        $COMMITS"

# ── Authors ──────────────────────────────────
AUTHORS=$(git log --since="$SINCE" --format="%aN" | sort -u | wc -l | tr -d ' ')
echo "👥 Authors:        $AUTHORS"

# ── Files changed ────────────────────────────
FILES=$(git log --since="$SINCE" --name-only --format="" | sort -u | grep -c . || true)
echo "📁 Files changed:  $FILES"

# ── Lines added / removed ────────────────────
NUMSTAT=$(git log --since="$SINCE" --numstat --format="" \
  | awk 'NF==3 && $1~/^[0-9]+$/ && $2~/^[0-9]+$/ { add+=$1; del+=$2 }
         END { printf "%d %d", add, del }')
ADDED=$(echo "$NUMSTAT" | cut -d' ' -f1)
REMOVED=$(echo "$NUMSTAT" | cut -d' ' -f2)
NET=$(( ADDED - REMOVED ))
echo "➕ Lines added:    $ADDED"
echo "➖ Lines removed:  $REMOVED"
echo "📊 Net change:     $NET"

# ── Top contributors ─────────────────────────
echo ""
echo "🏆 Top contributors:"
git log --since="$SINCE" --format="%aN" \
  | sort | uniq -c | sort -rn | head -10 \
  | awk '{ printf "   %4d commits — %s\n", $1, $2 }'

# ── Most changed files ───────────────────────
echo ""
echo "🔥 Most changed files:"
git log --since="$SINCE" --name-only --format="" \
  | grep -v '^$' | sort | uniq -c | sort -rn | head -10 \
  | awk '{ printf "   %4d times — %s\n", $1, $2 }'

# ── Recent commits ───────────────────────────
echo ""
echo "🕓 Recent commits:"
git log --since="$SINCE" --format="   %C(yellow)%h%Creset %C(dim)%ad%Creset %s %C(cyan)(%aN)%Creset" \
  --date=short | head -10

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
