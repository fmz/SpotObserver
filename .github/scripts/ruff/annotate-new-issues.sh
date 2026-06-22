#!/usr/bin/env bash
# =============================================================================
# Annotate NEW Issues Only
# =============================================================================
# Generates GitHub annotations filtered to rules with positive deltas.
# Only issues from rules that INCREASED get annotated, not inherited issues.
#
# Input files (from collect-stats.sh):
#   - rules_with_new_issues.txt: rule codes with positive deltas (one per line)
#   - pr_ruff_output.json: all ruff issues (JSON-lines format)
#
# Environment:
#   - GITHUB_WORKSPACE: checkout directory (for stripping absolute paths)
#
# Output:
#   - GitHub annotations (::error file=...,line=...) printed to stdout
# =============================================================================

set -euo pipefail

# Load shared utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

D=".ruff-stats"

log_section "Annotating new issues"

# Exit silently if no new issues
if [ ! -s "$D/rules_with_new_issues.txt" ]; then
  echo "  ✓ No new issues to annotate"
  exit 0
fi

# Build regex pattern from rules with new issues
# E.g., "SIM102|F401|RUF022"
PATTERN=$(paste -sd'|' "$D/rules_with_new_issues.txt")
RULE_COUNT=$(wc -l < "$D/rules_with_new_issues.txt" | tr -d ' ')

echo "  → Rules with new issues: $RULE_COUNT"
echo "  → Pattern: $PATTERN"

# Check environment
if [ -z "${GITHUB_WORKSPACE:-}" ]; then
  echo "  ⚠️  GITHUB_WORKSPACE not set (paths may be absolute)"
fi

# Get workspace prefix for path stripping (with trailing slash)
WS_PREFIX=$(get_workspace_prefix)

# Filter PR ruff output to only rules with positive deltas
# Then format as GitHub annotations
# Note: pr_ruff_output.json is JSON-lines (one object per line)
# Note: filenames are absolute, need to strip workspace prefix for GitHub
# Using -r (not -rs) for efficient line-by-line processing without slurping
# Guard: skip entries without .location (defensive, should not happen with ruff)

# Generate annotations and count them
# Store in temp file to both output and count
TEMP_ANNOTATIONS="$D/annotations.txt"
jq -r --arg pattern "$PATTERN" --arg ws "$WS_PREFIX" '
  select(.code | test($pattern)) |
  select(.location != null) |
  "::error file=\(.filename | ltrimstr($ws)),line=\(.location.row),col=\(.location.column)::\(.code): \(.message)"
' "$D/pr_ruff_output.json" > "$TEMP_ANNOTATIONS"

ANNOTATION_COUNT=$(wc -l < "$TEMP_ANNOTATIONS" | tr -d ' ')
echo "  → Annotations: $ANNOTATION_COUNT"

# Output annotations to stdout (for GitHub to pick up)
cat "$TEMP_ANNOTATIONS"
