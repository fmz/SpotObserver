#!/bin/bash
# =============================================================================
# collect-stats.sh - Ruff Statistics Collection
# =============================================================================
# Analyzes PR and main branches, calculates delta statistics.
#
# Prerequisites:
#   - Git repository with origin/main reference
#   - RUFF_OUTPUT_FORMAT should be unset (caller's responsibility)
#
# Outputs (in .ruff-stats/):
#   - stats.env: All computed variables for GitHub outputs
#   - Various intermediate files for table generation
#
# Exit codes:
#   0: Success (stats collected)
#   Non-zero: Script error (file operations, jq parsing, etc.)
#
# =============================================================================
# KEY DESIGN DECISIONS
# =============================================================================
#
# 1. DELTA-BASED PHILOSOPHY (Critical)
#    - NEW_ISSUES = sum of positive rule deltas (issues YOU introduced)
#    - FIXED_ISSUES = sum of negative rule deltas (issues YOU fixed)
#    - INHERITED_ISSUES = pre-existing issues in modified code
#    - Workflow PASSES if NEW_ISSUES = 0, regardless of INHERITED_ISSUES
#    - Rationale: Contributors shouldn't be blocked for pre-existing problems
#
# 2. RULE-LEVEL DELTA CALCULATION (Critical)
#    - Deltas calculated from individual rules, NOT categories
#    - Prevents masked regressions (e.g., RUF022 +4 hidden by RUF overall -6)
#    - Category-level calculation misses regressions within improving categories
#
# 3. AUTO-FIX SCOPING (NEW issues only)
#    - delta_fixable_issues.json filtered to PR-changed files ONLY
#    - Further filtered to rules with positive deltas (NEW issues only)
#    - Rationale: Don't recommend auto-fixing inherited issues or unrelated files
#    - DELTA_FIX_PCT = percentage of NEW issues that are auto-fixable
#
# 4. FILES_FILTER (Path Matching)
#    - Built from: git diff --name-only origin/main...HEAD
#    - Ruff outputs absolute paths, git outputs relative paths
#    - Must strip workspace prefix from ruff paths before matching
#
# =============================================================================

set -euo pipefail

# Load shared utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

# All temp files go in .ruff-stats/ to avoid clutter
D=".ruff-stats"
mkdir -p "$D"
# Preserve format_*.txt artifacts created earlier in the workflow; clear everything else.
 find "$D" -maxdepth 1 -type f ! -name 'format_*' -delete 2>/dev/null || true

# =============================================================================
# PR Branch Analysis
# =============================================================================
log_section "Analyzing PR branch"

PR_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
PR_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "  → Branch: $PR_BRANCH @ $PR_SHA"

ruff check . --output-format json-lines > "$D/pr_ruff_output.json" || true
validate_json "$D/pr_ruff_output.json" "PR output"

# Extract statistics with jq
STATS=$(jq -s '{total: length, fixable: [.[] | select(.fix)] | length}' "$D/pr_ruff_output.json" 2>/dev/null || echo '{"total":0,"fixable":0}')
PR_TOTAL=$(echo "$STATS" | jq -r '.total')
PR_FIXABLE=$(echo "$STATS" | jq -r '.fixable')
echo "  → Issues: $PR_TOTAL total, $PR_FIXABLE fixable"

# Extract category stats for PR (maintain "CAT COUNT" format)
jq_count_by_category "$D/pr_ruff_output.json" \
  > "$D/pr_category_counts.txt" 2>/dev/null || touch "$D/pr_category_counts.txt"
log_file "$D/pr_category_counts.txt" "Categories"

# Count fixable per category (maintain "CAT COUNT" format)
jq_count_fixable_by_category "$D/pr_ruff_output.json" \
  > "$D/pr_category_fixable.txt" 2>/dev/null || touch "$D/pr_category_fixable.txt"

# Extract top 5 specific rules (maintain "COUNT RULE" format for sorting)
jq -rs '[.[] | .code] | group_by(.) | map("\(length) \(.[0])")[]' "$D/pr_ruff_output.json" | \
  sort -rn | head -5 > "$D/pr_top_rules.txt" 2>/dev/null || touch "$D/pr_top_rules.txt"
if [ -s "$D/pr_top_rules.txt" ]; then
  echo "  → Top rules: $(head -1 "$D/pr_top_rules.txt" | awk '{print $2 " (" $1 ")"}')"
fi

# =============================================================================
# Main Branch Analysis
# =============================================================================
log_section "Analyzing main branch"

# Stash any uncommitted changes
git stash --include-untracked --quiet 2>/dev/null || true

# Checkout main branch
if git checkout origin/main --quiet 2>/dev/null; then
  MAIN_REF="origin/main"
elif git checkout main --quiet 2>/dev/null; then
  MAIN_REF="main"
else
  log_error "Failed to checkout main branch"
  exit 1
fi
MAIN_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "  → Branch: $MAIN_REF @ $MAIN_SHA"

ruff check . --output-format json-lines > "$D/main_ruff_output.json" || true
validate_json "$D/main_ruff_output.json" "Main output"

# Extract statistics with jq
MAIN_TOTAL=$(jq -s 'length' "$D/main_ruff_output.json" 2>/dev/null || echo '0')
echo "  → Issues: $MAIN_TOTAL total"

# Extract category stats for main (maintain "CAT COUNT" format)
jq_count_by_category "$D/main_ruff_output.json" \
  > "$D/main_category_counts.txt" 2>/dev/null || touch "$D/main_category_counts.txt"

# Return to PR branch
git checkout - --quiet 2>/dev/null || log_warn "Failed to checkout previous branch"
git stash pop --quiet 2>/dev/null || true

# =============================================================================
# New vs Inherited Analysis
# =============================================================================
log_section "Calculating deltas"

# Get files changed in this PR
git diff --name-only origin/main...HEAD > "$D/pr_changed_files.txt"
CHANGED_FILES=$(wc -l < "$D/pr_changed_files.txt" | tr -d ' ')
echo "  → Changed files: $CHANGED_FILES"

# Build jq filter for changed files
FILES_FILTER=$(text_to_json_array "$D/pr_changed_files.txt")

# Calculate NEW_ISSUES = sum of positive deltas (issues you introduced)
# Calculate FIXED_ISSUES = sum of negative deltas (issues you fixed)
# IMPORTANT: Must use rule-level deltas, not category-level, to catch regressions
# in individual rules that are masked by category-level improvements

# Get rule counts for both branches first (needed below)
jq_count_by_rule "$D/pr_ruff_output.json" \
  > "$D/pr_all_rules_early.txt" 2>/dev/null || touch "$D/pr_all_rules_early.txt"
jq_count_by_rule "$D/main_ruff_output.json" \
  > "$D/main_all_rules_early.txt" 2>/dev/null || touch "$D/main_all_rules_early.txt"

PR_RULES=$(wc -l < "$D/pr_all_rules_early.txt" | tr -d ' ')
MAIN_RULES=$(wc -l < "$D/main_all_rules_early.txt" | tr -d ' ')
echo "  → Unique rules: PR=$PR_RULES, Main=$MAIN_RULES"

# Calculate from rule-level deltas
rm -f "$D/positive_deltas.txt" "$D/negative_deltas.txt" "$D/rules_with_new_issues.txt"
touch "$D/positive_deltas.txt" "$D/negative_deltas.txt"  # Ensure files exist (may remain empty)

cat "$D/pr_all_rules_early.txt" "$D/main_all_rules_early.txt" | \
  awk '{print $1}' | sort -u | while read rule; do
  # Use || true to prevent grep exit code 1 from killing script (pipefail)
  pr_count=$(grep "^$rule " "$D/pr_all_rules_early.txt" 2>/dev/null | awk '{print $2}' || true)
  pr_count=${pr_count:-0}
  main_count=$(grep "^$rule " "$D/main_all_rules_early.txt" 2>/dev/null | awk '{print $2}' || true)
  main_count=${main_count:-0}
  delta=$((pr_count - main_count))

  if [ "$delta" -gt 0 ]; then
    echo "$delta" >> "$D/positive_deltas.txt"
    echo "$rule" >> "$D/rules_with_new_issues.txt"
  elif [ "$delta" -lt 0 ]; then
    echo "$((- delta))" >> "$D/negative_deltas.txt"  # Store absolute value
  fi
done

# Sum deltas from files (files guaranteed to exist, may be empty)
NEW_ISSUES=$(awk '{sum+=$1} END {print sum+0}' "$D/positive_deltas.txt")
FIXED_ISSUES=$(awk '{sum+=$1} END {print sum+0}' "$D/negative_deltas.txt")

# INHERITED_ISSUES = issues that existed on main and still exist in PR
# Formula: MAIN_TOTAL - FIXED_ISSUES = issues from main that weren't fixed
INHERITED_ISSUES=$((MAIN_TOTAL - FIXED_ISSUES))
if [ "$INHERITED_ISSUES" -lt 0 ]; then
  INHERITED_ISSUES=0
fi

# Show delta breakdown with visual indicators
if [ "$NEW_ISSUES" -gt 0 ]; then
  echo "  🔴 New issues: $NEW_ISSUES"
  if [ -f "$D/rules_with_new_issues.txt" ]; then
    echo "     Rules: $(paste -sd', ' "$D/rules_with_new_issues.txt")"
  fi
else
  echo "  ✓ New issues: 0"
fi

if [ "$FIXED_ISSUES" -gt 0 ]; then
  echo "  🟢 Fixed issues: $FIXED_ISSUES"
else
  echo "  → Fixed issues: 0"
fi

echo "  → Inherited: $INHERITED_ISSUES"

# =============================================================================
# Calculate Deltas
# =============================================================================
DELTA=$((PR_TOTAL - MAIN_TOTAL))
if [ "$DELTA" -gt 0 ]; then
  DELTA_STR="+$DELTA from main"
elif [ "$DELTA" -lt 0 ]; then
  DELTA_STR="**$DELTA from main** 🎉"
else
  DELTA_STR="same as main"
fi

if [ "$PR_TOTAL" -gt 0 ] && [ "$PR_FIXABLE" -gt 0 ]; then
  FIX_PCT=$((PR_FIXABLE * 100 / PR_TOTAL))
else
  FIX_PCT=0
fi

# =============================================================================
# Calculate Delta Fixable (fixable issues in files changed by PR)
# =============================================================================
log_section "Analyzing auto-fixable issues"

# Get fixable issues only in PR-changed files using jq
# Note: ruff outputs absolute paths, git diff outputs relative paths
# We need to convert ruff's absolute paths to relative before matching
WORKSPACE=$(get_workspace_prefix)
jq -s --argjson files "$FILES_FILTER" --arg workspace "$WORKSPACE" \
  '[.[] | select(.fix and (.filename | sub($workspace; "") as $rel | $files | index($rel)))]' \
  "$D/pr_ruff_output.json" > "$D/delta_fixable_issues.json"

DELTA_FIXABLE_IN_CHANGED=$(jq 'length' "$D/delta_fixable_issues.json")
echo "  → Fixable in changed files: $DELTA_FIXABLE_IN_CHANGED"

# =============================================================================
# Filter to NEW fixable issues only (not inherited fixable)
# =============================================================================
# Filter delta_fixable_issues.json to only NEW fixable (rules with positive deltas)
if [ -f "$D/rules_with_new_issues.txt" ] && [ -s "$D/rules_with_new_issues.txt" ]; then
  # Build jq array of rules with new issues
  RULES_WITH_NEW=$(text_to_json_array "$D/rules_with_new_issues.txt")
  jq --argjson rules "$RULES_WITH_NEW" \
    '[.[] | select(.code as $c | $rules | index($c))]' \
    "$D/delta_fixable_issues.json" > "$D/new_fixable_issues.json"
else
  echo '[]' > "$D/new_fixable_issues.json"
fi

# Count NEW fixable issues by category
jq -r '[.[] | .code | gsub("[0-9].*$"; "")] | group_by(.) | map("\(.[0]) \(length)")[]' "$D/new_fixable_issues.json" \
  > "$D/delta_category_fixable.txt" 2>/dev/null || touch "$D/delta_category_fixable.txt"

# =============================================================================
# Calculate Delta Fixable Total (auto-fixable NEW issues only)
# =============================================================================
DELTA_FIXABLE=$(jq 'length' "$D/new_fixable_issues.json")

# Calculate percentage of NEW issues that are auto-fixable
if [ "$NEW_ISSUES" -gt 0 ] && [ "$DELTA_FIXABLE" -gt 0 ]; then
  DELTA_FIX_PCT=$((DELTA_FIXABLE * 100 / NEW_ISSUES))
  echo "  → Auto-fixable NEW issues: $DELTA_FIXABLE of $NEW_ISSUES ($DELTA_FIX_PCT%)"
else
  DELTA_FIX_PCT=0
  if [ "$NEW_ISSUES" -gt 0 ]; then
    echo "  ⚠️  None of the $NEW_ISSUES new issues are auto-fixable"
  else
    echo "  ✓ No new issues to fix"
  fi
fi

# =============================================================================
# Detect Error-Prone Rules (B=Bugbear, F=Pyflakes)
# =============================================================================
# These rules catch real bugs, not just style issues
HAS_ERROR_PRONE="false"
ERROR_PRONE_RULES=""

# Check if B or F categories have positive delta
for cat in B F; do
  pr_count=$(grep "^$cat " "$D/pr_category_counts.txt" 2>/dev/null | awk '{print $2}' || true)
  pr_count=${pr_count:-0}
  main_count=$(grep "^$cat " "$D/main_category_counts.txt" 2>/dev/null | awk '{print $2}' || true)
  main_count=${main_count:-0}
  delta=$((pr_count - main_count))

  if [ "$delta" -gt 0 ]; then
    HAS_ERROR_PRONE="true"
    # Map category code to full name
    case "$cat" in
      F) cat_display="Pyflakes (F)" ;;
      B) cat_display="Bugbear (B)" ;;
      *) cat_display="$cat" ;;
    esac
    if [ -z "$ERROR_PRONE_RULES" ]; then
      ERROR_PRONE_RULES="$cat_display (+$delta)"
    else
      ERROR_PRONE_RULES="$ERROR_PRONE_RULES, $cat_display (+$delta)"
    fi
  fi
done

if [ "$HAS_ERROR_PRONE" = "true" ]; then
  echo ""
  echo "  🐛 Error-prone rules detected: $ERROR_PRONE_RULES"
  echo "     These catch real bugs, not just style issues!"
fi

# =============================================================================
# Summary
# =============================================================================
log_section "Summary"
echo "  PR:   $PR_TOTAL issues ($PR_FIXABLE fixable)"
echo "  Main: $MAIN_TOTAL issues"
echo "  Delta: $DELTA_STR"
echo ""
echo "  New:       $NEW_ISSUES (you introduced)"
echo "  Fixed:     $FIXED_ISSUES (you fixed)"
echo "  Inherited: $INHERITED_ISSUES (pre-existing)"

# =============================================================================
# Export to stats.env (for sourcing by workflow)
# Quote values to handle spaces/special chars in DELTA_STR and ERROR_PRONE_RULES
# =============================================================================
cat > "$D/stats.env" << EOF
PR_TOTAL="$PR_TOTAL"
PR_FIXABLE="$PR_FIXABLE"
MAIN_TOTAL="$MAIN_TOTAL"
DELTA="$DELTA"
DELTA_STR="$DELTA_STR"
DELTA_FIXABLE="$DELTA_FIXABLE"
DELTA_FIX_PCT="$DELTA_FIX_PCT"
FIX_PCT="$FIX_PCT"
NEW_ISSUES="$NEW_ISSUES"
FIXED_ISSUES="$FIXED_ISSUES"
INHERITED_ISSUES="$INHERITED_ISSUES"
HAS_ERROR_PRONE="$HAS_ERROR_PRONE"
ERROR_PRONE_RULES="$ERROR_PRONE_RULES"
EOF

log_section "Stats exported to $D/stats.env"
