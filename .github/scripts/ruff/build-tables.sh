#!/bin/bash
# =============================================================================
# build-tables.sh - Ruff Hierarchical Delta Tables Builder
# =============================================================================
# Builds hierarchical tables with categories and rules grouped together.
# Category rows are bold with emoji, rule rows are italicized without emoji.
#
# Outputs (in .ruff-stats/):
#   - new_table.md (new issues, includes Auto-fixable column)
#   - fixed_table.md (fixed issues, no Auto-fixable column)
#
# =============================================================================
# EMOJI CODING SYSTEM
# =============================================================================
#
# Category prefixes (visual classification):
#   🐛 Pyflakes/Bugbear (F, B) - catches bugs
#   🎨 pycodestyle (E, W) - style errors/warnings
#   ⚡ pyupgrade/simplify (UP, SIM) - modern syntax
#   📦 isort/Ruff (I, RUF) - import/package organization
#   📊 NumPy/Pandas (NPY, PD) - array/dataframe patterns
#
# =============================================================================

set -euo pipefail

# Load shared utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

D=".ruff-stats"
source "$D/stats.env"

# =============================================================================
# Helper: Get category display name, tool name, and description
# Format: display|tool|description
# =============================================================================
get_category_info() {
  local cat="$1"
  case "$cat" in
    ANN) echo "📝 $cat|flake8-annotations|Type annotation issues" ;;
    F) echo "🐛 $cat|Pyflakes|Undefined names, unused imports" ;;
    E) echo "🎨 $cat|pycodestyle|PEP 8 formatting violations" ;;
    W) echo "🎨 $cat|pycodestyle|PEP 8 style warnings" ;;
    B) echo "🐛 $cat|Bugbear|Likely bugs and design issues" ;;
    UP) echo "⚡ $cat|pyupgrade|Upgrade to modern Python syntax" ;;
    SIM) echo "⚡ $cat|flake8-simplify|Simplify complex expressions" ;;
    RUF) echo "📦 $cat|Ruff|Ruff-specific linting rules" ;;
    NPY) echo "📊 $cat|NumPy|NumPy-specific patterns" ;;
    PD) echo "📊 $cat|pandas-vet|pandas-specific patterns" ;;
    I) echo "📦 $cat|isort|Import order and grouping" ;;
    C90) echo "📐 $cat|mccabe|Cyclomatic complexity" ;;
    N) echo "📝 $cat|pep8-naming|PEP 8 naming conventions" ;;
    D) echo "📝 $cat|pydocstyle|Docstring conventions" ;;
    S) echo "🔒 $cat|Bandit|Security issues" ;;
    T) echo "🐛 $cat|flake8-print|Print statements" ;;
    ERA) echo "🧹 $cat|eradicate|Commented-out code" ;;
    PL) echo "🐛 $cat|Pylint|Pylint rules" ;;
    TRY) echo "🐛 $cat|tryceratops|Exception handling" ;;
    FLY) echo "⚡ $cat|flynt|f-string conversion" ;;
    PERF) echo "⚡ $cat|Perflint|Performance issues" ;;
    FURB) echo "⚡ $cat|refurb|Code modernization" ;;
    LOG) echo "📝 $cat|flake8-logging|Logging issues" ;;
    G) echo "📝 $cat|flake8-logging-format|Logging format" ;;
    *) echo "📋 $cat|$cat|$cat rules" ;;
  esac
}

# =============================================================================
# Helper: Extract category prefix from rule code (F401 -> F, RUF022 -> RUF)
# =============================================================================
get_category_from_rule() {
  echo "$1" | sed 's/[0-9].*//'
}

# =============================================================================
# Helper: Calculate auto-fixable string for a category/rule
# Usage: get_fix_str <category_or_rule> <delta> <is_category>
#
# Fix applicability (from ruff):
#   Safe   = Fix guaranteed to preserve semantics (apply automatically)
#   Unsafe = Fix may change behavior (review before applying)
#
# Output markers:
#   ✅ = All fixable issues are safe (safe to auto-apply)
#   ❓ = Some/all fixable issues are unsafe (review before applying)
#   ❌ = No auto-fixes available (manual fix required)
# =============================================================================
get_fix_str() {
  local key="$1"
  local delta="$2"
  local is_category="$3"

  local delta_fixable
  if [ "$is_category" = "true" ]; then
    delta_fixable=$(grep "^$key " "$D/delta_category_fixable.txt" 2>/dev/null | awk '{print $2}' || true)
  else
    delta_fixable=$(grep "^$key " "$D/delta_rule_fixable.txt" 2>/dev/null | awk '{print $2}' || true)
  fi
  delta_fixable=${delta_fixable:-0}

  if [ "$delta_fixable" -gt 0 ]; then
    local SAFE_COUNT UNSAFE_COUNT
    if [ "$is_category" = "true" ]; then
      # Use regex to ensure prefix is followed by digit (prevents E matching ERA, etc.)
      SAFE_COUNT=$(jq --arg key "$key" '[.[] | select((.code | test("^" + $key + "[0-9]")) and .fix.applicability == "safe")] | length' "$D/new_fixable_issues.json" 2>/dev/null || echo '0')
      UNSAFE_COUNT=$(jq --arg key "$key" '[.[] | select((.code | test("^" + $key + "[0-9]")) and .fix.applicability == "unsafe")] | length' "$D/new_fixable_issues.json" 2>/dev/null || echo '0')
    else
      SAFE_COUNT=$(jq --arg key "$key" '[.[] | select(.code == $key and .fix.applicability == "safe")] | length' "$D/new_fixable_issues.json" 2>/dev/null || echo '0')
      UNSAFE_COUNT=$(jq --arg key "$key" '[.[] | select(.code == $key and .fix.applicability == "unsafe")] | length' "$D/new_fixable_issues.json" 2>/dev/null || echo '0')
    fi

    local FIX_LABELS=""
    [ "$SAFE_COUNT" -gt 0 ] && FIX_LABELS="$SAFE_COUNT Safe"
    [ "$UNSAFE_COUNT" -gt 0 ] && FIX_LABELS="${FIX_LABELS:+$FIX_LABELS, }$UNSAFE_COUNT Unsafe"

    local fix_pct=$((delta_fixable * 100 / delta))
    if [ "$SAFE_COUNT" -gt 0 ]; then
      echo "$FIX_LABELS ($fix_pct%) ✅"
    else
      echo "$FIX_LABELS ($fix_pct%) ❓"
    fi
  else
    echo "0 ❌"
  fi
}

# =============================================================================
# Prepare rule counts and metadata
# =============================================================================
log_section "Preparing rule data"

jq_count_by_rule "$D/pr_ruff_output.json" > "$D/pr_all_rules.txt" 2>/dev/null || touch "$D/pr_all_rules.txt"
jq_count_by_rule "$D/main_ruff_output.json" > "$D/main_rule_counts.txt" 2>/dev/null || touch "$D/main_rule_counts.txt"
# Note: new_fixable_issues.json is a JSON array (not JSON-lines), so use -r not -rs
jq -r '[.[] | .code] | group_by(.) | map("\(.[0]) \(length)")[]' "$D/new_fixable_issues.json" > "$D/delta_rule_fixable.txt" 2>/dev/null || touch "$D/delta_rule_fixable.txt"
cat "$D/pr_all_rules.txt" "$D/main_rule_counts.txt" | awk '{print $1}' | sort -u > "$D/all_rules.txt"

# =============================================================================
# Build rule rows (will be grouped by category later)
# Format: category|abs_delta|rule|desc|main_count|pr_count|delta_str|fix_str|url
# =============================================================================
log_section "Building rule rows"

> "$D/all_rule_rows.txt"

while read rule; do
  pr_count=$(grep "^$rule " "$D/pr_all_rules.txt" 2>/dev/null | awk '{print $2}' || true)
  pr_count=${pr_count:-0}
  main_count=$(grep "^$rule " "$D/main_rule_counts.txt" 2>/dev/null | awk '{print $2}' || true)
  main_count=${main_count:-0}
  delta=$((pr_count - main_count))
  [ "$delta" -eq 0 ] && continue

  # Get description from actual violation; strip backtick-quoted identifiers to keep it generic.
  raw_desc=$(jq -rs --arg rule "$rule" '[.[] | select(.code == $rule)][0].message // empty' "$D/pr_ruff_output.json" 2>/dev/null || true)
  raw_url=$(jq -rs --arg rule "$rule" '[.[] | select(.code == $rule)][0].url // empty' "$D/pr_ruff_output.json" 2>/dev/null || true)

  if [ -z "$raw_desc" ]; then
    raw_desc=$(jq -rs --arg rule "$rule" '[.[] | select(.code == $rule)][0].message // empty' "$D/main_ruff_output.json" 2>/dev/null || true)
  fi

  if [ -z "$raw_url" ]; then
    raw_url=$(jq -rs --arg rule "$rule" '[.[] | select(.code == $rule)][0].url // empty' "$D/main_ruff_output.json" 2>/dev/null || true)
  fi

  DESC=$(printf '%s' "$raw_desc" | sed 's/`[^`]*`//g; s/  */ /g; s/^ //; s/ $//' | tr -d '|\n\r')
  URL="$raw_url"

  category=$(get_category_from_rule "$rule")

  if [ "$delta" -gt 0 ]; then
    # NEW issue
    fix_str=$(get_fix_str "$rule" "$delta" "false")
    echo "new|$category|$delta|$rule|$DESC|$main_count|$pr_count|+$delta 🔴|$fix_str|$URL" >> "$D/all_rule_rows.txt"
  else
    # FIXED issue
    abs_delta=$((-delta))
    echo "fixed|$category|$abs_delta|$rule|$DESC|$main_count|$pr_count|-$abs_delta 🟢||$URL" >> "$D/all_rule_rows.txt"
  fi
done < "$D/all_rules.txt"

# =============================================================================
# Calculate total fix string for new issues
# =============================================================================
if [ "$NEW_ISSUES" -gt 0 ] && [ "$DELTA_FIXABLE" -gt 0 ]; then
  TOTAL_SAFE=$(jq '[.[] | select(.fix.applicability == "safe")] | length' "$D/new_fixable_issues.json" 2>/dev/null || echo '0')
  TOTAL_UNSAFE=$(jq '[.[] | select(.fix.applicability == "unsafe")] | length' "$D/new_fixable_issues.json" 2>/dev/null || echo '0')
  TOTAL_FIX_LABELS=""
  [ "$TOTAL_SAFE" -gt 0 ] && TOTAL_FIX_LABELS="$TOTAL_SAFE Safe"
  [ "$TOTAL_UNSAFE" -gt 0 ] && TOTAL_FIX_LABELS="${TOTAL_FIX_LABELS:+$TOTAL_FIX_LABELS, }$TOTAL_UNSAFE Unsafe"
  if [ "$TOTAL_SAFE" -gt 0 ]; then
    TOTAL_FIX_STR="$TOTAL_FIX_LABELS ($DELTA_FIX_PCT%) ✅"
  else
    TOTAL_FIX_STR="$TOTAL_FIX_LABELS ($DELTA_FIX_PCT%) ❓"
  fi
elif [ "$NEW_ISSUES" -gt 0 ]; then
  TOTAL_FIX_STR="0 ❌"
else
  TOTAL_FIX_STR="-"
fi

# =============================================================================
# Generate per-category NEW issues tables
# =============================================================================
log_section "Building new issues tables"

if [ "$NEW_ISSUES" -gt 0 ]; then
  {
    # Build category summary data (for sorting and summary table)
    > "$D/new_category_summary.txt"
    grep "^new|" "$D/all_rule_rows.txt" | cut -d'|' -f2 | sort -u | while read -r cat; do
      cat_main=$(grep "^$cat " "$D/main_category_counts.txt" 2>/dev/null | awk '{print $2}' || echo "0")
      cat_main=${cat_main:-0}

      cat_pr=$(grep "^$cat " "$D/pr_category_counts.txt" 2>/dev/null | awk '{print $2}' || echo "0")
      cat_pr=${cat_pr:-0}

      # Sum positive rule deltas for this category; category net delta may be <= 0.
      cat_delta=$(awk -F'|' -v cat="$cat" '$1 == "new" && $2 == cat {sum += $3} END {print sum + 0}' "$D/all_rule_rows.txt")

      cat_info=$(get_category_info "$cat")
      cat_display=$(echo "$cat_info" | cut -d'|' -f1)
      cat_tool=$(echo "$cat_info" | cut -d'|' -f2)
      cat_fix_str=$(get_fix_str "$cat" "$cat_delta" "true")
      echo "$cat_delta|$cat|$cat_display|$cat_tool|$cat_main|$cat_pr|$cat_fix_str"
    done | sort -t'|' -k1 -rn > "$D/new_category_summary.txt"

    # Generate per-category tables
    while IFS='|' read -r cat_delta cat_code cat_display cat_tool cat_main cat_pr cat_fix_str; do
      echo "### $cat_display — $cat_tool"
      echo ""
      echo "| Rule | Description | Main | PR | Δ | Auto-fixable |"
      echo "|------|-------------|-----:|---:|--:|-------------:|"

      grep "^new|$cat_code|" "$D/all_rule_rows.txt" | sort -t'|' -k3 -rn | while IFS='|' read -r _ _ _ rule desc main_count pr_count delta_str fix_str url; do
        if [ -n "$url" ]; then
          rule_link="[$rule]($url)"
        else
          rule_link="$rule"
        fi
        echo "| $rule_link | $desc | $main_count | $pr_count | $delta_str | $fix_str |"
      done

      echo "| **Subtotal** | | **$cat_main** | **$cat_pr** | **+$cat_delta 🔴** | **$cat_fix_str** |"
      echo ""
    done < "$D/new_category_summary.txt"

    # Summary table
    echo "### Summary"
    echo ""
    echo "| Category | Tool | Main | PR | Δ | Auto-fixable |"
    echo "|----------|------|-----:|---:|--:|-------------:|"
    while IFS='|' read -r cat_delta cat_code cat_display cat_tool cat_main cat_pr cat_fix_str; do
      echo "| $cat_display | $cat_tool | $cat_main | $cat_pr | +$cat_delta 🔴 | $cat_fix_str |"
    done < "$D/new_category_summary.txt"
    echo "| **Total** | | **$MAIN_TOTAL** | **$PR_TOTAL** | **+$NEW_ISSUES 🔴** | **$TOTAL_FIX_STR** |"

  } > "$D/new_table.md"
  CAT_COUNT=$(wc -l < "$D/new_category_summary.txt" | tr -d ' ')
  echo "  → new_table.md: $NEW_ISSUES issues across $CAT_COUNT categories"
else
  > "$D/new_table.md"
  echo "  → new_table.md: empty (no new issues)"
fi

# =============================================================================
# Generate per-category FIXED issues tables
# =============================================================================
log_section "Building fixed issues tables"

if [ "$FIXED_ISSUES" -gt 0 ]; then
  {
    # Build category summary data (for sorting and summary table)
    > "$D/fixed_category_summary.txt"
    grep "^fixed|" "$D/all_rule_rows.txt" | cut -d'|' -f2 | sort -u | while read cat; do
      cat_main=$(grep "^$cat " "$D/main_category_counts.txt" 2>/dev/null | awk '{print $2}' || echo "0")
      cat_main=${cat_main:-0}
      cat_pr=$(grep "^$cat " "$D/pr_category_counts.txt" 2>/dev/null | awk '{print $2}' || echo "0")
      cat_pr=${cat_pr:-0}
      cat_delta=$((cat_pr - cat_main))
      abs_delta=$((-cat_delta))
      cat_info=$(get_category_info "$cat")
      cat_display=$(echo "$cat_info" | cut -d'|' -f1)
      cat_tool=$(echo "$cat_info" | cut -d'|' -f2)
      echo "$abs_delta|$cat|$cat_display|$cat_tool|$cat_main|$cat_pr"
    done | sort -t'|' -k1 -rn > "$D/fixed_category_summary.txt"

    # Generate per-category tables
    while IFS='|' read -r abs_delta cat_code cat_display cat_tool cat_main cat_pr; do
      echo "### $cat_display — $cat_tool"
      echo ""
      echo "| Rule | Description | Main | PR | Δ |"
      echo "|------|-------------|-----:|---:|--:|"

      grep "^fixed|$cat_code|" "$D/all_rule_rows.txt" | sort -t'|' -k3 -rn | while IFS='|' read -r _ _ _ rule desc main_count pr_count delta_str _ url; do
        if [ -n "$url" ]; then
          rule_link="[$rule]($url)"
        else
          rule_link="$rule"
        fi
        echo "| $rule_link | $desc | $main_count | $pr_count | $delta_str |"
      done

      echo "| **Subtotal** | | **$cat_main** | **$cat_pr** | **-$abs_delta 🟢** |"
      echo ""
    done < "$D/fixed_category_summary.txt"

    # Summary table
    echo "### Summary"
    echo ""
    echo "| Category | Tool | Main | PR | Δ |"
    echo "|----------|------|-----:|---:|--:|"
    while IFS='|' read -r abs_delta cat_code cat_display cat_tool cat_main cat_pr; do
      echo "| $cat_display | $cat_tool | $cat_main | $cat_pr | -$abs_delta 🟢 |"
    done < "$D/fixed_category_summary.txt"
    echo "| **Total** | | **$MAIN_TOTAL** | **$PR_TOTAL** | **-$FIXED_ISSUES 🟢** |"

  } > "$D/fixed_table.md"
  CAT_COUNT=$(wc -l < "$D/fixed_category_summary.txt" | tr -d ' ')
  echo "  → fixed_table.md: $FIXED_ISSUES issues across $CAT_COUNT categories"
else
  > "$D/fixed_table.md"
  echo "  → fixed_table.md: empty (no fixed issues)"
fi

log_section "Tables built"
