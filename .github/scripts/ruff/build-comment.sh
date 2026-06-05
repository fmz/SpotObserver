#!/bin/bash
# =============================================================================
# build-comment.sh - Ruff PR Comment Builder
# =============================================================================
# Assembles the final PR comment from stats and tables.
#
# Inputs (in .ruff-stats/):
#   - stats.env, category_delta_table.md, rules_delta_table.md, rules_rows_sorted.txt
#
# Environment variables (from caller):
#   - FORMAT_FAILED: "true" if format check failed
#
# Outputs (in .ruff-stats/):
#   - comment_body.md: Final PR comment body
#
# =============================================================================
# COMMENT STRUCTURE & PHILOSOPHY
# =============================================================================
#
# Title shows delta-based status:
#   - Passing: "🧹 Ruff — ✅ No new issues" or "✅ 5 issues fixed"
#   - Failing: "🧹 Ruff — ❌ 4 new issues" or "❌ 4 new issues, 2 fixed"
#
# Summary card (compact, scannable):
#   - Lint: Shows NEW/FIXED breakdown, NOT just net delta
#   - Top offender: Rule with biggest regression (if failing)
#   - Format: Pass/fail status
#   - Auto-fix: Percentage of NEW issues auto-fixable
#   - Inherited: Count of pre-existing issues in modified code
#
# Required Actions (only when failing):
#   - Numbered steps: auto-fix → format → manual fix → push
#   - Steps shown conditionally based on what's needed
#   - Doesn't ask you to fix inherited issues
#
# Context-aware tips:
#   - Bug-related issues (F, B): Different tip than style issues
#   - Emphasizes local testing before push
#
# =============================================================================

set -euo pipefail

# Load shared utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

D=".ruff-stats"
source "$D/stats.env"
FORMAT_FILES_CHECKED="${FORMAT_FILES_CHECKED:-0}"
FORMAT_NEW="${FORMAT_NEW:-0}"
FORMAT_FIXED="${FORMAT_FIXED:-0}"
FORMAT_INHERITED="${FORMAT_INHERITED:-0}"
DELTA_FIXABLE="${DELTA_FIXABLE:-0}"

# Determine pass/fail status
REGRESSION_STATUS="fail"
FORMAT_STATUS="fail"
[ "$NEW_ISSUES" -eq 0 ] && REGRESSION_STATUS="pass"
[ "$FORMAT_NEW" -eq 0 ] && FORMAT_STATUS="pass"

OVERALL_STATUS="pass"
[ "$REGRESSION_STATUS" = "fail" ] || [ "$FORMAT_STATUS" = "fail" ] && OVERALL_STATUS="fail"

# Calculate top new issue (rule with most new issues)
# Format: type|category|abs_delta|rule|desc|main_count|pr_count|delta_str|fix_str|url
TOP_NEW=""
if [ "$NEW_ISSUES" -gt 0 ] && [ -s "$D/all_rule_rows.txt" ]; then
  TOP_ROW=$(grep "^new|" "$D/all_rule_rows.txt" 2>/dev/null | sort -t'|' -k3 -rn | head -1 || true)
  if [ -n "$TOP_ROW" ]; then
    TOP_RULE=$(echo "$TOP_ROW" | cut -d'|' -f4)
    TOP_DELTA=$(echo "$TOP_ROW" | cut -d'|' -f8)
    TOP_NEW="$TOP_RULE $TOP_DELTA"
  fi
fi

# Calculate top fixed issue (rule with most fixed issues)
TOP_FIXED=""
if [ "$FIXED_ISSUES" -gt 0 ] && [ -s "$D/all_rule_rows.txt" ]; then
  TOP_ROW=$(grep "^fixed|" "$D/all_rule_rows.txt" 2>/dev/null | sort -t'|' -k3 -rn | head -1 || true)
  if [ -n "$TOP_ROW" ]; then
    TOP_RULE=$(echo "$TOP_ROW" | cut -d'|' -f4)
    TOP_DELTA=$(echo "$TOP_ROW" | cut -d'|' -f8)
    TOP_FIXED="$TOP_RULE $TOP_DELTA"
  fi
fi

# =============================================================================
# Build comment body
# =============================================================================

# Build title: 🧹 Ruff — [status] [what happened]
if [ "$OVERALL_STATUS" = "pass" ]; then
  if [ "$FIXED_ISSUES" -gt 0 ]; then
    TITLE="## 🧹 Ruff — ✅ $FIXED_ISSUES issues fixed"
  else
    TITLE="## 🧹 Ruff — ✅ No new issues"
  fi
else
  if [ "$REGRESSION_STATUS" = "fail" ]; then
    if [ "$FIXED_ISSUES" -gt 0 ]; then
      TITLE="## 🧹 Ruff — ❌ $NEW_ISSUES new issues, $FIXED_ISSUES fixed"
    else
      TITLE="## 🧹 Ruff — ❌ $NEW_ISSUES new issues"
    fi
  else
    TITLE="## 🧹 Ruff — ❌ $FORMAT_NEW files need formatting"
  fi
fi

cat > "$D/comment_body.md" << HEADER
<!-- ruff-pr-summary -->
$TITLE
HEADER

# Summary card
echo "" >> "$D/comment_body.md"
echo "| Check | Result |" >> "$D/comment_body.md"
echo "|-------|--------|" >> "$D/comment_body.md"

# Lint row with auto-fix and inherited inline
if [ "$REGRESSION_STATUS" = "pass" ]; then
  if [ "$FIXED_ISSUES" -gt 0 ]; then
    if [ "$INHERITED_ISSUES" -gt 0 ]; then
      echo "| **Lint** | ✅ $FIXED_ISSUES fixed ($INHERITED_ISSUES inherited) |" >> "$D/comment_body.md"
    else
      echo "| **Lint** | ✅ $FIXED_ISSUES fixed |" >> "$D/comment_body.md"
    fi
  elif [ "$INHERITED_ISSUES" -gt 0 ]; then
    echo "| **Lint** | ✅ Passing ($INHERITED_ISSUES inherited) |" >> "$D/comment_body.md"
  else
    echo "| **Lint** | ✅ Passing |" >> "$D/comment_body.md"
  fi
else
  # Build lint result: "N new (M auto-fixable), K inherited"
  LINT_RESULT="$NEW_ISSUES new"
  [ "$DELTA_FIXABLE" -gt 0 ] && LINT_RESULT="$LINT_RESULT ($DELTA_FIXABLE auto-fixable)"
  [ "$INHERITED_ISSUES" -gt 0 ] && LINT_RESULT="$LINT_RESULT, $INHERITED_ISSUES inherited"
  echo "| **Lint** | ❌ $LINT_RESULT |" >> "$D/comment_body.md"
fi

[ -n "$TOP_NEW" ] && echo "| Top new | $TOP_NEW |" >> "$D/comment_body.md"
[ -n "$TOP_FIXED" ] && echo "| Top fixed | $TOP_FIXED |" >> "$D/comment_body.md"

if [ "$FORMAT_STATUS" = "pass" ]; then
  if [ "$FORMAT_FIXED" -gt 0 ]; then
    echo "| **Format** | ✅ $FORMAT_FIXED fixed ($FORMAT_FILES_CHECKED files) |" >> "$D/comment_body.md"
  elif [ "$FORMAT_INHERITED" -gt 0 ]; then
    echo "| **Format** | ✅ Passing ($FORMAT_INHERITED inherited, $FORMAT_FILES_CHECKED files) |" >> "$D/comment_body.md"
  else
    echo "| **Format** | ✅ Passing ($FORMAT_FILES_CHECKED files) |" >> "$D/comment_body.md"
  fi
else
  if [ "$FORMAT_INHERITED" -gt 0 ]; then
    echo "| **Format** | ❌ $FORMAT_NEW file(s) need formatting, $FORMAT_INHERITED inherited ($FORMAT_FILES_CHECKED checked) |" >> "$D/comment_body.md"
  else
    echo "| **Format** | ❌ $FORMAT_NEW file(s) need formatting ($FORMAT_FILES_CHECKED checked) |" >> "$D/comment_body.md"
  fi
fi
echo "" >> "$D/comment_body.md"

# Format files list (if format failed)
# Try format_files.txt first, fall back to format_new.txt
FORMAT_FILES_LIST=""
if [ -s "$D/format_files.txt" ]; then
  FORMAT_FILES_LIST="$D/format_files.txt"
elif [ -s "$D/format_new.txt" ]; then
  FORMAT_FILES_LIST="$D/format_new.txt"
fi

if [ "$FORMAT_STATUS" = "fail" ] && [ -n "$FORMAT_FILES_LIST" ]; then
  echo "" >> "$D/comment_body.md"
  echo "**Files needing format:**" >> "$D/comment_body.md"
  echo '```' >> "$D/comment_body.md"
  cat "$FORMAT_FILES_LIST" >> "$D/comment_body.md"
  echo '```' >> "$D/comment_body.md"
fi

# High-priority alerts
if [ "$HAS_ERROR_PRONE" = "true" ]; then
  echo "" >> "$D/comment_body.md"
  echo "> [!IMPORTANT]" >> "$D/comment_body.md"
  echo "> 🐛 **$ERROR_PRONE_RULES** — catches real bugs, not just style issues. Review carefully." >> "$D/comment_body.md"
fi

# Required Actions section
if [ "$OVERALL_STATUS" = "fail" ]; then
  echo "" >> "$D/comment_body.md"
  echo "> [!IMPORTANT]" >> "$D/comment_body.md"

  STEP=1
  REMAINING=$((NEW_ISSUES - DELTA_FIXABLE))

  # Step 1: Auto-fix (if there are auto-fixable issues or format failed)
  if { [ "$REGRESSION_STATUS" = "fail" ] && [ "$DELTA_FIXABLE" -gt 0 ]; } || [ "$FORMAT_STATUS" = "fail" ]; then
    echo "> $STEP. Auto-fix issues:" >> "$D/comment_body.md"

    # Build VSCode commands based on what's needed
    VSCODE_CMDS=""
    if [ "$FORMAT_STATUS" = "fail" ]; then
      VSCODE_CMDS="\"Ruff: Format document\" + \"Ruff: Format imports\""
    fi
    if [ "$REGRESSION_STATUS" = "fail" ] && [ "$DELTA_FIXABLE" -gt 0 ]; then
      if [ -n "$VSCODE_CMDS" ]; then
        VSCODE_CMDS="$VSCODE_CMDS + \"Ruff: Fix all auto-fixable problems\""
      else
        VSCODE_CMDS="\"Ruff: Fix all auto-fixable problems\""
      fi
    fi
    echo ">    - **VSCode:** $VSCODE_CMDS" >> "$D/comment_body.md"

    # Build CLI command based on what's needed
    CLI_CMDS=""
    if [ "$REGRESSION_STATUS" = "fail" ] && [ "$DELTA_FIXABLE" -gt 0 ]; then
      CLI_CMDS="ruff check --fix ."
    fi
    if [ "$FORMAT_STATUS" = "fail" ]; then
      if [ -n "$CLI_CMDS" ]; then
        CLI_CMDS="$CLI_CMDS && ruff format ."
      else
        CLI_CMDS="ruff format ."
      fi
    fi
    echo ">    - **CLI:** \`$CLI_CMDS\`" >> "$D/comment_body.md"

    # Build bot command based on what's needed
    if [ "$REGRESSION_STATUS" = "fail" ] && [ "$DELTA_FIXABLE" -gt 0 ] && [ "$FORMAT_STATUS" = "fail" ]; then
      BOT_CMD="/ruff fix"
    elif [ "$REGRESSION_STATUS" = "fail" ] && [ "$DELTA_FIXABLE" -gt 0 ]; then
      BOT_CMD="/ruff check --fix"
    else
      BOT_CMD="/ruff format"
    fi
    echo ">    - **Bot:** Comment \`$BOT_CMD\` on this PR" >> "$D/comment_body.md"
    STEP=$((STEP + 1))
  fi

  # Step: Manual fixes (if there are non-auto-fixable NEW issues)
  if [ "$REGRESSION_STATUS" = "fail" ] && [ "$REMAINING" -gt 0 ]; then
    echo "> $STEP. Manually fix $REMAINING new issue(s) that can't be auto-fixed (see **Files changed**)" >> "$D/comment_body.md"
    STEP=$((STEP + 1))
  fi

  # Step: Push
  echo "> $STEP. Push" >> "$D/comment_body.md"

  # VSCode setup tips (only when failing - remediation focused)
  cat >> "$D/comment_body.md" << 'EOF'

> [!TIP]
> <details>
> <summary>VSCode: Set up auto-format and code actions on save</summary>
>
> **Settings UI:** Search "format on save" and "code actions on save"
>
> **settings.json:**
> ```json
> "editor.formatOnSave": true,
> "editor.codeActionsOnSave": {
>   "source.fixAll.ruff": "explicit",
>   "source.organizeImports.ruff": "explicit"
> }
> ```
>
> [Full setup guide →](https://docs.astral.sh/ruff/editors/setup/#vs-code)
>
> </details>
EOF
fi

# New Issues section (shown first)
if [ "$NEW_ISSUES" -gt 0 ] && [ -s "$D/new_table.md" ]; then
  echo "" >> "$D/comment_body.md"
  echo "## 🔴 New Issues ($NEW_ISSUES)" >> "$D/comment_body.md"
  echo "" >> "$D/comment_body.md"
  cat "$D/new_table.md" >> "$D/comment_body.md"
fi

# Fixed Issues section (shown last)
if [ "$FIXED_ISSUES" -gt 0 ] && [ -s "$D/fixed_table.md" ]; then
  echo "" >> "$D/comment_body.md"
  echo "## 🟢 Fixed Issues ($FIXED_ISSUES)" >> "$D/comment_body.md"
  echo "" >> "$D/comment_body.md"
  cat "$D/fixed_table.md" >> "$D/comment_body.md"
fi

# Bot commands reference (always shown - useful reference info)
# Uses print_bot_commands_markdown from lib.sh (single source of truth)
{
  echo ""
  echo "---"
  echo ""
  echo "<details>"
  echo "<summary><strong>Bot commands</strong> for this PR</summary>"
  echo ""
  echo "Type \`/ruff\` or \`@ruff\` followed by a command in a PR comment:"
  echo ""
  print_bot_commands_markdown
  echo ""
  echo "**Reactions:** 👀 = processing, 🚀 = all passing, 😕 = issues remain (details in reply)"
  echo ""
  echo "</details>"
} >> "$D/comment_body.md"

# Logs link (at very end)
if [ -n "${GITHUB_RUN_ID:-}" ] && [ -n "${GITHUB_REPOSITORY:-}" ]; then
  echo "" >> "$D/comment_body.md"
  echo "[View full logs →](https://github.com/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID)" >> "$D/comment_body.md"
fi

echo "=== Comment body ==="
cat "$D/comment_body.md"
