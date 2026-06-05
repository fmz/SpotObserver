#!/bin/bash
# =============================================================================
# lib.sh - Shared utilities for ruff scripts
# =============================================================================
# Common functions for logging and validation.
# Default output is maximally useful - no flags needed.
#
# Usage: source "$(dirname "$0")/lib.sh"
#
# =============================================================================

# =============================================================================
# Logging - always outputs useful info, no flags
# =============================================================================

# Section header
log_section() {
  echo "=== $* ==="
}

# Checkpoint: key metric with optional warning threshold
# Usage: log_checkpoint "description" <value> [warn_if_zero]
log_checkpoint() {
  local desc="$1"
  local value="$2"
  local warn_if_zero="${3:-false}"

  if [ "$warn_if_zero" = "true" ] && [ "$value" = "0" ]; then
    echo "  ⚠️  $desc: $value"
  else
    echo "  → $desc: $value"
  fi
}

# Warning: always shown, prefixed for visibility
log_warn() {
  echo "⚠️  $*" >&2
}

# Error: always shown to stderr
log_error() {
  echo "❌ ERROR: $*" >&2
}

# Success indicator
log_ok() {
  echo "  ✓ $*"
}

# =============================================================================
# File inspection - concise output
# =============================================================================

# Log file line count, warn if empty when shouldn't be
# Usage: log_file <file> <description> [warn_if_empty]
log_file() {
  local file="$1"
  local desc="$2"
  local warn_if_empty="${3:-false}"

  if [ ! -f "$file" ]; then
    log_error "$desc: FILE NOT FOUND"
    return 1
  fi

  local lines
  lines=$(wc -l < "$file" | tr -d ' ')

  if [ "$lines" -eq 0 ] && [ "$warn_if_empty" = "true" ]; then
    echo "  ⚠️  $desc: empty"
  else
    echo "  → $desc: $lines lines"
  fi
}

# Log JSON-lines file with issue count
# Usage: log_issues <file> <description>
log_issues() {
  local file="$1"
  local desc="$2"

  if [ ! -f "$file" ]; then
    log_error "$desc: FILE NOT FOUND"
    return 1
  fi

  local count
  count=$(wc -l < "$file" | tr -d ' ')
  echo "  → $desc: $count issues"
}

# =============================================================================
# Validation helpers
# =============================================================================

# Validate JSON file, show error details if invalid
# Usage: validate_json <file> <description>
# Returns: 0 if valid, 1 if invalid
validate_json() {
  local file="$1"
  local desc="$2"

  if jq -e . "$file" >/dev/null 2>&1; then
    log_ok "$desc: valid JSON"
    return 0
  else
    log_error "$desc: INVALID JSON - downstream processing will fail"
    echo "  First 3 lines:" >&2
    head -3 "$file" | sed 's/^/    /' >&2
    return 1
  fi
}

# Check if file has expected minimum lines
# Usage: expect_lines <file> <min_lines> <description>
expect_lines() {
  local file="$1"
  local min="$2"
  local desc="$3"

  local lines
  lines=$(wc -l < "$file" 2>/dev/null | tr -d ' ')
  lines=${lines:-0}

  if [ "$lines" -lt "$min" ]; then
    log_warn "$desc: only $lines lines (expected at least $min)"
  fi
}

# =============================================================================
# Bot commands reference (single source of truth)
# =============================================================================
# Used by: build-comment.sh (PR comments)
# Referenced by: ruff-commands.yml (header docs)

# Print bot commands table in markdown format
print_bot_commands_markdown() {
  cat << 'EOF'
| Command | Description |
|---------|-------------|
| `check` | Run lint check only (no changes) |
| `check --fix` | Auto-fix lint issues, commit to PR |
| `format` | Auto-format code, commit to PR |
| `format --check` | Check formatting only (no changes) |
| `fix` | **Recommended:** check --fix + format |
EOF
}

# =============================================================================
# Path utilities
# =============================================================================

# Get workspace prefix for jq --arg (includes trailing slash)
# Usage: ws=$(get_workspace_prefix)
# Returns: "/path/to/workspace/" or "$(pwd)/" if GITHUB_WORKSPACE unset
get_workspace_prefix() {
  echo "${GITHUB_WORKSPACE:-$(pwd)}/"
}

# =============================================================================
# jq helpers for rule extraction
# =============================================================================

# Count occurrences grouped by rule code
# Input: JSON-lines ruff output file
# Output: "RULE COUNT" per line (e.g., "F401 5")
jq_count_by_rule() {
  local input="$1"
  jq -rs '[.[] | .code] | group_by(.) | map("\(.[0]) \(length)")[]' "$input"
}

# Count occurrences grouped by category (rule prefix before digits)
# Input: JSON-lines ruff output file
# Output: "CATEGORY COUNT" per line (e.g., "F 12")
jq_count_by_category() {
  local input="$1"
  jq -rs '[.[] | .code | gsub("[0-9].*$"; "")] | group_by(.) | map("\(.[0]) \(length)")[]' "$input"
}

# Count fixable issues grouped by category
# Input: JSON-lines ruff output file
# Output: "CATEGORY COUNT" per line (fixable only)
jq_count_fixable_by_category() {
  local input="$1"
  jq -rs '[.[] | select(.fix) | .code | gsub("[0-9].*$"; "")] | group_by(.) | map("\(.[0]) \(length)")[]' "$input"
}

# Convert newline-separated text file to JSON array
# Input: text file path
# Output: JSON array of non-empty lines
text_to_json_array() {
  local file="$1"
  jq -R -s 'split("\n") | map(select(length > 0))' "$file"
}
