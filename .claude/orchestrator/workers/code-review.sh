#!/bin/bash
set -e
cd '/mnt/c/Users/Danny/Documents/Github/CS4650-Nvidia-Nemotron-Challenge'
PROMPT=$(cat '/mnt/c/Users/Danny/Documents/Github/CS4650-Nvidia-Nemotron-Challenge/.claude/orchestrator/workers/code-review.prompt')
# Prefer claude-code for Max plan compatibility (uses session auth, not API credits)
# Falls back to claude (API mode) if claude-code is unavailable
# Note: Bash intentionally excluded for security - reviewers don't need shell access
if command -v claude-code &> /dev/null; then
  claude-code --allowedTools 'Read,Glob,Grep,Write' --permission-mode 'bypassPermissions' --mcp-config '{"mcpServers":{}}' -p "$PROMPT" 2>&1 | tee '/mnt/c/Users/Danny/Documents/Github/CS4650-Nvidia-Nemotron-Challenge/.claude/orchestrator/workers/code-review.log'
else
  claude --allowedTools 'Read,Glob,Grep,Write' --permission-mode 'bypassPermissions' --mcp-config '{"mcpServers":{}}' -p "$PROMPT" 2>&1 | tee '/mnt/c/Users/Danny/Documents/Github/CS4650-Nvidia-Nemotron-Challenge/.claude/orchestrator/workers/code-review.log'
fi
echo 'REVIEWER_EXITED' >> '/mnt/c/Users/Danny/Documents/Github/CS4650-Nvidia-Nemotron-Challenge/.claude/orchestrator/workers/code-review.log'
