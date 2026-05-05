#!/bin/bash
set -e
cd '/mnt/c/Users/DangT/Documents/GitHub/CS4650-Nvidia-Nemotron-Challenge'
PROMPT=$(cat '/mnt/c/Users/DangT/Documents/GitHub/CS4650-Nvidia-Nemotron-Challenge/.claude/orchestrator/workers/feature-6.prompt')
# Prefer claude-code for Max plan compatibility (uses session auth, not API credits)
# Falls back to claude (API mode) if claude-code is unavailable
# Worker flags configured via env vars: CLAUDE_SWARM_ALLOWED_TOOLS, CLAUDE_SWARM_PERMISSION_MODE, CLAUDE_SWARM_MCP_SERVERS
if command -v claude-code &> /dev/null; then
  claude-code --allowedTools 'Bash,Read,Write,Edit,Glob,Grep' --permission-mode 'bypassPermissions' --mcp-config '{"mcpServers":{}}' -p "$PROMPT" 2>&1 | tee '/mnt/c/Users/DangT/Documents/GitHub/CS4650-Nvidia-Nemotron-Challenge/.claude/orchestrator/workers/feature-6.log'
else
  claude --allowedTools 'Bash,Read,Write,Edit,Glob,Grep' --permission-mode 'bypassPermissions' --mcp-config '{"mcpServers":{}}' -p "$PROMPT" 2>&1 | tee '/mnt/c/Users/DangT/Documents/GitHub/CS4650-Nvidia-Nemotron-Challenge/.claude/orchestrator/workers/feature-6.log'
fi
echo 'WORKER_EXITED' >> '/mnt/c/Users/DangT/Documents/GitHub/CS4650-Nvidia-Nemotron-Challenge/.claude/orchestrator/workers/feature-6.log'
