# Agent Loop Design — Castor

## Summary

Async Python agent loop: non-streaming, user-priority interruption, parallel tool execution with conflict avoidance.

## Architecture

```
src/castor/
├── __init__.py
├── types.py       # Message, ToolCall, ToolResult, Response
├── loop.py        # agent_loop() — the core
├── provider.py    # LLM provider interface (abstract)
└── tool.py        # Tool registry + base tool (abstract)
```

## Core Principles

1. **Honest assistant messages** — never modify, fabricate, or inject assistant messages. Only real LLM output goes in the assistant role. Incomplete output is discarded, not patched.
2. **Append-only history** — never mutate past messages. Interruptions are recorded as new entries (synthetic tool results, prefixed user messages), not edits to existing ones.
3. **Agent awareness of state changes** — when tools are interrupted mid-execution, the agent must know what completed and what didn't. Cancelled tools get explicit `[cancelled by user]` results so the LLM understands the current state of the world.

## Data Types

### Message

Union of three roles:
- `user` — user input (may include interruption prefix)
- `assistant` — LLM output: text content + optional tool calls
- `tool` — tool execution result, linked to a tool call by ID

### ToolCall

LLM-requested tool invocation: id, tool name, arguments dict.

### ToolResult

Execution outcome: tool call id, output string, success/error flag.

### Response

LLM response: text content + list of tool calls (may be empty).

## Agent Loop

Non-streaming. LLM returns complete response, tools execute, loop repeats until no tool calls.

```
User Message
    ↓
[Build Context]
    ↓
┌─→ [Call LLM] → [Parse Response]
│        ↓
│   Has tool calls?
│   ├─ No  → Return response
│   └─ Yes → [Execute Tools] → [Append Results] → loop back
│
│   *** At any point: new user message → abort + restart ***
└────────────────────────────────────────────────────────────
```

### Interruption Model

New user message always takes priority. `asyncio.Event` as cancel signal, checked at each loop boundary. Max iterations cap (20) as safety net.

```
Case 1: Interrupt during first LLM call (no complete assistant message)
  → Discard incomplete output, merge user messages
  user: "do X"
  user: "[user interrupted: 'do X'] do Y"

Case 2: Interrupt during tool execution
  → Keep assistant + completed tool results, cancel pending tools
  user: "do X"
  assistant: "I'll do X" [tool_calls...]
  tool: [completed results + "[cancelled by user]" for pending]
  user: "[user interrupted previous task] do Y"

Case 3: Interrupt during LLM call mid-loop (after tools ran)
  → Keep prior assistant + tool results, discard incomplete assistant
  user: "do X"
  assistant: "I'll do X" [tool_calls...]
  tool: [results...]
  ← discard incomplete second assistant message →
  user: "[user interrupted previous task] do Y"
```

### Parallel Tool Execution

Tools execute in parallel by default, same-resource operations run sequentially.

- Each tool declares its target resource
- Tools with same resource are grouped and run sequentially
- Groups run in parallel via asyncio.gather
- On cancel: finished tools keep results, pending get `[cancelled by user]`
- Resource grouping deferred to tool registry implementation

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Streaming | Non-streaming | Simpler parsing, parallel tools easier, streaming layered on later |
| Interruption | User-priority abort | User always wins, partial work preserved honestly |
| Tool execution | Parallel with resource grouping | Fast by default, safe for conflicting resources |
| Partial results | Keep completed, cancel pending | LLM needs state awareness to avoid redoing work |
| Failed tools | LLM decides retry | Error in tool result, LLM can retry or pivot |
| Incomplete LLM output | Discard | Partial JSON is unusable |
| History mutation | Append-only | Honest, auditable, cache-friendly |
