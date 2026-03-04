# Agent Loop Design — Castor

## Overview

Non-streaming agent loop with user-priority interruption and parallel tool execution.

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

## Design Decisions

### 1. Non-Streaming

LLM calls return complete responses. Streaming is a presentation concern layered on top later.
- Simpler tool call parsing (full response available)
- Parallel tool execution is easier with complete tool call list upfront

### 2. User-Priority Interruption

New user message always takes priority over the current loop.

**Mechanism:**
- `asyncio.Event` as cancel signal, checked at each loop boundary
- On new user message: set cancel event → cancel current LLM call → cancel running tools
- Max iterations cap (e.g. 20) as safety net

**Partial results handling — keep completed work:**
- Completed tool results stay in history
- Cancelled/orphaned tool calls get synthetic `[cancelled by user]` tool result (keeps API contract: every tool call has a result)
**Interruption cases:**

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

**Core Principles:**
1. **Honest assistant messages** — never modify, fabricate, or inject assistant messages. Only real LLM output goes in the assistant role. Incomplete output is discarded, not patched.
2. **Append-only history** — never mutate past messages. Interruptions are recorded as new entries (synthetic tool results, prefixed user messages), not edits to existing ones.
3. **Agent awareness of state changes** — when tools are interrupted mid-execution, the agent must know what completed and what didn't. Cancelled tools get explicit `[cancelled by user]` results so the LLM understands the current state of the world.

### 3. Parallel Tool Execution with Conflict Avoidance

Tools execute in parallel by default, but same-resource operations run sequentially.

**Strategy: resource-grouped execution**
```
Tool calls: [write(auth.py, L10), write(auth.py, L20), write(test.py)]

Group by resource:
  auth.py → [write L10, write L20]  ← sequential within group
  test.py → [write]                 ← parallel across groups
```

**Conflict detection:**
- Each tool declares its target resource (file path, URL, etc.)
- Tools with same resource are grouped and run sequentially
- Groups run in parallel via asyncio.gather
- Tools with no declared resource run in parallel (assumed independent)

**Cancellation during parallel execution:**
- Cancel event checked after each tool group completes
- On cancel: already-finished tools keep results, pending tools get `[cancelled]`

## Loop Pseudocode

```python
async def agent_loop(messages, tools, cancel_event, max_iter=20):
    for i in range(max_iter):
        if cancel_event.is_set():
            return messages

        response = await llm_call(messages, tools, cancel_event)
        if cancel_event.is_set():
            # keep partial assistant message if any
            return messages

        messages.append(assistant_message(response))

        if not response.tool_calls:
            return messages

        # group tool calls by resource, execute with parallelism
        results = await execute_tool_groups(response.tool_calls, cancel_event)
        messages.append(tool_results(results))

    return messages  # max iterations reached
```

## Resolved Questions

- **Tool resource grouping**: Deferred to tool registry implementation. Each tool will declare its resource. Loop provides the hook point for grouped execution.
- **Cancelled LLM calls**: Discard partial output. Incomplete tool call JSON is unusable.
- **Retry policy**: LLM decides. Failed tool results are appended to history — LLM can retry or pivot.
