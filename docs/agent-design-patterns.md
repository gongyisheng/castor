# Common Agent Design Patterns

Derived from reviewing Bub, Nanobot, Pi, and ZeroClaw agent designs.

## Core Layered Architecture

All agents converge on four layers:
- **Channel** (I/O) → **Agent Loop** (orchestration) → **Tools** (capabilities) → **Memory** (state)

## Key Patterns

### 1. Append-Only Session History (JSONL)
Canonical format for conversation history. Enables replay, auditing, prompt caching (prefix stability), and branching without mutation.

### 2. Agent Loop
Central orchestration: consume input → call LLM → route/execute tool calls → feed results back → repeat.

### 3. Tool Registry + Dynamic Discovery
Registry pattern — tools registered with schemas, dispatched by name. Progressive/lazy loading: summaries in prompt, full schemas on demand. MCP integration common.

### 4. Multi-Channel Architecture
Agent core decoupled from I/O channels (Telegram, Discord, Slack, etc.) via queues/adapters. Same agent serves multiple platforms with per-sender/session isolation.

### 5. Skill/Extension System
User-defined behaviors — markdown skills (Bub, Nanobot), TypeScript extensions (Pi), WASM plugins (ZeroClaw). Pattern: summaries in context, full content loaded on demand.

### 6. Memory Management with Compaction
Strategies: LLM-driven summarization, context compaction, tape forking, memory decay with classification. Goal: keep context relevant without permanent information loss.

### 7. Session Isolation
Per-user/per-conversation isolated state prevents cross-contamination. Implementations: forked tapes, session trees, per-channel sessions, per-sender history.

### 8. Scheduling / Cron
Built-in scheduler for recurring or one-shot tasks — cron expressions, intervals, timestamps.

### 9. Provider Abstraction
Multiple LLM providers behind a unified interface. Advanced: fallback chains, health checks, cost tracking, streaming event normalization.
