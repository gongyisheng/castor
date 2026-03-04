from __future__ import annotations
import asyncio

from castor.provider import Provider
from castor.tool import ToolRegistry
from castor.types import (
    AssistantMessage, Message, ToolMessage, UserMessage,
)


async def agent_loop(
    messages: list[Message],
    provider: Provider,
    registry: ToolRegistry,
    cancel: asyncio.Event,
    max_iter: int = 20,
) -> list[Message]:
    for _ in range(max_iter):
        if cancel.is_set():
            return messages

        response = await provider.complete(messages, registry.schemas())
        if cancel.is_set():
            return messages

        messages.append(response)

        if not response.tool_calls:
            return messages

        results = await _execute_tools(response, registry, cancel)
        messages.extend(results)

    return messages


async def _execute_tools(
    response: AssistantMessage,
    registry: ToolRegistry,
    cancel: asyncio.Event,
) -> list[ToolMessage]:
    results: list[ToolMessage] = []

    tasks = {
        tc.id: asyncio.create_task(registry.execute(tc))
        for tc in response.tool_calls
    }

    for tc in response.tool_calls:
        if cancel.is_set():
            for remaining_tc in response.tool_calls:
                if remaining_tc.id not in {r.tool_call_id for r in results}:
                    tasks[remaining_tc.id].cancel()
                    results.append(ToolMessage.cancelled(remaining_tc.id))
            break

        try:
            tool_result = await tasks[tc.id]
            results.append(
                ToolMessage(
                    tool_call_id=tool_result.tool_call_id,
                    content=tool_result.output,
                    is_error=tool_result.is_error,
                )
            )
        except asyncio.CancelledError:
            results.append(ToolMessage.cancelled(tc.id))

    return results
