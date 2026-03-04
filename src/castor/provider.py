from __future__ import annotations
from abc import ABC, abstractmethod

from castor.types import AssistantMessage, Message


class Provider(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> AssistantMessage: ...
