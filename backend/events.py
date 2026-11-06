"""In-process event bus for live progress (jobs, agent workflow stages, tool calls) streamed over SSE.

Producers run in worker threads; subscribers are asyncio queues in the API's event loop.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import threading
import time
from collections import defaultdict, deque
from typing import Any

current_run: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_run", default=None)


class EventBus:
    def __init__(self, history: int = 1000):
        self._history: dict[str, deque] = defaultdict(lambda: deque(maxlen=history))
        self._subs: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = defaultdict(list)
        self._lock = threading.Lock()
        self._seq = 0

    def publish(self, channel: str, event: dict[str, Any]) -> dict:
        with self._lock:
            self._seq += 1
            ev = {"seq": self._seq, "time": time.time(), "channel": channel, **event}
            self._history[channel].append(ev)
            subs = list(self._subs[channel]) + list(self._subs["*"])
        for loop, q in subs:
            try:
                loop.call_soon_threadsafe(q.put_nowait, ev)
            except RuntimeError:  # loop closed
                pass
        return ev

    def history(self, channel: str) -> list[dict]:
        with self._lock:
            return list(self._history.get(channel, ()))

    def subscribe(self, channel: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subs[channel].append((asyncio.get_running_loop(), q))
        return q

    def unsubscribe(self, channel: str, q: asyncio.Queue) -> None:
        with self._lock:
            self._subs[channel] = [(l, s) for l, s in self._subs[channel] if s is not q]


bus = EventBus()


def sse_format(event: dict) -> str:
    return f"id: {event.get('seq', '')}\nevent: {event.get('type', 'message')}\ndata: {json.dumps(event, default=str)}\n\n"
