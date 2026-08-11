# stream_events.py
"""
习题 6.1：流式事件模型（教学版）

比官方 hello_agents.core.streaming 更轻量，面向「打字机效果」教学：
- token：模型吐出的文本片段
- tool_start / tool_result：工具生命周期（可选）
- status：步骤状态（如 Thought / Action）
- final：本轮完整回复
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional
import time

# StreamEventType：流式事件类型别名，用 Literal 限定 type 只能是下列字面量之一（便于类型检查与 IDE 提示）
StreamEventType = Literal["token", "tool_start", "tool_result", "status", "final", "error"]


@dataclass
class StreamEvent:
    """统一流式事件，UI 层按 type 订阅即可做打字机 / 工具进度条。"""

    type: StreamEventType
    data: Any
    agent_name: str = ""
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def token(cls, text: str, agent_name: str = "") -> "StreamEvent":
        return cls(type="token", data=text, agent_name=agent_name)

    @classmethod
    def status(cls, message: str, agent_name: str = "") -> "StreamEvent":
        return cls(type="status", data=message, agent_name=agent_name)

    @classmethod
    def tool_start(cls, tool_name: str, params: Any = None, agent_name: str = "") -> "StreamEvent":
        return cls(
            type="tool_start",
            data={"tool_name": tool_name, "params": params},
            agent_name=agent_name,
        )

    @classmethod
    def tool_result(cls, tool_name: str, result: str, agent_name: str = "") -> "StreamEvent":
        return cls(
            type="tool_result",
            data={"tool_name": tool_name, "result": result},
            agent_name=agent_name,
        )

    @classmethod
    def final(cls, text: str, agent_name: str = "") -> "StreamEvent":
        return cls(type="final", data=text, agent_name=agent_name)

    @classmethod
    def error(cls, message: str, agent_name: str = "") -> "StreamEvent":
        return cls(type="error", data=message, agent_name=agent_name)

    def as_text(self) -> Optional[str]:
        """仅 token / final 可直接拼成可见文本。"""
        if self.type in ("token", "final") and isinstance(self.data, str):
            return self.data
        return None
