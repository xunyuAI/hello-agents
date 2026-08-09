# plugins/demo_plugins.py
"""示例插件：演示如何在不改框架核心的前提下扩展 Agent / Tool / Provider。"""

from __future__ import annotations

from typing import Any, Optional

from hello_agents import SimpleAgent, HelloAgentsLLM, Config, Message

from plugin_system import Plugin, HelloAgentsApp


class EchoAgent(SimpleAgent):
    """
    继承官方 SimpleAgent：复用历史 / 配置，run 内调用真实 LLM 生成回复。
    （名称保留 Echo，语义为「插件注册的示例 Agent」，不再硬编码假回复。）
    """

    def run(self, input_text: str, **kwargs) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": input_text})

        raw = self.llm.invoke(messages, **kwargs)
        reply = raw.content if hasattr(raw, "content") else str(raw)

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(reply, "assistant"))
        return reply


class EchoAgentPlugin(Plugin):
    name = "echo-agent"
    version = "1.0.0"

    def setup(self, app: HelloAgentsApp) -> None:
        app.register_agent_type("echo", EchoAgent)


class GreeterToolPlugin(Plugin):
    name = "greeter-tool"
    version = "1.0.0"

    def setup(self, app: HelloAgentsApp) -> None:
        def greet(name: str = "朋友") -> str:
            return f"你好，{name}！欢迎使用 HelloAgents 插件系统。"

        app.register_function(
            name="greet",
            description="向用户打招呼",
            func=greet,
        )


class HelloAgentsProviderPlugin(Plugin):
    """注册真实 HelloAgentsLLM 工厂（读取环境变量 / .env）。"""

    name = "helloagents-provider"
    version = "1.0.0"

    def setup(self, app: HelloAgentsApp) -> None:
        def factory(**kwargs: Any) -> HelloAgentsLLM:
            return HelloAgentsLLM(**kwargs)

        app.register_provider("helloagents", factory)
