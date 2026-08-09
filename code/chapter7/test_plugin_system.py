# test_plugin_system.py
"""
习题 6.3：插件系统测试（真实 LLM）

验证：
- Plugin.setup 通过 HelloAgentsApp 注册 Agent / Tool / Provider
- 注册的 Agent 必须继承 Agent（拒绝无关类型）
- create_agent + 真实 LLM 能产出模型回复
- load_from_path / unload / discover
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM
from hello_agents.core.agent import Agent

from plugin_system import HelloAgentsApp, PluginManager
from plugins.demo_plugins import (
    EchoAgent,
    EchoAgentPlugin,
    GreeterToolPlugin,
    HelloAgentsProviderPlugin,
)

_CHAPTER7 = Path(__file__).resolve().parent
if str(_CHAPTER7) not in sys.path:
    sys.path.insert(0, str(_CHAPTER7))

load_dotenv(_CHAPTER7 / ".env")

llm = HelloAgentsLLM()


def test_register_and_create_echo_agent():
    app = HelloAgentsApp()
    mgr = PluginManager(app)
    mgr.load_plugin(EchoAgentPlugin())

    assert "echo" in app.agent_types
    assert issubclass(EchoAgent, Agent)
    assert app.agent_types["echo"] is EchoAgent

    agent = app.create_agent(
        "echo",
        name="回声",
        llm=llm,
        system_prompt="用不超过20字简短回答。",
    )
    question = "用一个词形容晴天"
    print(f"\n=== 插件 Agent: {question} ===")
    reply = agent.run(question)
    print(f"真实回复: {reply}")

    assert isinstance(reply, str) and reply.strip()
    assert "[EchoAgent]" not in reply  # 不再使用硬编码假回复
    assert len(agent.get_history()) == 2
    assert agent.get_history()[1].content == reply


def test_reject_non_agent_type():
    app = HelloAgentsApp()

    class NotAnAgent:
        pass

    try:
        app.register_agent_type("bad", NotAnAgent)  # type: ignore[arg-type]
        assert False, "应拒绝非 Agent 子类"
    except TypeError as e:
        assert "Agent" in str(e)


def test_tool_and_provider_plugins():
    app = HelloAgentsApp()
    mgr = PluginManager(app)
    mgr.load_plugin(GreeterToolPlugin())
    mgr.load_plugin(HelloAgentsProviderPlugin())

    tools = app.tool_registry.list_tools()
    funcs = list(getattr(app.tool_registry, "_functions", {}).keys())
    assert "greet" in tools or "greet" in funcs

    result = app.tool_registry.execute_tool("greet", "小明")
    text = getattr(result, "text", None) or str(result)
    assert "小明" in text

    created = app.create_llm("helloagents")
    assert isinstance(created, HelloAgentsLLM)
    raw = created.invoke([{"role": "user", "content": "只回复两个字：通过"}])
    content = raw.content if hasattr(raw, "content") else str(raw)
    print(f"\n=== Provider 真实调用: {content} ===")
    assert isinstance(content, str) and content.strip()


def test_load_from_path_and_unload():
    app = HelloAgentsApp()
    mgr = PluginManager(app)
    plugin = mgr.load_from_path("plugins.demo_plugins:EchoAgentPlugin")
    assert plugin.name == "echo-agent"
    assert "echo" in app.agent_types

    mgr.unload_plugin("echo-agent")
    assert "echo-agent" not in mgr.plugins


def test_discover_plugins_dir():
    app = HelloAgentsApp()
    mgr = PluginManager(app)
    loaded = mgr.discover(_CHAPTER7 / "plugins")
    names = {p.name for p in loaded}
    assert "echo-agent" in names or "echo-agent" in mgr.plugins
    assert "helloagents-provider" in mgr.plugins
    assert len(mgr.plugins) >= 1


if __name__ == "__main__":
    test_register_and_create_echo_agent()
    test_reject_non_agent_type()
    test_tool_and_provider_plugins()
    test_load_from_path_and_unload()
    test_discover_plugins_dir()
    print("\n✅ test_plugin_system 全部通过（真实 LLM）")
