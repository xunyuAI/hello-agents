# test_streaming_agent.py
"""
习题 6.1：流式输出测试（真实 LLM）

验证：
- StreamingAgent 继承 MySimpleAgent，复用历史
- stream() 产出 token / final 事件（内容来自真实模型）
- stream_run() 打字机文本拼接等于 final
- 仅暴露 invoke 时，降级路径仍返回真实模型整段回复

提示词故意要求「较长完整段落」，方便肉眼观察流式打字效果。
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM

from my_streaming_agent import StreamingAgent

_ENV_DIR = Path(__file__).resolve().parent
load_dotenv(_ENV_DIR / ".env")

llm = HelloAgentsLLM()

# 统一要求成段输出，避免「一句话/一个字」看不出流式
LONG_SYSTEM = (
    "你是讲解型 AI 助手。请用中文完整作答，写成连贯段落，"
    "不要只回一句话或几个字；目标篇幅大约 200～400 字，可分 2～4 段。"
)


class InvokeOnlyLLM:
    """包装真实 LLM，故意隐藏流式接口，用于验证降级路径仍走真实模型。"""

    def __init__(self, inner: HelloAgentsLLM):
        self._inner = inner
        self.model = getattr(inner, "model", "wrapped")
        self.provider = getattr(inner, "provider", "wrapped")

    def invoke(self, messages, **kwargs):
        return self._inner.invoke(messages, **kwargs)


def _print_typewriter(label: str, chunks) -> str:
    """边收边打印，方便观察流式效果。"""
    print(f"\n=== {label} ===")
    print("实时输出: ", end="", flush=True)
    parts = []
    for chunk in chunks:
        parts.append(chunk)
        print(chunk, end="", flush=True)
    print()  # 换行
    full = "".join(parts)
    print(f"（共 {len(parts)} 个片段，约 {len(full)} 字）")
    return full


def test_stream_events_tokens_and_final():
    """
    测试 1：结构化流式协议是否正确。

    作用：验证 agent.stream() 产出完整事件链，且内容与历史一致。
    检查点：
    - 有 status（开始处理等）
    - 有足够多的 token（真流式，不是一整段）
    - 最后一个事件是 final
    - 所有 token 拼接起来 == final
    - 对话历史写入了 user + assistant 两条
    """
    agent = StreamingAgent(
        name="流式助手",
        llm=llm,
        system_prompt=LONG_SYSTEM,
        enable_tool_calling=False,  # 本测只关心纯对话流式，关掉工具
    )

    # 故意问开放性长答，方便拆成多个 token 片段
    question = (
        "请比较详细地介绍什么是人工智能：包括定义、主要技术方向"
        "（机器学习、深度学习、大模型）、以及一两个生活中的应用例子。"
    )
    print(f"\n=== stream 提问 ===\n{question}")

    events = []
    print("实时输出: ", end="", flush=True)
    # 边收 StreamEvent 边打印 token，模拟 UI 打字机
    for event in agent.stream(question):
        events.append(event)
        if event.type == "token" and isinstance(event.data, str):
            print(event.data, end="", flush=True)
    print()

    types = [e.type for e in events]
    tokens = [e.data for e in events if e.type == "token"]
    # 协议约定：流结束时最后一个事件应为 final，携带完整回复
    final = events[-1].data if events and events[-1].type == "final" else ""

    print(f"token 片段数: {len(tokens)}，最终约 {len(final)} 字")
    print(f"最终回复预览: {final[:120]}...")

    # --- 事件协议断言 ---
    assert "status" in types  # 至少有一步状态事件
    assert types.count("token") >= 5, "片段太少，看不出流式效果"
    assert types[-1] == "final"  # 必须以 final 收尾
    assert isinstance(final, str) and len(final.strip()) >= 80, "回复过短，请检查提示词"
    assert "".join(tokens) == final  # token 拼起来必须等于完整答案

    # --- 历史副作用断言：流式结束后也应落库，供下一轮对话复用 ---
    history = agent.get_history()
    assert len(history) == 2
    assert history[0].role == "user" and history[0].content == question
    assert history[1].role == "assistant" and history[1].content == final


def test_stream_run_typewriter():
    """
    测试 2：兼容接口 stream_run() 的打字机文本流。

    作用：验证 stream_run 只 yield 纯字符串 chunk（不暴露 StreamEvent），
    适合「只要边打边显示」的调用方；内容仍来自真实 LLM 长回复。
    """
    agent = StreamingAgent(
        name="打字机",
        llm=llm,
        system_prompt=LONG_SYSTEM,
        enable_tool_calling=False,
    )
    question = (
        "请用散文风格描写春天：涵盖气候、植物、人的活动与心情，"
        "写得生动一些，至少三个自然段。"
    )
    # _print_typewriter 边收边打印，并返回拼接后的全文
    full = _print_typewriter(f"stream_run: {question[:24]}...", agent.stream_run(question))

    # 只断言「确实有像样长度」；片段拆分依赖模型/API，不强制 chunk 个数
    assert len(full.strip()) >= 80
    # stream_run 内部过滤了 status/final，只透传 token 文本


def test_stream_fallback_without_stream_invoke():
    """
    测试 3：无流式接口时的降级路径。

    作用：用 InvokeOnlyLLM 包住真实 LLM，故意不暴露 stream_invoke/think，
    验证 StreamingAgent 仍能走 invoke → 整段 yield 一次 token → final，
    对外协议不变（只是体验从「多片段」退化成「单片段」）。
    """
    wrapped = InvokeOnlyLLM(llm)  # 只有 invoke，没有 stream_invoke
    agent = StreamingAgent(
        name="降级助手",
        llm=wrapped,
        system_prompt=LONG_SYSTEM,
        enable_tool_calling=False,
    )
    question = (
        "请分步骤说明为什么要在 Agent 框架里做流式输出："
        "从用户体验、延迟感知、以及和完整 invoke 的差异三个角度展开。"
    )
    print(f"\n=== fallback stream（无 stream_invoke，整段降级）===\n{question}")
    events = list(agent.stream(question))
    tokens = [e.data for e in events if e.type == "token"]
    final = events[-1].data

    print(f"降级回复（约 {len(final)} 字）:\n{final}\n")
    # 降级时通常只有 1 个大 token，但仍须满足协议
    assert len(tokens) >= 1
    assert events[-1].type == "final"
    assert isinstance(final, str) and len(final.strip()) >= 80
    assert "".join(tokens) == final


def test_inherits_my_simple_agent_run():
    """
    测试 4：继承完整性——同步 run() 仍可用。

    作用：确认 StreamingAgent 只是「叠加」流式能力，没有破坏父类
    MySimpleAgent.run()；非流式调用应一次返回完整字符串。
    """
    agent = StreamingAgent(
        name="同步",
        llm=llm,
        system_prompt=LONG_SYSTEM,
        enable_tool_calling=False,
    )
    question = (
        "请用一段完整文字说明：北京作为中国首都，"
        "在政治、文化、交通方面各有什么代表性特点。"
    )
    print(f"\n=== run（非流式对照）===\n{question}")
    reply = agent.run(question)  # 走父类同步路径，不是 stream
    print(f"同步回复（约 {len(reply)} 字）:\n{reply}\n")
    assert isinstance(reply, str) and len(reply.strip()) >= 80


if __name__ == "__main__":
    test_stream_events_tokens_and_final()
    test_stream_run_typewriter()
    test_stream_fallback_without_stream_invoke()
    test_inherits_my_simple_agent_run()
    print("\n✅ test_streaming_agent 全部通过（真实 LLM，长回复流式）")
