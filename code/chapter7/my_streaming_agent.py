# my_streaming_agent.py
"""
习题 6.1：流式输出 Agent

设计要点：
- 继承 MySimpleAgent，复用 run / 工具解析 / 历史管理，避免重复实现对话逻辑
- 新增 stream()：产出 StreamEvent（token / tool_* / final）
- stream_run()：仅产出文本 chunk，兼容旧测试与「打字机」UI
- 无 stream_invoke 时降级：调用父类 run() 后一次性 yield
"""

from __future__ import annotations

from typing import Iterator, Optional

from hello_agents import HelloAgentsLLM, Config, Message

from my_simple_agent import MySimpleAgent
from stream_events import StreamEvent


class StreamingAgent(MySimpleAgent):
    """在 MySimpleAgent 之上叠加结构化流式输出。"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry=None,
        enable_tool_calling: bool = True,
    ):
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            config=config,
            tool_registry=tool_registry,
            enable_tool_calling=enable_tool_calling,
        )

    def stream(self, input_text: str, **kwargs) -> Iterator[StreamEvent]:
        """
        结构化流式入口。

        Yields:
            StreamEvent: status / token / tool_* / final / error
        """
        yield StreamEvent.status(f"开始处理: {input_text}", agent_name=self.name)

        messages = self._build_stream_messages(input_text)

        # 工具开启时：先走一轮非流式工具循环，再对流式最终答案（教学简化）
        if self.enable_tool_calling and self.tool_registry:
            yield from self._stream_with_tools(messages, input_text, **kwargs)
            return

        yield from self._stream_tokens(messages, input_text, **kwargs)

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """兼容父类接口：只 yield 文本 token（打字机效果）。"""
        for event in self.stream(input_text, **kwargs):
            if event.type == "token" and isinstance(event.data, str):
                yield event.data

    def _build_stream_messages(self, input_text: str) -> list:
        messages = []
        enhanced = self._get_enhanced_system_prompt()
        messages.append({"role": "system", "content": enhanced})
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": input_text})
        return messages

    def _stream_tokens(
        self,
        messages: list,
        input_text: str,
        save_history: bool = True,
        **kwargs,
    ) -> Iterator[StreamEvent]:
        """调用 llm.stream_invoke，边生成边 yield token；失败则降级为整段回复。"""
        full_response = ""
        stream_fn = None
        for name in ("stream_invoke", "think"):
            candidate = getattr(self.llm, name, None)
            if callable(candidate):
                stream_fn = candidate
                break

        try:
            if stream_fn is None:
                # 降级：非流式拼完再 yield 一次
                raw = self.llm.invoke(messages, **kwargs)
                full_response = raw.content if hasattr(raw, "content") else str(raw)
                yield StreamEvent.token(full_response, agent_name=self.name)
            else:
                for chunk in stream_fn(messages, **kwargs):
                    text = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if not text:
                        continue
                    full_response += text
                    yield StreamEvent.token(text, agent_name=self.name)
        except Exception as e:
            yield StreamEvent.error(str(e), agent_name=self.name)
            raise

        if save_history:
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(full_response, "assistant"))

        yield StreamEvent.final(full_response, agent_name=self.name)

    def _stream_with_tools(
        self,
        messages: list,
        input_text: str,
        max_tool_iterations: int = 3,
        **kwargs,
    ) -> Iterator[StreamEvent]:
        """
        工具轮次用 invoke（便于解析 TOOL_CALL），最终答案再 stream。
        复用父类 _parse_tool_calls / _execute_tool_call，不重写解析逻辑。
        """
        current_iteration = 0
        working = list(messages)

        while current_iteration < max_tool_iterations:
            raw = self.llm.invoke(working, **kwargs)
            response = raw.content if hasattr(raw, "content") else str(raw)
            tool_calls = self._parse_tool_calls(response)

            if not tool_calls:
                # 最终答案：对流式生成助手回复，并写入历史
                yield StreamEvent.status("生成最终回答", agent_name=self.name)
                yield from self._stream_tokens(working, input_text, save_history=True, **kwargs)
                return

            yield StreamEvent.status(
                f"检测到 {len(tool_calls)} 个工具调用",
                agent_name=self.name,
            )
            clean_response = response
            tool_results = []
            for call in tool_calls:
                yield StreamEvent.tool_start(
                    call["tool_name"],
                    call["parameters"],
                    agent_name=self.name,
                )
                result = self._execute_tool_call(call["tool_name"], call["parameters"])
                tool_results.append(result)
                clean_response = clean_response.replace(call["original"], "")
                yield StreamEvent.tool_result(
                    call["tool_name"],
                    result,
                    agent_name=self.name,
                )

            working.append({"role": "assistant", "content": clean_response})
            tool_results_text = "\n\n".join(tool_results)
            working.append(
                {
                    "role": "user",
                    "content": (
                        f"工具执行结果：\n{tool_results_text}\n\n"
                        "请基于这些结果给出完整的回答。"
                    ),
                }
            )
            current_iteration += 1

        # 超过迭代上限：最后再流式要一次答案
        yield StreamEvent.status("达到工具迭代上限，强制生成回答", agent_name=self.name)
        yield from self._stream_tokens(working, input_text, save_history=True, **kwargs)
