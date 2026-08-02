"""
ReAct 智能体 — JSON 结构化输出版（课后练习 2 优化实现）

对比原版 ReAct.py（正则解析 Thought/Action）：
- 原版脆弱点：标签拼写/大小写不一致、多段 Thought、Action 含换行或嵌套 []、
  模型输出中文冒号/多余前言等都会导致解析失败。
- 本版方案：强制模型输出单一 JSON 对象，用 json.loads + 字段校验解析；
  兼容 ```json ... ``` 代码块，并在解析失败时把错误回灌给模型重试。

优缺点简述见文件末尾 COMPARISON 常量，或运行本文件时的打印说明。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

# TYPE_CHECKING 是 typing 模块提供的特殊常量：
# - 静态类型检查时（mypy / pyright 等）为 True，下面的 import 会被分析，用于注解
# - 运行时始终为 False，因此这些 import 不会真正执行，可避免循环依赖与多余运行时依赖
if TYPE_CHECKING:
    from llm_client import HelloAgentsLLM
    from tools import ToolExecutor


# ---------------------------------------------------------------------------
# Prompt：强制 JSON 协议（比纯文本 Thought:/Action: 更易机器解析）
# ---------------------------------------------------------------------------

REACT_JSON_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下：
{tools}

你必须且只能输出一个 JSON 对象（不要输出其它解释文字），格式严格如下：

{{
  "thought": "你的思考过程：分析问题、拆解任务、规划下一步",
  "action": {{
    "type": "tool",
    "name": "工具名",
    "input": "工具输入参数"
  }}
}}

当你已收集到足够信息、可以给出最终答案时，改用：

{{
  "thought": "说明为何可以结束",
  "action": {{
    "type": "finish",
    "answer": "最终答案"
  }}
}}

约束：
1. 只输出一个 JSON 对象；可用 ```json ... ``` 包裹，但不要夹杂其它自然语言。
2. action.type 只能是 "tool" 或 "finish"。
3. type 为 "tool" 时必须包含 name、input；type 为 "finish" 时必须包含 answer。
4. 工具名必须来自上方可用工具列表。

现在，请开始解决以下问题：
Question: {question}
History: {history}
"""


COMPARISON = """
【正则解析 vs JSON 结构化解析】

1) 正则方案（ReAct.py）
   优点：提示词短、人类可读、实现简单、延迟略低（无需严格 JSON）。
   缺点：对格式极其敏感；Thought/Action 标签漂移即失败；Action 内嵌 []、
         多轮输出、中文标点、前言废话都会击穿解析；难做字段级校验与自动纠错。

2) JSON 方案（本文件）
   优点：字段语义明确；可用 schema 校验；解析失败可把错误回灌让模型重试；
         易扩展（加 confidence、need_more_info 等字段）；与 Function Calling 思路一致。
   缺点：弱模型偶发输出非法 JSON；提示词更长；需额外处理代码块/尾随逗号等脏数据。

工程上更优路径（本课可后续扩展）：模型原生 Function/Tool Calling，或
Pydantic + instructor / outlines 等约束解码，比手写正则或裸 JSON 更稳。
"""


@dataclass
class ParsedStep:
    """一轮 ReAct 解析结果。"""
    thought: str
    action_type: str  # "tool" | "finish"
    tool_name: Optional[str] = None
    tool_input: Optional[str] = None
    final_answer: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


class JsonOutputParser:
    """从 LLM 文本中提取并校验 ReAct JSON 步骤。"""

    # 允许从混杂文本中捞出第一个 JSON 对象
    _OBJECT_RE = re.compile(r"\{[\s\S]*\}")

    def parse(self, text: str) -> ParsedStep:
        data = self._loads_json(text)
        return self._validate(data)

    def _loads_json(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("模型返回为空，无法解析 JSON。")

        candidates = [text.strip()]

        # 兼容 ```json ... ``` / ``` ... ```
        fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        candidates.extend(block.strip() for block in fenced)

        # 兼容前后夹杂自然语言：截取首个大括号对象
        obj_match = self._OBJECT_RE.search(text)
        if obj_match:
            candidates.append(obj_match.group(0))

        last_error: Optional[Exception] = None
        for candidate in candidates:
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
                last_error = ValueError(f"JSON 根类型必须是 object，实际为 {type(data).__name__}")
            except json.JSONDecodeError as e:
                last_error = e
                # 轻量修复：去掉尾随逗号后再试一次
                repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
                try:
                    data = json.loads(repaired)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError as e2:
                    last_error = e2

        raise ValueError(f"无法从模型输出中解析合法 JSON：{last_error}\n原始输出:\n{text}")

    def _validate(self, data: Dict[str, Any]) -> ParsedStep:
        thought = data.get("thought")
        action = data.get("action")

        if not isinstance(thought, str) or not thought.strip():
            raise ValueError('字段 "thought" 必须是非空字符串。')
        if not isinstance(action, dict):
            raise ValueError('字段 "action" 必须是对象。')

        action_type = action.get("type")
        if action_type not in ("tool", "finish"):
            raise ValueError('字段 "action.type" 只能是 "tool" 或 "finish"。')

        if action_type == "tool":
            name = action.get("name")
            tool_input = action.get("input")
            if not isinstance(name, str) or not name.strip():
                raise ValueError('当 type="tool" 时，"action.name" 必须是非空字符串。')
            if not isinstance(tool_input, str):
                # 允许非字符串输入时统一转成字符串，便于现有 ToolExecutor
                tool_input = json.dumps(tool_input, ensure_ascii=False)
            return ParsedStep(
                thought=thought.strip(),
                action_type="tool",
                tool_name=name.strip(),
                tool_input=tool_input,
                raw=data,
            )

        answer = action.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError('当 type="finish" 时，"action.answer" 必须是非空字符串。')
        return ParsedStep(
            thought=thought.strip(),
            action_type="finish",
            final_answer=answer.strip(),
            raw=data,
        )


class ReActJsonAgent:
    """使用 JSON 结构化输出的 ReAct 智能体。"""

    def __init__(
        self,
        llm_client: "HelloAgentsLLM",
        tool_executor: "ToolExecutor",
        max_steps: int = 5,
        max_parse_retries: int = 2,
    ):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.max_parse_retries = max_parse_retries
        self.parser = JsonOutputParser()
        self.history: List[str] = []

    def run(self, question: str) -> Optional[str]:
        self.history = []
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            step, response_text = self._think_and_parse(question)
            if step is None:
                print("错误：多次解析失败，流程终止。")
                print(f"最后一次原始输出:\n{response_text}")
                break

            print(f"🤔 思考: {step.thought}")

            if step.action_type == "finish":
                print(f"🎉 最终答案: {step.final_answer}")
                return step.final_answer

            tool_name = step.tool_name or ""
            tool_input = step.tool_input or ""
            print(f"🎬 行动: {tool_name}[{tool_input}]")

            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = (
                    f"错误：未找到名为 '{tool_name}' 的工具。"
                    f"可用工具：{list(self.tool_executor.tools.keys())}"
                )
            else:
                observation = tool_function(tool_input)

            print(f"👀 观察: {observation}")
            # history 用 JSON，便于下一轮模型对齐协议
            self.history.append(
                json.dumps(
                    {
                        "action": {"type": "tool", "name": tool_name, "input": tool_input},
                        "observation": observation,
                    },
                    ensure_ascii=False,
                )
            )

        print("已达到最大步数，流程终止。")
        return None

    def _think_and_parse(self, question: str) -> Tuple[Optional[ParsedStep], str]:
        """调用 LLM 并解析；失败时把校验错误回灌，最多重试 max_parse_retries 次。"""
        tools_desc = self.tool_executor.getAvailableTools()
        history_str = "\n".join(self.history) if self.history else "(empty)"
        prompt = REACT_JSON_PROMPT_TEMPLATE.format(
            tools=tools_desc,
            question=question,
            history=history_str,
        )
        messages = [{"role": "user", "content": prompt}]
        last_response = ""

        for attempt in range(self.max_parse_retries + 1):
            response_text = self.llm_client.think(messages=messages)
            last_response = response_text or ""
            if not response_text:
                return None, last_response

            try:
                return self.parser.parse(response_text), response_text
            except ValueError as e:
                if attempt >= self.max_parse_retries:
                    break
                # 解析失败 → 回灌错误，要求模型只修正 JSON
                repair_prompt = (
                    "你上一次的输出无法被解析为合法 JSON，请只重新输出一个符合协议的 JSON 对象，"
                    f"不要解释。\n校验错误: {e}\n你上次的输出:\n{response_text}"
                )
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response_text},
                    {"role": "user", "content": repair_prompt},
                ]
                print(f"⚠️ JSON 解析失败，正在重试 ({attempt + 1}/{self.max_parse_retries})：{e}")

        return None, last_response


def _demo_offline_parser():
    """不调用 LLM，演示解析器对脏输出的鲁棒性。"""
    parser = JsonOutputParser()
    dirty_cases = [
        # 标准输出
        '{"thought":"需要搜索","action":{"type":"tool","name":"Search","input":"华为最新手机"}}',
        # 带代码块 + 前言
        '好的，我来处理。\n```json\n{"thought":"已足够","action":{"type":"finish","answer":"Mate 70"}}\n```',
        # 尾随逗号（会触发轻量修复）
        '{"thought":"算了","action":{"type":"finish","answer":"完成",},}',
    ]
    print("\n=== 离线解析演示 ===")
    for i, case in enumerate(dirty_cases, 1):
        step = parser.parse(case)
        print(f"[{i}] type={step.action_type}, thought={step.thought!r}, "
              f"tool={step.tool_name}, answer={step.final_answer}")


if __name__ == "__main__":
    import sys

    print(COMPARISON)
    _demo_offline_parser()

    # 仅演示解析器：python ReAct_json.py --offline
    if "--offline" in sys.argv:
        raise SystemExit(0)

    # 与原版相同的在线运行入口（需要 .env 中的 LLM / SerpAPI 配置）
    from llm_client import HelloAgentsLLM
    from tools import ToolExecutor, search

    llm = HelloAgentsLLM()
    tool_executor = ToolExecutor()
    search_desc = (
        "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，"
        "应使用此工具。"
    )
    tool_executor.registerTool("Search", search_desc, search)
    agent = ReActJsonAgent(llm_client=llm, tool_executor=tool_executor)
    question = "华为最新的手机是哪一款？它的主要卖点是什么？"
    agent.run(question)
