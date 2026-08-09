# my_reflection_agent.py
"""
框架化 ReflectionAgent（对应第 7.4.3 节）

在「执行 → 反思 → 优化」循环中增加质量评分：
每次反思后让 LLM 对当前输出打分，分数达到阈值则提前终止，避免无效迭代。
"""

from __future__ import annotations

import re
from string import Formatter
from typing import Any, Dict, Optional, Set

from hello_agents import ReflectionAgent, HelloAgentsLLM, Config, Message

_TEMPLATE_KEYS = {"task", "content", "last_attempt", "feedback"}

# 教材默认提示词（通用场景）
DEFAULT_PROMPTS = {
    "initial": """
        请根据以下要求完成任务:
        任务: {task}
        请提供一个完整、准确的回答。
        """,
            "reflect": """
        请仔细审查以下回答，并找出可能的问题或改进空间:

        # 原始任务:
        {task}

        # 当前回答:
        {content}

        请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
        如果回答已经很好，请回答"无需改进"。
        """,
            "refine": """
        请根据反馈意见改进你的回答:

        # 原始任务:
        {task}

        # 上一轮回答:
        {last_attempt}

        # 反馈意见:
        {feedback}

        请提供一个改进后的回答。
        """,
}

# 质量评分提示词（习题扩展）
SCORE_PROMPT = """请对下面回答打分（0-10 的之间，可以保留一位小数），只输出数字。
任务: {task}
回答: {content}
评分标准: 正确性、完整性、清晰度。
分数:"""


def _extract_text(response: Any) -> str:
    """兼容 LLMResponse 对象与纯字符串返回值"""
    if response is None:
        return ""
    if hasattr(response, "content"):
        return response.content or ""
    return str(response)


def _parse_score(score_text: str) -> float:
    """从模型输出中解析分数，失败时返回 0（继续优化）"""
    if not score_text:
        return 0.0
    match = re.search(r"(\d+(?:\.\d+)?)", score_text)
    if not match:
        return 0.0
    try:
        score = float(match.group(1))
    except ValueError:
        return 0.0
    # 合理钳制到 0-10，避免模型偶发输出异常大数
    return max(0.0, min(10.0, score))


class MyReflectionAgent(ReflectionAgent):
    """
    重写的 Reflection Agent：支持 custom_prompts + 质量评分早停
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        custom_prompts: Optional[Dict[str, str]] = None,
        score_threshold: float = 9.9,
        enable_quality_score: bool = True,
    ):
        # 教学版默认不挂工具，聚焦反思循环本身
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            config=config,
            max_iterations=max_iterations,
            enable_tool_calling=False,
        )
        self.prompts = {**DEFAULT_PROMPTS, **(custom_prompts or {})}
        self.score_threshold = score_threshold
        self.enable_quality_score = enable_quality_score
        print(
            f"✅ {name} 初始化完成，最大迭代: {max_iterations}，"
            f"质量评分: {'启用' if enable_quality_score else '禁用'}"
            + (f"（阈值 {score_threshold}）" if enable_quality_score else "")
        )

    def run(self, input_text: str, **kwargs) -> str:
        """运行 Reflection：initial → (reflect → score? → refine)*N"""
        print(f"\n🤖 {self.name} 开始处理任务: {input_text}")

        # 1. 初始执行
        print("\n--- 正在进行初始尝试 ---")
        content = self._call_prompt("initial", task=input_text, **kwargs)
        print(f"✅ 初始回答长度: {len(content)} 字符")

        # 2. 迭代：反思 →（可选评分）→ 优化
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i + 1}/{self.max_iterations} 轮迭代 ---")

            print("\n-> 正在进行反思...")
            feedback = self._call_prompt(
                "reflect",
                task=input_text,
                content=content,
                **kwargs,
            )
            print(f"📝 反思反馈预览: {feedback}...")

            if "无需改进" in feedback or "no need for improvement" in feedback.lower():
                print("\n✅ 反思认为结果已无需改进，提前结束。")
                break

            # 质量评分：分数达标则不再 refine
            if self.enable_quality_score:
                print("\n-> 正在进行质量评分...")
                score_text = self._invoke(
                    SCORE_PROMPT.format(task=input_text, content=content),
                    **kwargs,
                )
                score = _parse_score(score_text)
                print(f"📊 当前质量分: {score} / 阈值 {self.score_threshold}")
                if score >= self.score_threshold:
                    print("\n✅ 分数已达阈值，提前终止优化。")
                    break

            print("\n-> 正在进行优化...")
            content = self._call_prompt(
                "refine",
                task=input_text,
                content=content,
                last_attempt=content,
                feedback=feedback,
                **kwargs,
            )
            print(f"✅ 本轮优化完成，回答长度: {len(content)} 字符")

        print(f"\n--- 任务完成 ---\n最终结果预览:\n{content[:200]}...")

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(content, "assistant"))
        return content

    def _call_prompt(self, key: str, **format_kwargs: Any) -> str:
        """按 prompts[key] 渲染并调用 LLM；仅填充模板实际用到的占位符。"""
        invoke_kwargs = {
            k: v for k, v in format_kwargs.items() if k not in _TEMPLATE_KEYS
        }
        template = self.prompts[key]
        needed = _template_field_names(template)
        template_vars = {
            k: format_kwargs[k] for k in needed if k in format_kwargs
        }
        prompt = template.format(**template_vars)
        return self._invoke(prompt, **invoke_kwargs)

    def _invoke(self, prompt: str, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        return _extract_text(self.llm.invoke(messages, **kwargs))


def _template_field_names(template: str) -> Set[str]:
    """提取 format 模板中的字段名（忽略字面量 {{ }}）"""
    names: Set[str] = set()
    for _, field_name, _, _ in Formatter().parse(template):
        if field_name:
            # 支持简单字段名；忽略属性/下标访问的复杂形式
            names.add(field_name.split(".")[0].split("[")[0])
    return names
