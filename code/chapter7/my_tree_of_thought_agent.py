# my_tree_of_thought_agent.py
"""
Tree-of-Thought (ToT) Agent（对应第 7 章习题 4.3）

每一步：
1. Expand  — 从当前状态生成多条候选思考路径
2. Evaluate — 对每条候选打分
3. Select  — 选择最优路径继续，直到给出最终答案或达到深度上限

继承 hello_agents 的 Agent 基类，复用 Message 历史管理。
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

from hello_agents import HelloAgentsLLM, Config, Message
from hello_agents.core.agent import Agent

EXPAND_PROMPT = """你是一位善于多路径推理的助手。请针对问题，基于「当前推理状态」提出 {breadth} 条互不相同的下一步思考。

# 原始问题
{question}

# 当前推理状态
{state}

# 深度约束
当前深度: {depth}/{max_depth}
{final_rule}

要求：
1. 每条思考应是可继续推进的具体思路（只推进一步，不要一次性算完全部）
2. 严格按下面格式输出（不要输出其它内容）：
1. ...
2. ...
3. ...
"""

EVALUATE_PROMPT = """请评估下面这条「下一步思考」对解决原始问题的帮助程度。

# 原始问题
{question}

# 当前推理状态
{state}

# 候选下一步思考
{candidate}

评分标准（0-10）：
- 是否更接近正确答案
- 逻辑是否清晰、可执行
- 是否避免明显错误或死胡同

请只输出一个 0-10 的数字。
"""

SYNTHESIZE_PROMPT = """请根据原始问题与已选中的推理路径，给出最终答案（简洁明确）。

# 原始问题
{question}

# 已选推理路径
{path}

请直接给出最终答案，不要重复推理过程。
"""

def _extract_text(response: Any) -> str:
    if response is None:
        return ""
    if hasattr(response, "content"):
        return response.content or ""
    return str(response)


def _parse_score(text: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)", text or "")
    if not match:
        return 0.0
    try:
        return max(0.0, min(10.0, float(match.group(1))))
    except ValueError:
        return 0.0


def _parse_candidates(text: str, breadth: int) -> List[str]:
    """从编号列表中解析候选；失败时把整段当作单一候选。"""
    candidates: List[str] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\d+[\.\)、]\s*(.+)$", line)
        if m:
            candidates.append(m.group(1).strip())
    if not candidates and text.strip():
        candidates = [text.strip()]
    return candidates[:breadth]


class MyTreeOfThoughtAgent(Agent):
    """Tree-of-Thought Agent：扩展 → 评估 → 选择"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        breadth: int = 3,
        max_depth: int = 3,
        min_depth_for_final: int = 1,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.breadth = max(1, breadth)
        self.max_depth = max(1, max_depth)
        # 未达到该深度前，即使模型写出 FINAL 也继续循环（便于演示多层 ToT）
        self.min_depth_for_final = max(1, min(min_depth_for_final, self.max_depth))
        print(
            f"✅ {name} 初始化完成（ToT：breadth={self.breadth}, "
            f"max_depth={self.max_depth}, min_depth_for_final={self.min_depth_for_final}）"
        )

    def run(self, input_text: str, **kwargs) -> str:
        """运行 ToT：逐层扩展候选路径并择优深入。"""
        print(f"\n🌳 {self.name} 开始 Tree-of-Thought 推理")
        print(f"问题: {input_text}")

        state = "（尚无推理，请从问题本身出发提出第一步思路）"
        path_steps: List[str] = []

        for depth in range(1, self.max_depth + 1):
            print(f"\n===== 深度 {depth}/{self.max_depth} =====")

            candidates = self._expand(input_text, state, depth=depth, **kwargs)
            if not candidates:
                print("⚠️ 未能生成候选路径，结束搜索。")
                break

            print(f"🌱 扩展出 {len(candidates)} 条候选：")
            for i, c in enumerate(candidates, 1):
                print(f"  [{i}] {c}")

            allow_final = depth >= self.min_depth_for_final
            best, best_score = self._select(
                input_text,
                state,
                candidates,
                allow_final=allow_final,
                **kwargs,
            )
            print(
                f"⭐ 选中候选（得分 {best_score}）: "
                f"{best}"
            )

            path_steps.append(best)

            if allow_final and self._is_final(best):
                final_answer = self._extract_final(best)
                print(f"\n✅ 在深度 {depth} 得到最终答案")
                self._save_history(input_text, final_answer)
                return final_answer

            if not allow_final and self._is_final(best):
                print(
                    f"⏳ 深度 {depth} < min_depth_for_final={self.min_depth_for_final}，"
                    "忽略提前 FINAL，继续下一层循环"
                )

            state = self._merge_state(state, best, depth)

        print("\n🧭 达到深度上限，综合已选路径生成最终答案...")
        path_text = "\n".join(f"{i}. {s}" for i, s in enumerate(path_steps, 1)) or state
        final_answer = self._synthesize(input_text, path_text, **kwargs)
        self._save_history(input_text, final_answer)
        return final_answer

    def _expand(self, question: str, state: str, depth: int = 1, **kwargs) -> List[str]:
        if depth < self.min_depth_for_final:
            final_rule = (
                "本层禁止输出 FINAL / 最终答案；只给出「下一步中间推理」，"
                "把问题再推进一步即可。"
            )
        else:
            final_rule = (
                "若已能直接给出最终答案，请写成：FINAL: <最终答案>；"
                "否则继续给出中间推理。"
            )
        prompt = EXPAND_PROMPT.format(
            question=question,
            state=state,
            breadth=self.breadth,
            depth=depth,
            max_depth=self.max_depth,
            final_rule=final_rule,
        )
        text = self._invoke(prompt, **kwargs)
        return _parse_candidates(text, self.breadth)

    def _evaluate(self, question: str, state: str, candidate: str, **kwargs) -> float:
        prompt = EVALUATE_PROMPT.format(
            question=question,
            state=state,
            candidate=candidate,
        )
        return _parse_score(self._invoke(prompt, **kwargs))

    def _select(
        self,
        question: str,
        state: str,
        candidates: List[str],
        allow_final: bool = True,
        **kwargs,
    ) -> Tuple[str, float]:
        scored: List[Tuple[str, float]] = []
        for i, cand in enumerate(candidates, 1):
            if allow_final and self._is_final(cand):
                score = 10.0
                print(f"  📊 候选[{i}] 已含 FINAL，视为满分 {score}")
            else:
                score = self._evaluate(question, state, cand, **kwargs)
                print(f"  📊 候选[{i}] 得分: {score}")
            scored.append((cand, score))

        # list.sort 原地排序；key=lambda x: x[1] 取元组第 2 项（分数）作比较键；
        # reverse=True 表示降序，分数最高的排到 scored[0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0]

    def _is_final(self, thought: str) -> bool:
        t = thought or ""
        return bool(re.search(r"\bFINAL\s*:", t, flags=re.IGNORECASE)) or (
            "最终答案" in t
        )

    def _extract_final(self, thought: str) -> str:
        t = thought.strip()
        m = re.search(r"\bFINAL\s*:\s*(.+)$", t, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
        if "最终答案" in t:
            return re.sub(
                r"^.*最终答案[:：\s]*", "", t, count=1, flags=re.DOTALL
            ).strip()
        return t

    def _merge_state(self, prev_state: str, best: str, depth: int) -> str:
        if prev_state.startswith("（尚无推理"):
            return f"步骤1: {best}"
        return f"{prev_state}\n步骤{depth}: {best}"

    def _synthesize(self, question: str, path: str, **kwargs) -> str:
        prompt = SYNTHESIZE_PROMPT.format(question=question, path=path)
        return self._invoke(prompt, **kwargs).strip()

    def _invoke(self, prompt: str, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return _extract_text(self.llm.invoke(messages, **kwargs))

    def _save_history(self, user_text: str, assistant_text: str) -> None:
        self.add_message(Message(user_text, "user"))
        self.add_message(Message(assistant_text, "assistant"))
        print(f"\n--- 任务完成 ---\n最终答案: {assistant_text}")
