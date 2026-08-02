"""
方案 3 示例：外层状态机 + 内层单阶段任务（AutoGen）

对比 autogen_software_team.py 的 RoundRobin：
- 这里不用「整队轮询到底」
- 用 phase 决定当前阶段；审查结果决定前进 / 回退
- 每个阶段只调用相关角色（内层），阶段跳转由外层 while 控制

运行（需配置 .env）：
  ../.venv 或项目根 .venv：
  /path/to/.venv/bin/python software_team_state_machine.py
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console


# ---------------------------------------------------------------------------
# 1) 阶段定义：外层状态机只认这些状态
# ---------------------------------------------------------------------------

class Phase(str, Enum):
    PM_PLAN = "pm_plan"           # 需求分析
    ENGINEER = "engineer"         # 编码
    REVIEW = "review"             # 审查（决定前进或回退）
    QA = "qa"                     # 验收
    DONE = "done"


class ReviewDecision(str, Enum):
    PASS = "pass"                 # 通过 → QA
    NEED_FIX = "need_fix"         # 代码问题 → 回退 Engineer
    NEED_REQ_CHANGE = "need_req_change"  # 需求问题 → 回退 PM


# ---------------------------------------------------------------------------
# 2) 共享上下文：各阶段读写同一份「项目黑板」
# ---------------------------------------------------------------------------

@dataclass
class ProjectContext:
    user_task: str
    requirements: str = ""
    code: str = ""
    review_feedback: str = ""
    qa_report: str = ""
    # 防止 PM ↔ Engineer ↔ Review 无限循环
    rollback_count: int = 0
    max_rollbacks: int = 3
    history: list[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        print(msg)
        self.history.append(msg)


# ---------------------------------------------------------------------------
# 3) 模型与角色（复用课程案例思路）
# ---------------------------------------------------------------------------

def create_model_client() -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(
        model=os.getenv("LLM_MODEL_ID", "gpt-4o"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.UNKNOWN,
            "structured_output": True,
        },
    )


def create_agents(model_client):
    pm = AssistantAgent(
        name="ProductManager",
        model_client=model_client,
        system_message=(
            "你是产品经理。根据用户任务与（如有）审查反馈，输出清晰需求："
            "功能列表、验收标准、非目标。末尾写：PHASE_DONE"
        ),
    )
    engineer = AssistantAgent(
        name="Engineer",
        model_client=model_client,
        system_message=(
            "你是工程师。根据需求与（如有）审查意见，输出完整可运行 Python 代码，"
            "放在 ```python 代码块中。末尾写：PHASE_DONE"
        ),
    )
    reviewer = AssistantAgent(
        name="CodeReviewer",
        model_client=model_client,
        system_message=(
            "你是代码审查员。对照需求审查代码。\n"
            "必须在回复最后一行单独输出三选一决策标签（只能有一个）：\n"
            "DECISION: PASS\n"
            "DECISION: NEED_FIX\n"
            "DECISION: NEED_REQ_CHANGE\n"
            "说明：PASS=可验收；NEED_FIX=实现有问题需工程师改；"
            "NEED_REQ_CHANGE=需求本身要改，应退回产品经理。"
        ),
    )
    qa = AssistantAgent(
        name="QA",
        model_client=model_client,
        system_message=(
            "你是测试工程师。根据需求与代码给出测试要点与结论 PASS/FAIL。"
            "末尾写：PHASE_DONE"
        ),
    )
    return pm, engineer, reviewer, qa


# ---------------------------------------------------------------------------
# 4) 内层：跑「单一阶段」——这里用 1 人 RoundRobin + 最多 2 条消息
#    （真实项目里也可以改成多人小群聊，但阶段边界仍由外层决定）
# ---------------------------------------------------------------------------

async def run_phase_agent(agent: AssistantAgent, prompt: str) -> str:
    """内层执行：只让当前阶段的一个角色工作一轮。"""
    team = RoundRobinGroupChat(
        participants=[agent],
        termination_condition=MaxMessageTermination(2),
    )
    result = await Console(team.run_stream(task=prompt))
    # 取该角色最后一条文本
    texts = []
    for m in getattr(result, "messages", []) or []:
        if getattr(m, "source", None) == agent.name and isinstance(
            getattr(m, "content", None), str
        ):
            texts.append(m.content)
    return texts[-1] if texts else ""


_CODE_RE = re.compile(r"```(?:python|py)?\s*\n([\s\S]*?)```", re.I)


def extract_code(text: str) -> str:
    blocks = _CODE_RE.findall(text or "")
    return max(blocks, key=len).strip() if blocks else text


def parse_review_decision(text: str) -> ReviewDecision:
    """从审查员输出解析结构化决策（外层状态机的关键输入）。"""
    upper = (text or "").upper()
    if "DECISION: NEED_REQ_CHANGE" in upper:
        return ReviewDecision.NEED_REQ_CHANGE
    if "DECISION: NEED_FIX" in upper:
        return ReviewDecision.NEED_FIX
    if "DECISION: PASS" in upper:
        return ReviewDecision.PASS
    # 兜底：看关键词
    if "需求" in (text or "") and ("变更" in text or "不清晰" in text):
        return ReviewDecision.NEED_REQ_CHANGE
    if "修改" in (text or "") or "问题" in (text or ""):
        return ReviewDecision.NEED_FIX
    return ReviewDecision.PASS


# ---------------------------------------------------------------------------
# 5) 外层状态机：唯一决定下一阶段的地方
# ---------------------------------------------------------------------------

async def run_state_machine(user_task: str) -> ProjectContext:
    ctx = ProjectContext(user_task=user_task)
    model_client = create_model_client()
    pm, engineer, reviewer, qa = create_agents(model_client)

    phase = Phase.PM_PLAN

    while phase != Phase.DONE:
        ctx.log(f"\n========== 当前阶段: {phase.value} ==========")

        if phase == Phase.PM_PLAN:
            prompt = (
                f"用户任务：\n{ctx.user_task}\n\n"
                f"审查反馈（可能为空）：\n{ctx.review_feedback or '无'}\n"
            )
            ctx.requirements = await run_phase_agent(pm, prompt)
            ctx.log("✅ 需求分析完成 → 进入编码")
            phase = Phase.ENGINEER

        elif phase == Phase.ENGINEER:
            prompt = (
                f"需求文档：\n{ctx.requirements}\n\n"
                f"审查意见（可能为空，有则按此修改）：\n{ctx.review_feedback or '无'}\n"
            )
            eng_out = await run_phase_agent(engineer, prompt)
            ctx.code = extract_code(eng_out)
            ctx.log("✅ 编码完成 → 进入审查")
            phase = Phase.REVIEW

        elif phase == Phase.REVIEW:
            prompt = (
                f"需求：\n{ctx.requirements}\n\n"
                f"代码：\n```python\n{ctx.code}\n```\n"
            )
            review_out = await run_phase_agent(reviewer, prompt)
            ctx.review_feedback = review_out
            decision = parse_review_decision(review_out)
            ctx.log(f"🧾 审查决策: {decision.value}")

            if decision == ReviewDecision.PASS:
                ctx.log("✅ 审查通过 → 进入 QA")
                phase = Phase.QA

            elif decision == ReviewDecision.NEED_FIX:
                ctx.rollback_count += 1
                if ctx.rollback_count > ctx.max_rollbacks:
                    ctx.log("⛔ 回退次数耗尽，强制进入 QA")
                    phase = Phase.QA
                else:
                    ctx.log(
                        f"↩️ 回退工程师改代码 "
                        f"({ctx.rollback_count}/{ctx.max_rollbacks})"
                    )
                    phase = Phase.ENGINEER  # ★ 动态回退

            elif decision == ReviewDecision.NEED_REQ_CHANGE:
                ctx.rollback_count += 1
                if ctx.rollback_count > ctx.max_rollbacks:
                    ctx.log("⛔ 回退次数耗尽，强制进入 QA")
                    phase = Phase.QA
                else:
                    ctx.log(
                        f"↩️ 回退产品经理改需求 "
                        f"({ctx.rollback_count}/{ctx.max_rollbacks})"
                    )
                    phase = Phase.PM_PLAN  # ★ 动态回退到 PM

        elif phase == Phase.QA:
            prompt = (
                f"需求：\n{ctx.requirements}\n\n"
                f"代码：\n```python\n{ctx.code}\n```\n"
            )
            ctx.qa_report = await run_phase_agent(qa, prompt)
            ctx.log("✅ QA 完成 → 结束")
            phase = Phase.DONE

    return ctx


async def main():
    if not os.getenv("LLM_API_KEY"):
        print("❌ 请在 .env 中配置 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_ID")
        return

    task = (
        "开发一个比特币价格显示的 Streamlit 小应用："
        "显示 USD 现价、24h 涨跌幅，带刷新按钮和错误处理。"
    )
    ctx = await run_state_machine(task)

    print("\n" + "=" * 60)
    print("📋 状态机运行摘要")
    print(f"- 回退次数: {ctx.rollback_count}")
    print(f"- 需求字数: {len(ctx.requirements)}")
    print(f"- 代码字数: {len(ctx.code)}")
    print(f"- QA 字数: {len(ctx.qa_report)}")
    if ctx.code:
        out = os.path.join(os.path.dirname(__file__), "generated_from_state_machine.py")
        with open(out, "w", encoding="utf-8") as f:
            f.write(ctx.code if "import" in ctx.code else ctx.code)
        print(f"- 代码已尝试写入: {out}")


if __name__ == "__main__":
    asyncio.run(main())
