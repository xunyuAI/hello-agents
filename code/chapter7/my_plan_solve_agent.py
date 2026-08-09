# my_plan_solve_agent.py
"""
框架化 PlanAndSolveAgent 实现（对应第 7.4.4 节）

相对第四章脚本版，本实现接入统一 Agent 基类、Message 历史管理，
并支持通过 custom_prompts 分别定制规划器 / 执行器提示词。

说明：当前 hello_agents 包导出名为 PlanSolveAgent（官方实现已改为 Function Calling）。
本文件按教材要求，保留「提示词约束 + Python 列表解析」的教学实现。
"""

import ast
from typing import Optional, List, Dict, Any

from hello_agents import PlanSolveAgent, HelloAgentsLLM, Config, Message

# 默认规划器提示词模板
DEFAULT_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

# 默认执行器提示词模板
DEFAULT_EXECUTOR_PROMPT = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""


def _extract_text(response: Any) -> str:
    """兼容 LLMResponse 对象与纯字符串返回值"""
    if response is None:
        return ""
    if hasattr(response, "content"):
        return response.content or ""
    return str(response)


class Planner:
    """规划器 - 将复杂问题分解为可执行步骤列表（提示词 + 列表解析）"""

    def __init__(self, llm_client: HelloAgentsLLM, prompt_template: Optional[str] = None):
        self.llm_client = llm_client
        self.prompt_template = prompt_template or DEFAULT_PLANNER_PROMPT

    def plan(self, question: str, **kwargs) -> List[str]:
        prompt = self.prompt_template.format(question=question)
        messages = [{"role": "user", "content": prompt}]

        print("--- 正在生成计划 ---")
        response_text = _extract_text(self.llm_client.invoke(messages, **kwargs))
        print(f"✅ 计划已生成:\n{response_text}")

        try:
            # 优先从 ```python 代码块中提取列表
            if "```python" in response_text:
                plan_str = response_text.split("```python")[1].split("```")[0].strip()
            elif "```" in response_text:
                # 兼容仅使用 ``` 包裹的情况（含文档中省略标记的写法）
                plan_str = response_text.split("```")[1].split("```")[0].strip()
                if plan_str.startswith("python"):
                    plan_str = plan_str[len("python"):].strip()
            else:
                # 兜底：尝试直接解析整段响应中的列表字面量
                start = response_text.find("[")
                end = response_text.rfind("]")
                if start == -1 or end == -1 or end <= start:
                    raise ValueError("响应中未找到可解析的列表")
                plan_str = response_text[start : end + 1]

            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_text}")
            return []
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []


class Executor:
    """执行器 - 按计划逐步执行并累积历史上下文"""

    def __init__(self, llm_client: HelloAgentsLLM, prompt_template: Optional[str] = None):
        self.llm_client = llm_client
        self.prompt_template = prompt_template or DEFAULT_EXECUTOR_PROMPT

    def execute(self, question: str, plan: List[str], **kwargs) -> str:
        history = ""
        final_answer = ""

        print("\n--- 正在执行计划 ---")
        for i, step in enumerate(plan, 1):
            print(f"\n-> 正在执行步骤 {i}/{len(plan)}: {step}")
            prompt = self.prompt_template.format(
                question=question,
                plan=plan,
                history=history if history else "无",
                current_step=step,
            )
            messages = [{"role": "user", "content": prompt}]

            response_text = _extract_text(self.llm_client.invoke(messages, **kwargs))

            history += f"步骤 {i}: {step}\n结果: {response_text}\n\n"
            final_answer = response_text
            print(f"✅ 步骤 {i} 已完成，结果: {final_answer}")

        return final_answer


class MyPlanAndSolveAgent(PlanSolveAgent):
    """
    重写的 PlanAndSolve Agent - 分解规划与逐步执行

    改进点（相对第四章 Plan_and_solve.py）：
    1. 继承框架 PlanSolveAgent，统一 name / llm / config / 历史管理接口
    2. 使用 Message + get_history() 管理对话历史
    3. 支持 custom_prompts 分别覆盖 planner / executor 提示词
    4. LLM 调用统一为 invoke，并透传 **kwargs
    5. run() 始终返回 str，失败时也写入历史并返回可读错误信息
    6. 计划解析增加多种格式兜底，提升稳定性
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[Dict[str, str]] = None,
    ):
        # 先走父类初始化，复用框架历史管理等能力
        # 父类会创建 Function Calling 版 planner/executor，下面用教材版覆盖
        super().__init__(name, llm, system_prompt, config)

        if custom_prompts:
            planner_prompt = custom_prompts.get("planner")
            executor_prompt = custom_prompts.get("executor")
        else:
            planner_prompt = None
            executor_prompt = None

        self.planner = Planner(self.llm, planner_prompt)
        self.executor = Executor(self.llm, executor_prompt)
        print(f"✅ {name} 初始化完成（PlanAndSolve / 提示词解析版）")

    def run(self, input_text: str, **kwargs) -> str:
        """运行 Plan-and-Solve：先规划，再逐步执行"""
        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")

        # 1. 生成计划
        plan = self.planner.plan(input_text, **kwargs)
        if not plan:
            final_answer = "无法生成有效的行动计划，任务终止。"
            print(f"\n--- 任务终止 ---\n{final_answer}")
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(final_answer, "assistant"))
            return final_answer
            

        # 2. 执行计划
        final_answer = self.executor.execute(input_text, plan, **kwargs)
        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")

        # 3. 写入统一历史
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))

        return final_answer
