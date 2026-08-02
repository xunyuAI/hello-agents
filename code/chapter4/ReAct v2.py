"""
ReAct 智能体 — JSON 结构化输出版（课后练习 2）

对比原版 ReAct.py：用 JSON 替代 Thought:/Action: 正则解析，更易校验、更抗格式漂移。
"""

import json
import re
from llm_client import HelloAgentsLLM
from tools import ToolExecutor, search

REACT_JSON_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下：
{tools}

你必须且只能输出一个 JSON 对象，格式如下：

调用工具：
{{"thought": "思考过程", "action": {{"type": "tool", "name": "工具名", "input": "参数"}}}}

给出最终答案：
{{"thought": "为何可以结束", "action": {{"type": "finish", "answer": "最终答案"}}}}

现在，请开始解决以下问题：
Question: {question}
History: {history}
"""

class ReActJsonAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        self.history = []
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_JSON_PROMPT_TEMPLATE.format(
                tools=tools_desc, question=question, history=history_str
            )

            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)
            if not response_text:
                print("错误：LLM未能返回有效响应。")
                break

            thought, action = self._parse_output(response_text)
            print(f"👀 提取的 JSON: {thought}, {action}")
            if thought:
                print(f"🤔 思考: {thought}")
            if not action:
                print("警告：未能解析出有效的 Action，流程终止。")
                break

            # finish → 结束
            if action.get("type") == "finish":
                final_answer = action.get("answer", "")
                print(f"🎉 最终答案: {final_answer}")
                return final_answer

            # tool → 调用工具
            tool_name = action.get("name")
            tool_input = action.get("input", "")
            if not tool_name:
                self.history.append("Observation: 无效的 Action 格式，请检查。")
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")
            tool_function = self.tool_executor.getTool(tool_name)
            observation = (
                tool_function(tool_input)
                if tool_function
                else f"错误：未找到名为 '{tool_name}' 的工具。"
            )

            print(f"👀 观察: {observation}")
            self.history.append(f"Action: {json.dumps(action, ensure_ascii=False)}")
            self.history.append(f"Observation: {observation}")

        print("已达到最大步数，流程终止。")
        return None

    def _parse_output(self, text: str):
        """从模型输出中提取 JSON，返回 (thought, action)。"""
        data = self._extract_json(text)
        if not data:
            return None, None
        thought = data.get("thought")
        action = data.get("action")
        if not isinstance(action, dict):
            return thought, None
        return thought, action

    def _extract_json(self, text: str):
        """兼容纯 JSON 或 ```json ... ``` 代码块。"""
        # 优先取代码块内内容
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        candidate = fenced.group(1).strip() if fenced else text.strip()

        # 再兜底截取首个 {...}
        if not candidate.startswith("{"):
            match = re.search(r"\{[\s\S]*\}", candidate)
            if not match:
                return None
            candidate = match.group(0)

        try:
            data = json.loads(candidate)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None


if __name__ == "__main__":
    llm = HelloAgentsLLM()
    tool_executor = ToolExecutor()
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_desc, search)
    agent = ReActJsonAgent(llm_client=llm, tool_executor=tool_executor)
    question = "华为最新的手机是哪一款？它的主要卖点是什么？"
    agent.run(question)