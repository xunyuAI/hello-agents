from dotenv import load_dotenv
# 加载 .env 文件中的环境变量
load_dotenv()

import ast
import operator as op
import os
from typing import Dict, Any

from serpapi import SerpApiClient


def search(query: str) -> str:
    """
    一个基于SerpApi的实战网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "错误：SERPAPI_API_KEY 未在 .env 文件中配置。"

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",  # 国家代码
            "hl": "zh-cn",  # 语言代码
        }

        client = SerpApiClient(params)
        results = client.get_dict()

        # 智能解析：优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)

        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"


# AST 白名单：仅允许四则运算与括号，禁止任意 eval
_CALC_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _CALC_OPS:
        return _CALC_OPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_OPS:
        return _CALC_OPS[type(node.op)](_eval_ast(node.operand))
    raise ValueError(f"不支持的表达式节点: {type(node).__name__}")


def calculator(expression: str) -> str:
    """
    安全计算数学表达式。仅允许数字与 + - * / // % ** ()。
    输入示例: (123+456)*789/12
    """
    print(f"🧮 正在执行计算器: {expression}")
    if not expression or not str(expression).strip():
        return (
            "错误：计算器输入为空。"
            "请提供算术表达式，例如 Calculator[(123+456)*789/12]。"
        )

    # 归一化常见中文/全角运算符，便于模型直接传入题面
    expr = (
        str(expression)
        .strip()
        .replace("×", "*")
        .replace("÷", "/")
        .replace("（", "(")
        .replace("）", ")")
        .replace(" ", "")
    )

    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_ast(tree)
        return str(result)
    except ZeroDivisionError:
        return "错误：除数不能为 0。请检查表达式后重试。"
    except Exception as e:
        return f"错误：无法计算表达式。请检查格式后重试。详情: {e}"


CALCULATOR_DESCRIPTION = (
    "用于精确数学计算。输入为算术表达式，如 (123+456)*789/12。"
    "当问题涉及加减乘除等运算时，应使用此工具，不要用搜索来算数。"
)


class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    """
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: callable):
        """
        向工具箱中注册一个新工具。
        """
        if name in self.tools:
            print(f"警告：工具 '{name}' 已存在，将被覆盖。")

        self.tools[name] = {"description": description, "func": func}
        print(f"工具 '{name}' 已注册。")

    def getTool(self, name: str) -> callable:
        """
        根据名称获取一个工具的执行函数。
        """
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串。
        """
        return "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])


# --- 工具初始化与使用示例 ---
if __name__ == '__main__':
    # 1. 初始化工具执行器
    toolExecutor = ToolExecutor()

    # 2. 注册我们的实战搜索工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool("Search", search_description, search)
    toolExecutor.registerTool("Calculator", CALCULATOR_DESCRIPTION, calculator)

    # 3. 打印可用的工具
    print("\n--- 可用的工具 ---")
    print(toolExecutor.getAvailableTools())

    # 4. 智能体的 Action 调用，这次我们问一个实时性的问题
    print("\n--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---")
    tool_name = "Search"
    tool_input = "英伟达最新的GPU型号是什么"

    tool_function = toolExecutor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误：未找到名为 '{tool_name}' 的工具。")

    # 5. 计算器示例（教材习题）
    print("\n--- 执行 Action: Calculator[(123+456)*789/12] ---")
    calc_fn = toolExecutor.getTool("Calculator")
    print("--- 观察 (Observation) ---")
    print(calc_fn("(123+456)*789/12") if calc_fn else "错误：未找到 Calculator 工具。")
