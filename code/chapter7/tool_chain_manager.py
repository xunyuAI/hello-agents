# tool_chain_manager.py
"""7.5.4 工具链式调用：把多个已注册工具按顺序串起来，用 context 传递中间结果。"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Union

from hello_agents import ToolRegistry


def _result_to_text(result: Any) -> str:
    """
    将 ToolRegistry.execute_tool 的返回值统一成 str。

    当前 hello_agents 返回 ToolResponse；教材早期版本按 str 书写。
    取 .text 既兼容新协议，也方便后续步骤用模板引用。
    """
    if result is None:
        return ""
    text = getattr(result, "text", None)
    if text is not None:
        return str(text)
    return str(result)


class ToolChain:
    """工具链 - 支持多个工具的顺序执行"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.steps: List[Dict[str, Any]] = []

    def add_step(self, tool_name: str, input_template: str, output_key: str = None):
        """
        添加工具执行步骤

        Args:
            tool_name: 工具名称
            input_template: 输入模板，支持变量替换
            output_key: 输出结果的键名，用于后续步骤引用
        """
        self.steps.append({
            "tool_name": tool_name,
            "input_template": input_template,
            "output_key": output_key or f"step_{len(self.steps)}_result"
        })

    def execute(
        self,
        registry: ToolRegistry,
        initial_input: str,
        context: Dict[str, Any] = None,
    ) -> str:
        """执行工具链"""
        if not self.steps:
            return "❌ 工具链执行失败:尚未添加任何步骤"

        context = context or {}
        context["input"] = initial_input

        print(f"🔗 开始执行工具链: {self.name}")

        for i, step in enumerate(self.steps, 1):
            tool_name = step["tool_name"]
            input_template = step["input_template"]
            output_key = step["output_key"]

            # 替换模板中的变量
            try:
                tool_input = input_template.format(**context)
            except KeyError as e:
                return f"❌ 工具链执行失败:模板变量 {e} 未找到"

            print(f"  步骤 {i}: 使用 {tool_name} 处理 '{tool_input[:50]}...'")

            # 执行工具（结果统一成 str，写入 context 供下一步引用）
            result = _result_to_text(registry.execute_tool(tool_name, tool_input))
            context[output_key] = result

            print(f"  ✅ 步骤 {i} 完成，结果长度: {len(result)} 字符")

        # 返回最后一步的结果
        final_result = context[self.steps[-1]["output_key"]]
        print(f"🎉 工具链 '{self.name}' 执行完成")
        return final_result


class ToolChainManager:
    """工具链管理器"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.chains: Dict[str, ToolChain] = {}

    def register_chain(self, chain: ToolChain):
        """注册工具链"""
        self.chains[chain.name] = chain
        print(f"✅ 工具链 '{chain.name}' 已注册")

    def execute_chain(
        self,
        chain_name: str,
        input_data: str,
        context: Dict[str, Any] = None,
    ) -> str:
        """执行指定的工具链"""
        if chain_name not in self.chains:
            return f"❌ 工具链 '{chain_name}' 不存在"

        chain = self.chains[chain_name]
        return chain.execute(self.registry, input_data, context)

    def list_chains(self) -> List[str]:
        """列出所有工具链"""
        return list(self.chains.keys())


# 使用示例（与教材一致：search -> my_calculator）
def create_research_chain() -> ToolChain:
    """创建一个研究工具链:搜索 -> 计算"""
    chain = ToolChain(
        name="research_and_calculate",
        description="搜索信息并进行相关计算"
    )

    # 步骤1:搜索信息
    chain.add_step(
        tool_name="search",
        input_template="{input}",
        output_key="search_result"
    )

    # 步骤2:基于搜索结果进行计算（如果需要）
    # 注意：教材模板带中文说明，适合「能理解自然语言的计算工具」。
    # 若接 AST 安全计算器，测试里会用纯表达式模板的演示链。
    chain.add_step(
        tool_name="my_calculator",
        input_template="根据以下信息计算相关数值:{search_result}",
        output_key="calculation_result"
    )

    return chain


def create_demo_three_step_chain() -> ToolChain:
    """
    可运行的三步演示链（对应习题「≥3 个工具」思路）：
    search(拿到表达式文本) -> extract_expr(抽出算式) -> my_calculator(求值)
    """
    chain = ToolChain(
        name="search_extract_calculate",
        description="搜索 → 抽取表达式 → 安全计算",
    )
    chain.add_step("search", "{input}", "search_result")
    chain.add_step("extract_expr", "{search_result}", "expr")
    chain.add_step("my_calculator", "{expr}", "calculation_result")
    return chain


if __name__ == "__main__":
    # 仅展示结构；完整可运行演示见 test_tool_chain_manager.py
    chain = create_research_chain()
    print(f"链名: {chain.name}, 步骤数: {len(chain.steps)}")
    for i, step in enumerate(chain.steps, 1):
        print(f"  {i}. {step['tool_name']} <- {step['input_template']!r} -> {step['output_key']}")
