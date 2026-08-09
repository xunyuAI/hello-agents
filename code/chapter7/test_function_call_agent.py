# test_function_call_agent.py
"""
测试第 7.4.5 节 MyFunctionCallAgent（原生 Function Calling）

运行（请在 chapter7 目录，或任意目录均可，.env 按脚本目录加载）:
  ../.venv/bin/python test_function_call_agent.py
  # 或
  /Users/mima0000/Desktop/ai/project/hello-agents/.venv/bin/python test_function_call_agent.py
"""

from pathlib import Path

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM, ToolRegistry
from hello_agents.tools import CalculatorTool

from my_function_call_agent import MyFunctionCallAgent

# 按脚本所在目录加载 .env
_ENV_DIR = Path(__file__).resolve().parent
load_dotenv(_ENV_DIR / ".env")


def _mock_weather(location: str) -> str:
    """教材 OpenAI 示例同款：模拟天气查询（不依赖外网）。"""
    demo = {
        "boston": "Boston: 12°C, cloudy, light wind",
        "beijing": "Beijing: 26°C, sunny",
        "shanghai": "Shanghai: 28°C, rainy",
    }
    key = location.strip().lower()
    for city, info in demo.items():
        if city in key:
            return info
    return f"{location}: 22°C, partly cloudy (mock)"


def build_registry() -> ToolRegistry:
    """注册计算器 + 模拟天气工具。"""
    registry = ToolRegistry()

    calculator = CalculatorTool()
    registry.register_tool(calculator)

    # 函数工具：对应教材里 get_current_weather 示例的简化版
    def get_current_weather(input: str) -> str:
        return _mock_weather(input)

    registry.register_function(
        get_current_weather,
        name="get_current_weather",
        description=(
            "Get the current weather in a given location. "
            "Input should be city name, e.g. Boston or Beijing."
        ),
    )
    return registry

def test_live_function_calling():
    """调用真实 LLM：原生 Function Calling 走通工具循环。"""
    print("=" * 60)
    print("在线测试：原生 Function Calling")
    print("=" * 60)

    llm = HelloAgentsLLM()
    registry = build_registry()
    agent = MyFunctionCallAgent(
        name="FunctionCalling助手",
        llm=llm,
        tool_registry=registry,
        system_prompt=(
            "你是助手。遇到计算请调用 python_calculator；"
            "遇到天气请调用 get_current_weather。"
            "拿到工具结果后再用中文简洁回答。"
        ),
        max_tool_iterations=5,
    )

    # 测试1：计算（对照教材 / SimpleAgent 工具场景）
    print("\n📊 测试1：数学计算")
    q1 = "请帮我计算：(25 + 15) * 3 - 8，只给出最终数字并简短说明。"
    try:
        r1 = agent.run(q1)
        print(f"🎯 结果1: {r1}")
    except Exception as e:
        print(f"❌ 测试1失败: {e}")

    # 测试2：天气（对照教材 OpenAI 原生示例）
    print("\n🌤️ 测试2：天气查询（模拟）")
    q2 = "What's the weather like in Boston today? 请用中文回答。"
    try:
        r2 = agent.run(q2)
        print(f"🎯 结果2: {r2}")
    except Exception as e:
        print(f"❌ 测试2失败: {e}")

    print(f"\n📝 对话历史: {len(agent.get_history())} 条消息")
    print("在线测试完成\n")


if __name__ == "__main__":
    test_live_function_calling()
    print("✨ 全部测试流程结束")
