# test_tree_of_thought_agent.py
"""Tree-of-Thought Agent 测试（对应第 7 章习题 4.3）

刻意选用「需要分步拆解」的多约束规划题，并设置 min_depth_for_final=max_depth，
确保 Expand → Evaluate → Select 会完整循环多层，而不是首轮就 FINAL。
"""

from pathlib import Path

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM

from my_tree_of_thought_agent import MyTreeOfThoughtAgent

# 按脚本所在目录加载 .env
_ENV_DIR = Path(__file__).resolve().parent
load_dotenv(_ENV_DIR / ".env")

llm = HelloAgentsLLM()

# 强制走满 3 层：未到第 3 层不允许提前 FINAL
agent = MyTreeOfThoughtAgent(
    name="我的ToT助手",
    llm=llm,
    breadth=2,
    max_depth=3,
    min_depth_for_final=1,
)

# 多约束规划题：需要先拆约束、再组合方案，适合多层 ToT
question = (
    "小明要在周末安排一天行程，约束如下：\n"
    "1) 必须完成：买菜、复习高等数学、运动；\n"
    "2) 买菜只能在上午 9:00-11:00 的菜市场时段；\n"
    "3) 复习高等数学需要连续 2 小时安静环境；\n"
    "4) 运动建议在饭后 1 小时后进行，时长约 1 小时；\n"
    "5) 中午 12:00-13:00 要吃饭休息。\n"
    "请给出一个合理的时间表（列出上午/中午/下午各时段安排），"
    "并简要说明为什么这样安排能同时满足上述约束。"
)

result = agent.run(question)
print(f"\n测试输出:\n{result}")
print(f"对话历史: {len(agent.get_history())} 条消息")
