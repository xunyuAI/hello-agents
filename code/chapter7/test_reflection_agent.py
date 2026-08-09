# test_reflection_agent.py
from pathlib import Path

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM
from my_reflection_agent import MyReflectionAgent

# 按脚本所在目录加载 .env（避免从仓库根目录运行时找不到配置）
_ENV_DIR = Path(__file__).resolve().parent
load_dotenv(_ENV_DIR / ".env")

llm = HelloAgentsLLM()

# 使用默认通用提示词（含质量评分早停）
general_agent = MyReflectionAgent(name="我的反思助手", llm=llm)

# 使用自定义代码生成提示词（类似第四章）
code_prompts = {
    "initial": "你是Python专家，请编写函数：{task}",
    "reflect": "请审查代码的算法效率：\n任务：{task}\n代码：{content}",
    "refine": "请根据反馈优化代码：\n任务：{task}\n反馈：{feedback}",
}
code_agent = MyReflectionAgent(
    name="我的代码生成助手",
    llm=llm,
    custom_prompts=code_prompts,
)

# 测试使用
result = general_agent.run("写一篇关于人工智能发展历程的简短文章")
print(f"最终结果: {result}")
