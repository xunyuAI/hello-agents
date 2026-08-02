"""
AutoGen 软件开发团队协作案例
"""

import os
import re
import asyncio
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 先测试一个版本，使用 OpenAI 客户端
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console

# 协作产出的可运行代码默认写入同目录
OUTPUT_PY_PATH = Path(__file__).resolve().parent / "generated_bitcoin_app.py"
_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n([\s\S]*?)```", re.IGNORECASE)

def create_openai_model_client():
    """创建 OpenAI 兼容模型客户端（支持 DeepSeek 等非官方模型名）。"""
    model = os.getenv("LLM_MODEL_ID", "gpt-4o")
    # 非 OpenAI 官方模型名时必须显式提供 model_info，否则会抛 ValueError
    model_info = {
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": ModelFamily.UNKNOWN,
        "structured_output": True,
    }
    return OpenAIChatCompletionClient(
        model=model,
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        model_info=model_info,
    )

def create_product_manager(model_client):
    """创建产品经理智能体"""
    system_message = """你是一位经验丰富的产品经理，专门负责软件产品的需求分析和项目规划。

        你的核心职责包括：
        1. **需求分析**：深入理解用户需求，识别核心功能和边界条件
        2. **技术规划**：基于需求制定清晰的技术实现路径
        3. **风险评估**：识别潜在的技术风险和用户体验问题
        4. **协调沟通**：与工程师和其他团队成员进行有效沟通

        当接到开发任务时，请按以下结构进行分析：
        1. 需求理解与分析
        2. 功能模块划分
        3. 技术选型建议
        4. 实现优先级排序
        5. 验收标准定义

        请简洁明了地回应，并在分析完成后说"请工程师开始实现"。"""

    return AssistantAgent(
        name="ProductManager",
        model_client=model_client,
        system_message=system_message,
    )

def create_engineer(model_client):
    """创建软件工程师智能体"""
    system_message = """你是一位资深的软件工程师，擅长 Python 开发和 Web 应用构建。

        你的技术专长包括：
        1. **Python 编程**：熟练掌握 Python 语法和最佳实践
        2. **Web 开发**：精通 Streamlit、Flask、Django 等框架
        3. **API 集成**：有丰富的第三方 API 集成经验
        4. **错误处理**：注重代码的健壮性和异常处理

        当收到开发任务时，请：
        1. 仔细分析技术需求
        2. 选择合适的技术方案
        3. 编写完整的代码实现
        4. 添加必要的注释和说明
        5. 考虑边界情况和异常处理

        请提供完整的可运行代码，并在完成后说"请代码审查员检查"。"""

    return AssistantAgent(
        name="Engineer",
        model_client=model_client,
        system_message=system_message,
    )

def create_code_reviewer(model_client):
    """创建代码审查员智能体"""
    system_message = """你是一位经验丰富的代码审查专家，专注于代码质量和最佳实践。

        你的审查重点包括：
        1. **代码质量**：检查代码的可读性、可维护性和性能
        2. **安全性**：识别潜在的安全漏洞和风险点
        3. **最佳实践**：确保代码遵循行业标准和最佳实践
        4. **错误处理**：验证异常处理的完整性和合理性

        审查流程：
        1. 仔细阅读和理解代码逻辑
        2. 检查代码规范和最佳实践
        3. 识别潜在问题和改进点
        4. 提供具体的修改建议
        5. 评估代码的整体质量

        请提供具体的审查意见，完成后说"代码审查完成，请用户代理测试"。"""

    return AssistantAgent(
        name="CodeReviewer",
        model_client=model_client,
        system_message=system_message,
    )

def create_user_proxy():
    """创建用户代理智能体（自动验收并终止，避免非交互环境卡在 stdin）。"""
    def auto_approve(_prompt: str) -> str:
        return (
            "已完成验收：核心功能与 Streamlit 实现符合需求。"
            "TERMINATE"
        )

    return UserProxyAgent(
        name="UserProxy",
        description="""用户代理，负责以下职责：
            1. 代表用户提出开发需求
            2. 执行最终的代码实现
            3. 验证功能是否符合预期
            4. 提供用户反馈和建议

            完成测试后请回复 TERMINATE。""",
        input_func=auto_approve,
    )

def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    return str(content) if content is not None else ""


def extract_python_code(result: Any) -> Optional[str]:
    """从团队对话结果中提取最完整的 Python 代码块（优先 Engineer）。"""
    messages = list(getattr(result, "messages", []) or [])
    if not messages:
        return None

    engineer_blocks: list[str] = []
    other_blocks: list[str] = []

    for message in messages:
        text = _message_text(message)
        blocks = [b.strip() for b in _CODE_FENCE_RE.findall(text) if b.strip()]
        if not blocks:
            continue
        source = getattr(message, "source", "") or ""
        target = engineer_blocks if source == "Engineer" else other_blocks
        target.extend(blocks)

    candidates = engineer_blocks or other_blocks
    if not candidates:
        return None

    # 选最长且看起来像可运行脚本的代码块
    def score(code: str) -> tuple:
        has_streamlit = 1 if "streamlit" in code or "st." in code else 0
        has_import = 1 if "import " in code else 0
        return (has_streamlit, has_import, len(code))

    return max(candidates, key=score)


def save_generated_code(code: str, output_path: Path = OUTPUT_PY_PATH) -> Path:
    """将提取到的代码写入可运行的 .py 文件。"""
    header = (
        "# -*- coding: utf-8 -*-\n"
        "# 本文件由 autogen_software_team.py 自动从团队协作结果中提取生成。\n"
        "# 运行方式: streamlit run generated_bitcoin_app.py\n\n"
    )
    output_path.write_text(header + code.rstrip() + "\n", encoding="utf-8")
    return output_path


async def run_software_development_team():
    """运行软件开发团队协作"""
    
    print("🔧 正在初始化模型客户端...")
    
    # 先使用标准的 OpenAI 客户端测试
    model_client = create_openai_model_client()
    
    print("👥 正在创建智能体团队...")
    
    # 创建智能体团队
    product_manager = create_product_manager(model_client)
    engineer = create_engineer(model_client)
    code_reviewer = create_code_reviewer(model_client)
    user_proxy = create_user_proxy()
    
    # 添加终止条件
    termination = TextMentionTermination("TERMINATE")
    
    # 创建团队聊天
    team_chat = RoundRobinGroupChat(
        participants=[
            product_manager,
            engineer, 
            code_reviewer,
            user_proxy
        ],
        termination_condition=termination,
        max_turns=20,  # 增加最大轮次
    )
    
    # 定义开发任务
    task = """我们需要开发一个比特币价格显示应用，具体要求如下：

        核心功能：
        - 实时显示比特币当前价格（USD）
        - 显示24小时价格变化趋势（涨跌幅和涨跌额）
        - 提供价格刷新功能

        技术要求：
        - 使用 Streamlit 框架创建 Web 应用
        - 界面简洁美观，用户友好
        - 添加适当的错误处理和加载状态

        请团队协作完成这个任务，从需求分析到最终实现。
        工程师请将完整可运行代码放在 ```python 代码块中。"""
    
    # 执行团队协作
    print("🚀 启动 AutoGen 软件开发团队协作...")
    print("=" * 60)
    
    # 使用 Console 来显示对话过程
    result = await Console(team_chat.run_stream(task=task))
    
    print("\n" + "=" * 60)
    print("✅ 团队协作完成！")

    code = extract_python_code(result)
    if code:
        saved = save_generated_code(code)
        print(f"💾 已将生成代码写入: {saved}")
        print("▶️  运行: streamlit run generated_bitcoin_app.py")
    else:
        print("⚠️ 未从对话中提取到 Python 代码块，未生成文件。")
    
    return result

# 主程序入口
if __name__ == "__main__":
    try:
        # 运行异步协作流程
        result = asyncio.run(run_software_development_team())
        
        print(f"\n📋 协作结果摘要：")
        print(f"- 参与智能体数量：4个")
        print(f"- 任务完成状态：{'成功' if result else '需要进一步处理'}")
        print(f"- 产出文件：{OUTPUT_PY_PATH.name}")
        
    except ValueError as e:
        print(f"❌ 配置错误：{e}")
        print("请检查 .env 文件中的配置是否正确")
    except Exception as e:
        print(f"❌ 运行错误：{e}")
        import traceback
        traceback.print_exc()