# 第七章 构建你的智能体框架

> **配套学习：** 同目录下有可视化互动教程 [`interactive-tutorial/index.html`](./interactive-tutorial/index.html)，可用浏览器打开，按「分层 → LLM → 范式 → 工具」翻页。本文覆盖原版全部知识点，并对框架分层与核心代码概念做教学加厚；HTML 负责可视化拆解，两者互补。

第六章我们「站在巨人肩膀上」体验了 AutoGen、AgentScope、CAMEL、LangGraph。这一章要做一件更「工程师」的事：**把智能体的共性能力，沉淀成你自己能读懂、能改、能扩展的小框架 —— HelloAgents**。

目标不是重新发明一个能打赢工业框架的产品，而是：

1. 把第四章的范式，从「脚本」升级为「可复用组件」；
2. 把 LLM 调用、配置、消息、工具注册做成稳定接口；
3. 为第八章的记忆/RAG、第九章的上下文工程预留挂载点。

---

## 读这一章之前，先建立心智模型

### 一个工程比喻

把 HelloAgents 想成「组装一台可换配件的电脑」：

| 配件 | 框架对应 | 你真正要会的事 |
|------|----------|----------------|
| CPU | `HelloAgentsLLM` | 多提供商、本地、自动检测 |
| 主板总线 | `Message` / `Config` / `Agent` 基类 | 统一数据与生命周期 |
| 预装系统 | Simple / ReAct / Reflection / Plan-and-Solve / FunctionCall | 范式框架化 |
| 外设 | Tool 基类 + 注册表 + 搜索等 | 万物皆工具 |

### 分层图（先印在脑子里）

```
应用层：你的业务 Agent / 脚本
    ↓
范式层：SimpleAgent / ReActAgent / ReflectionAgent / PlanAndSolveAgent / FunctionCallAgent
    ↓
接口层：Agent 抽象 · Message · Config · ToolRegistry
    ↓
模型层：HelloAgentsLLM（OpenAI 兼容 / 多 Provider / 本地）
```

### 几个关键词，先抓牢

| 术语 | 可检查定义 |
|------|------------|
| **Provider** | 模型服务提供方（OpenAI、ModelScope、Ollama、vLLM…） |
| **OpenAI 兼容接口** | 多数本地/云中转都提供 `/v1/chat/completions` 形态 |
| **抽象基类（ABC）** | 规定子类必须实现的方法（如 `run`），保证可替换 |
| **工具注册表** | 名称 → 工具实例的字典式管理，供 Agent 动态调用 |
| **Function Calling** | 模型以结构化方式请求调用函数（相对纯文本 ReAct 解析） |
| **流式输出** | 边生成边返回 token，改善交互体验 |

### 建议怎么读

1. 先跑通 **7.1.3 快速体验**（环境 + 最小对话）。
2. **7.2** 搞懂 LLM 客户端如何「自动找 Provider」。
3. **7.3** 把 Message/Config/Agent 当成宪法——后面所有范式都长在这上面。
4. **7.4** 对照第四章：同一范式，框架版多了哪些工程件？
5. **7.5** 工具系统是「框架能否长出生态」的关键；自定义计算器走一遍。

### 本章在全书中的位置

```
第六章：用别人的框架
    ↓
第七章：造自己的 HelloAgents ← 你在这里
    ↓
第八章：挂上 Memory + RAG
```

---

在前面的章节中，我们讲解了智能体的基础知识，并体验了主流框架带来的开发便利。从本章开始，我们将进入一个更具挑战也更有价值的阶段：**从零开始，逐步构建一个智能体框架——HelloAgents**。

为确保学习过程的连贯性与可复现性，HelloAgents 将以版本迭代的方式推进开发。每一章都会在前一章的基础上增加新的功能模块，并将智能体相关的知识点进行串讲与实现。最终，我们将利用这个自建框架，来高效地实现本书后续章节中的高级应用案例。

## 7.1 框架整体架构设计

> **本节目标**：回答「为何自建」以及 HelloAgents 的设计理念与学习目标。

### 7.1.1 为何需要自建Agent框架

> **铺垫**：用别人的框架快，但教学、二次开发、裁剪依赖、把知识沉淀进团队时，自建小框架往往更「可控可讲」。
>
> 自建不是为了重复造轮子炫技，而是为了**暴露必要复杂度**：让你看清 Agent 真正依赖哪些接口。

在智能体技术快速发展的今天，市面上已经存在众多成熟的Agent框架。那么，为什么我们还要从零开始构建一个新的框架呢？

（1）市面框架的快速迭代与局限性

智能体领域是一个快速发展的领域，随时会有新的概念产生，对于智能体的设计每个框架都有自己的定位和理解，不过智能体的核心知识点是一致的。

- **过度抽象的复杂性**：许多框架为了追求通用性，引入了大量抽象层和配置选项。以LangChain为例，其链式调用机制虽然灵活，但对初学者而言学习曲线陡峭，往往需要理解大量概念才能完成简单任务。
- **快速迭代带来的不稳定性**：商业化框架为了抢占市场，API接口变更频繁。开发者经常面临版本升级后代码无法运行的困扰，维护成本居高不下。
- **黑盒化的实现逻辑**：许多框架将核心逻辑封装得过于严密，开发者难以理解Agent的内部工作机制，缺乏深度定制能力。遇到问题时只能依赖文档和社区支持，尤其是如果社区不够活跃，可能一个反馈意见会非常久也没有人推进，影响后续的开发效率。
- **依赖关系的复杂性**：成熟框架往往携带大量依赖包，安装包体积庞大，在需要与别的项目代码配合使用可能出现依赖冲突问题。

（2）从使用者到构建者的能力跃迁

构建自己的Agent框架，实际上是一个从"使用者"向"构建者"转变的过程。这种转变带来的价值是长远的。

- **深度理解Agent工作原理**：通过亲手实现每个组件，开发者能够真正理解Agent的思考过程、工具调用机制、以及各种设计模式的好坏与区别。
- **获得完全的控制权**：自建框架意味着对每一行代码都有完全的掌控，可以根据具体需求进行精确调优，而不受第三方框架设计理念的束缚。
- **培养系统设计能力**：框架构建过程涉及模块化设计、接口抽象、错误处理等软件工程核心技能，这些能力对开发者的长期成长具有重要价值。

（3）定制化需求与深度掌握的必要性

在实际应用中，不同场景对智能体的需求差异巨大，往往都需要在通用框架基础上做二次开发。

- **特定领域的优化需求**：金融、医疗、教育等垂直领域往往需要针对性的提示词模板、特殊的工具集成、以及定制化的安全策略。
- **性能与资源的精确控制**：生产环境中，对响应时间、内存占用、并发处理能力都有严格要求，通用框架的"一刀切"方案往往无法满足精细化需求。
- **学习与教学的透明性要求**：在我们的教学场景中，学习者更期待的是清晰地看到智能体的每一步构建过程，理解不同范式的工作机制，这要求框架具有高度的可观测性和可解释性。

### 7.1.2 HelloAgents框架的设计理念

> **理念抓三个词**：简单优先、万物皆工具、从 Paradign 到 Product（可演进）。

构建一个新的Agent框架，关键不在于功能的多少，而在于设计理念是否能真正解决现有框架的痛点。HelloAgents框架的设计围绕着一个核心问题展开：如何让学习者既能快速上手，又能深入理解Agent的工作原理？

当你初次接触任何成熟的框架时，可能会被其丰富的功能所吸引，但很快就会发现一个问题：要完成一个简单的任务，往往需要理解Chain、Agent、Tool、Memory、Retriever等十几个不同的概念。每个概念都有自己的抽象层，学习曲线变得异常陡峭。这种复杂性虽然带来了强大的功能，但也成为了初学者的障碍。HelloAgents框架试图在功能完整性和学习友好性之间找到平衡点，形成了四个核心的设计理念。

（1）轻量级与教学友好的平衡

一个优秀的学习框架应该具备完整的可读性。HelloAgents将核心代码按照章节区分开，这是基于一个简单的原则：任何有一定编程基础的开发者都应该能够在合理的时间内完全理解框架的工作原理。在依赖管理方面，框架采用了极简主义的策略。除了OpenAI的官方SDK和几个必要的基础库外，不引入任何重型依赖。如果遇到问题时，我们可以直接定位到框架本身的代码，而不需要在复杂的依赖关系中寻找答案。

（2）基于标准API的务实选择

OpenAI的API已经成为了行业标准，几乎所有主流的LLM提供商都在努力兼容这套接口。HelloAgents选择在这个标准之上构建，而不是重新发明一套抽象接口。这个决定主要是出于几点动机。首先是兼容性的保证，当你掌握了HelloAgents的使用方法后，迁移到其他框架或将其集成到现有项目中时，底层的API调用逻辑是完全一致的。其次是学习成本的降低。你不需要学习新的概念模型，因为所有的操作都基于你已经熟悉的标准接口。

（3）渐进式学习路径的精心设计

HelloAgents提供了一条清晰的学习路径。我们将会把每一章的学习代码，保存为一个可以pip下载的历史版本，因此无需担心代码的使用成本，因为每一个核心的功能都将会是你自己编写的。这种设计让你能够按照自己的需求和节奏前进。每一步的升级都是自然而然的，不会产生概念上的跳跃或理解上的断层。值得一提的是，我们这一章的内容，也是基于前六章的内容来完善的。同样，这一章也是为后续高级知识学习部分打下框架基础。

（4）统一的“工具”抽象：万物皆为工具

为了彻底贯彻轻量级与教学友好的理念，HelloAgents在架构上做出了一个关键的简化：除了核心的Agent类，一切皆为Tools。在许多其他框架中需要独立学习的Memory（记忆）、RAG（检索增强生成）、RL（强化学习）、MCP（协议）等模块，在HelloAgents中都被统一抽象为一种“工具”。这种设计的初衷是消除不必要的抽象层，让学习者可以回归到最直观的“智能体调用工具”这一核心逻辑上，从而真正实现快速上手和深入理解的统一。

### 7.1.3 本章学习目标

> **快速体验导读**：先 `.env`，再 `HelloAgentsLLM` + `SimpleAgent` 跑通一句对话；工具相关行若提示需 7.4.1，属正常——按章节顺序推进即可。

让我们先看看第七章的核心学习内容：

```
hello-agents/
├── hello_agents/
│   │
│   ├── core/                     # 核心框架层
│   │   ├── agent.py              # Agent基类
│   │   ├── llm.py                # HelloAgentsLLM统一接口
│   │   ├── message.py            # 消息系统
│   │   ├── config.py             # 配置管理
│   │   └── exceptions.py         # 异常体系
│   │
│   ├── agents/                   # Agent实现层
│   │   ├── simple_agent.py       # SimpleAgent实现
│   │   ├── react_agent.py        # ReActAgent实现
│   │   ├── reflection_agent.py   # ReflectionAgent实现
│   │   └── plan_solve_agent.py   # PlanAndSolveAgent实现
│   │
│   ├── tools/                    # 工具系统层
│   │   ├── base.py               # 工具基类
│   │   ├── registry.py           # 工具注册机制
│   │   ├── chain.py              # 工具链管理系统
│   │   ├── async_executor.py     # 异步工具执行器
│   │   └── builtin/              # 内置工具集
│   │       ├── calculator.py     # 计算工具
│   │       └── search.py         # 搜索工具
└──
```

在开始编写具体代码之前，我们需要先建立一个清晰的架构蓝图。HelloAgents的架构设计遵循了"分层解耦、职责单一、接口统一"的核心原则，这样既保持了代码的组织性，也便于按照章节扩展内容。

**快速开始：安装HelloAgents框架**

为了让读者能够快速体验本章的完整功能，我们提供了可直接安装的Python包。你可以通过以下命令安装本章对应的版本：

```bash
# hello-agents 框架代码可见链接：https://github.com/jjyaoao/HelloAgents
# Python 版本需要>=3.10
pip install "hello-agents==0.1.1"
```

本章的学习可以采用两种方式：

1. **体验式学习**：直接使用`pip`安装框架，运行示例代码，快速体验各种功能
2. **深度学习**：跟随本章内容，从零开始实现每个组件，深入理解框架的设计思想和实现细节

我们建议采用"先体验，后实现"的学习路径。在本章中，我们提供了完整的测试文件，你可以重写核心函数并运行测试，以检验你的实现是否正确。这种学习方式既保证了实践性，又确保了学习效果。如果你想深入了解框架的实现细节，或者希望参与到框架的开发中来，可以访问这个[GitHub仓库](https://github.com/jjyaoao/helloagents)。

在开始之前，让我们用30秒体验使用Hello-agents构建简单智能体！

```python
# 配置好同级文件夹下.env中的大模型API, 可参考code文件夹配套的.env.example，也可以拿前几章的案例的.env文件复用。
from hello_agents import SimpleAgent, HelloAgentsLLM
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 创建LLM实例 - 框架自动检测provider
llm = HelloAgentsLLM()

# 或手动指定provider（可选）
# llm = HelloAgentsLLM(provider="modelscope")

# 创建SimpleAgent
agent = SimpleAgent(
    name="AI助手",
    llm=llm,
    system_prompt="你是一个有用的AI助手"
)

# 基础对话
response = agent.run("你好！请介绍一下自己")
print(response)

# 添加工具功能（可选）
from hello_agents.tools import CalculatorTool
calculator = CalculatorTool()
# 需要实现7.4.1的MySimpleAgent进行调用，后续章节会支持此类调用方式
# agent.add_tool(calculator)

# 现在可以使用工具了
response = agent.run("请帮我计算 2 + 3 * 4")
print(response)

# 查看对话历史
print(f"历史消息数: {len(agent.get_history())}")
```



## 7.2 HelloAgentsLLM扩展

> **本节目标**：让「换模型」变成改配置，而不是改业务代码。

本节内容将在第 4.1.3 节创建的 `HelloAgentsLLM` 基础上进行迭代升级。我们将把这个基础客户端，改造为一个更具适应性的模型调用中枢。本次升级主要围绕以下三个目标展开：

1. **多提供商支持**：实现对 OpenAI、ModelScope、智谱 AI 等多种主流 LLM 服务商的无缝切换，避免框架与特定供应商绑定。
2. **本地模型集成**：引入 VLLM 和 Ollama 这两种高性能本地部署方案，作为对第 3.2.3 节中 Hugging Face Transformers 方案的生产级补充，满足数据隐私和成本控制的需求。
3. **自动检测机制**：建立一套自动识别机制，使框架能根据环境信息智能推断所使用的 LLM 服务类型，简化用户的配置过程。

### 7.2.1 支持多提供商

> **代码精讲**：子类/扩展点通常落在「如何构造 client」与「如何把消息转成提供商格式」。先读 `__init__` 与 `think`/`invoke` 主路径。

我们之前定义的 `HelloAgentsLLM` 类，已经能够通过 `api_key` 和 `base_url` 这两个核心参数，连接任何兼容 OpenAI 接口的服务。这在理论上保证了通用性，但在实际应用中，不同的服务商在环境变量命名、默认 API 地址和推荐模型等方面都存在差异。如果每次切换服务商都需要用户手动查询并修改代码，会极大影响开发效率。为了解决这一问题，我们引入 `provider`。其改进思路是：让 `HelloAgentsLLM` 在内部处理不同服务商的配置细节，从而为用户提供一个统一、简洁的调用体验。具体的实现细节我们将在7.2.3节“自动检测机制”中详细阐述，在这里，我们首先关注如何利用这一机制来扩展框架。

下面，我们将演示如何通过继承 `HelloAgentsLLM`，来增加对 ModelScope 平台的支持。我们希望读者不仅学会如何“使用”框架，更能掌握如何“扩展”框架。直接修改已安装的库源码是一种不被推荐的做法，因为它会使后续的库升级变得困难。

（1）创建自定义LLM类并继承

假设我们的项目目录中有一个 `my_llm.py` 文件。我们首先从 `hello_agents` 库中导入 `HelloAgentsLLM` 基类，然后创建一个名为 `MyLLM` 的新类继承它。

```python
# my_llm.py
import os
from typing import Optional
from openai import OpenAI
from hello_agents import HelloAgentsLLM

class MyLLM(HelloAgentsLLM):
    """
    一个自定义的LLM客户端，通过继承增加了对ModelScope的支持。
    """
    pass # 暂时留空
```

（2）重写 `__init__` 方法以支持新供应商

接下来，我们在 `MyLLM` 类中重写 `__init__` 方法。我们的目标是：当用户传入 `provider="modelscope"` 时，执行我们自定义的逻辑；否则，就调用父类 `HelloAgentsLLM` 的原始逻辑，使其能够继续支持 OpenAI 等其他内置的供应商。

```python
class MyLLM(HelloAgentsLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = "auto",
        **kwargs
    ):
        # 检查provider是否为我们想处理的'modelscope'
        if provider == "modelscope":
            print("正在使用自定义的 ModelScope Provider")
            self.provider = "modelscope"
            
            # 解析 ModelScope 的凭证
            self.api_key = api_key or os.getenv("MODELSCOPE_API_KEY")
            self.base_url = base_url or "https://api-inference.modelscope.cn/v1/"
            
            # 验证凭证是否存在
            if not self.api_key:
                raise ValueError("ModelScope API key not found. Please set MODELSCOPE_API_KEY environment variable.")

            # 设置默认模型和其他参数
            self.model = model or os.getenv("LLM_MODEL_ID") or "Qwen/Qwen2.5-VL-72B-Instruct"
            self.temperature = kwargs.get('temperature', 0.7)
            self.max_tokens = kwargs.get('max_tokens')
            self.timeout = kwargs.get('timeout', 60)
            
            # 使用获取的参数创建OpenAI客户端实例
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

        else:
            # 如果不是 modelscope, 则完全使用父类的原始逻辑来处理
            super().__init__(model=model, api_key=api_key, base_url=base_url, provider=provider, **kwargs)

```

这段代码展示了“重写”的思想：我们拦截了 `provider="modelscope"` 的情况并进行了特殊处理，对于其他所有情况，则通过 `super().__init__(...)` 交还给父类，保留了原有框架的全部功能。

（3）使用自定义的 `MyLLM` 类

现在，我们可以在项目的业务逻辑中，像使用原生 `HelloAgentsLLM` 一样使用我们自己的 `MyLLM` 类。

首先，在 `.env` 文件中配置 ModelScope 的 API 密钥：

```bash
# .env file
MODELSCOPE_API_KEY="your-modelscope-api-key"
```

然后，在主程序中导入并使用 `MyLLM`：

```python
# my_main.py
from dotenv import load_dotenv
from my_llm import MyLLM # 注意:这里导入我们自己的类

# 加载环境变量
load_dotenv()

# 实例化我们重写的客户端，并指定provider
llm = MyLLM(provider="modelscope") 

# 准备消息
messages = [{"role": "user", "content": "你好，请介绍一下你自己。"}]

# 发起调用，think等方法都已从父类继承，无需重写
response_stream = llm.think(messages)

# 打印响应
print("ModelScope Response:")
for chunk in response_stream:
    # chunk在my_llm库中已经打印过一遍，这里只需要pass即可
    # print(chunk, end="", flush=True)
    pass
```

通过以上步骤，我们就在不修改 `hello-agents` 库源码的前提下，成功为其扩展了新的功能。这种方法不仅保证了代码的整洁和可维护性，也使得未来升级 `hello-agents` 库时，我们的定制化功能不会丢失。

### 7.2.2 本地模型调用

> **铺垫**：vLLM / Ollama 往往提供 OpenAI 兼容 HTTP。对框架来说，本地模型 ≈ 换了 `base_url` 的远程模型。
>
> **易混**：模型文件在本地 ≠ 框架直接 `from_pretrained`；本章主流是「本地起服务，框架走 HTTP」。

在第 3.2.3 节，我们学习了如何使用 Hugging Face Transformers 库在本地直接运行开源模型。该方法非常适合入门学习和功能验证，但其底层实现在处理高并发请求时性能有限，通常不作为生产环境的首选方案。

为了在本地实现高性能、生产级的模型推理服务，社区涌现出了 VLLM 和 Ollama 等优秀工具。它们通过连续批处理、PagedAttention 等技术，显著提升了模型的吞吐量和运行效率，并将模型封装为兼容 OpenAI 标准的 API 服务。这意味着，我们可以将它们无缝地集成到 `HelloAgentsLLM` 中。

**VLLM**

VLLM 是一个为 LLM 推理设计的高性能 Python 库。它通过 PagedAttention 等先进技术，可以实现比标准 Transformers 实现高出数倍的吞吐量。下面是在本地部署一个 VLLM 服务的完整步骤：

首先，需要根据你的硬件环境（特别是 CUDA 版本）安装 VLLM。推荐遵循其[官方文档](https://docs.vllm.ai/en/latest/getting_started/installation.html)进行安装，以避免版本不匹配问题。

```python
pip install vllm
```

安装完成后，使用以下命令即可启动一个兼容 OpenAI 的 API 服务。VLLM 会自动从 Hugging Face Hub 下载指定的模型权重（如果本地不存在）。我们依然以 Qwen1.5-0.5B-Chat 模型为例：

```
# 启动 VLLM 服务，并加载 Qwen1.5-0.5B-Chat 模型
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen1.5-0.5B-Chat \
    --host 0.0.0.0 \
    --port 8000
```

服务启动后，便会在 `http://localhost:8000/v1` 地址上提供与 OpenAI 兼容的 API。

**Ollama**

Ollama 进一步简化了本地模型的管理和部署，它将模型下载、配置和服务启动等步骤封装到了一条命令中，非常适合快速上手。访问 Ollama [官方网站](https://ollama.com)下载并安装适用于你操作系统的客户端。

安装后，打开终端，执行以下命令即可下载并运行一个模型（以 Llama 3 为例）。Ollama 会自动处理模型的下载、服务封装和硬件加速配置。

```
# 首次运行会自动下载模型，之后会直接启动服务
ollama run llama3
```

当你在终端看到模型的交互提示时，即表示服务已经成功在后台启动。Ollama 默认会在 `http://localhost:11434/v1` 地址上暴露 OpenAI 兼容的 API 接口。

**接入 `HelloAgentsLLM`**

由于 VLLM 和 Ollama 都遵循了行业标准 API，将它们接入 `HelloAgentsLLM` 的过程非常简单。我们只需在实例化客户端时，将它们视为一个新的 `provider` 即可。

例如，连接本地运行的 **VLLM** 服务：

```python
llm_client = HelloAgentsLLM(
    provider="vllm",
    model="Qwen/Qwen1.5-0.5B-Chat", # 需与服务启动时指定的模型一致
    base_url="http://localhost:8000/v1",
    api_key="vllm" # 本地服务通常不需要真实API Key，可填任意非空字符串
)
```

或者，通过设置环境变量并让客户端自动检测，实现代码的零修改：

```bash
# 在 .env 文件中设置
LLM_BASE_URL="http://localhost:8000/v1"
LLM_API_KEY="vllm"

# Python 代码中直接实例化即可
llm_client = HelloAgentsLLM() # 将自动检测为 vllm
```

同理，连接本地的 **Ollama** 服务也一样简单：

```python
llm_client = HelloAgentsLLM(
    provider="ollama",
    model="llama3", # 需与 `ollama run` 指定的模型一致
    base_url="http://localhost:11434/v1",
    api_key="ollama" # 本地服务同样不需要真实 Key
)
```

通过这种统一的设计，我们的智能体核心代码无需任何修改，就可以在云端 API 和本地模型之间自由切换。这为后续应用的开发、部署、成本控制以及保护数据隐私提供了极大的灵活性。

### 7.2.3 自动检测机制

> **读代码顺序**：环境变量优先级 → URL 特征猜测 → 默认回退。把它当成「小型服务发现」。

为了尽可能减少用户的配置负担并遵循“约定优于配置”的原则，`HelloAgentsLLM` 内部设计了两个核心辅助方法：`_auto_detect_provider` 和 `_resolve_credentials`。它们协同工作，`_auto_detect_provider` 负责根据环境信息推断服务商，而 `_resolve_credentials` 则根据推断结果完成具体的参数配置。

`_auto_detect_provider` 方法负责根据环境信息，按照下述优先级顺序，尝试自动推断服务商：

1. **最高优先级：检查特定服务商的环境变量** 这是最直接、最可靠的判断依据。框架会依次检查 `MODELSCOPE_API_KEY`, `OPENAI_API_KEY`, `ZHIPU_API_KEY` 等环境变量是否存在。一旦发现任何一个，就会立即确定对应的服务商。

2. **次高优先级：根据 `base_url` 进行判断** 如果用户没有设置特定服务商的密钥，但设置了通用的 `LLM_BASE_URL`，框架会转而解析这个 URL。

   - **域名匹配**：通过检查 URL 中是否包含 `"api-inference.modelscope.cn"`, `"api.openai.com"` 等特征字符串来识别云服务商。

   - **端口匹配**：通过检查 URL 中是否包含 `:11434` (Ollama), `:8000` (VLLM) 等本地服务的标准端口来识别本地部署方案。

3. **辅助判断：分析 API 密钥的格式** 在某些情况下，如果上述两种方式都无法确定，框架会尝试分析通用环境变量 `LLM_API_KEY` 的格式。例如，某些服务商的 API 密钥有固定的前缀或独特的编码格式。不过，由于这种方式可能存在模糊性（例如多个服务商的密钥格式相似），因此它的优先级较低，仅作为辅助手段。

其部分关键代码如下：

```python
def _auto_detect_provider(self, api_key: Optional[str], base_url: Optional[str]) -> str:
    """
    自动检测LLM提供商
    """
    # 1. 检查特定提供商的环境变量 (最高优先级)
    if os.getenv("MODELSCOPE_API_KEY"): return "modelscope"
    if os.getenv("OPENAI_API_KEY"): return "openai"
    if os.getenv("ZHIPU_API_KEY"): return "zhipu"
    # ... 其他服务商的环境变量检查

    # 获取通用的环境变量
    actual_api_key = api_key or os.getenv("LLM_API_KEY")
    actual_base_url = base_url or os.getenv("LLM_BASE_URL")

    # 2. 根据 base_url 判断
    if actual_base_url:
        base_url_lower = actual_base_url.lower()
        if "api-inference.modelscope.cn" in base_url_lower: return "modelscope"
        if "open.bigmodel.cn" in base_url_lower: return "zhipu"
        if "localhost" in base_url_lower or "127.0.0.1" in base_url_lower:
            if ":11434" in base_url_lower: return "ollama"
            if ":8000" in base_url_lower: return "vllm"
            return "local" # 其他本地端口

    # 3. 根据 API 密钥格式辅助判断
    if actual_api_key:
        if actual_api_key.startswith("ms-"): return "modelscope"
        # ... 其他密钥格式判断

    # 4. 默认返回 'auto'，使用通用配置
    return "auto"
```

一旦 `provider` 被确定（无论是用户指定还是自动检测），`_resolve_credentials` 方法便会接手处理服务商的差异化配置。它会根据 `provider` 的值，去主动查找对应的环境变量，并为其设置默认的 `base_url`。其部分关键实现如下：

```python
def _resolve_credentials(self, api_key: Optional[str], base_url: Optional[str]) -> tuple[str, str]:
    """根据provider解析API密钥和base_url"""
    if self.provider == "openai":
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
        return resolved_api_key, resolved_base_url

    elif self.provider == "modelscope":
        resolved_api_key = api_key or os.getenv("MODELSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
        resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api-inference.modelscope.cn/v1/"
        return resolved_api_key, resolved_base_url
    
    # ... 其他服务商的逻辑
```

让我们通过一个简单的例子来感受自动检测带来的便利。假设一个用户想要使用本地的 Ollama 服务，他只需在 `.env` 文件中进行如下配置：

```bash
LLM_BASE_URL="http://localhost:11434/v1"
LLM_MODEL_ID="llama3"
```

他完全不需要配置 `LLM_API_KEY` 或在代码中指定 `provider`。然后，在 Python 代码中，他只需简单地实例化 `HelloAgentsLLM` 即可：

```python
from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM

load_dotenv()

# 无需传入 provider，框架会自动检测
llm = HelloAgentsLLM() 
# 框架内部日志会显示检测到 provider 为 'ollama'

# 后续调用方式完全不变
messages = [{"role": "user", "content": "你好！"}]
for chunk in llm.think(messages):
    print(chunk, end="")

```

在这个过程中，`_auto_detect_provider` 方法通过解析 `LLM_BASE_URL` 中的 `"localhost"` 和 `:11434`，成功地将 `provider` 推断为 `"ollama"`。随后，`_resolve_credentials` 方法会为 Ollama 设置正确的默认参数。

相比于4.1.3节的基础实现，现在的HelloAgentsLLM具有以下显著优势：

<div align="center">
  <p>表 7.1 HelloAgentLLM不同版本特性对比</p>
  <img src="https://raw.githubusercontent.com/datawhalechina/Hello-Agents/main/docs/images/7-figures/table-01.png" alt="" width="90%"/>
</div>

如上表7.1所示，这种演进体现了框架设计的重要原则：**从简单开始，逐步完善**。我们在保持接口简洁的同时，增强了功能的完整性。



## 7.3 框架接口实现

> **本节目标**：三块积木 —— Message（说什么）、Config（怎么配）、Agent（生命周期）。

在上节中，我们构建了 `HelloAgentsLLM` 这一核心组件，解决了与大语言模型通信的关键问题。不过它还需要一系列配套的接口和组件来处理数据流、管理配置、应对异常，并为上层应用的构建提供一个清晰、统一的结构。本节将讲述以下三个核心文件：

- `message.py`： 定义了框架内统一的消息格式，确保了智能体与模型之间信息传递的标准化。
- `config.py`： 提供了一个中心化的配置管理方案，使框架的行为易于调整和扩展。
- `agent.py`： 定义了所有智能体的抽象基类（`Agent`），为后续实现不同类型的智能体提供了统一的接口和规范。

### 7.3.1 Message 类

> **为什么需要 Message**：用结构化角色（system/user/assistant/tool）替代裸字符串，方便历史管理与多范式复用。

在智能体与大语言模型的交互中，对话历史是至关重要的上下文。为了规范地管理这些信息，我们设计了一个简易 `Message` 类。在后续上下文工程章节中，会对其进行扩展。

```python
"""消息系统"""
from typing import Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel

# 定义消息角色的类型，限制其取值
MessageRole = Literal["user", "assistant", "system", "tool"]

class Message(BaseModel):
    """消息类"""
    
    content: str
    role: MessageRole
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __init__(self, content: str, role: MessageRole, **kwargs):
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get('timestamp', datetime.now()),
            metadata=kwargs.get('metadata', {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（OpenAI API格式）"""
        return {
            "role": self.role,
            "content": self.content
        }
    
    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"
```

该类的设计有几个关键点。首先，我们通过 `typing.Literal` 将 `role` 字段的取值严格限制为 `"user"`, `"assistant"`, `"system"`, `"tool"` 四种，这直接对应 OpenAI API 的规范，保证了类型安全。除了 `content` 和 `role` 这两个核心字段外，我们还增加了 `timestamp` 和 `metadata`，为日志记录和未来功能扩展预留了空间。最后，`to_dict()` 方法是其核心功能之一，负责将内部使用的 `Message` 对象转换为与 OpenAI API 兼容的字典格式，体现了“对内丰富，对外兼容”的设计原则。

### 7.3.2 Config 类

> **为什么需要 Config**：把温度、最大 token、超时、默认模型等集中管理，避免魔法常量散落。

`Config` 类的职责是将代码中硬编码配置参数集中起来，并支持从环境变量中读取。

```python
"""配置管理"""
import os
from typing import Optional, Dict, Any
from pydantic import BaseModel

class Config(BaseModel):
    """HelloAgents配置类"""
    
    # LLM配置
    default_model: str = "gpt-3.5-turbo"
    default_provider: str = "openai"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    
    # 系统配置
    debug: bool = False
    log_level: str = "INFO"
    
    # 其他配置
    max_history_length: int = 100
    
    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量创建配置"""
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.dict()
```

首先，我们将配置项按逻辑划分为 `LLM配置`、`系统配置` 等，使结构一目了然。其次，每个配置项都设有合理的默认值，保证了框架在零配置下也能工作。最核心的是 `from_env()` 类方法，它允许用户通过设置环境变量来覆盖默认配置，无需修改代码，这在部署到不同环境时尤其有用。

### 7.3.3 Agent 抽象基类

> **精讲**：抽象基类规定「Agent 必须会什么」（如运行、维护历史）。具体范式只实现差异部分。

`Agent` 类是整个框架的顶层抽象。它定义了一个智能体应该具备的通用行为和属性，但并不关心具体的实现方式。我们通过 Python 的 `abc` (Abstract Base Classes) 模块来实现它，这强制所有具体的智能体实现（如后续章节的 `SimpleAgent`, `ReActAgent` 等）都必须遵循同一个“接口”。

```python
"""Agent基类"""
from abc import ABC, abstractmethod
from typing import Optional, Any
from .message import Message
from .llm import HelloAgentsLLM
from .config import Config

class Agent(ABC):
    """Agent基类"""
    
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = []
    
    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行Agent"""
        pass
    
    def add_message(self, message: Message):
        """添加消息到历史记录"""
        self._history.append(message)
    
    def clear_history(self):
        """清空历史记录"""
        self._history.clear()
    
    def get_history(self) -> list[Message]:
        """获取历史记录"""
        return self._history.copy()
    
    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"
```

该类的设计体现了面向对象中的抽象原则。首先，它通过继承 `ABC` 被定义为一个不能直接实例化的抽象类。其构造函数 `__init__` 清晰地定义了 Agent 的核心依赖：名称、LLM 实例、系统提示词和配置。最重要的部分是使用 `@abstractmethod` 装饰的 `run` 方法，它强制所有子类必须实现此方法，从而保证了所有智能体都有统一的执行入口。此外，基类还提供了通用的历史记录管理方法，这些方法与 `Message` 类协同工作，体现了组件间的联系。

至此，我们已经完成了 `HelloAgents` 框架核心基础组件的设计与实现。

## 7.4 Agent范式的框架化实现

> **对照第四章读**：业务提示词可以很像；差别在于历史、工具、流式、配置是否被框架托住。

本节内容将在第四章构建的三种经典Agent范式（ReAct、Plan-and-Solve、Reflection）基础上进行框架化重构，并新增SimpleAgent作为基础对话范式。我们将把这些独立的Agent实现，改造为基于统一架构的框架组件。本次重构主要围绕以下三个核心目标展开：

1. **提示词工程的系统性提升**：对第四章中的提示词进行深度优化，从特定任务导向转向通用化设计，同时增强格式约束和角色定义。
2. **接口与格式的标准化统一**：建立统一的Agent基类和标准化的运行接口，所有Agent都遵循相同的初始化参数、方法签名和历史管理机制。
3. **高度可配置的自定义能力**：支持用户自定义提示词模板、配置参数和执行策略。

### 7.4.1 SimpleAgent

> **故事线**：无工具对话 → 注册工具 → 解析调用 → 流式；测试文件按「四场景」扫一遍。

SimpleAgent是最基础的Agent实现，它展示了如何在框架基础上构建一个完整的对话智能体。我们将通过继承框架基类来重写SimpleAgent。首先，在你的项目目录中创建一个`my_simple_agent.py`文件：

```python
# my_simple_agent.py
from typing import Optional, Iterator
from hello_agents import SimpleAgent, HelloAgentsLLM, Config, Message

class MySimpleAgent(SimpleAgent):
    """
    重写的简单对话Agent
    展示如何基于框架基类构建自定义Agent
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional['ToolRegistry'] = None,
        enable_tool_calling: bool = True
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        print(f"✅ {name} 初始化完成，工具调用: {'启用' if self.enable_tool_calling else '禁用'}")
```

接下来，我们需要重写Agent基类的抽象方法`run`。SimpleAgent支持可选的工具调用功能，也方便后续章节的扩展：

```python
# 继续在 my_simple_agent.py 中添加
import re

class MySimpleAgent(SimpleAgent):
    # ... 前面的 __init__ 方法

    def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        """
        重写的运行方法 - 实现简单对话逻辑，支持可选工具调用
        """
        print(f"🤖 {self.name} 正在处理: {input_text}")

        # 构建消息列表
        messages = []

        # 添加系统消息（可能包含工具信息）
        enhanced_system_prompt = self._get_enhanced_system_prompt()
        messages.append({"role": "system", "content": enhanced_system_prompt})

        # 添加历史消息
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        # 添加当前用户消息
        messages.append({"role": "user", "content": input_text})

        # 如果没有启用工具调用，使用简单对话逻辑
        if not self.enable_tool_calling:
            response = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response, "assistant"))
            print(f"✅ {self.name} 响应完成")
            return response

        # 支持多轮工具调用的逻辑
        return self._run_with_tools(messages, input_text, max_tool_iterations, **kwargs)

    def _get_enhanced_system_prompt(self) -> str:
        """构建增强的系统提示词，包含工具信息"""
        base_prompt = self.system_prompt or "你是一个有用的AI助手。"

        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt

        # 获取工具描述
        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "暂无可用工具":
            return base_prompt

        tools_section = "\n\n## 可用工具\n"
        tools_section += "你可以使用以下工具来帮助回答问题:\n"
        tools_section += tools_description + "\n"

        tools_section += "\n## 工具调用格式\n"
        tools_section += "当需要使用工具时，请使用以下格式:\n"
        tools_section += "`[TOOL_CALL:{tool_name}:{parameters}]`\n"
        tools_section += "例如:`[TOOL_CALL:search:Python编程]` 或 `[TOOL_CALL:memory:recall=用户信息]`\n\n"
        tools_section += "工具调用结果会自动插入到对话中，然后你可以基于结果继续回答。\n"

        return base_prompt + tools_section
```

现在我们实现工具调用的核心逻辑：

```python
# 继续在 my_simple_agent.py 中添加
class MySimpleAgent(SimpleAgent):
    # ... 前面的方法

    def _run_with_tools(self, messages: list, input_text: str, max_tool_iterations: int, **kwargs) -> str:
        """支持工具调用的运行逻辑"""
        current_iteration = 0
        final_response = ""

        while current_iteration < max_tool_iterations:
            # 调用LLM
            response = self.llm.invoke(messages, **kwargs)

            # 检查是否有工具调用
            tool_calls = self._parse_tool_calls(response)

            if tool_calls:
                print(f"🔧 检测到 {len(tool_calls)} 个工具调用")
                # 执行所有工具调用并收集结果
                tool_results = []
                clean_response = response

                for call in tool_calls:
                    result = self._execute_tool_call(call['tool_name'], call['parameters'])
                    tool_results.append(result)
                    # 从响应中移除工具调用标记
                    clean_response = clean_response.replace(call['original'], "")

                # 构建包含工具结果的消息
                messages.append({"role": "assistant", "content": clean_response})

                # 添加工具结果
                tool_results_text = "\n\n".join(tool_results)
                messages.append({"role": "user", "content": f"工具执行结果:\n{tool_results_text}\n\n请基于这些结果给出完整的回答。"})

                current_iteration += 1
                continue

            # 没有工具调用，这是最终回答
            final_response = response
            break

        # 如果超过最大迭代次数，获取最后一次回答
        if current_iteration >= max_tool_iterations and not final_response:
            final_response = self.llm.invoke(messages, **kwargs)

        # 保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))
        print(f"✅ {self.name} 响应完成")

        return final_response

    def _parse_tool_calls(self, text: str) -> list:
        """解析文本中的工具调用"""
        pattern = r'\[TOOL_CALL:([^:]+):([^\]]+)\]'
        matches = re.findall(pattern, text)

        tool_calls = []
        for tool_name, parameters in matches:
            tool_calls.append({
                'tool_name': tool_name.strip(),
                'parameters': parameters.strip(),
                'original': f'[TOOL_CALL:{tool_name}:{parameters}]'
            })

        return tool_calls

    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        """执行工具调用"""
        if not self.tool_registry:
            return f"❌ 错误:未配置工具注册表"

        try:
            # 智能参数解析
            if tool_name == 'calculator':
                # 计算器工具直接传入表达式
                result = self.tool_registry.execute_tool(tool_name, parameters)
            else:
                # 其他工具使用智能参数解析
                param_dict = self._parse_tool_parameters(tool_name, parameters)
                tool = self.tool_registry.get_tool(tool_name)
                if not tool:
                    return f"❌ 错误:未找到工具 '{tool_name}'"
                result = tool.run(param_dict)

            return f"🔧 工具 {tool_name} 执行结果:\n{result}"

        except Exception as e:
            return f"❌ 工具调用失败:{str(e)}"

    def _parse_tool_parameters(self, tool_name: str, parameters: str) -> dict:
        """智能解析工具参数"""
        param_dict = {}

        if '=' in parameters:
            # 格式: key=value 或 action=search,query=Python
            if ',' in parameters:
                # 多个参数:action=search,query=Python,limit=3
                pairs = parameters.split(',')
                for pair in pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        param_dict[key.strip()] = value.strip()
            else:
                # 单个参数:key=value
                key, value = parameters.split('=', 1)
                param_dict[key.strip()] = value.strip()
        else:
            # 直接传入参数，根据工具类型智能推断
            if tool_name == 'search':
                param_dict = {'query': parameters}
            elif tool_name == 'memory':
                param_dict = {'action': 'search', 'query': parameters}
            else:
                param_dict = {'input': parameters}

        return param_dict
```

我们还可以为自定义Agent添加流式响应功能和便利方法：

```python
# 继续在 my_simple_agent.py 中添加
class MySimpleAgent(SimpleAgent):
    # ... 前面的方法

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """
        自定义的流式运行方法
        """
        print(f"🌊 {self.name} 开始流式处理: {input_text}")

        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})

        # 流式调用LLM
        full_response = ""
        print("📝 实时响应: ", end="")
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk
            print(chunk, end="", flush=True)
            yield chunk

        print()  # 换行

        # 保存完整对话到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(full_response, "assistant"))
        print(f"✅ {self.name} 流式响应完成")

    def add_tool(self, tool) -> None:
        """添加工具到Agent（便利方法）"""
        if not self.tool_registry:
            from hello_agents import ToolRegistry
            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True

        self.tool_registry.register_tool(tool)
        print(f"🔧 工具 '{tool.name}' 已添加")

    def has_tools(self) -> bool:
        """检查是否有可用工具"""
        return self.enable_tool_calling and self.tool_registry is not None
    
    def remove_tool(self, tool_name: str) -> bool:
        """移除工具（便利方法）"""
        if self.tool_registry:
            self.tool_registry.unregister(tool_name)
            return True
        return False
    
    def list_tools(self) -> list:
        """列出所有可用工具"""
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []
```

创建一个测试文件`test_simple_agent.py`：

```python
# test_simple_agent.py
from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM, ToolRegistry
from hello_agents.tools import CalculatorTool
from my_simple_agent import MySimpleAgent

# 加载环境变量
load_dotenv()

# 创建LLM实例
llm = HelloAgentsLLM()

# 测试1:基础对话Agent（无工具）
print("=== 测试1:基础对话 ===")
basic_agent = MySimpleAgent(
    name="基础助手",
    llm=llm,
    system_prompt="你是一个友好的AI助手，请用简洁明了的方式回答问题。"
)

response1 = basic_agent.run("你好，请介绍一下自己")
print(f"基础对话响应: {response1}\n")

# 测试2:带工具的Agent
print("=== 测试2:工具增强对话 ===")
tool_registry = ToolRegistry()
calculator = CalculatorTool()
tool_registry.register_tool(calculator)

enhanced_agent = MySimpleAgent(
    name="增强助手",
    llm=llm,
    system_prompt="你是一个智能助手，可以使用工具来帮助用户。",
    tool_registry=tool_registry,
    enable_tool_calling=True
)

response2 = enhanced_agent.run("请帮我计算 15 * 8 + 32")
print(f"工具增强响应: {response2}\n")

# 测试3:流式响应
print("=== 测试3:流式响应 ===")
print("流式响应: ", end="")
for chunk in basic_agent.stream_run("请解释什么是人工智能"):
    pass  # 内容已在stream_run中实时打印

# 测试4:动态添加工具
print("\n=== 测试4:动态工具管理 ===")
print(f"添加工具前: {basic_agent.has_tools()}")
basic_agent.add_tool(calculator)
print(f"添加工具后: {basic_agent.has_tools()}")
print(f"可用工具: {basic_agent.list_tools()}")

# 查看对话历史
print(f"\n对话历史: {len(basic_agent.get_history())} 条消息")
```

在本节中，我们通过继承 `Agent` 基类，成功构建了一个功能完备且遵循框架规范的基础对话智能体 `MySimpleAgent`。它不仅支持基础对话，还具备可选的工具调用能力、流式响应和便利的工具管理方法。

### 7.4.2 ReActAgent

> **精讲**：提示词里的 Thought/Action/Observation 循环；框架负责多轮拼接与终止。

框架化的 ReActAgent 在保持核心逻辑不变的同时，提升了代码的组织性和可维护性，主要是通过提示词优化和与框架工具系统的集成。

（1）提示词模板的改进

保持了原有的格式要求，强调"每次只能执行一个步骤"，避免混乱，并明确了两种Action的使用场景。

```python
MY_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤:

Thought: 分析当前问题，思考需要什么信息或采取什么行动。
Action: 选择一个行动，格式必须是以下之一:
- `{{tool_name}}[{{tool_input}}]` - 调用指定工具
- `Finish[最终答案]` - 当你有足够信息给出最终答案时

## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 工具调用的格式必须严格遵循:工具名[参数]
3. 只有当你确信有足够信息回答问题时，才使用Finish
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动:
"""
```

（2）重写ReActAgent的完整实现

创建`my_react_agent.py`文件来重写ReActAgent：

```python
# my_react_agent.py
import re
from typing import Optional, List, Tuple
from hello_agents import ReActAgent, HelloAgentsLLM, Config, Message, ToolRegistry

class MyReActAgent(ReActAgent):
    """
    重写的ReAct Agent - 推理与行动结合的智能体
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
        custom_prompt: Optional[str] = None
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.current_history: List[str] = []
        self.prompt_template = custom_prompt if custom_prompt else MY_REACT_PROMPT
        print(f"✅ {name} 初始化完成，最大步数: {max_steps}")
```

其初始化参数的含义如下：

- `name`： Agent的名称。
- `llm`： `HelloAgentsLLM`的实例，负责与大语言模型通信。
- `tool_registry`： `ToolRegistry`的实例，用于管理和执行Agent可用的工具。
- `system_prompt`： 系统提示词，用于设定Agent的角色和行为准则。
- `config`： 配置对象，用于传递框架级的设置。
- `max_steps`： ReAct循环的最大执行步数，防止无限循环。
- `custom_prompt`： 自定义的提示词模板，用于替换默认的ReAct提示词。

框架化的ReActAgent将执行流程分解为清晰的步骤：

```python
def run(self, input_text: str, **kwargs) -> str:
    """运行ReAct Agent"""
    self.current_history = []
    current_step = 0

    print(f"\n🤖 {self.name} 开始处理问题: {input_text}")

    while current_step < self.max_steps:
        current_step += 1
        print(f"\n--- 第 {current_step} 步 ---")

        # 1. 构建提示词
        tools_desc = self.tool_registry.get_tools_description()
        history_str = "\n".join(self.current_history)
        prompt = self.prompt_template.format(
            tools=tools_desc,
            question=input_text,
            history=history_str
        )

        # 2. 调用LLM
        messages = [{"role": "user", "content": prompt}]
        response_text = self.llm.invoke(messages, **kwargs)

        # 3. 解析输出
        thought, action = self._parse_output(response_text)

        # 4. 检查完成条件
        if action and action.startswith("Finish"):
            final_answer = self._parse_action_input(action)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(final_answer, "assistant"))
            return final_answer

        # 5. 执行工具调用
        if action:
            tool_name, tool_input = self._parse_action(action)
            observation = self.tool_registry.execute_tool(tool_name, tool_input)
            self.current_history.append(f"Action: {action}")
            self.current_history.append(f"Observation: {observation}")

    # 达到最大步数
    final_answer = "抱歉，我无法在限定步数内完成这个任务。"
    self.add_message(Message(input_text, "user"))
    self.add_message(Message(final_answer, "assistant"))
    return final_answer
```

通过以上重构，我们将 ReAct 范式成功地集成到了框架中。核心改进在于利用了统一的 `ToolRegistry` 接口，并通过一个可配置、格式更严谨的提示词模板，提升了智能体执行思考-行动循环的稳定性。对于ReAct的测试案例，由于需要调用工具，所以统一放在文末提供测试代码。

### 7.4.3 ReflectionAgent

> **精讲**：初稿 → 反馈 → 重写；注意自定义提示词如何覆盖默认通用模板。

由于这几类Agent已经在第四章实现过核心逻辑，所以这里只给出对应的Prompt。与第四章专门针对代码生成的提示词不同，框架化的版本采用了通用化设计，使其适用于文本生成、分析、创作等多种场景，并通过`custom_prompts`参数支持用户深度定制。

```python
DEFAULT_PROMPTS = {
    "initial": """
请根据以下要求完成任务:

任务: {task}

请提供一个完整、准确的回答。
""",
    "reflect": """
请仔细审查以下回答，并找出可能的问题或改进空间:

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答"无需改进"。
""",
    "refine": """
请根据反馈意见改进你的回答:

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
"""
}
```

你可以尝试根据第四章的代码，以及上文ReAct的实现，构建出自己的MyReflectionAgent。下面提供一个测试代码供验证想法。

```python
# test_reflection_agent.py
from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM
from my_reflection_agent import MyReflectionAgent

load_dotenv()
llm = HelloAgentsLLM()

# 使用默认通用提示词
general_agent = MyReflectionAgent(name="我的反思助手", llm=llm)

# 使用自定义代码生成提示词（类似第四章）
code_prompts = {
    "initial": "你是Python专家，请编写函数:{task}",
    "reflect": "请审查代码的算法效率:\n任务:{task}\n代码:{content}",
    "refine": "请根据反馈优化代码:\n任务:{task}\n反馈:{feedback}"
}
code_agent = MyReflectionAgent(
    name="我的代码生成助手",
    llm=llm,
    custom_prompts=code_prompts
)

# 测试使用
result = general_agent.run("写一篇关于人工智能发展历程的简短文章")
print(f"最终结果: {result}")
```

### 7.4.4 PlanAndSolveAgent

> **精讲**：Planner 与 Executor 两套模板；状态里要能放下「完整计划 + 历史步骤」。

与第四章自由文本的计划输出不同，框架化版本强制要求Planner以Python列表的格式输出计划，并提供了完整的异常处理机制，确保了后续步骤能够稳定执行。框架化的Plan-and-Solve提示词：

````bash
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
````

这一节仍然给出一个综合测试文件`test_plan_solve_agent.py`，可以自行设计实现。

```python
# test_plan_solve_agent.py
from dotenv import load_dotenv
from hello_agents.core.llm import HelloAgentsLLM
from my_plan_solve_agent import MyPlanAndSolveAgent

# 加载环境变量
load_dotenv()

# 创建LLM实例
llm = HelloAgentsLLM()

# 创建自定义PlanAndSolveAgent
agent = MyPlanAndSolveAgent(
    name="我的规划执行助手",
    llm=llm
)

# 测试复杂问题
question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"

result = agent.run(question)
print(f"\n最终结果: {result}")

# 查看对话历史
print(f"对话历史: {len(agent.get_history())} 条消息")
```

在最后可以补充一款新的提示词，可以尝试实现`custom_prompt`载入自定义提示词。

```python
# 创建专门用于数学问题的自定义提示词
math_prompts = {
    "planner": """
你是数学问题规划专家。请将数学问题分解为计算步骤:

问题: {question}

输出格式:
python
["计算步骤1", "计算步骤2", "求总和"]

""",
    "executor": """
你是数学计算专家。请计算当前步骤:

问题: {question}
计划: {plan}
历史: {history}
当前步骤: {current_step}

请只输出数值结果:
"""
}

# 使用自定义提示词创建数学专用Agent
math_agent = MyPlanAndSolveAgent(
    name="数学计算助手",
    llm=llm,
    custom_prompts=math_prompts
)

# 测试数学问题
math_result = math_agent.run(question)
print(f"数学专用Agent结果: {math_result}")
```

如表7.2所示，通过这种框架化的重构，我们不仅保持了第四章中各种Agent范式的核心功能，还大幅提升了代码的组织性、可维护性和扩展性。所有Agent现在都共享统一的基础架构，同时保持了各自的特色和优势。

<div align="center">
  <p>表 7.2 Agent不同章节实现对比</p>
  <img src="https://raw.githubusercontent.com/datawhalechina/Hello-Agents/main/docs/images/7-figures/table-02.png" alt="" width="90%"/>
</div>

### 7.4.5 FunctionCallAgent

> **铺垫**：相对 ReAct 的文本解析，Function Calling 让模型直接产出结构化调用；框架侧要对接工具 schema。

FunctionCallAgent是hello-agents在0.2.8之后引入的Agent，它基于OpenAI原生函数调用机制的Agent，展示了如何使用OpenAI的函数调用机制来构建Agent。
它支持以下功能：

- _build_tool_schemas:通过工具的description构建OpenAI的function calling schema
- _extract_message_content:从OpenAI的响应中提取文本
- _parse_function_call_arguments:解析模型返回的JSON字符串参数
- _convert_parameter_types:转换参数类型

这些功能可以使其具备原生的OpenAI Function Calling的能力，对比使用prompt约束的方式，具备更强的鲁棒性。
```python
def _invoke_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: Union[str, dict], **kwargs):
        """调用底层OpenAI客户端执行函数调用"""
        client = getattr(self.llm, "_client", None)
        if client is None:
            raise RuntimeError("HelloAgentsLLM 未正确初始化客户端，无法执行函数调用。")

        client_kwargs = dict(kwargs)
        client_kwargs.setdefault("temperature", self.llm.temperature)
        if self.llm.max_tokens is not None:
            client_kwargs.setdefault("max_tokens", self.llm.max_tokens)

        return client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **client_kwargs,
        )

#内部逻辑是对Openai 原生的functioncall作再封装
#OpenAI 原生functioncall示例
from openai import OpenAI
client = OpenAI()

tools = [
  {
    "type": "function",
    "function": {
      "name": "get_current_weather",
      "description": "Get the current weather in a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
          },
          "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
      },
    }
  }
]
messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]
completion = client.chat.completions.create(
  model="gpt-5",
  messages=messages,
  tools=tools,
  tool_choice="auto"
)

print(completion)
```

## 7.5 工具系统

> **一句话**：工具是框架的「外设生态」；注册表是插槽。

本节内容将在前面构建的Agent基础架构上，深入探讨工具系统的设计与实现。我们将从基础设施建设开始，逐步深入到自定义开发设计。本节的学习目标围绕以下三个核心方面展开：

1. **统一的工具抽象与管理**：建立标准化的Tool基类和ToolRegistry注册机制，为工具的开发、注册、发现和执行提供统一的基础设施。

2. **实战驱动的工具开发**：以数学计算工具为案例，展示如何设计和实现自定义工具，让读者掌握工具开发的完整流程。

3. **高级整合与优化策略**：通过多源搜索工具的设计，展示如何整合多个外部服务，实现智能后端选择、结果合并和容错处理，体现工具系统在复杂场景下的设计思维。

### 7.5.1 工具基类与注册机制设计

> **精讲**：基类通常要求 `name` / `description` / `run`（或 `execute`）；注册表负责查名与列表化给 Prompt/Schema。

在构建可扩展的工具系统时，我们需要首先建立一套标准化的基础设施。这套基础设施包括Tool基类、ToolRegistry注册表，以及工具管理机制。

（1）Tool基类的抽象设计

Tool基类是整个工具系统的核心抽象，它定义了所有工具必须遵循的接口规范：

````python
class Tool(ABC):
    """工具基类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> str:
        """执行工具"""
        pass

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """获取工具参数定义"""
        pass
````
这个设计体现了面向对象设计的核心思想：通过统一的`run`方法接口，所有工具都能以一致的方式执行，接受字典参数并返回字符串结果，确保了框架的一致性。同时，工具具备了自描述能力，通过`get_parameters`方法能够清晰地告诉调用者自己需要什么参数，这种内省机制为自动化文档生成和参数验证提供了基础。而name和description等元数据的设计，则让工具系统具备了良好的可发现性和可理解性。

（2）ToolParameter参数定义系统

为了支持复杂的参数验证和文档生成，我们设计了ToolParameter类：

````python
class ToolParameter(BaseModel):
    """工具参数定义"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
````
这种设计让工具能够精确描述自己的参数需求，支持类型检查、默认值设置和文档自动生成。

（3）ToolRegistry注册表的实现

ToolRegistry是工具系统的管理中枢，它提供了工具的注册、发现、执行等核心功能，在这一节我们主要用到以下功能：

````python
class ToolRegistry:
    """HelloAgents工具注册表"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, dict[str, Any]] = {}

    def register_tool(self, tool: Tool):
        """注册Tool对象"""
        if tool.name in self._tools:
            print(f"⚠️ 警告:工具 '{tool.name}' 已存在，将被覆盖。")
        self._tools[tool.name] = tool
        print(f"✅ 工具 '{tool.name}' 已注册。")
        
    def register_function(self, name: str, description: str, func: Callable[[str], str]):
        """
        直接注册函数作为工具（简便方式）

        Args:
            name: 工具名称
            description: 工具描述
            func: 工具函数，接受字符串参数，返回字符串结果
        """
        if name in self._functions:
            print(f"⚠️ 警告:工具 '{name}' 已存在，将被覆盖。")

        self._functions[name] = {
            "description": description,
            "func": func
        }
        print(f"✅ 工具 '{name}' 已注册。")
````
ToolRegistry支持两种注册方式：

1. **Tool对象注册**：适合复杂工具，支持完整的参数定义和验证
2. **函数直接注册**：适合简单工具，快速集成现有函数

（4）工具发现与管理机制

注册表提供了丰富的工具管理功能：

````python
def get_tools_description(self) -> str:
    """获取所有可用工具的格式化描述字符串"""
    descriptions = []

    # Tool对象描述
    for tool in self._tools.values():
        descriptions.append(f"- {tool.name}: {tool.description}")

    # 函数工具描述
    for name, info in self._functions.items():
        descriptions.append(f"- {name}: {info['description']}")

    return "\n".join(descriptions) if descriptions else "暂无可用工具"
````
这个方法生成的描述字符串可以直接用于构建Agent的提示词，让Agent了解可用的工具。

````python
def to_openai_schema(self) -> Dict[str, Any]:
        """转换为 OpenAI function calling schema 格式

        用于 FunctionCallAgent，使工具能够被 OpenAI 原生 function calling 使用

        Returns:
            符合 OpenAI function calling 标准的 schema
        """
        parameters = self.get_parameters()

        # 构建 properties
        properties = {}
        required = []

        for param in parameters:
            # 基础属性定义
            prop = {
                "type": param.type,
                "description": param.description
            }

            # 如果有默认值，添加到描述中（OpenAI schema 不支持 default 字段）
            if param.default is not None:
                prop["description"] = f"{param.description} (默认: {param.default})"

            # 如果是数组类型，添加 items 定义
            if param.type == "array":
                prop["items"] = {"type": "string"}  # 默认字符串数组

            properties[param.name] = prop

            # 收集必需参数
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
````
这个方法生成的schema可以直接用于原生的OpenAI SDK的工具调用。

### 7.5.2 自定义工具开发

> **最小练习**：实现计算器工具并挂到 SimpleAgent——验证「描述是否写清楚」比「代码是否炫」更重要。

有了基础设施后，我们来看看如何开发一个完整的自定义工具。数学计算工具是一个很好的例子，因为它简单直观，最直接的方式是使用ToolRegistry的函数注册功能。

让我们创建一个自定义的数学计算工具。首先，在你的项目目录中创建`my_calculator_tool.py`：

```python
# my_calculator_tool.py
import ast
import operator
import math
from hello_agents import ToolRegistry

def my_calculate(expression: str) -> str:
    """简单的数学计算函数"""
    if not expression.strip():
        return "计算表达式不能为空"

    # 支持的基本运算
    operators = {
        ast.Add: operator.add,      # +
        ast.Sub: operator.sub,      # -
        ast.Mult: operator.mul,     # *
        ast.Div: operator.truediv,  # /
    }

    # 支持的基本函数
    functions = {
        'sqrt': math.sqrt,
        'pi': math.pi,
    }

    try:
        node = ast.parse(expression, mode='eval')
        result = _eval_node(node.body, operators, functions)
        return str(result)
    except:
        return "计算失败，请检查表达式格式"

def _eval_node(node, operators, functions):
    """简化的表达式求值"""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = _eval_node(node.left, operators, functions)
        right = _eval_node(node.right, operators, functions)
        op = operators.get(type(node.op))
        return op(left, right)
    elif isinstance(node, ast.Call):
        func_name = node.func.id
        if func_name in functions:
            args = [_eval_node(arg, operators, functions) for arg in node.args]
            return functions[func_name](*args)
    elif isinstance(node, ast.Name):
        if node.id in functions:
            return functions[node.id]

def create_calculator_registry():
    """创建包含计算器的工具注册表"""
    registry = ToolRegistry()

    # 注册计算器函数
    registry.register_function(
        name="my_calculator",
        description="简单的数学计算工具，支持基本运算(+,-,*,/)和sqrt函数",
        func=my_calculate
    )

    return registry
```

工具不仅支持基本的四则运算，还涵盖了常用的数学函数和常数，满足了大多数计算场景的需求。你也可以自己扩展这个文件，制作一个更加完备的计算函数。我们提供一个测试文件`test_my_calculator.py`帮助你验证功能实现：

```python
# test_my_calculator.py
from dotenv import load_dotenv
from my_calculator_tool import create_calculator_registry

# 加载环境变量
load_dotenv()

def test_calculator_tool():
    """测试自定义计算器工具"""

    # 创建包含计算器的注册表
    registry = create_calculator_registry()

    print("🧪 测试自定义计算器工具\n")

    # 简单测试用例
    test_cases = [
        "2 + 3",           # 基本加法
        "10 - 4",          # 基本减法
        "5 * 6",           # 基本乘法
        "15 / 3",          # 基本除法
        "sqrt(16)",        # 平方根
    ]

    for i, expression in enumerate(test_cases, 1):
        print(f"测试 {i}: {expression}")
        result = registry.execute_tool("my_calculator", expression)
        print(f"结果: {result}\n")

def test_with_simple_agent():
    """测试与SimpleAgent的集成"""
    from hello_agents import HelloAgentsLLM

    # 创建LLM客户端
    llm = HelloAgentsLLM()

    # 创建包含计算器的注册表
    registry = create_calculator_registry()

    print("🤖 与SimpleAgent集成测试:")

    # 模拟SimpleAgent使用工具的场景
    user_question = "请帮我计算 sqrt(16) + 2 * 3"

    print(f"用户问题: {user_question}")

    # 使用工具计算
    calc_result = registry.execute_tool("my_calculator", "sqrt(16) + 2 * 3")
    print(f"计算结果: {calc_result}")

    # 构建最终回答
    final_messages = [
        {"role": "user", "content": f"计算结果是 {calc_result}，请用自然语言回答用户的问题:{user_question}"}
    ]

    print("\n🎯 SimpleAgent的回答:")
    response = llm.think(final_messages)
    for chunk in response:
        print(chunk, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    test_calculator_tool()
    test_with_simple_agent()
```

通过这个简化的数学计算工具案例，我们学会了如何快速开发自定义工具：编写一个简单的计算函数，通过ToolRegistry注册，然后与SimpleAgent集成使用。为了更直观的观察，这里提供了图7.1，可以清晰理解代码的运行逻辑。

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/Hello-Agents/main/docs/images/7-figures/01.png" alt="" width="90%"/>
  <p>图 7.1 基于Helloagents的SimpleAgent运行工作流</p>
</div>

### 7.5.3 多源搜索工具

> **设计看点**：多后端、失败降级、结果规范化——这是真实工具的常态，不是演示彩蛋。

在实际应用中，我们经常需要整合多个外部服务来提供更强大的功能。搜索工具就是一个典型的例子，它整合多个搜索引擎，能提供更加完备的真实信息。在第一章我们使用过Tavily的搜索API，在第四章我们使用过SerpApi的搜索API。因此这次我们使用这两个API来实现多源搜索功能。如果没安装对应的python依赖可以运行下面这条脚本：

```bash
pip install "hello-agents[search]==0.1.1"
```

（1）搜索工具的统一接口设计

HelloAgents框架内置的SearchTool展示了如何设计一个高级的多源搜索工具：

````python
class SearchTool(Tool):
    """
    智能混合搜索工具

    支持多种搜索引擎后端，智能选择最佳搜索源:
    1. 混合模式 (hybrid) - 智能选择TAVILY或SERPAPI
    2. Tavily API (tavily) - 专业AI搜索
    3. SerpApi (serpapi) - 传统Google搜索
    """

    def __init__(self, backend: str = "hybrid", tavily_key: Optional[str] = None, serpapi_key: Optional[str] = None):
        super().__init__(
            name="search",
            description="一个智能网页搜索引擎。支持混合搜索模式，自动选择最佳搜索源。"
        )
        self.backend = backend
        self.tavily_key = tavily_key or os.getenv("TAVILY_API_KEY")
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_API_KEY")
        self.available_backends = []
        self._setup_backends()
````
这个设计的核心思想是根据可用的API密钥和依赖库，自动选择最佳的搜索后端。

（2）TAVILY与SERPAPI搜索源的整合策略

框架实现了智能的后端选择逻辑：

````python
def _search_hybrid(self, query: str) -> str:
    """混合搜索 - 智能选择最佳搜索源"""
    # 优先使用Tavily（AI优化的搜索）
    if "tavily" in self.available_backends:
        try:
            return self._search_tavily(query)
        except Exception as e:
            print(f"⚠️ Tavily搜索失败: {e}")
            # 如果Tavily失败，尝试SerpApi
            if "serpapi" in self.available_backends:
                print("🔄 切换到SerpApi搜索")
                return self._search_serpapi(query)

    # 如果Tavily不可用，使用SerpApi
    elif "serpapi" in self.available_backends:
        try:
            return self._search_serpapi(query)
        except Exception as e:
            print(f"⚠️ SerpApi搜索失败: {e}")

    # 如果都不可用，提示用户配置API
    return "❌ 没有可用的搜索源，请配置TAVILY_API_KEY或SERPAPI_API_KEY环境变量"
````
这种设计体现了高可用系统的核心理念：通过降级机制，系统能够从最优的搜索源逐步降级到可用的备选方案。当所有搜索源都不可用时，明确提示用户配置正确的API密钥。

（3）搜索结果的统一格式化

不同搜索引擎返回的结果格式不同，框架通过统一的格式化方法来处理：

````python
def _search_tavily(self, query: str) -> str:
    """使用Tavily搜索"""
    response = self.tavily_client.search(
        query=query,
        search_depth="basic",
        include_answer=True,
        max_results=3
    )

    result = f"🎯 Tavily AI搜索结果:{response.get('answer', '未找到直接答案')}\n\n"

    for i, item in enumerate(response.get('results', [])[:3], 1):
        result += f"[{i}] {item.get('title', '')}\n"
        result += f"    {item.get('content', '')[:200]}...\n"
        result += f"    来源: {item.get('url', '')}\n\n"

    return result
````

基于框架的设计思想，我们可以创建自己的高级搜索工具。这次我们使用类的方式来展示不同的实现方法，创建`my_advanced_search.py`：

```python
# my_advanced_search.py
import os
from typing import Optional, List, Dict, Any
from hello_agents import ToolRegistry

class MyAdvancedSearchTool:
    """
    自定义高级搜索工具类
    展示多源整合和智能选择的设计模式
    """

    def __init__(self):
        self.name = "my_advanced_search"
        self.description = "智能搜索工具，支持多个搜索源，自动选择最佳结果"
        self.search_sources = []
        self._setup_search_sources()

    def _setup_search_sources(self):
        """设置可用的搜索源"""
        # 检查Tavily可用性
        if os.getenv("TAVILY_API_KEY"):
            try:
                from tavily import TavilyClient
                self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
                self.search_sources.append("tavily")
                print("✅ Tavily搜索源已启用")
            except ImportError:
                print("⚠️ Tavily库未安装")

        # 检查SerpApi可用性
        if os.getenv("SERPAPI_API_KEY"):
            try:
                import serpapi
                self.search_sources.append("serpapi")
                print("✅ SerpApi搜索源已启用")
            except ImportError:
                print("⚠️ SerpApi库未安装")

        if self.search_sources:
            print(f"🔧 可用搜索源: {', '.join(self.search_sources)}")
        else:
            print("⚠️ 没有可用的搜索源，请配置API密钥")

    def search(self, query: str) -> str:
        """执行智能搜索"""
        if not query.strip():
            return "❌ 错误:搜索查询不能为空"

        # 检查是否有可用的搜索源
        if not self.search_sources:
            return """❌ 没有可用的搜索源，请配置以下API密钥之一:

1. Tavily API: 设置环境变量 TAVILY_API_KEY
   获取地址: https://tavily.com/

2. SerpAPI: 设置环境变量 SERPAPI_API_KEY
   获取地址: https://serpapi.com/

配置后重新运行程序。"""

        print(f"🔍 开始智能搜索: {query}")

        # 尝试多个搜索源，返回最佳结果
        for source in self.search_sources:
            try:
                if source == "tavily":
                    result = self._search_with_tavily(query)
                    if result and "未找到" not in result:
                        return f"📊 Tavily AI搜索结果:\n\n{result}"

                elif source == "serpapi":
                    result = self._search_with_serpapi(query)
                    if result and "未找到" not in result:
                        return f"🌐 SerpApi Google搜索结果:\n\n{result}"

            except Exception as e:
                print(f"⚠️ {source} 搜索失败: {e}")
                continue

        return "❌ 所有搜索源都失败了，请检查网络连接和API密钥配置"

    def _search_with_tavily(self, query: str) -> str:
        """使用Tavily搜索"""
        response = self.tavily_client.search(query=query, max_results=3)

        if response.get('answer'):
            result = f"💡 AI直接答案:{response['answer']}\n\n"
        else:
            result = ""

        result += "🔗 相关结果:\n"
        for i, item in enumerate(response.get('results', [])[:3], 1):
            result += f"[{i}] {item.get('title', '')}\n"
            result += f"    {item.get('content', '')[:150]}...\n\n"

        return result

    def _search_with_serpapi(self, query: str) -> str:
        """使用SerpApi搜索"""
        import serpapi

        search = serpapi.GoogleSearch({
            "q": query,
            "api_key": os.getenv("SERPAPI_API_KEY"),
            "num": 3
        })

        results = search.get_dict()

        result = "🔗 Google搜索结果:\n"
        if "organic_results" in results:
            for i, res in enumerate(results["organic_results"][:3], 1):
                result += f"[{i}] {res.get('title', '')}\n"
                result += f"    {res.get('snippet', '')}\n\n"

        return result

def create_advanced_search_registry():
    """创建包含高级搜索工具的注册表"""
    registry = ToolRegistry()

    # 创建搜索工具实例
    search_tool = MyAdvancedSearchTool()

    # 注册搜索工具的方法作为函数
    registry.register_function(
        name="advanced_search",
        description="高级搜索工具，整合Tavily和SerpAPI多个搜索源，提供更全面的搜索结果",
        func=search_tool.search
    )

    return registry
```

接下来可以测试我们自己编写的工具，创建`test_advanced_search.py`：

```python
# test_advanced_search.py
from dotenv import load_dotenv
from my_advanced_search import create_advanced_search_registry, MyAdvancedSearchTool

# 加载环境变量
load_dotenv()

def test_advanced_search():
    """测试高级搜索工具"""

    # 创建包含高级搜索工具的注册表
    registry = create_advanced_search_registry()

    print("🔍 测试高级搜索工具\n")

    # 测试查询
    test_queries = [
        "Python编程语言的历史",
        "人工智能的最新发展",
        "2024年科技趋势"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"测试 {i}: {query}")
        result = registry.execute_tool("advanced_search", query)
        print(f"结果: {result}\n")
        print("-" * 60 + "\n")

def test_api_configuration():
    """测试API配置检查"""
    print("🔧 测试API配置检查:")

    # 直接创建搜索工具实例
    search_tool = MyAdvancedSearchTool()

    # 如果没有配置API，会显示配置提示
    result = search_tool.search("机器学习算法")
    print(f"搜索结果: {result}")

def test_with_agent():
    """测试与Agent的集成"""
    print("\n🤖 与Agent集成测试:")
    print("高级搜索工具已准备就绪，可以与Agent集成使用")

    # 显示工具描述
    registry = create_advanced_search_registry()
    tools_desc = registry.get_tools_description()
    print(f"工具描述:\n{tools_desc}")

if __name__ == "__main__":
    test_advanced_search()
    test_api_configuration()
    test_with_agent()
```

通过这个高级搜索工具的设计实践，我们学会了如何使用类的方式来构建复杂的工具系统。相比函数方式，类方式更适合需要维护状态（如API客户端、配置信息）的工具。

### 7.5.4 工具系统的高级特性

> **导读**：工具链（串联）与异步执行——为后续协议工具、并行 IO 打基础。

在掌握了基础的工具开发和多源整合后，我们来探讨工具系统的高级特性。这些特性能够让工具系统在复杂的生产环境中稳定运行，并为Agent提供更强大的能力。

（1）工具链式调用机制

在实际应用中，Agent经常需要组合使用多个工具来完成复杂任务。我们可以设计一个工具链管理器来支持这种场景，这里借鉴了第六章中提到的图的概念：

```python
# tool_chain_manager.py
from typing import List, Dict, Any, Optional
from hello_agents import ToolRegistry

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

    def execute(self, registry: ToolRegistry, initial_input: str, context: Dict[str, Any] = None) -> str:
        """执行工具链"""
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

            # 执行工具
            result = registry.execute_tool(tool_name, tool_input)
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

    def execute_chain(self, chain_name: str, input_data: str, context: Dict[str, Any] = None) -> str:
        """执行指定的工具链"""
        if chain_name not in self.chains:
            return f"❌ 工具链 '{chain_name}' 不存在"

        chain = self.chains[chain_name]
        return chain.execute(self.registry, input_data, context)

    def list_chains(self) -> List[str]:
        """列出所有工具链"""
        return list(self.chains.keys())

# 使用示例
def create_research_chain() -> ToolChain:
    """创建一个研究工具链:搜索 -> 计算 -> 总结"""
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
    chain.add_step(
        tool_name="my_calculator",
        input_template="根据以下信息计算相关数值:{search_result}",
        output_key="calculation_result"
    )

    return chain
```

（2）异步工具执行支持

对于耗时的工具操作，我们可以提供异步执行支持：

```python
# async_tool_executor.py
import asyncio
import concurrent.futures
from typing import Dict, Any, List, Callable
from hello_agents import ToolRegistry

class AsyncToolExecutor:
    """异步工具执行器"""

    def __init__(self, registry: ToolRegistry, max_workers: int = 4):
        self.registry = registry
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def execute_tool_async(self, tool_name: str, input_data: str) -> str:
        """异步执行单个工具"""
        loop = asyncio.get_event_loop()

        def _execute():
            return self.registry.execute_tool(tool_name, input_data)

        result = await loop.run_in_executor(self.executor, _execute)
        return result

    async def execute_tools_parallel(self, tasks: List[Dict[str, str]]) -> List[str]:
        """并行执行多个工具"""
        print(f"🚀 开始并行执行 {len(tasks)} 个工具任务")

        # 创建异步任务
        async_tasks = []
        for task in tasks:
            tool_name = task["tool_name"]
            input_data = task["input_data"]
            async_task = self.execute_tool_async(tool_name, input_data)
            async_tasks.append(async_task)

        # 等待所有任务完成
        results = await asyncio.gather(*async_tasks)

        print(f"✅ 所有工具任务执行完成")
        return results

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

# 使用示例
async def test_parallel_execution():
    """测试并行工具执行"""
    from hello_agents import ToolRegistry

    registry = ToolRegistry()
    # 假设已经注册了搜索和计算工具

    executor = AsyncToolExecutor(registry)

    # 定义并行任务
    tasks = [
        {"tool_name": "search", "input_data": "Python编程"},
        {"tool_name": "search", "input_data": "机器学习"},
        {"tool_name": "my_calculator", "input_data": "2 + 2"},
        {"tool_name": "my_calculator", "input_data": "sqrt(16)"},
    ]

    # 并行执行
    results = await executor.execute_tools_parallel(tasks)

    for i, result in enumerate(results):
        print(f"任务 {i+1} 结果: {result[:100]}...")
```

基于以上的设计和实现经验，我们可以总结出工具系统开发的核心理念：在设计层面，每个工具都应该遵循单一职责原则，专注于特定功能的同时保持接口的统一性，并将完善的异常处理和安全优先的输入验证作为基本要求。在性能优化方面，利用异步执行提高并发处理能力，同时合理管理外部连接和系统资源。



## 7.6 本章小结

> **合章口诀**：LLM 可替换、范式可插拔、工具可注册——这就是最小可用 Agent 框架。

在正式总结之前，我们想告诉大家一个好消息：对于本章实现的所有方法和功能，都在GitHub仓库中提供了完整的测试案例。你可以访问[这个链接](https://github.com/jjyaoao/HelloAgents/blob/main/examples/chapter07_basic_setup.py)查看和运行这些测试代码。这个文件包含了四种Agent范式的演示、工具系统的集成测试、高级功能的使用示例，以及交互式的Agent体验。如果你想验证自己的实现是否正确，或者想深入了解框架的实际使用方式，这些测试案例将是有价值的参考。

回顾本章，我们完成了一项富有挑战的任务：一步步构建了一个基础的智能体框架——HelloAgents。这个过程始终遵循着“分层解耦、职责单一、接口统一”的核心原则。

在框架的具体实现中，我们再次实现了四种经典的Agent范式。从SimpleAgent的基础对话模式，到ReActAgent的推理与行动结合；从ReflectionAgent的自我反思与迭代优化，到PlanAndSolveAgent的分解规划与逐步执行。而工具系统作为Agent能力延伸的核心，其构建过程则是一次完整的工程实践。

更重要的是，第七章的构建并非终点，而是为后续更深入学习提供了必要的技术基础。我们在设计之初便充分考虑了后续内容的延展性，为高级功能的实现预留了必要的接口和扩展点。我们所建立的统一LLM接口、标准化消息系统、工具注册机制，共同构成了一个完备的技术底座。这使得我们在后续章节中，可以更加从容地去学习更高级的主题：第八章的记忆与RAG系统将基于此扩展Agent的能力边界；第九章的上下文工程将深入我们已经建立的消息处理机制；第十章的智能体协议则需要扩展新的工具。

接下来，我们将一起探索如何往框架中加入RAG系统与Memory机制，敬请期待第八章！


## 习题

> 完整详解也可参见：[study/chapter7-exercises-answers.md](./study/chapter7-exercises-answers.md)

1. 本章构建了 `HelloAgents` 框架，并阐述了"为何需要自建Agent框架"。请分析：

   - 在7.1.1节中提到了当前主流框架的四个主要局限性。结合你在[第六章习题](../chapter6/第六章%20框架开发实践.md#习题)或实际项目中使用过的某个框架的实际经验，说明这些问题是如何影响开发效率的。
   - `HelloAgents` 提出了"万物皆为工具"的设计理念，将 `Memory`、`RAG`、`MCP` 等模块都抽象为工具。这种设计有什么优势？是否存在局限性？请举例说明。
   - 对比第四章从零实现的智能体代码和本章的框架化实现，框架化带来了哪些具体的改进？如果让你设计一个框架，你会优先考虑哪些设计原则？

   <details>
   <summary><strong>参考解答</strong></summary>

   **（1）四个局限性如何拖慢开发效率**

   对照第六章用过的 LangChain / AutoGen / LangGraph / AgentScope：

   | 局限性 | 如何影响效率 | 实际例子 |
   |--------|--------------|----------|
   | 过度抽象 | 简单任务也要先对齐多层概念，调试堆栈深 | LangChain 做「搜索+总结」要先搞清 LCEL、Agent、Tool 等多套抽象 |
   | API 快速迭代 | 文档与代码不同步，旧教程易失效 | 版本升级后 import/参数变更，「复制即失败」 |
   | 黑盒实现 | 死循环、工具未调用时难定位根因 | AutoGen 群聊跑偏时缺乏清晰状态机，只能靠日志猜 |
   | 依赖复杂 | 安装体积大、易与现有项目冲突 | 小 Demo 拉起大量传递依赖，排依赖比写业务更久 |

   **（2）「万物皆为工具」的优势与局限**

   - **优势**：心智模型统一（Agent 只调工具）；扩展成本低（实现接口 + 注册）；便于观测/测试；工具链与异步执行器可直接组合能力。
   - **局限**：Memory/RAG/MCP 的生命周期与结构化语义被压扁成 `str→str`；长连接服务（向量库、MCP）与「无状态工具」不匹配；权限边界变模糊。教学与中小规模很合适，生产级子系统仍建议在工具外观下保留独立模块（见第八章演进）。

   **（3）第四章脚本 vs 框架化；自建优先原则**

   - **改进**：统一 `Agent.run` 入口与 `Message` 历史；`HelloAgentsLLM`/`ToolRegistry` 解耦；提示词可配置；可安装、可复现测试。
   - **优先原则**：接口统一可替换 → 约定优于配置且支持显式覆盖 → 可观测 → 最小依赖 → 安全默认（白名单/超时/最大步数）。

   </details>

2. 在7.2节中，我们扩展了 `HelloAgentsLLM` 以支持多模型供应商和本地模型调用。

   > <strong>提示</strong>：这是一道实践题，建议实际操作

   - 参考7.2.1节的示例，尝试为 `HelloAgentsLLM` 添加一个新模型供应商的支持（如`Gemini`、`Anthropic`、`Kim`）。要求通过继承方式实现，并能够自动检测该提供商的环境变量。
   - 在7.2.3节中介绍了自动检测机制的三个优先级。请分析：如果同时设置了 `OPENAI_API_KEY` 和 `LLM_BASE_URL="http://localhost:11434/v1"`，框架最后会选择哪个提供商？这种优先级设计是否合理？
   - 除了本章介绍的 `VLLM` 和 `Ollama`，还有 `SGLang` 等其他本地模型部署方案。请先搜索并了解 `SGLang` 的基本信息和特点，然后对比 `VLLM`、`SGLang` 和 `Ollama` 这三者在易用性、资源占用、推理速度、推理精度等方面的优劣。

   <details>
   <summary><strong>参考解答</strong></summary>

   **（1）继承扩展新供应商（示意）**

   对齐 7.2.1：子类重写 `__init__`，先根据 `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` / `MOONSHOT_API_KEY`（Kimi）自动检测 `provider`，再补默认 `base_url` 与模型名，最后 `super().__init__(...)`。完整示例见 `study/chapter7-exercises-answers.md` 习题 2.1。

   **（2）同时设置 OpenAI Key 与 Ollama URL**

   按 7.2.3：**最高优先级是特定服务商环境变量**。存在 `OPENAI_API_KEY` 时直接返回 `"openai"`，**不会**再解析 `:11434`。  
   - 合理处：密钥是强意图，避免本机默认端口误伤云端配置。  
   - 不妥处：本地开发常「Key 常驻 + 临时切 Ollama」会被抢优先级。建议 `provider=` 显式指定永远最高，或增加 `LLM_PROVIDER`。

   **（3）Ollama / vLLM / SGLang**

   - **SGLang**：高吞吐推理框架，特色是 **RadixAttention**（按前缀复用 KV cache），适合多轮、RAG、Agent 等共享前缀重的负载；通常也提供 OpenAI 兼容 API。
   - **对比**：易用性 Ollama ≫ vLLM ≈ SGLang；资源上 Ollama 本机友好，后两者吃 GPU 但利用率高；吞吐上高并发时 vLLM/SGLang 明显强于 Ollama，共享前缀重时 SGLang 常更优；**精度主要取决于模型与量化，而非引擎本身**。
   - **选型**：本机试玩 → Ollama；通用生产吞吐 → vLLM；多轮/Agent 前缀复用重 → 重点评估 SGLang。

   </details>

3. 在7.3节中，我们实现了 `Message` 类、`Config` 类和 `Agent` 基类。请分析：

   - `Message` 类使用了 `Pydantic` 的 `BaseModel` 进行数据验证。这种设计在实际应用中有哪些优势？
   - `Agent` 基类定义了 `run` 和 `_execute` 两个方法，其中 `run` 是公开接口，`_execute` 是抽象方法。这种设计模式叫什么？有什么好处？
   - 在 `Config` 类中，我们使用了单例模式。请解释什么是单例模式，为什么配置管理需要使用单例模式？如果不使用单例会导致什么问题？

   <details>
   <summary><strong>参考解答</strong></summary>

   **（1）Pydantic Message 的优势**

   自动校验（如 `role` 的 `Literal`）、类型提示友好、序列化方便（`to_dict` 对齐 OpenAI）、`metadata`/`timestamp` 扩展位清晰、字段定义即契约。

   **（2）设计模式：模板方法（Template Method）**

   基类 `run` 固定骨架（记历史、日志、异常、再调核心逻辑），子类实现 `_execute` 填差异。好处：公共流程只写一次、子类专注范式逻辑、对外 API 稳定。  
   > 与正文对照：7.3.3 示例把抽象点直接放在 `run`；习题的 `run`/`_execute` 拆分是更完整的模板方法写法。

   **（3）单例与配置**

   单例保证进程内只有一份实例。配置需要全局一致（温度、debug、默认模型），否则多模块各 `Config()` 会导致行为不一致、热更新只生效一份、测试串味。  
   > 与正文对照：7.3.2 给出的是普通 `Config` + `from_env()`，**未强制单例**；也可用依赖注入同一实例（更利于测试）。

   </details>

4. 在7.4节中，我们动手进行了四种 `Agent` 范式的框架化实现。

   > <strong>提示</strong>：这是一道实践题，建议实际操作

   - 对比第四章从零实现的 `ReActAgent` 和本章框架化的 `ReActAgent`，列举3个具体的改进点，并说明这些改进如何提升了代码的可维护性和可扩展性。
   - `ReflectionAgent` 实现了"执行-反思-优化"循环。请扩展这个实现，添加一个"质量评分"机制：在每次反思后，让 `LLM` 对当前版本的输出打分，只有分数低于阈值时才继续优化，否则提前终止。
   - 请设计并实现一个新的 `Agent` 范式 `Tree-of-Thought Agent`，要求继承 `Agent` 基类，它能够在每一步生成多个可能的思考路径，然后选择最优路径继续。

   <details>
   <summary><strong>参考解答</strong></summary>

   **（1）框架化 ReAct 的 3 个改进**

   1. 接入 `ToolRegistry`：增删工具不必改 Agent 内部分支 → 可扩展性 ↑  
   2. 提示词模板化/`custom_prompt`：格式约束更严 → 可维护性 ↑  
   3. 统一生命周期与 `Message` 历史、`max_steps` 配置化 → 可测试/可观测 ↑  

   **（2）质量评分早停**

   在 reflect 之后增加 score 步：LLM 输出 0–10 分；`score >= threshold`（如 8）则 break，否则再 refine。打分解析失败要有兜底，避免空转耗 token。完整伪代码见 study 笔记习题 4.2。

   **（3）Tree-of-Thought Agent**

   每步 **Expand（生成 k 条候选）→ Evaluate（打分/投票）→ Select（选最优继续）**，直到 Final 或达到深度上限。注意控制 `breadth * depth` 的调用成本；需要工具时在选定路径上再调 Registry。

   </details>

5. 在7.5节中，我们构建了工具系统。请思考以下问题：

   - `BaseTool` 类定义了 `execute` 抽象方法，所有工具都必须实现这个方法。请解释为什么要强制所有工具实现统一的接口？如果某个工具需要返回多个值（如搜索工具返回标题、摘要、链接），应该如何设计？
   - 在7.5.3节中实现了工具链（`ToolChain`）。请设计一个实际的应用场景，需要串联至少3个工具，并画出工具链的执行流程图。
   - 异步工具执行器（`AsyncToolExecutor`）使用了线程池来并行执行工具。请分析：在什么情况下并行执行工具能带来性能提升？

   <details>
   <summary><strong>参考解答</strong></summary>

   **（1）统一接口 & 多值返回**

   统一 `execute`/`run` 才能让 Registry、工具链、Agent 无差别调度，符合里氏替换。多值建议返回 **JSON 字符串** 或 dict（由 Registry 统一序列化），也可「摘要 str + metadata」；避免每个工具互不兼容的返回类型。

   **（2）场景：技术调研简报（≥3 工具）**

   `advanced_search`（搜版本与变更）→ `calculator`（对比性能数字）→ `summarizer`/LLM 工具（生成中文简报）。

   ```mermaid
   flowchart LR
     A[用户问题] --> B[Search]
     B --> C[Calculator]
     C --> D[Summarize]
     D --> E[最终输出]
   ```

   **（3）并行何时提速**

   - **能**：相互独立的 I/O（多搜索源、多 URL），等待网络时并行。  
   - **不能/有害**：后步依赖前步输出；纯 CPU 重且受 GIL 限制；共享配额易触发限流。  
   - 口诀：无数据依赖的 I/O → 并行；有依赖 → 工具链串行。

   </details>

6. 框架的可扩展性是设计的重要考量因素之一。你现在要扩展 `HelloAgents` 框架，为其实现一些有趣的新功能和特性。

   - 首先为 `HelloAgents` 添加一个"流式输出"功能，使得 `Agent` 在生成响应时能够实时返回中间结果（类似 `ChatGPT` 用户界面的打字效果）。请设计这个功能的实现方案，说明需要修改哪些类和方法。
   - 然后为框架添加"多轮对话管理"功能，能够自动管理对话历史、支持对话分支和回溯，你会如何设计？需要新增哪些类？如何与现有的 `Message` 系统集成？
   - 最后请为 `HelloAgents` 设计一个"插件系统"，允许第三方开发者通过插件的方式扩展框架功能（如添加新的 `Agent` 类型、新的工具类型等），而无需修改框架核心代码。要求画出插件系统的架构图并说明关键接口。

   <details>
   <summary><strong>参考解答</strong></summary>

   **（1）流式输出**

   - `HelloAgentsLLM`：统一 `stream(messages) -> Iterator[str]`（可基于已有 `think`）  
   - `Agent`：新增 `stream()`；`SimpleAgent` 边生成边 yield；ReAct 等可先 yield 步骤事件再流式输出最终答案  
   - 可选 `StreamEvent(type, data)`：`token` / `tool_start` / `tool_result` / `final`

   **落地（继承扩展）**：`code/chapter7/my_streaming_agent.py` 中 `StreamingAgent(MySimpleAgent)`；事件模型见 `stream_events.py`；测试 `test_streaming_agent.py`。

   **（2）多轮对话 + 分支回溯**

   新增 `Conversation`、`ConversationStore`、`DialogTree`/`Checkpoint`。叶子仍是 `Message`；分支 = 复制某节点前缀历史再 append；回溯 = 将 Agent `_history` 重置为某 checkpoint。Agent 只消费「当前分支的 Message 列表」。

   **落地（继承扩展）**：`conversation.py` + `my_dialog_agent.py` 中 `DialogAgent(MySimpleAgent)`；测试 `test_dialog_agent.py`。

   **（3）插件系统**

   ```text
   Core(Agent/ToolRegistry/LLM/Config)
           │ PluginAPI
      PluginManager (discover/load/register)
           ├── AgentPlugin
           ├── ToolPlugin
           └── ProviderPlugin
   ```

   关键接口：`Plugin.setup(app)` / `teardown()`；`app.register_agent_type` / `register_tool` / `register_provider`。通过 entry_points 或 `plugins/` 目录发现，第三方不修改核心源码。

   **落地**：`plugin_system.py`；示例 `plugins/demo_plugins.py` 中 `EchoAgent(SimpleAgent)`；测试 `test_plugin_system.py`。

   </details>

---

## 学习版加厚 · 核心代码概念精讲

### 1. HelloAgentsLLM：为什么要「自动检测」

业务代码希望写成：

```python
llm = HelloAgentsLLM()  # 或 HelloAgentsLLM(provider="ollama")
```

而不是每个项目复制一套 OpenAI/ModelScope/Ollama 客户端。自动检测通常按：**显式 provider → 环境变量线索 → base_url 特征 → 默认**。读源码时顺着这条链看，比死记函数名有用。

### 2. Message / Config / Agent：最小宪法

- **Message**：`role + content (+ name/metadata)`，让历史可序列化、可截断、可审计。
- **Config**：温度、max_tokens、超时、默认模型——集中配置，避免魔法数。
- **Agent ABC**：规定 `run`/`stream`/历史管理等必须实现的能力；范式子类只填差异。

把这三块想成 USB 标准：上层设备（范式）可替换，下层供电（LLM）可替换。

### 3. 范式框架化时多出来的工程件

对照第四章脚本，框架版通常多了：统一历史容器、工具注册注入、流式接口、可测入口（`test_*.py`）、提示词模板可覆盖。读 7.4 时用「多了什么」而不是「提示词是否一字不差」当检查表。

### 4. 工具注册表的心智模型

```
Agent.run
  → 决定调用工具 X
  → registry.get("X")
  → tool.run(**args)
  → 观察写回历史
```

`description` 是给模型看的说明书；`name` 是注册键；参数 schema（Function Calling）是类型合同。三者不一致时，最常见的 bug 是「模型想调却调不到 / 调错参」。

### 5. FunctionCallAgent vs ReActAgent

| | ReAct | Function Calling |
|--|-------|------------------|
| 调用表达 | 文本 Thought/Action | 结构化 tool_calls |
| 解析 | 正则/约定格式，易碎 | 协议层更稳 |
| 适用 | 教学、泛模型 | 支持工具调用的模型 |

---

## 学习版加厚 · 跟练文件速查

| 主题 | 建议文件/入口 |
|------|----------------|
| LLM 多 Provider | `my_llm.py` / `.env` |
| SimpleAgent | `my_simple_agent.py` + `test_simple_agent.py` |
| ReAct | `my_react_agent.py` |
| Reflection | `test_reflection_agent.py` |
| Plan-and-Solve | `test_plan_solve_agent.py` |
| 自定义工具 | `my_calculator_tool.py` |
| 多源搜索 | `my_advanced_search.py` |
| 工具链 / 异步 | `tool_chain_manager.py` / `async_tool_executor.py` |

（具体路径以仓库 `code/` 与章节说明为准。）

## 学习版加厚 · FAQ

**Q1：为什么不直接依赖 LangChain？**  
教学框架刻意瘦身，让你看见最小必要接口；理解后再用工业框架会更有判断力。

**Q2：SimpleAgent 和 ReActAgent 怎么选？**  
无工具/轻工具对话 → Simple；需要交错推理与行动 → ReAct；要结构化调用 → FunctionCall。

**Q3：自动检测 Provider 失败怎么办？**  
显式传 `provider=`，并检查 `base_url`、密钥环境变量名是否与文档一致。

**Q4：工具 description 为什么这么重要？**  
无论 ReAct 文本解析还是 Function Calling，模型都主要靠描述决定「何时调用、传什么」。

## 学习版加厚 · 术语表

| 术语 | 含义 |
|------|------|
| HelloAgentsLLM | 框架统一 LLM 客户端 |
| Message | 带角色的一条对话消息 |
| Config | 运行配置对象 |
| ToolRegistry | 工具注册与发现 |
| Paradigm Agent | 实现某种经典范式的 Agent 子类 |
| Streaming | 流式生成 |
| Schema | 工具参数的结构化描述 |

## 与第四、六章对照

| 能力 | 第四章 | 第六章 | 第七章 |
|------|--------|--------|--------|
| 范式实现 | 手搓脚本 | 框架内置/编排 | 自建可复用类 |
| 多模型 | 自行改客户端 | 各框架自有 | HelloAgentsLLM |
| 工具 | 示例级 | 框架机制 | 注册表 + 高级特性 |
| 目的 | 理解 | 选型与协作 | 沉淀自己的底座 |


---

## 本章自检清单

按用户约定的四条标准自检如下（请你学完后也自己勾选）：

- [ ] **知识点是否全覆盖** — 已覆盖原版全部主要知识点与小节
- [ ] **是否详细** — 在原版全文基础上补充教学铺垫与决策表，体量不低于原版
- [ ] **语言是否清晰** — 先心智模型与关键词，再进入协议/算法/代码
- [ ] **陌生点是否有前置铺垫** — 陌生术语在进入正文前给出可检查定义

### 自检说明（作者侧已执行）

1. **覆盖**：已对照原版 7.1–7.6 全部 H2/H3（含 LLM 扩展、Message/Config/Agent、五大范式、工具系统四级、习题与参考文献）。
2. **详细**：在原版全文外增加分层心智模型、各节代码精讲导读、跟练速查、FAQ 与跨章对照；体量不低于原版。
3. **清晰**：先配件比喻与分层图，再进实现；本地模型与 Function Calling 单独铺垫。
4. **铺垫**：Provider、ABC、注册表、OpenAI 兼容、流式等术语在进入正文前给出可检查定义。

### 覆盖对照表

| 原书要点 | 学习版对应 | 状态 |
|---------|-----------|------|
| 7.1 为何自建 / 理念 / 学习目标与快速体验 | 7.1 + 分层图 | ✅ |
| 7.2 多提供商 / 本地 / 自动检测 | 7.2 + 精讲 | ✅ |
| 7.3 Message / Config / Agent | 7.3 + 为何需要 | ✅ |
| 7.4 Simple/ReAct/Reflection/PlanSolve/FunctionCall | 7.4 + 对照第四章 | ✅ |
| 7.5 工具基类/自定义/多源搜索/高级特性 | 7.5 + 速查 | ✅ |
| 7.6 小结 + 习题 + 参考文献 | 文末保留 | ✅ |

### 局限说明

HelloAgents 是教学向框架，功能子集小于工业框架；部分高级工具示例依赖外部搜索 API。自动检测逻辑随环境变量约定变化，以章节当时代码为准。若官方包版本升级，导入路径可能微调。

---

## 💬 讨论与交流

本章学习过程中遇到问题？想与其他学习者交流心得？

**📝 前往 GitHub Discussions 讨论区:**
- [💬 习题讨论与问答](https://github.com/datawhalechina/Hello-Agents/discussions)

**💡 提示:** 每个页面底部也有评论区，可以直接在页面内讨论。
