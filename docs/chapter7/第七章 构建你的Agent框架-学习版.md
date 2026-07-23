# 第七章 构建你的智能体框架

前六章我们认识了智能体、大模型、经典范式，也体验了低代码平台和主流框架。从这一章开始，路线换了：不再只当「框架的使用者」，而是亲自搭一套能长大的底座——**HelloAgents**。

你可以把它理解成：前面是学会开车，这一章是拆开引擎、看懂变速箱，再按图纸装回一台能继续加涡轮（第八章 Memory/RAG）的车。

为确保学习过程的连贯性与可复现性，HelloAgents 将以**版本迭代**的方式推进开发。每一章都会在前一章的基础上增加新的功能模块，并将智能体相关的知识点进行串讲与实现。最终，我们将利用这个自建框架，来高效地实现本书后续章节中的高级应用案例。

---

## 读这一章之前，先建立心智模型

### 一个工程比喻

| 盖房子 | HelloAgents | 你要记住的一句话 |
|--------|-------------|------------------|
| 地基与水电规范 | `core/`（LLM / Message / Config / Agent） | 没有统一接口，上层范式会各写一套重复代码 |
| 不同户型平面图 | `agents/`（五种范式） | 同一套房型规范，不同生活流线 |
| 电钻、梯子、外接设备 | `tools/` | 能力外延都挂工具柜，不另开科室 |
| 「一切外接能力都走插座」 | **万物皆为工具** | Memory/RAG/MCP 也是 Tool |

### 几个关键词，先抓牢

- **provider**：告诉框架用哪家模型服务（`openai` / `modelscope` / `ollama` / `vllm` / `auto`）。
- **继承扩展**：用 `MyLLM`、`MySimpleAgent` 定制，**不要改**已安装库的源码——否则升级会把定制冲掉。
- **ToolRegistry**：工具柜——注册、发现、按名执行。
- **Agent.run**：所有范式统一入口；抽象基类强制子类实现。
- **约定优于配置**：尽量靠环境变量自动检测，少写样板代码。
- **万物皆工具**：把 Memory / RAG / MCP 等统一抽象成 Tool，减少概念层数。
- **对内丰富，对外兼容**：`Message` 内部可有 timestamp/metadata，发给 API 时只留 role+content。

### 建议怎么读

1. **先 30 秒跑通**：`pip install "hello-agents==0.1.1"`（Python ≥ 3.10），配好 `.env`，跑 SimpleAgent，确认链路通。
2. **对照目录树**：先搞清 `core` / `agents` / `tools` 各管什么，再钻进某一文件。
3. **跟写 My\* 类**：扩展一律继承；每写完一块就跑对应 `test_*.py`。
4. **互动教程 + 本文对照**：浏览器翻页建立地图与自测；本文补「为什么这样设计」与习题导读。
5. **代码位置**：仓库 `code/chapter7/`；官方示例 [chapter07_basic_setup.py](https://github.com/jjyaoao/HelloAgents/blob/main/examples/chapter07_basic_setup.py)。

读代码时建议每次问四个问题（和第四章同一套方法）：这段是在定义能力还是组织流程？它操作的是模型、工具、状态还是解析？删掉它会失去什么？它是业务逻辑还是工程保护（重试、步数上限、异常）？

### 本章在全书中的位置

```
第 1–6 章：认知 + 范式 + 低代码 + 主流框架实践
        ↓
第 7 章：自建 HelloAgents 底座（本章）
        ↓
第 8 章：Memory / RAG（仍挂成 Tool）
第 9 章：上下文工程（深化 Message）
第 10 章：智能体协议（新工具形态）
```

---

## 7.1 框架整体架构设计

这一节回答三件事：**为什么还要自建**、**HelloAgents 信什么**、**你要建什么**。读完你应能向别人用两分钟讲清楚「为何不继续只用 LangChain」。

### 7.1.1 为何需要自建 Agent 框架

在智能体技术快速发展的今天，市面上已经存在众多成熟的 Agent 框架。那么，为什么我们还要从零开始构建一个新的框架呢？

#### （1）市面框架的快速迭代与局限性

智能体领域随时会有新概念；各框架定位不同，但核心知识点是一致的。常见痛点：

1. **过度抽象的复杂性**：为追求通用性引入大量抽象层。以 LangChain 为例，链式调用灵活，但对初学者陡峭，往往要理解大量概念才能完成简单任务。
2. **快速迭代带来的不稳定性**：商业化框架 API 变更频繁，版本升级后代码无法运行，维护成本居高不下。
3. **黑盒化的实现逻辑**：核心封装过严，难以理解内部机制，缺乏深度定制能力。出问题只能等文档和社区；社区不活跃时反馈可能很久无人推进。
4. **依赖关系的复杂性**：携带大量依赖，安装包体积大，与其他项目配合时易出现依赖冲突。

#### （2）从使用者到构建者的能力跃迁

构建自己的 Agent 框架，是从「使用者」向「构建者」转变，价值长远：

- **深度理解 Agent 工作原理**：亲手实现每个组件，才能真正理解思考过程、工具调用机制、以及各种设计模式的好坏与区别。
- **获得完全的控制权**：对每一行代码完全掌控，可按需求精确调优，不受第三方设计理念束缚。
- **培养系统设计能力**：模块化设计、接口抽象、错误处理等软件工程核心技能，对长期成长重要。

#### （3）定制化需求与深度掌握的必要性

不同场景差异巨大，往往需要在通用框架上二次开发：

- **特定领域优化**：金融、医疗、教育需要针对性提示词、特殊工具集成、定制化安全策略。
- **性能与资源精确控制**：生产环境对响应时间、内存、并发有严格要求，「一刀切」常不够。
- **学习与教学的透明性**：学习者更期待清晰看到每一步构建过程，理解不同范式机制 → 要求高度可观测、可解释。

> **类比**：别人的成品屋漂亮，但你拆不开墙；自建后你知道每块砖为什么在那儿。

### 7.1.2 HelloAgents 框架的设计理念

构建新框架的关键不在功能多少，而在设计理念是否解决痛点。HelloAgents 围绕一个核心问题：

> 如何让学习者既能快速上手，又能深入理解 Agent 的工作原理？

初次接触成熟框架时，你可能被功能吸引，但很快发现：完成简单任务也要理解 Chain、Agent、Tool、Memory、Retriever 等十几个概念，学习曲线陡峭。HelloAgents 试图在**功能完整性**与**学习友好性**之间找平衡，形成四条理念。

#### （1）轻量级与教学友好的平衡

优秀的学习框架应具备完整可读性。HelloAgents 将核心代码按章节区分开——有一定编程基础的开发者应能在合理时间内完全理解工作原理。依赖管理采用极简主义：除 OpenAI 官方 SDK 与几个必要基础库外，不引入重型依赖。出问题可直接定位框架本身，而不必在复杂依赖里找答案。

#### （2）基于标准 API 的务实选择

OpenAI API 已成为行业标准，主流 LLM 提供商都在兼容这套接口。HelloAgents 选择在其上构建，而不是重新发明抽象：

- **兼容性**：掌握用法后，迁移到其他框架或集成现有项目时，底层 API 调用逻辑一致。
- **学习成本**：不必学新概念模型，操作基于已熟悉的标准接口。

#### （3）渐进式学习路径的精心设计

每一章的学习代码可保存为可 `pip` 下载的历史版本；每一个核心功能都是你自己编写的。按自己的节奏前进，升级自然衔接，不会产生概念跳跃。本章基于前六章完善，并为后续高级知识打下框架基础。

#### （4）统一的「工具」抽象：万物皆为工具

除核心 Agent 类外，一切皆为 Tools。许多其他框架中要独立学习的 Memory、RAG、RL、MCP 等，在 HelloAgents 中统一抽象为「工具」。初衷是消除不必要抽象层，让学习者回归最直观的「智能体调用工具」逻辑，实现快速上手与深入理解的统一。

> **习题预告**：这种设计有什么优势？是否存在局限性？例如某些模块的生命周期是否不完全等于「一次工具调用」？

### 7.1.3 本章学习目标

架构设计遵循 **「分层解耦、职责单一、接口统一」**，既保持组织性，也便于按章节扩展。

```
hello-agents/
├── hello_agents/
│   ├── core/                     # 核心框架层
│   │   ├── agent.py              # Agent基类
│   │   ├── llm.py                # HelloAgentsLLM统一接口
│   │   ├── message.py            # 消息系统
│   │   ├── config.py             # 配置管理
│   │   └── exceptions.py         # 异常体系
│   ├── agents/                   # Agent实现层
│   │   ├── simple_agent.py
│   │   ├── react_agent.py
│   │   ├── reflection_agent.py
│   │   └── plan_solve_agent.py
│   └── tools/                    # 工具系统层
│       ├── base.py
│       ├── registry.py
│       ├── chain.py
│       ├── async_executor.py
│       └── builtin/
│           ├── calculator.py
│           └── search.py
```

| 路径 | 一句话职责 |
|------|------------|
| `core/llm.py` | 模型调用枢纽（多提供商 / 本地 / 自动检测） |
| `core/message.py` | 对话消息标准信封 |
| `core/config.py` | 集中配置 + `from_env` |
| `core/agent.py` | Agent 抽象基类（强制 `run`） |
| `agents/*.py` | 各范式具体实现 |
| `tools/base.py` | Tool / ToolParameter |
| `tools/registry.py` | 注册、发现、执行 |
| `tools/chain.py` | 顺序工具链 |
| `tools/async_executor.py` | 异步/并行执行 |

#### 快速开始：安装

```bash
# https://github.com/jjyaoao/HelloAgents
# Python >= 3.10
pip install "hello-agents==0.1.1"
```

两种学习方式：

1. **体验式学习**：直接 `pip` 安装，运行示例，快速体验。
2. **深度学习**：跟随本章从零实现每个组件，深入理解设计思想。

建议采用 **「先体验，后实现」**：重写核心函数并运行测试，检验实现是否正确。想参与框架开发可访问 [GitHub 仓库](https://github.com/jjyaoao/helloagents)。

#### 30 秒体验

配置好同级文件夹下 `.env` 中的大模型 API（可参考 `code` 文件夹配套的 `.env.example`，或复用前几章案例的 `.env`）：

```python
from hello_agents import SimpleAgent, HelloAgentsLLM
from dotenv import load_dotenv

load_dotenv()

llm = HelloAgentsLLM()  # 框架自动检测 provider
# 或手动：HelloAgentsLLM(provider="modelscope")

agent = SimpleAgent(
    name="AI助手",
    llm=llm,
    system_prompt="你是一个有用的AI助手",
)

response = agent.run("你好！请介绍一下自己")
print(response)

from hello_agents.tools import CalculatorTool
calculator = CalculatorTool()
# 需要实现 7.4.1 的 MySimpleAgent 进行调用；后续章节会支持此类调用
# agent.add_tool(calculator)

response = agent.run("请帮我计算 2 + 3 * 4")
print(response)
print(f"历史消息数: {len(agent.get_history())}")
```

---
## 7.2 HelloAgentsLLM 扩展

本节在第 4.1.3 节创建的 `HelloAgentsLLM` 基础上迭代升级，把它改造成更具适应性的模型调用中枢。三个目标：

1. **多提供商支持**：OpenAI、ModelScope、智谱 AI 等无缝切换，避免与特定供应商绑定。
2. **本地模型集成**：引入 VLLM 和 Ollama，作为对第 3.2.3 节 Hugging Face Transformers 的生产级补充，满足隐私与成本需求。
3. **自动检测机制**：根据环境信息智能推断 LLM 服务类型，简化配置。

> **类比**：HelloAgentsLLM 像万能插座转接头——墙上不管哪种孔（云端/本地），插头统一成 OpenAI 兼容口。

### 7.2.1 支持多提供商

之前的类已能通过 `api_key` 和 `base_url` 连接任何兼容 OpenAI 接口的服务，理论上通用。但实际中，不同服务商在环境变量命名、默认 API 地址和推荐模型上存在差异。每次切换都让用户查文档改代码，会严重影响效率。因此引入 `provider`：让 `HelloAgentsLLM` **内部**处理配置细节，对外统一简洁。

具体自动检测细节在 7.2.3；这里先学如何**扩展**框架。

> **重要**：直接修改已安装库源码不被推荐——会使后续升级困难。正确做法是继承。

#### （1）创建自定义 LLM 类并继承

```python
# my_llm.py
import os
from typing import Optional
from openai import OpenAI
from hello_agents import HelloAgentsLLM

class MyLLM(HelloAgentsLLM):
    """自定义 LLM 客户端，通过继承增加 ModelScope 支持。"""
    pass  # 暂时留空
```

#### （2）重写 `__init__` 以支持新供应商

目标：`provider="modelscope"` 时走自定义逻辑；否则 `super().__init__(...)` 保留父类全部能力。

```python
class MyLLM(HelloAgentsLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = "auto",
        **kwargs,
    ):
        if provider == "modelscope":
            print("正在使用自定义的 ModelScope Provider")
            self.provider = "modelscope"
            self.api_key = api_key or os.getenv("MODELSCOPE_API_KEY")
            self.base_url = base_url or "https://api-inference.modelscope.cn/v1/"
            if not self.api_key:
                raise ValueError("ModelScope API key not found. Please set MODELSCOPE_API_KEY.")
            self.model = model or os.getenv("LLM_MODEL_ID") or "Qwen/Qwen2.5-VL-72B-Instruct"
            self.temperature = kwargs.get("temperature", 0.7)
            self.max_tokens = kwargs.get("max_tokens")
            self.timeout = kwargs.get("timeout", 60)
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        else:
            super().__init__(model=model, api_key=api_key, base_url=base_url, provider=provider, **kwargs)
```

这就是「重写」思想：拦截目标 provider，其余交还父类。

#### （3）使用自定义的 MyLLM

```bash
# .env
MODELSCOPE_API_KEY="your-modelscope-api-key"
```

```python
# my_main.py
from dotenv import load_dotenv
from my_llm import MyLLM

load_dotenv()
llm = MyLLM(provider="modelscope")
messages = [{"role": "user", "content": "你好，请介绍一下你自己。"}]
response_stream = llm.think(messages)  # think 等方法已从父类继承
for chunk in response_stream:
    pass
```

不改库源码 → 代码整洁可维护 → 未来升级定制不丢失。跟练见 `code/chapter7/my_llm.py`、`my_main.py`。

### 7.2.2 本地模型调用

第 3.2.3 节用 Transformers 本地跑开源模型，适合入门与验证，但高并发性能有限，通常不是生产首选。VLLM、Ollama 等通过连续批处理、PagedAttention 等提升吞吐量，并把模型封装为兼容 OpenAI 标准的 API 服务——因此可无缝接入 `HelloAgentsLLM`。

#### VLLM

高性能推理库。按硬件（尤其 CUDA）按[官方文档](https://docs.vllm.ai/en/latest/getting_started/installation.html)安装，避免版本不匹配。

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen1.5-0.5B-Chat \
    --host 0.0.0.0 \
    --port 8000
# → http://localhost:8000/v1
```

#### Ollama

把下载、配置、启动封装成一条命令，适合快速上手。访问 [ollama.com](https://ollama.com) 安装客户端后：

```bash
ollama run llama3
# → http://localhost:11434/v1
```

#### 接入 HelloAgentsLLM

把它们视为新的 `provider` 即可：

```python
llm_client = HelloAgentsLLM(
    provider="vllm",
    model="Qwen/Qwen1.5-0.5B-Chat",  # 需与服务启动时一致
    base_url="http://localhost:8000/v1",
    api_key="vllm",  # 本地通常不需要真实 Key
)
```

或通过环境变量零修改：

```bash
LLM_BASE_URL="http://localhost:8000/v1"
LLM_API_KEY="vllm"
```

```python
llm_client = HelloAgentsLLM()  # 自动检测为 vllm
```

Ollama 同理，端口改为 `11434`，`provider="ollama"`，`model` 与 `ollama run` 一致。

**统一设计的收益**：智能体核心代码无需修改，即可在云端 API 与本地模型间自由切换——利于开发、部署、成本控制与保护数据隐私。

### 7.2.3 自动检测机制

为减少配置负担并遵循「约定优于配置」，内部有两个核心辅助方法：

- `_auto_detect_provider`：根据环境推断服务商
- `_resolve_credentials`：根据推断结果完成参数配置

#### 检测优先级

1. **最高：特定服务商环境变量**  
   依次检查 `MODELSCOPE_API_KEY`、`OPENAI_API_KEY`、`ZHIPU_API_KEY` 等，发现即确定。

2. **次高：根据 base_url**  
   - 域名：如含 `api-inference.modelscope.cn`、`api.openai.com`  
   - 端口：`:11434` → Ollama；`:8000` → VLLM；其他 localhost → `local`

3. **辅助：API 密钥格式**  
   例如 `ms-` 前缀 → ModelScope。可能模糊，优先级低。

4. 仍不明 → 返回 `"auto"`，使用通用配置。

关键逻辑示意：

```python
def _auto_detect_provider(self, api_key, base_url) -> str:
    if os.getenv("MODELSCOPE_API_KEY"): return "modelscope"
    if os.getenv("OPENAI_API_KEY"): return "openai"
    if os.getenv("ZHIPU_API_KEY"): return "zhipu"
    actual_api_key = api_key or os.getenv("LLM_API_KEY")
    actual_base_url = base_url or os.getenv("LLM_BASE_URL")
    if actual_base_url:
        u = actual_base_url.lower()
        if "api-inference.modelscope.cn" in u: return "modelscope"
        if "localhost" in u or "127.0.0.1" in u:
            if ":11434" in u: return "ollama"
            if ":8000" in u: return "vllm"
            return "local"
    if actual_api_key and actual_api_key.startswith("ms-"):
        return "modelscope"
    return "auto"
```

provider 确定后，`_resolve_credentials` 按家查找环境变量并设默认 `base_url`（如 openai → `https://api.openai.com/v1`，modelscope → ModelScope 推理地址）。

#### 便利示例：只配 Ollama

```bash
LLM_BASE_URL="http://localhost:11434/v1"
LLM_MODEL_ID="llama3"
```

```python
from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM
load_dotenv()
llm = HelloAgentsLLM()  # 自动 → ollama
for chunk in llm.think([{"role": "user", "content": "你好！"}]):
    print(chunk, end="")
```

#### 表 7.1：从简单到完善

相较 4.1.3，现版增加多提供商、本地部署、自动检测，同时保持接口简洁。这体现框架设计原则：**从简单开始，逐步完善**。

> **习题 2**：若同时设置 `OPENAI_API_KEY` 与 `LLM_BASE_URL="http://localhost:11434/v1"`，框架会选哪个？这种优先级是否合理？你若想「有本地 URL 就优先本地」，会怎么改？

---
## 7.3 框架接口实现

`HelloAgentsLLM` 解决了与大模型通信的关键问题，还需要配套组件处理数据流、管理配置、应对异常，并为上层提供清晰统一结构。本节三个核心文件：

- `message.py`：统一消息格式
- `config.py`：中心化配置
- `agent.py`：所有智能体的抽象基类

### 7.3.1 Message 类

对话历史是至关重要的上下文。简易 `Message` 类在后续上下文工程章节还会扩展。

设计要点：

1. 用 `typing.Literal` 把 `role` 限制为 `"user" | "assistant" | "system" | "tool"`，对齐 OpenAI，保证类型安全。
2. 除 `content`、`role` 外增加 `timestamp`、`metadata`，为日志与未来扩展预留空间。
3. `to_dict()` 转为 API 兼容字典——**对内丰富，对外兼容**。

```python
MessageRole = Literal["user", "assistant", "system", "tool"]

class Message(BaseModel):
    content: str
    role: MessageRole
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}
```

> **习题 3**：使用 Pydantic `BaseModel` 做数据验证，在实际应用中有哪些优势？（提示：校验、文档、序列化、非法输入早失败。）

### 7.3.2 Config 类

职责：把硬编码参数集中起来，并支持从环境变量读取。

逻辑分组：LLM 配置（model / provider / temperature / max_tokens）、系统配置（debug / log_level）、其他（如 `max_history_length`）。每项有合理默认值 → 零配置也能工作。核心是 `from_env()`：部署到不同环境时用环境变量覆盖，无需改代码。

```python
@classmethod
def from_env(cls) -> "Config":
    return cls(
        debug=os.getenv("DEBUG", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        temperature=float(os.getenv("TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,
    )
```

> **单例直觉**：若应用里多处各自 `Config()` 且互不共享，可能出现温度、debug 标志不一致。配置管理常用「进程内一份」避免漂移。习题会要求你解释单例模式。

#### 【补缺口】习题写「单例」，正文却没强制单例

原书习题称「Config 使用了单例模式」，但 **7.3.2 正文给出的是普通 Pydantic 模型 + `from_env()`**，并未展示 `__new__` / 模块级唯一实例等经典单例写法。

答题时建议分三层：

1. **意图**：全进程共享一份有效配置，避免行为漂移。  
2. **实现可选**：单例、依赖注入同一 `config` 对象、或启动时读一次环境再传递——都能达成意图。  
3. **对照正文**：注明「示例是可序列化配置对象；单例是工程上的可选强化」。

### 7.3.3 Agent 抽象基类

`Agent` 是顶层抽象：定义通用行为与属性，不关心具体实现。通过 `abc` 强制所有具体智能体（Simple / ReAct / …）遵循同一接口。

```python
class Agent(ABC):
    def __init__(self, name, llm, system_prompt=None, config=None):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        pass

    def add_message(self, message: Message): ...
    def clear_history(self): ...
    def get_history(self) -> list[Message]:
        return self._history.copy()
```

设计体现：

- 不能直接实例化抽象类
- 构造函数清晰声明依赖：名称、LLM、系统提示、配置
- `@abstractmethod run` 保证统一执行入口
- 历史管理与 `Message` 协同

这是面向对象中的抽象 + **模板方法**思想：公共骨架在基类，变化点在子类。

> 习题若提到 `run` 与 `_execute` 的分工，请以你跟写版本的源码为准；考点都是「公开入口稳定、内部实现可替换」。

至此，核心基础组件设计完成。

---
## 7.4 Agent 范式的框架化实现

在第四章三种经典范式（ReAct、Plan-and-Solve、Reflection）基础上框架化重构，并新增 SimpleAgent。三个核心目标：

1. **提示词工程系统性提升**：从特定任务导向转向通用化，增强格式约束与角色定义。
2. **接口与格式标准化统一**：同一基类、初始化参数、方法签名、历史管理。
3. **高度可配置**：自定义提示词模板、配置参数、执行策略。

### 7.4.1 SimpleAgent

最基础的对话智能体，展示如何在框架上构建完整 Agent。创建 `my_simple_agent.py`：

```python
class MySimpleAgent(SimpleAgent):
    def __init__(self, name, llm, system_prompt=None, config=None,
                 tool_registry=None, enable_tool_calling=True):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
```

#### run：无工具 vs 有工具

1. `_get_enhanced_system_prompt()`：若启用工具，注入工具描述与调用格式说明。
2. 拼系统消息 + 历史 + 当前 user。
3. 未启用工具：一次 `llm.invoke`，写入历史，返回。
4. 启用工具：进入 `_run_with_tools`，多轮迭代。

工具调用约定格式：

```text
[TOOL_CALL:{tool_name}:{parameters}]
例如: [TOOL_CALL:search:Python编程]
```

`max_tool_iterations`（默认 3）防止死循环。工具结果写回对话后再继续问模型。

便利能力：`stream_run` 流式输出；`add_tool` / `remove_tool` / `list_tools` / `has_tools` 动态管理注册表。

建议用 `test_simple_agent.py` 覆盖：基础对话、带工具、流式、动态添加工具、历史条数。

### 7.4.2 ReActAgent

接 `ToolRegistry`，可配 `max_steps`。提示词结构通常包含：可用工具、工作流程、重要提醒、当前任务、执行历史。

硬约束帮助工程稳定：

- 每次只输出一步 Thought + Action
- Action 为 `tool_name[input]` 或 `Finish[最终答案]`
- 解析严格；格式跑偏要能处理或重试（框架化相对第四章的重要改进之一）

循环：填模板 → LLM → 解析 Thought/Action → 若 Finish 则返回 → 否则 `registry.execute_tool` → 追加 Observation → 下一步直至上限。

**何时用**：路径不确定、需要边查边决策。纯闲聊或单次工具可优先 Simple。

文件：`my_react_agent.py`、`test_react_agent.py`。

### 7.4.3 ReflectionAgent

默认三段提示词：

| 阶段 | 作用 |
|------|------|
| initial | 针对任务生成初稿 |
| reflect | 批判初稿，给出改进意见 |
| refine | 按反馈改写 |

相对第四章「偏代码生成」的特化提示，本章强调**通用化**，并支持 `custom_prompts` 深度定制（例如代码生成专用）。若反思认为「无需改进」，可提前终止，节省调用。

文件：`test_reflection_agent.py`。

### 7.4.4 PlanAndSolveAgent

- **Planner**：必须输出可解析的步骤列表字符串，例如 `["步骤1", "步骤2", ...]`——这是相对第四章自由文本计划的关键格式升级，方便程序解析。
- **Executor**：输入含原始问题、完整计划、历史步骤结果、当前步骤；**仅输出针对当前步骤的回答**。

可用 `custom_prompts` 换成数学专用模板（planner 要求输出计算步骤列表；executor 只输出数值）。注意完整异常处理，避免一步失败拖垮整条链。

文件：`test_plan_solve_agent.py`。

### 7.4.5 FunctionCallAgent

`hello-agents` 在 0.2.8 之后引入。基于 OpenAI 原生函数调用，相对 Prompt 约束的 `[TOOL_CALL:...]` 鲁棒性更强。

关键能力：

- `_build_tool_schemas`：由工具 description 构建 schema
- `_extract_message_content`：从响应提取文本
- `_parse_function_call_arguments`：解析 JSON 参数字符串
- `_convert_parameter_types`：参数类型转换
- `_invoke_with_tools`：对 `client.chat.completions.create(..., tools=..., tool_choice=...)` 再封装

原生示例思路：先声明 tools schema，再把 messages 与 tools 一并交给模型，由模型决定是否调用函数。

#### 表 7.2：第四章 vs 本章

框架化后：组织性、可维护性、扩展性上升；统一初始化 / 历史 / ToolRegistry；提示词更通用、格式更严——同时保留各范式特色与优势。

| 范式 | 一句话 | 典型场景 |
|------|--------|----------|
| Simple | 对话底座 + 可选文本工具约定 | 聊天、轻量工具 |
| ReAct | 边想边做 | 路径不确定 |
| Reflection | 初稿→批判→改写 | 质量敏感 |
| PlanSolve | 先拆列表再逐步执行 | 步骤清晰的长任务 |
| FunctionCall | 原生 schema 工具调用 | 需要更稳参数解析 |

---
## 7.5 工具系统

在 Agent 基础架构上，深入工具系统的设计与实现。学习目标：

1. **统一的工具抽象与管理**：Tool 基类 + ToolRegistry。
2. **实战驱动的工具开发**：数学计算工具走完流程。
3. **高级整合与优化**：多源搜索、智能后端选择、容错；以及链与异步。

### 7.5.1 工具基类与注册机制设计

#### （1）Tool 基类

```python
class Tool(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        pass
```

统一 `run`：字典参数进、字符串出，保证框架一致性。`get_parameters` 提供自描述，支撑文档与校验。`name`/`description` 让工具可发现、可理解。

#### （2）ToolParameter

```python
class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
```

#### （3）ToolRegistry

维护 `_tools` 与 `_functions` 两套存储：

- `register_tool(tool)`：适合复杂工具
- `register_function(name, description, func)`：适合简单函数，快速集成

重复注册会警告并覆盖。

#### （4）发现与 schema

`get_tools_description()` 生成可直接写入 Agent 提示词的列表。

`to_openai_schema()` 转为 Function Calling 标准格式：数组类型默认 `items` 为 string；若有默认值则写入 description（OpenAI schema 本身无 default 字段）。

### 7.5.2 自定义工具开发

最直接方式：写函数 + `register_function`。计算器示例用 AST 白名单求值：

- 支持 `+ - * /`、`sqrt`、`pi`
- 空表达式返回错误提示
- **禁止裸 `eval`**，降低任意代码执行风险

```python
def create_calculator_registry():
    registry = ToolRegistry()
    registry.register_function(
        name="my_calculator",
        description="简单的数学计算工具，支持基本运算(+,-,*,/)和sqrt函数",
        func=my_calculate,
    )
    return registry
```

用 `test_my_calculator.py` 测：直接计算、与 SimpleAgent 集成。图 7.1 展示工作流：用户提问 → SimpleAgent → 工具注册表（可选）→ 结果回填 → LLM 组织自然语言回答。

跟练：`code/chapter7/my_calculator_tool.py`。

### 7.5.3 多源搜索工具

整合 Tavily（第一章用过）与 SerpApi（第四章用过）：

```bash
pip install "hello-agents[search]==0.1.1"
```

#### 统一接口与 hybrid

`SearchTool(backend="hybrid"|"tavily"|"serpapi")` 在初始化时探测可用后端。

hybrid 逻辑体现高可用：**优先 Tavily → 失败降级 SerpApi → 都不可用则明确提示配置密钥**。

Tavily 侧常取 `answer` + 若干条 `results` 摘要；最终格式化为统一可读字符串，屏蔽不同引擎 JSON 差异。

#### 用类实现自己的高级搜索

`MyAdvancedSearchTool` 展示：需要持有客户端状态、配置、可用源列表时，**类比函数更合适**。`_setup_search_sources` 检查密钥与依赖；`search` 逐源尝试；都失败返回可读错误。再把 `search` 方法通过 `register_function` 挂到注册表。

跟练：`my_advanced_search.py`、`test_advanced_search.py`。

### 7.5.4 工具系统的高级特性

#### （1）工具链 ToolChain

借鉴第六章「图」的顺序组合。Agent 常需组合多工具完成复杂任务：

```python
chain.add_step("search", "{input}", "search_result")
chain.add_step("my_calculator", "根据以下信息计算相关数值:{search_result}", "calculation_result")
```

`execute` 时 `context["input"] = initial_input`，每步用 `str.format(**context)` 生成工具输入，结果写入 `output_key`。缺变量则失败返回错误字符串。

`ToolChainManager` 负责注册与按名执行多条链。

#### （2）异步 AsyncToolExecutor

耗时 I/O 可用线程池并行：

- `execute_tool_async`：`run_in_executor` 跑同步 `registry.execute_tool`
- `execute_tools_parallel`：`asyncio.gather` 多任务

适合多个独立搜索/API；有依赖的步骤仍应用 ToolChain。析构时 `shutdown` 线程池。

#### 设计理念总结

设计层：单一职责、接口统一、完善异常处理、安全优先的输入验证。  
性能层：异步提高并发，合理管理外部连接与资源。

---
## 7.6 本章小结

好消息：本章方法与功能在 GitHub 提供了完整测试案例——[chapter07_basic_setup.py](https://github.com/jjyaoao/HelloAgents/blob/main/examples/chapter07_basic_setup.py)。其中包含多种 Agent 范式演示、工具集成、高级功能与交互体验。可用它校验自己的实现。

回顾：我们一步步构建了基础智能体框架 HelloAgents，始终遵循：

> **分层解耦 · 职责单一 · 接口统一**

具体落地：

- 再次实现经典范式：Simple（基础对话）、ReAct（推理与行动）、Reflection（自我反思）、PlanAndSolve（分解规划）；并引入 FunctionCall（原生函数调用）。
- 工具系统作为能力延伸：从基类、注册，到自定义、多源、链与异步，是一次完整工程实践。

更重要的是：第七章不是终点，而是后续深入学习的技术基础。设计之初已预留扩展点：

- 第八章：记忆与 RAG —— 扩展 Agent 能力边界（仍以 Tool 形态接入）
- 第九章：上下文工程 —— 深化已建立的消息处理机制
- 第十章：智能体协议 —— 扩展新的工具形态

统一 LLM 接口、标准化消息系统、工具注册机制，共同构成完备技术底座。

### 自测清单

- [ ] 能说清自建框架的四类痛点，以及「万物皆工具」的意图
- [ ] 能画出 `core` / `agents` / `tools` 分层并解释职责
- [ ] 会用继承扩展 provider，能口述自动检测优先级
- [ ] 理解 Message / Config / `Agent.run` 契约与「对内丰富对外兼容」
- [ ] 能对比五种范式的适用场景与关键格式约束
- [ ] 能注册自定义工具，并说明 ToolChain 与并行执行的边界
- [ ] 知道完整代码与测试在 `code/chapter7/` 与 GitHub examples

---

## 习题导读

完整题面以原书文末为准。下面是新手向拆解，帮你知道「考什么、从哪答」。

### 第 1 题 · 为何自建与万物皆工具

- 结合第六章或真实项目，举一个「抽象过重 / API 变更 / 黑盒 / 依赖冲突」如何浪费你时间的例子。
- 万物皆工具：优势（概念少、接入统一、教学清晰）；局限（生命周期、状态管理、权限模型可能被过度简化）——各举一例。
- 对比第四章裸实现与本章：统一历史、统一注册表、提示词通用化、可 pip 版本……列出你会坚持的 3 条设计原则。

### 第 2 题 · 扩展 LLM（实践）

- 按 7.2.1 模式继承添加 Gemini/Anthropic 等，并支持专用环境变量检测。
- 双变量优先级：按正文会先命中 `OPENAI_API_KEY`；讨论「是否合理」时要区分「确定性」与「用户直觉」。
- 调研 SGLang，与 VLLM、Ollama 比易用性、资源、速度、精度（可列表）。

### 第 3 题 · Message / Config / Agent

- Pydantic：校验、错误信息、与 IDE/文档工具协作。
- 模板方法：稳定对外 API，变化封装在子类。**注意**：正文抽象的是 `run`；习题里的 `_execute` 是「可扩展写法」，答题时写清对照关系（见 7.3 补缺口）。
- 单例：全局一份配置；不用则多处配置漂移。**注意**：正文 `Config` 未强制单例实现，写清「意图 vs 实现」（见 7.3 补缺口）。

### 第 4 题 · 范式实践

- 框架化 ReAct 的改进例：统一基类与历史、ToolRegistry、提示词格式更严、可配置 max_steps/custom_prompt。
- Reflection + 质量评分：reflect 后让 LLM 打分，低于阈值才 refine。
- Tree-of-Thought：每步多候选 → 选优继续；注意分支预算与终止条件。

### 第 5 题 · 工具系统

- 统一接口保证 Agent/Registry 不用特殊分支；多值可返回拼接字符串，或 JSON 字符串再解析。
- 设计 ≥3 步链（如 搜索→抽取数字→计算→格式化报告）并画流程图。
- 并行：独立 I/O 等待可重叠时划算；有依赖或强 CPU 争用时不划算。

### 第 6 题 · 可扩展性畅想

- 流式：在 LLM 层与 Agent.`stream_run` 贯通，注意工具循环中的流式边界。
- 多轮分支/回溯：会话树节点引用 Message 列表；与现有 `_history` 如何切换。
- 插件系统：发现、注册、版本、沙箱；关键接口画架构图。

---

## 附录 A · 跟练文件速查

| 主题 | 建议文件 |
|------|----------|
| 扩展 LLM | `code/chapter7/my_llm.py`、`my_main.py` |
| SimpleAgent | `my_simple_agent.py`、`test_simple_agent.py` |
| ReAct | `my_react_agent.py`、`test_react_agent.py` |
| Reflection | `test_reflection_agent.py` |
| PlanSolve | `test_plan_solve_agent.py` |
| 计算器工具 | `my_calculator_tool.py`、`test_my_calculator.py` |
| 多源搜索 | `my_advanced_search.py`、`test_advanced_search.py` |

## 附录 B · 资源对应关系

| 资源 | 路径 |
|------|------|
| 原书（完整代码与图） | `docs/chapter7/第七章 构建你的Agent框架.md` |
| 英文版（可选） | `docs/chapter7/Chapter7-Building-Your-Agent-Framework.md` |
| 互动教程（翻页巩固） | `docs/chapter7/interactive-tutorial/index.html` |
| 本学习版（新手地图） | `docs/chapter7/第七章 构建你的Agent框架-学习版.md` |

三者分工：原书保真；学习版讲「为什么 + 怎么学」；互动教程用闪卡与选择题巩固。对照阅读效果最好。

## 附录 C · 原书知识点覆盖自检

- [x] 7.1.1 为何自建（局限 / 跃迁 / 定制）
- [x] 7.1.2 四条设计理念
- [x] 7.1.3 目录树、分层原则、安装、30 秒体验
- [x] 7.2.1 多提供商与继承扩展 MyLLM
- [x] 7.2.2 VLLM / Ollama 接入
- [x] 7.2.3 自动检测与凭证解析、表 7.1 启示
- [x] 7.3.1 Message
- [x] 7.3.2 Config
- [x] 7.3.3 Agent ABC
- [x] 7.4.1 SimpleAgent（工具循环 / 流式 / 动态工具）
- [x] 7.4.2 ReActAgent
- [x] 7.4.3 ReflectionAgent
- [x] 7.4.4 PlanAndSolveAgent
- [x] 7.4.5 FunctionCallAgent
- [x] 表 7.2 范式对比
- [x] 7.5.1 Tool / Parameter / Registry / schema
- [x] 7.5.2 自定义计算器与图 7.1 工作流
- [x] 7.5.3 多源搜索 hybrid
- [x] 7.5.4 ToolChain 与 AsyncToolExecutor
- [x] 7.6 小结与后续章节地图
- [x] 习题 1–6 导读
