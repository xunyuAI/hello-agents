# 第四章 · 智能体经典范式构建 · 互动教程

面向零基础/新手的浏览器互动学习页，覆盖 Hello-Agents 第四章全部知识点（约 **108** 页幻灯片，相对初版约 38 页扩写近 3 倍）。与第 1–3 章同属「工业蓝图实验室」产品线（Lab 04 琥珀信号色 + 紫罗兰辅色微调）。

## 如何本地打开

1. 用 Finder 进入本目录，双击 `index.html`（多数浏览器可直接打开）。
2. 或在终端执行：

```bash
open "/Users/mima0000/Desktop/ai/project/hello-agents/docs/chapter4/interactive-tutorial/index.html"
```

3. 公式渲染依赖 KaTeX CDN；原书配图（图 4.1–4.4）通过 GitHub raw 链接加载（需联网）。封面示意图在本地 `assets/ch4_cover.svg`。

## 操作提示

- 左侧目录跳转；顶部「上一页 / 下一页」
- 键盘 `←` `→`（或 PageUp / PageDown）翻页
- 可翻转概念卡、闪卡、选择题、范式流程步进请点击交互
- 手机：点「目录」打开侧栏

## 内容结构（与原书对齐）

| 分组 | 覆盖小节 |
|------|----------|
| 开篇 | 分水岭、能力边界、造轮子、三种范式地图、路线图 |
| 4.1 | 依赖、.env、HelloAgentsLLM、读代码四问、自测 |
| 4.2 ReAct | 流程/公式/图4.1、工具与 SerpApi、Agent 编码与运行轨迹、特点局限调试 |
| 4.3 Plan-and-Solve | 两阶段/图4.2、Planner/Executor、水果店实例、对比 |
| 4.4 Reflection | 三步循环/图4.3、Memory、三角色 Prompt、素数终稿、成本收益 |
| 4.5+ | 小结、表4.1/图4.4、三坑、自测清单、习题 1–7 引导、文献 |

## 与前三章的关系

- 第一章：`docs/chapter1/interactive-tutorial/`（初识智能体）
- 第二章：`docs/chapter2/interactive-tutorial/`（智能体发展史）
- 第三章：`docs/chapter3/interactive-tutorial/`（大语言模型基础）
- 本章接续同一视觉语言与交互模式，主题转向 **ReAct / Plan-and-Solve / Reflection** 编排思维。可独立打开，按课程顺序学习即可。

## 完整代码实践

教程内代码为关键片段与流程可视化；完整可运行脚本见课程仓库：

- `code/chapter4/llm_client.py`
- `code/chapter4/tools.py`
- `code/chapter4/ReAct.py`
- `code/chapter4/Plan_and_solve.py`
- `code/chapter4/Reflection.py`

## 文件结构

```
interactive-tutorial/
├── index.html      # 主教程（单文件自包含逻辑与样式）
├── assets/         # 本地配图（封面 SVG）
│   └── ch4_cover.svg
└── README.md       # 本说明
```

## 源文档

- 主源：`../第四章 智能体经典范式构建-学习版.md`
- 补充：`../第四章 智能体经典范式构建.md`
