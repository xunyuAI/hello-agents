# 第十章 · 智能体通信协议 · 互动教程

面向零基础/新手的浏览器互动学习页，覆盖 Hello-Agents 第十章全部知识点。与第 1–9 章同属「工业蓝图实验室」产品线（Lab 10 **电光青 / 霓虹青绿**信号色）。

## 如何本地打开

1. 用 Finder 进入本目录，双击 `index.html`（多数浏览器可直接打开）。
2. 或在终端执行：

```bash
open "/Users/mima0000/Desktop/ai/project/hello-agents/docs/chapter10/interactive-tutorial/index.html"
```

3. 原书配图（图 10.1–10.11、表 10.1–10.8）通过 GitHub raw 链接加载（需联网）；离线时会自动降级为文字要点卡。封面示意图在本地 `assets/ch10_cover.svg`。

## 操作提示

- 左侧目录跳转；顶部「上一页 / 下一页」
- 键盘 `←` `→`（或 PageUp / PageDown）翻页
- 可翻转概念卡、展开卡片、闪卡、选择题、MCP 握手/工具流步进、A2A 任务与请求生命周期、ANP 三步流、传输方式芯片实验室
- 手机：点「目录」打开侧栏

## 与前九章的关系

- 第一章至第九章：`docs/chapterN/interactive-tutorial/`
- 第九章侧重 **Context Engineering（装进窗口）**
- 本章侧重 **Agent 通信协议（MCP / A2A / ANP）**
- 可独立打开；建议按课程顺序在第九章之后学习

## 代码实操补充

教程内为知识点与关键片段可视化；完整实现见 `hello_agents.protocols` 与 `MCPTool` / `A2ATool` / `ANPTool`。运行需配置 LLM 的 `.env`（Agent 集成场景），并安装：

```bash
pip install "hello-agents[protocol]==0.2.2"
```

使用社区 MCP 服务器时通常还需要本机 Node.js（`npx`）。

## 文件结构

```
interactive-tutorial/
├── index.html      # 主教程（单文件自包含逻辑与样式）
├── assets/         # 本地配图（封面 SVG）
│   └── ch10_cover.svg
└── README.md       # 本说明
```

## 源文档

- 主源：`../第十章 智能体通信协议.md`
- 英文补充（可选）：`../Chapter10-Agent-Communication-Protocols.md`
