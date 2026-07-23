# 第十三章 · 智能旅行助手 · 互动教程

面向零基础/新手的浏览器互动学习页，覆盖 Hello-Agents 第十三章全部知识点。与第 1–12 章同属「工业蓝图实验室」产品线（Lab 13 **天空蓝 / azure** 信号色）。

## 如何本地打开

1. 用 Finder 进入本目录，双击 `index.html`（多数浏览器可直接打开）。
2. 或在终端执行：

```bash
open "/Users/mima0000/Desktop/ai/project/hello-agents/docs/chapter13/interactive-tutorial/index.html"
```

3. 原书配图（图 13.1–13.8 等）通过 GitHub raw 链接加载（需联网）；离线时会自动降级为文字要点卡。封面示意图在本地 `assets/ch13_cover.svg`。

## 操作提示

- 左侧目录跳转；顶部「上一页 / 下一页」
- 键盘 `←` `→`（或 PageUp / PageDown）翻页
- 可翻转概念卡、展开卡片、闪卡、选择题、请求链路 / 多 Agent / MCP / 加载进度步进
- 手机：点「目录」打开侧栏

## 与前面章节的关系

- 第一章：简易旅行助手演示 Thought-Action-Observation
- 第七–十章：框架、工具、记忆、上下文、通信协议
- **本章**：把上述能力做成完整全栈应用（Vue3 + FastAPI + 四 Agent + 高德 MCP）

## 代码实操补充

教程内为知识点与关键片段可视化；完整项目见课程仓库对应的 `helloagents-trip-planner/`（backend + frontend）。运行需配置 LLM、高德、Unsplash 的 `.env`。

## 文件结构

```
interactive-tutorial/
├── index.html      # 主教程（单文件自包含逻辑与样式）
├── assets/         # 本地配图（封面 SVG）
│   └── ch13_cover.svg
└── README.md       # 本说明
```

## 源文档

- 主源：`../第十三章 智能旅行助手.md`
- 学习版：`../第十三章 智能旅行助手-学习版.md`
- 英文补充（可选）：`../Chapter13-Intelligent-Travel-Assistant.md`
