# 第九章 · 上下文工程 · 互动教程

面向零基础/新手的浏览器互动学习页，覆盖 Hello-Agents 第九章全部知识点。与第 1–8 章同属「工业蓝图实验室」产品线（Lab 09 **珊瑚橙 / 暖琥珀**信号色）。

## 如何本地打开

1. 用 Finder 进入本目录，双击 `index.html`（多数浏览器可直接打开）。
2. 或在终端执行：

```bash
open "/Users/mima0000/Desktop/ai/project/hello-agents/docs/chapter9/interactive-tutorial/index.html"
```

3. 原书配图（图 9.1–9.3）通过 GitHub raw 链接加载（需联网）；离线时会自动降级为文字要点卡。封面示意图在本地 `assets/ch9_cover.svg`。

## 操作提示

- 左侧目录跳转；顶部「上一页 / 下一页」
- 键盘 `←` `→`（或 PageUp / PageDown）翻页
- 可翻转概念卡、展开卡片、闪卡、选择题、GSSC / 长程策略 / 笔记操作 / 安全层 / 案例流程步进、**窗口预算滑块实验室**
- 手机：点「目录」打开侧栏

## 与前八章的关系

- 第一章至第八章：`docs/chapterN/interactive-tutorial/`
- 第八章侧重 **Memory + RAG（记与找）**
- 本章侧重 **Context Engineering（装进窗口：预算、筛选、装配、压缩、JIT）**
- 可独立打开；建议按课程顺序在第八章之后学习

## 代码实操补充

教程内为知识点与关键片段可视化；完整实现见框架包 `hello_agents.context` / `NoteTool` / `TerminalTool`，以及课程仓库对应章节代码。运行需配置 LLM 的 `.env`，并安装：

```bash
pip install "hello-agents[all]==0.2.8"
```

## 文件结构

```
interactive-tutorial/
├── index.html      # 主教程（单文件自包含逻辑与样式）
├── assets/         # 本地配图（封面 SVG）
│   └── ch9_cover.svg
└── README.md       # 本说明
```

## 源文档

- 主源：`../第九章 上下文工程.md`
- 学习版（新手细读 / 复盘）：`../第九章 上下文工程-学习版.md`
- 英文补充（可选）：`../Chapter9-Context-Engineering.md`
