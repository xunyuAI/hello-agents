# 第十一章 · Agentic-RL · 互动教程

面向零基础/新手的浏览器互动学习页，覆盖 Hello-Agents 第十一章全部知识点。与第 1–10 章同属「工业蓝图实验室」产品线（Lab 11 **翡翠绿 / 奖励信号绿**信号色）。

## 如何本地打开

1. 用 Finder 进入本目录，双击 `index.html`（多数浏览器可直接打开）。
2. 或在终端执行：

```bash
open "/Users/mima0000/Desktop/ai/project/hello-agents/docs/chapter11/interactive-tutorial/index.html"
```

3. 原书配图（图 11.1–11.9、表 11.1–11.9）通过 GitHub raw 链接加载（需联网）；离线时会自动降级为文字要点卡。封面示意图在本地 `assets/ch11_cover.svg`。

## 操作提示

- 左侧目录跳转；顶部「上一页 / 下一页」
- 键盘 `←` `→`（或 PageUp / PageDown）翻页
- 可翻转概念卡、展开卡片、闪卡、选择题、训练流水线 / GRPO 循环 / Pipeline 步进
- 手机：点「目录」打开侧栏

## 与前十章的关系

- 第一章至第十章：`docs/chapterN/interactive-tutorial/`
- 前几章构建范式、工具、记忆、上下文与通信协议
- 本章侧重 **用强化学习训练智能体**（SFT 打底 → GRPO 拔高 → 评估部署）
- 可独立打开；建议按课程顺序在第十章之后学习

## 代码实操补充

教程内为知识点与关键片段可视化；完整可运行示例见原书与框架 `hello_agents.tools.RLTrainingTool`、`hello_agents.rl`。运行需 GPU/显存（或小样本快速测），并安装：

```bash
pip install "hello-agents[rl]==0.2.5"
# 或从源码：pip install -e ".[rl]"
```

## 文件结构

```
interactive-tutorial/
├── index.html      # 主教程（单文件自包含逻辑与样式）
├── assets/         # 本地配图（封面 SVG）
│   └── ch11_cover.svg
└── README.md       # 本说明
```

## 源文档

- 主源：`../第十一章 Agentic-RL.md`
- 英文补充（可选）：`../Chapter11-Agentic-RL.md`
- 新手学习版：`../第十一章 Agentic-RL-学习版.md`
