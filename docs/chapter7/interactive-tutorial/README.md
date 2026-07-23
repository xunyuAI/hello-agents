# 第七章 · 构建你的 Agent 框架 · 互动教程

面向零基础/新手的浏览器互动学习页，覆盖 Hello-Agents 第七章全部知识点。与第 1–6 章同属「工业蓝图实验室」产品线（Lab 07 **品红 / 玫红**信号色）。

## 如何本地打开

1. 用 Finder 进入本目录，双击 `index.html`（多数浏览器可直接打开）。
2. 或在终端执行：

```bash
open "/Users/mima0000/Desktop/ai/project/hello-agents/docs/chapter7/interactive-tutorial/index.html"
```

3. 原书配图（表 7.1、表 7.2、图 7.1）通过 GitHub raw 链接加载（需联网）；离线时会自动降级为文字要点卡。封面示意图在本地 `assets/ch7_cover.svg`。
4. 建议与学习版 MD 对照：`../第七章 构建你的Agent框架-学习版.md`（含 exceptions / 习题措辞等补缺口说明）。

## 操作提示

- 左侧目录跳转；顶部「上一页 / 下一页」
- 键盘 `←` `→`（或 PageUp / PageDown）翻页
- 可翻转概念卡、展开卡片、闪卡、选择题、架构/循环步进请点击交互
- 手机：点「目录」打开侧栏

## 与前六章的关系

- 第一章：`docs/chapter1/interactive-tutorial/`（初识智能体）
- 第二章：`docs/chapter2/interactive-tutorial/`（智能体发展史）
- 第三章：`docs/chapter3/interactive-tutorial/`（大语言模型基础）
- 第四章：`docs/chapter4/interactive-tutorial/`（智能体经典范式构建）
- 第五章：`docs/chapter5/interactive-tutorial/`（低代码平台搭建）
- 第六章：`docs/chapter6/interactive-tutorial/`（框架开发实践）
- 本章接续同一视觉语言与交互模式，主题转向 **自建 HelloAgents**：核心抽象、LLM 枢纽、范式框架化、工具系统。可独立打开，按课程顺序学习即可。

## 代码实操补充

教程内为知识点与关键片段可视化；完整可运行项目见课程仓库 `code/` 目录与官方 examples（如 `chapter07_basic_setup.py`）。正文只讲原理与设计意图。

## 文件结构

```
interactive-tutorial/
├── index.html      # 主教程（单文件自包含逻辑与样式）
├── assets/         # 本地配图（封面 SVG）
│   └── ch7_cover.svg
└── README.md       # 本说明
```

## 源文档

- 主源：`../第七章 构建你的Agent框架.md`
- 学习版（新手详解 / 补缺口 / 习题导读）：`../第七章 构建你的Agent框架-学习版.md`
- 英文补充（可选）：`../Chapter7-Building-Your-Agent-Framework.md`
