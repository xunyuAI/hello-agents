# 第十五章 · 构建赛博小镇 · 互动教程

面向零基础/新手的浏览器互动学习页，覆盖 Hello-Agents 第十五章全部知识点。与第 1–14 章同属「工业蓝图实验室」产品线（Lab 15 **赛博霓虹品红**信号色）。

## 如何本地打开

1. 用 Finder 进入本目录，双击 `index.html`（多数浏览器可直接打开）。
2. 或在终端执行：

```bash
open "/Users/mima0000/Desktop/ai/project/hello-agents/docs/chapter15/interactive-tutorial/index.html"
```

3. 原书配图（图 15.1–15.15 等）通过 GitHub raw 链接加载（需联网）；离线时会自动降级为文字要点卡。封面示意图在本地 `assets/ch15_cover.svg`。

## 操作提示

- 左侧目录跳转；顶部「上一页 / 下一页」
- 键盘 `←` `→`（或 PageUp / PageDown）翻页
- 可翻转概念卡、展开卡片、闪卡、选择题、架构/对话流/记忆/玩家/接线步进、自测清单
- 手机：点「目录」打开侧栏

## 与前面章节的关系

- 第七章：`SimpleAgent` 对话循环
- 第八章：`WorkingMemory` / `EpisodicMemory`
- 本章：把 Agent + 记忆接到 **Godot 像素办公室**，并加上好感度与 FastAPI

## 代码实操补充

教程内为知识点与关键片段可视化；完整工程见：

```
code/chapter15/Helloagents-AI-Town/
├── helloagents-ai-town/   # Godot
└── backend/               # FastAPI + HelloAgents
```

运行需配置 LLM 的 `.env`，安装后端依赖后启动 `python main.py`，再用 Godot 打开主场景按 `F5`。

## 文件结构

```
interactive-tutorial/
├── index.html      # 主教程（单文件自包含逻辑与样式）
├── assets/         # 本地配图（封面 SVG）
│   └── ch15_cover.svg
└── README.md       # 本说明
```

## 源文档

- 主源：`../第十五章 构建赛博小镇.md`
- 学习版：`../第十五章 构建赛博小镇-学习版.md`
- 英文补充（可选）：`../Chapter15-Building-Cyber-Town.md`
