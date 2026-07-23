# 第十二章 · 智能体性能评估 · 互动教程

面向零基础/新手的浏览器互动学习页，覆盖 Hello-Agents 第十二章全部知识点。与第 1–11 章同属「工业蓝图实验室」产品线（Lab 12 **天蓝 / Azure** 信号色）。

## 如何本地打开

1. 用 Finder 进入本目录，双击 `index.html`（多数浏览器可直接打开）。
2. 或在终端执行：

```bash
open "/Users/mima0000/Desktop/ai/project/hello-agents/docs/chapter12/interactive-tutorial/index.html"
```

3. 原书配图（图 12.1–12.7、表图等）通过 GitHub raw 链接加载（需联网）；离线时会自动降级为文字要点卡。封面示意图在本地 `assets/ch12_cover.svg`。

## 操作提示

- 左侧目录跳转；顶部「上一页 / 下一页」
- 键盘 `←` `→`（或 PageUp / PageDown）翻页
- 可翻转概念卡、展开卡片、闪卡、选择题、步进条请点击交互
- 手机：点「目录」打开侧栏

## 与前后章的关系

- 前面章节已搭建范式、工具、记忆、上下文与训练能力
- 本章补上 **Evaluation System**：BFCL · GAIA · 数据生成质量
- 可独立打开；建议按课程顺序，在具备 SimpleAgent / Tool 基础后再学

## 代码实操补充

教程内为知识点与关键片段可视化；完整可运行项目见课程仓库 `code/chapter12/`（如 `04_run_bfcl_evaluation.py`、`data_generation/`）。运行需配置 LLM API、`HF_TOKEN`（GAIA），并按书中说明处理 `bfcl-eval` / `numpy` 依赖。

## 文件结构

```
interactive-tutorial/
├── index.html      # 主教程（单文件自包含逻辑与样式）
├── assets/         # 本地配图（封面 SVG）
│   └── ch12_cover.svg
└── README.md       # 本说明
```

## 源文档

- 主源：`../第十二章 智能体性能评估.md`
- 学习版：`../第十二章 智能体性能评估-学习版.md`
- 英文补充（可选）：`../Chapter12-Agent-Performance-Evaluation.md`
