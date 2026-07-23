# 第十六章 · 毕业设计 · 互动教程

面向零基础/新手的浏览器互动学习页，覆盖 Hello-Agents 第十六章全部知识点。与第 1–15 章同属「工业蓝图实验室」产品线（Lab 16 **毕业设计 / Capstone**，毕业金信号色）。

## 如何本地打开

1. 用 Finder 进入本目录，双击 `index.html`（多数浏览器可直接打开）。
2. 或在终端执行：

```bash
open "/Users/mima0000/Desktop/ai/project/hello-agents/docs/chapter16/interactive-tutorial/index.html"
```

3. 原书配图（图 16.1–16.4）通过 GitHub raw 链接加载（需联网）；离线时会自动降级为文字要点卡。封面示意图在本地 `assets/ch16_cover.svg`。

## 操作提示

- 左侧目录跳转；顶部「上一页 / 下一页」
- 键盘 `←` `→`（或 PageUp / PageDown）翻页
- 可翻转概念卡、展开卡片、闪卡、选择题、选题/Notebook/Review/Agent 步进、**选题自检实验室**
- 手机：点「目录」打开侧栏

## 与前面章节的关系

- 第一章至第十五章：`docs/chapterN/interactive-tutorial/`（部分章节持续补齐中）
- 第 13–15 章是三项完整实战；本章要求你**独立选题并开源提交**
- 可独立打开；建议在完成实战章后再做毕业设计

## 代码实操补充

教程内为流程与规范可视化；完整示例见仓库 `Co-creation-projects/jjyaoao-CodeReviewAgent/` 与原书 16.6。运行需配置 LLM（`.env` 或 Notebook 环境变量），并安装：

```bash
pip install "hello-agents[all]"
```

## 文件结构

```
interactive-tutorial/
├── index.html      # 主教程（单文件自包含逻辑与样式，约 45 页）
├── assets/         # 本地配图（封面 SVG）
│   └── ch16_cover.svg
└── README.md       # 本说明
```

## 源文档

- 主源：`../第十六章 毕业设计.md`
- 英文补充（可选）：`../Chapter16-Graduation-Project.md`
- 学习版笔记：`../第十六章 毕业设计-学习版.md`
