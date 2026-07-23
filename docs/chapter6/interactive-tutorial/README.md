# 第六章 · 框架开发实践 · 互动教程

面向零基础/新手的浏览器互动学习页，覆盖 Hello-Agents 第六章全部知识点。与第 1–5 章同属「工业蓝图实验室」产品线（Lab 06 天蓝 / 钴蓝信号色）。

## 如何本地打开

1. 用 Finder 进入本目录，双击 `index.html`（多数浏览器可直接打开）。
2. 或在终端执行：

```bash
open "/Users/mima0000/Desktop/ai/project/hello-agents/docs/chapter6/interactive-tutorial/index.html"
```

3. 原书配图（图 6.1–6.3、表 6.1）通过 GitHub raw 链接加载（需联网）；离线时会自动降级为文字要点卡。封面示意图在本地 `assets/ch6_cover.svg`。

## 操作提示

- 左侧目录跳转；顶部「上一页 / 下一页」
- 键盘 `←` `→`（或 PageUp / PageDown）翻页
- 可翻转概念卡、展开卡片、闪卡、选择题、流程步进请点击交互
- 手机：点「目录」打开侧栏

## 与前五章的关系

- 第一章：`docs/chapter1/interactive-tutorial/`（初识智能体）
- 第二章：`docs/chapter2/interactive-tutorial/`（智能体发展史）
- 第三章：`docs/chapter3/interactive-tutorial/`（大语言模型基础）
- 第四章：`docs/chapter4/interactive-tutorial/`（智能体经典范式构建）
- 第五章：`docs/chapter5/interactive-tutorial/`（低代码平台搭建）
- 本章接续同一视觉语言与交互模式，主题转向 **AutoGen / AgentScope / CAMEL / LangGraph** 框架实践。可独立打开，按课程顺序学习即可。

## 代码实操补充

教程内为知识点与关键片段可视化；完整可运行项目见课程仓库 `code/` 目录（正文只讲原理）。

## 文件结构

```
interactive-tutorial/
├── index.html      # 主教程（单文件自包含逻辑与样式）
├── assets/         # 本地配图（封面 SVG）
│   └── ch6_cover.svg
└── README.md       # 本说明
```

## 源文档

- 主源：`../第六章 框架开发实践.md`
- 新手学习版：`../第六章 框架开发实践-学习版.md`
- 英文补充（可选）：`../Chapter6-Framework-Development-Practice.md`

## 篇幅说明

- 互动教程约 **100 页**（由约 41 页提纲级扩写为知识点详解，约 2.5 倍）
- 覆盖原书 6.1–6.6 全部小节、四框架案例、习题导读与参考文献
- 与学习版 Markdown 互补：HTML 偏可视化翻页，学习版偏线性精读与附录加厚
