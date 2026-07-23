# 第十四章 · 自动化深度研究智能体 · 互动教程

面向零基础/新手的浏览器互动学习页，覆盖 Hello-Agents 第十四章全部知识点。与第 1–13 章同属「工业蓝图实验室」产品线（Lab 14 **翠绿 / 青绿**信号色）。

## 如何本地打开

1. 用 Finder 进入本目录，双击 `index.html`（多数浏览器可直接打开）。
2. 或在终端执行：

```bash
open "/Users/mima0000/Desktop/ai/project/hello-agents/docs/chapter14/interactive-tutorial/index.html"
```

3. 原书配图（图 14.1–14.10、表 14.1–14.2）通过 GitHub raw 链接加载（需联网）；离线时会自动降级为文字要点卡。封面示意图在本地 `assets/ch14_cover.svg`。

## 操作提示

- 左侧目录跳转；顶部「上一页 / 下一页」
- 键盘 `←` `→`（或 PageUp / PageDown）翻页
- 可翻转概念卡、展开卡片、闪卡、选择题、架构/三阶段/协作/工具/SSE 步进器
- 手机：点「目录」打开侧栏

## 与前后章的关系

- 第十三章：旅行助手多智能体产品体验
- **本章**：知识密集型「深度研究」——规划拆题、多源检索、可引用报告
- 第十五章预告：赛博小镇（游戏引擎 + 多 Agent 社会）

## 代码实操补充

教程内为知识点与关键片段可视化；完整工程见课程仓库 `helloagents-deepresearch/`（backend FastAPI + frontend Vue3）。运行需配置 `.env` 中的 LLM 与搜索 API。

## 文件结构

```
interactive-tutorial/
├── index.html      # 主教程（约 40 页，单文件自包含）
├── assets/
│   └── ch14_cover.svg
└── README.md
```

## 源文档

- 主源：`../第十四章 自动化深度研究智能体.md`
- 学习版：`../第十四章 自动化深度研究智能体-学习版.md`
- 英文补充（可选）：`../Chapter14-Automated-Deep-Research-Agent.md`
