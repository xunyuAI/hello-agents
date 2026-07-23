# 第三章 · 大语言模型基础 · 互动教程

面向零基础/新手的浏览器互动学习页，覆盖 Hello-Agents 第三章全部知识点。当前约 **117** 页互动幻灯（相对初版约 39 页扩至约 3 倍），与原书节级一一对应。与第 1–2 章同属「工业蓝图实验室」产品线（Lab 03 用电青绿 / 信号蓝色号微调）。

## 如何本地打开

1. 用 Finder 进入本目录，双击 `index.html`（多数浏览器可直接打开）。
2. 或在终端执行：

```bash
open "/Users/mima0000/Desktop/ai/project/hello-agents/docs/chapter3/interactive-tutorial/index.html"
```

3. 公式渲染依赖 KaTeX CDN；原书配图通过 GitHub raw 链接加载（需联网）。封面示意图在本地 `assets/ch3_cover.svg`。

## 操作提示

- 左侧目录跳转；顶部「上一页 / 下一页」
- 键盘 `←` `→`（或 PageUp / PageDown）翻页
- 可翻转概念卡、闪卡、选择题、流程步进请点击交互
- 手机：点「目录」打开侧栏

## 与前两章的关系

- 第一章：`docs/chapter1/interactive-tutorial/`（初识智能体）
- 第二章：`docs/chapter2/interactive-tutorial/`（智能体发展史）
- 本章接续同一视觉语言与交互模式，主题转向 LLM 原理、交互与边界。可独立打开，按课程顺序学习即可。

## 完整代码实践

教程内代码为关键片段；完整可运行脚本见课程仓库：

- `code/chapter3/N_gram.py`
- `code/chapter3/Word_Embedding.py`
- `code/chapter3/Transformer.py`
- `code/chapter3/BPE.py`
- `code/chapter3/Qwen.py`

## 文件结构

```
interactive-tutorial/
├── index.html      # 主教程（单文件自包含逻辑与样式）
├── assets/         # 本地配图（封面 SVG）
└── README.md       # 本说明
```

## 源文档

- 主源：`../第三章 大语言模型基础-学习版.md`
- 补充：`../第三章 大语言模型基础.md`
