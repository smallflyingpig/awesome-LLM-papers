# Code Agent 的构建

这是一个可直接阅读、编辑和重新编译的中文 LaTeX 技术报告项目。报告系统梳理 Code Agent 从代码基座训练到可执行软件工程智能体的完整构建路径，并区分模型能力、Agent harness、执行环境、评估系统与 test-time compute 的作用边界。

作者：Jiguo Li（jiguolee@gmail.com）。报告在 OpenAI Codex 的协助下完成，相关说明已作为标题页脚注写入正文。

## 文件说明

- `code-agent-construction-report.tex`：自包含的完整 LaTeX 源文件，包括全局格式、正文、TikZ 架构图、数据与轨迹样例，以及 54 条参考文献；不依赖外部章节或 BibTeX 文件。
- `code-agent-construction-report.pdf`：最终编译版 PDF，共 22 页。
- `README.md`：项目说明、内容范围与编译方式。

## 内容范围

报告覆盖以下主题：

- 从工具调用、Search Agent 和 Deep Research 逐步建立 Agent 概念；
- 代码预训练数据、清洗去重、FIM、repo-level 组织与开发历史数据；
- mid-training、依赖感知数据组织、长上下文扩展与课程设计；
- SFT、偏好学习、verifier、execution RL、process reward、self-play 与在线 rollout；
- Agent Runtime、工具协议、沙箱、上下文管理，以及 Harness 的显式状态机；
- Claude Code 公开控制面的 Harness 架构映射；
- 分层评估、污染治理、等 token 消融、上线门槛和典型应用场景；
- 预训练、mid-training、偏好数据及多类轨迹数据的 JSONL 样例与来源边界。

## 编译

本项目使用 XeLaTeX 语义和 `ctexart` 中文文档类。当前 PDF 由 Tectonic 0.16.9 生成：

```bash
mkdir -p build
tectonic --outdir build code-agent-construction-report.tex
```

若使用完整 TeX Live，也可以运行：

```bash
latexmk -xelatex -interaction=nonstopmode code-agent-construction-report.tex
```

## 字体说明

当前 macOS 构建优先使用 Times New Roman、Songti SC、Heiti SC、Menlo 和 STIX Two Math。系统中的黑体文件可以是 `/System/Library/Fonts/STHeiti Medium.ttc` 与 `/System/Library/Fonts/STHeiti Light.ttc`，但 LaTeX 应通过注册字体族名 `Heiti SC` 选择它们。源码已为缺少这些字体的环境配置可用回退字体；跨平台重新编译时，字形和分页可能略有变化。
