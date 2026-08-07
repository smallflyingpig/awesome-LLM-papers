# 一文详解 GPT-Live 的技术架构与实现方案

这是一个可直接编辑和重新编译的中文 LaTeX 技术报告项目。

作者：Jiguo Li（jiguolee@gmail.com）。报告在 OpenAI Codex 的协助下完成，相关说明已作为标题页脚注写入正文。

## 文件说明

- `main.tex`：论文入口与全局格式。
- `sections/`：按主题拆分的正文，共 10 个一级章节。
- `references.bib`：38 条参考文献，优先使用官方材料、论文原文和 IETF 标准/草案。
- `evidence-matrix.md`：32 条关键命题的证据类型、来源与写作边界。
- `quality-review.md`：交付前质量评分与自动检查结果。
- `gpt-live-architecture-report.pdf`：最终编译版 PDF。
- `rendered/`：本地视觉抽检的生成目录，由 Git 忽略。
- `build*/`、`tmp/`：LaTeX 编译与 PDF 检查的中间目录，由 Git 忽略。

## 证据标签

- `[F1]`：OpenAI 官方公开事实。
- `[F2]`：论文、开源项目或标准中可验证的公开机制。
- `[H]`：由 F1/F2 约束的架构假设，不能视作 GPT-Live 未披露内幕。

## 编译

本项目使用 XeLaTeX 语义和 `ctexart` 中文文档类。当前 PDF 由 Tectonic 0.16.9 生成：

```bash
tectonic -X compile --outdir build --outfmt pdf --keep-logs main.tex
```

若使用完整 TeX Live，也可以运行：

```bash
latexmk -xelatex -interaction=nonstopmode main.tex
```

首次在另一台机器编译时，中文字体可能与 macOS 默认宋体略有差异，但正文、TikZ 图和 BibTeX 均为可编辑源文件。
