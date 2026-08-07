# LLM Scaling Law：从经验幂律、Compute-Optimal 到数据与推理扩展

这是一个可直接编辑和重新编译的中文 LaTeX 技术报告项目。报告面向 LLM 入门者与预训练工程实践，梳理 scaling law 的问题起源、数学形式、技术演进、拟合方法和决策边界，并以 2020--2026 年 32 个公开模型家族验证其工程实践。

作者：Jiguo Li（jiguolee@gmail.com）。报告在 OpenAI Codex 的协助下完成，相关说明已作为标题页脚注写入正文。

## 文件说明

- `llm-scaling-law-paper.tex`：自包含的完整 LaTeX 源文件，包括全局格式、全部正文、公式、代码示例、附录和 50 条参考文献；不依赖外部章节或 BibTeX 文件。
- `llm-scaling-law-paper.pdf`：最终编译版 PDF，共 13 页。
- `build/`、`rendered/`、`tmp/`：可用于本地编译和视觉检查的中间目录，不属于报告交付文件。

## 证据等级

- `A`：同架构、同 tokenizer、同数据分布，系统改变参数量、训练 token 或计算量，并公开 loss 或 checkpoints；适合拟合 scaling law。
- `B`：同代模型家族，关键训练信息公开，但尺寸点较少，或 token/recipe 随尺寸改变；适合趋势校验，不宜直接拟合通用系数。
- `C`：跨代模型或单一 endpoint，架构、数据、多模态或 post-training 改动显著；只能作为工程趋势或披露边界。

附录中的 `A+`、`A-`、`B+`、`B-`、`C+` 表示同一等级内部的相对证据强弱，不改变上述适用边界。公开模型尺寸表不是一条 universal scaling curve；跨家族参数、token 和 benchmark 散点不能识别参数规模的独立因果效应。

## 编译

本项目使用 XeLaTeX 语义和 `ctexart` 中文文档类。当前 PDF 由 Tectonic 0.17.0 生成：

```bash
mkdir -p build
tectonic --outdir build --keep-logs llm-scaling-law-paper.tex
```

若使用完整 TeX Live，也可以运行：

```bash
latexmk -xelatex -interaction=nonstopmode llm-scaling-law-paper.tex
```

当前排版使用 Times New Roman、Arial、Menlo 和 Songti SC。首次在非 macOS 环境编译时，需要安装这些字体或在源文件中替换为可用的等价字体；字体替换可能改变分页和长表格换行。
