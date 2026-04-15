---
name: survey
description: |
  生成高质量学术 Survey 综述笔记，覆盖任意 AI/ML/系统方向主题。结构对标 ACM Computing Surveys、IEEE TPAMI、NeurIPS/ICML Survey Track 高引用综述论文。

  触发条件（遇到以下任意情形立即使用）：
  - 用户说"帮我做一个 XX 的 survey / 综述 / 调研"
  - 用户说"整理一下 XX 领域的论文 / 方法 / 进展"
  - 用户说"我想了解 XX 方向的技术脉络 / 发展历史"
  - 用户提供一批论文链接，希望整理成综述
  - 用户说 /survey 或 @survey

  不触发：用户只想读单篇论文（用 paper skill）；只想做代码实现。
---

# Survey 综述生成器

## 第一步：判断 Survey 类型

根据用户主题，判断属于哪种类型，然后读取对应模板：

| 类型 | 适用主题示例 | 模板文件 |
|------|------------|---------|
| **方法综述型**（Method Survey） | RLHF、LoRA、Chain-of-Thought、Attention、PEFT | `references/templates/method-survey.md` |
| **系统综述型**（System Survey） | LLM Agent、RAG、推理加速、长上下文、MoE | `references/templates/system-survey.md` |
| **数据集/Benchmark 综述型** | Code Benchmark、Alignment 评测、多模态数据集 | `references/templates/dataset-benchmark.md` |
| **领域全景型**（Domain Survey） | Code LLM、Vision-Language Model、具身智能、医疗 LLM | `references/templates/domain-survey.md` |

不确定类型时，默认用**领域全景型**，它覆盖最全面。

## 第二步：读取执行流程

读取 `references/workflow.md`，按 Phase 1-5 依次执行。

## 第三步：质量把关

生成初稿后，必须读取 `references/quality-review.md` 执行评分与迭代，通过后才写入文件。

## 输出

- **存储路径**：`/Users/lijiguo/workspace/awesome-LLM-papers/notes/`
- **命名规范**：`{YYYYMM}-{主题简称}-Survey.md`
- **完成后**：同步更新 README.md 对应分类

## 语言

中文为主，技术术语保留英文（首次出现括注中文）。详见 `references/writing-guide.md`。
