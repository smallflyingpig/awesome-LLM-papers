# 质量评审记录

评审日期：2026-08-07

## 第 1 轮 Review

### 当前版本已覆盖

- GPT-Live 官方架构、与 turn-based omni model 的差异和演进路线。
- 全双工连续推理形式化、音频表示、双流路由、控制/action channel 与异步认知。
- 数据构建和 post-training、KV cache、deadline-aware serving、compaction/handoff。
- WebRTC/WARP、SPED/SNAP/DTLS 1.3、弱网和时钟漂移。
- 四层评测体系、可归因消融、证据约束重建及开源复现方案。

### 质量评分

| 维度 | 分值 | 说明 |
|---|---:|---|
| 文献覆盖度 | 5/5 | 覆盖 2009 人类轮次研究、2021--2023 codec/serving 基础、2024--2026 全双工前沿。 |
| 技术深度 | 5/5 | 包含双流概率过程、码率/KV/延迟/成本公式、调度伪代码。 |
| 分类体系 | 5/5 | 按表示、流路由、动作、认知与系统层分解，避免只按模型名罗列。 |
| 数据准确性 | 4/5 | 关键数字均来自原文；GPT-Live 未公开数字明确留空。 |
| 时间线完整性 | 5/5 | 从级联、turn-based omni、单模型 full-duplex 到 interaction-thinking 解耦。 |
| 批判性分析 | 5/5 | 比较 codec/codec-free、fusion/cross-attention、单/双模型与各自失败模式。 |
| Benchmark/数据集 | 4/5 | 覆盖 Full-Duplex-Bench/v2 和 DuplexChat；公开基准仍处早期。 |
| 可读性 | 5/5 | 15 页技术报告版式，4 张 TikZ 图、3 张表、1 个算法；版式与《Transformer 位置编码的前世今生》保持同一风格。 |
| 引用规范 | 5/5 | 38 条 BibTeX 记录，正文编号链接可追溯。 |
| 引用覆盖率 | 4/5 | 关键公开事实和数字有引用；方案性建议明确为作者设计。 |
| 语言规范 | 4/5 | 中文为主，必要系统/协议/模型标识保留英文。 |
| **总分** | **51/55** | **通过：总分不低于 44，且无单项低于 3。** |

### 局限与处理

- GPT-Live 参数量、tokenizer、frame rate、attention、训练数据及集群规模未公开：不猜数，用 `[H]` 标记可行重建。
- 不同工作的 latency 起点、硬件和网络条件不同：正文禁止直接排行榜式比较。
- IETF SPED/SNAP 为 Internet-Draft：均标明 work in progress，不以正式 RFC 口吻表述。

## 自动与视觉检查

- Tectonic 最终编译退出码：0。
- PDF：15 页；可提取文本 65,435 字符（含空白）。
- 参考文献：38 条；正文引用 key 无缺失。
- 日志：0 个 unresolved citation/reference；0 个 overfull box。
- 模板复核：标准 `ctexart` 10pt 定制；左右 19 mm、上 18 mm、下 19 mm；中文 Songti SC、英文 Times New Roman、代码 Menlo；正文 `linespread=1.13`、段间距 2.5 pt、公式上下间距 6 pt，章节标题使用深蓝色。
- 视觉抽检：第 1、2、3、5、6、7、9、10、13、15 页；自定义深蓝标题页、阅读提要框、独立单页目录、灰色动态页眉、段首缩进、公式、表格、TikZ 图、术语脚注和参考文献未见裁切或重叠。
- 章节结构复核：10 个一级章节中，仅形式化、公开模型设计轴、stateful serving 和证据约束重建保留信息性导读；其余 6 章直接进入首个小节。
- 术语复核：中文负责论述与衔接；token、wall-clock、codec、cross-attention、stateful serving、deadline、session affinity、KV cache 等固定技术表达保留英文，正文不再使用“词元”“墙钟”。
- 缩写复核：RTT、MOS、WER、DTLS 等术语均在首次出现处用脚注给出英文全称和中文释义；WebRTC、VAD/ASR/LLM/TTS、SCTP/DTLS/ICE/UDP 等相关缩写采用合并脚注，兼顾可查阅性与版面紧凑度。
- 研究缘起复核：摘要和第一章自然引入 OpenAI 2026-08-03 工程博客，说明笔者的兴趣来源，并由博客披露的系统边界引出全文技术问题。
- 非阻断警告：窄表格/长 URL 产生 underfull box；Tectonic 读取 macOS 中文系统字体，跨机器字形可能略有差异。
