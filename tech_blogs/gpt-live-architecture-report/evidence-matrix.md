# GPT-Live 技术报告证据矩阵

更新日期：2026-08-07。标签含义：`F1` 为 OpenAI 官方公开事实；`F2` 为论文、开源项目或标准中得到验证的公开机制；`H` 为基于 F1/F2 的架构假设，不能写成 GPT-Live 的未披露事实。

| ID | 标签 | 报告中的命题 | 直接证据 | 写作边界 |
|---|---|---|---|---|
| E01 | F1 | GPT-Live 是持续监听、持续决策、允许说听重叠的 full-duplex 系统。 | OpenAI, *Introducing GPT-Live* (2026) | 可陈述产品行为；不可推断内部 token 格式。 |
| E02 | F1 | 音频主路径不依赖传统 turn detector 才能推进生成。 | OpenAI, *How We Built...* (2026) | “无 turn detector in audio path”不等于系统没有任何状态估计。 |
| E03 | F1 | 交互模型持续处理输入输出，复杂推理和工具调用可异步委派给更强模型。 | OpenAI, *Introducing GPT-Live*；*How We Built...* | 可画为 interaction plane 与 cognition plane；连接协议未披露。 |
| E04 | F1 | 媒体 fast path 与应用逻辑之间存在异步 RPC 边界。 | OpenAI, *How We Built...* | RPC 实现、序列化协议、部署拓扑不得臆测。 |
| E05 | F1 | 前端/推理服务由 Python asyncio 迁移到 Go 后，新系统 p95 达到旧系统 p50 水平。 | OpenAI, *How We Built...* | 仅引用相对结论；无公开绝对延迟值。 |
| E06 | F1 | 传输使用 WebRTC，并处理丢包、时钟漂移及音频追赶/拉伸。 | OpenAI, *How We Built...* | 不把某一种 jitter-buffer 算法写成官方实现。 |
| E07 | F1 | 长会话采用有状态推理与模型实例无缝切换：预热替代实例、prefill 上下文、并行运行、切流。 | OpenAI, *How We Built...* | KV 的布局和迁移粒度未披露。 |
| E08 | F1 | 上下文 compaction 被移出实时音频路径；压缩会使原 KV cache 失效，后台预填充替代实例。 | OpenAI, *How We Built...* | 可讨论语义摘要风险；具体压缩模型未知。 |
| E09 | F1 | 委派链路会预创建/预填充 frontier model session，并利用 affinity 与 prompt caching。 | OpenAI, *How We Built...* | 不声称缓存命中策略或路由键细节。 |
| E10 | F1 | 连续语音之上，应用层会推导离散 turns；转写分为 speculative 与 authoritative。 | OpenAI, *How We Built...* | turn 是应用语义抽象，不等同于阻塞式音频门控。 |
| E11 | F1 | WARP 将媒体与 data channel 建连从 6 RTT 压缩到 1 RTT，涉及 SPED、DTLS 1.3、SNAP 和预协商 data channel。 | OpenAI, *How We Built...*；IETF drafts | IETF 草案仍是 work in progress。 |
| E12 | F1 | Instant Connect 通过预协商 SDP，使会话可由一个 UDP packet 启动。 | OpenAI, *How We Built...* | 不能泛化为所有 WebRTC 部署。 |
| E13 | F1 | 容量指标以能否按时处理 frame 的并发 session 为核心，而不是普通 GPU request throughput。 | OpenAI, *How We Built...* | 报告中将 deadline miss 作为核心 SLO 是工程推导。 |
| E14 | F1 | GPT-4o 是单一神经网络统一处理文本、视觉和音频的 omni 模型；公开音频响应最低 232 ms、平均 320 ms。 | OpenAI, *GPT-4o System Card* | 数值属于 GPT-4o 公布条件，不能当作 GPT-Live 延迟。 |
| E15 | F1 | GPT-Live 的安全控制随会话持续运行，可 steer、interrupt、播放安全回复或结束通话。 | OpenAI, *GPT-Live System Card* | 具体分类器数量和阈值未披露。 |
| E16 | F2 | Moshi 使用用户/助手平行音频流、残差量化 codec token 与时间对齐 Inner Monologue。 | Défossez et al. (2024) | 是可比公开实现，不是 GPT-Live 结构证据。 |
| E17 | F2 | SyncLLM 用固定时间 chunk 与同步 token 将 LLM 对齐到真实时钟，并评估最高 240 ms 网络延迟。 | Veluri et al., EMNLP 2024 | 支持“统一时钟”设计轴。 |
| E18 | F2 | Neural-FSM 让 LLM 以控制 token 决定 start/wait/interrupt；报告平均响应延迟相对 half-duplex 降低超过 3 倍。 | Wang et al., NeurIPS 2024 | 结果只适用于论文配置。 |
| E19 | F2 | LSLM 使用 streaming SSL encoder 与 token-based decoder-only TTS；middle fusion 在文中权衡最好。 | Ma et al., AAAI 2025 | 支持 user-stream fusion 讨论。 |
| E20 | F2 | OmniFlatten 以 flatten operation 统一不同模态和任务，采用 modality alignment→half-duplex→full-duplex 三阶段 post-training。 | Zhang et al. (2024) | 支持渐进训练路线，不证明 GPT-Live 相同。 |
| E21 | F2 | Freeze-Omni 保持文本 LLM 冻结，以三阶段训练接入语音输入输出，并用多任务训练获得 duplex 能力。 | Wang et al. (2024) | 作为模块化端到端路线。 |
| E22 | F2 | SALMONN-omni 不向 LLM token 空间注入 codec token，以 dynamic thinking 学习说/听状态转换。 | Yu et al., NeurIPS 2025 | 其“至少 30%”为论文相对结果。 |
| E23 | F2 | Full-Duplex-Bench 将交互能力拆为 pause、backchannel、turn-taking、interruption。 | Lin et al. (2025) | 作为行为维度，不等于线上 QoE 全集。 |
| E24 | F2 | DuplexOmni 明确拆分异步并行的 interaction layer 与可插拔 thinking layer。 | Huang et al. (2026) | 与 GPT-Live 官方描述同构，但两者实现关系未知。 |
| E25 | F2 | DuplexSLA 在共享 160 ms 时间线上联合解码 assistant audio 与限速 action stream。 | Zhang et al. (2026) | 说明“动作原生通道”是另一条路线。 |
| E26 | F2 | 双流路由存在 channel fusion 与 cross-attention 的语义整合/上下文抗污染权衡。 | Lu et al. (2026) | 适合设计消融，不应直接指定 GPT-Live 选择。 |
| E27 | F2 | DuplexChat-Pipe 从播客构建 speaker-separated 数据，公开规模为英语 282,634 h、日语 132,723 h。 | Nakata et al. (2026) | 数值为作者报告；需保留数据许可与污染风险说明。 |
| E28 | F2 | WebRTC 以 SRTP 传媒体，data channel 典型为 SCTP over DTLS over ICE/UDP。 | RFC 8825, RFC 8831 | 标准事实；WARP 是其建连优化。 |
| E29 | F2 | PagedAttention 处理动态增长 KV cache 的碎片与共享问题；FlashAttention 通过 IO-aware tiling 降低 HBM 访问。 | Kwon et al. (2023); Dao et al. (2022) | 用于提出 serving 方案，非 GPT-Live 官方披露。 |
| E30 | H | 最可信的 GPT-Live 架构重建是：低延迟 full-duplex interaction core + 异步 reasoning/tool delegate + deadline-aware stateful serving + WebRTC/WARP transport。 | E01–E13 与 E24 的交集 | 必须标为“证据约束下的重建”。 |
| E31 | H | 交互核心可能采用双输入/输出时间格或等价并行流表示，并维护 self-playback/echo 相关状态。 | E01、E06、E16–E26 | 官方未公布 tokenizer、fusion 或 echo reference 注入方式。 |
| E32 | H | 端到端模型与级联系统不是二元选择：GPT-Live 更像实时交互端到端、复杂认知模块化的混合架构。 | E03、E04、E24、E25 | 是架构解释，不是官方命名。 |

## 使用规则

1. 正文中的关键数字必须紧跟引用；跨模型数字不得横向直接比较，除非数据集、硬件与延迟定义一致。
2. `H` 类命题在标题、图注和正文首次出现时都使用“推测”“重建”或“可能”字样。
3. GPT-Live 未公开的参数量、audio tokenizer、frame rate、训练数据规模、RL 配方、attention 拓扑和集群规模一律留空，不以相邻工作填充。
4. IETF Internet-Draft 只作为正在演进的协议设计，不以正式 RFC 的语气表述。
