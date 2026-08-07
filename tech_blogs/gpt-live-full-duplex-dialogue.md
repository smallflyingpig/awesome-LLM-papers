# 从 Omni 到 GPT‑Live：关于全双工语音模型的一次技术对话

> 整理时间：2026 年 8 月 6 日  
> 本文由一组关于 GPT‑Live 的连续问答整理而成，保留原始的“我—AI”对话格式。文中明确区分 OpenAI 官方披露、公开论文机制与基于二者的合理推测；GPT‑Live 尚未公开的模型结构和训练方法，不作为已确认事实陈述。

---

> **我**
>
> 总结一下 GPT‑Live 的架构，以及演进路线。

> **AI**

GPT‑Live 的核心变化，是从“等用户说完再处理”的回合制语音，转为“语音持续流动、模型边听边说”的全双工实时系统。

```text
Client
  │  WebRTC / WARP / Instant Connect
  ▼
Media fast path（Go）
  │  连续音频帧，低抖动、非阻塞
  ▼
GPT‑Live voice model（stateful, full-duplex）
  │                 │
  │                 └─ 异步 delegation RPC
  │                     ▼
  │                  GPT‑5.5 + Search / Tools / Reasoning
  │                     ▼
  └──── 持续语音回复 ← 结果注入，不阻塞媒体流

并行的应用侧：
partial transcript + timing → speculative turns → finalized transcript
```

架构可以拆成六层：

- **实时媒体层**：客户端经 WebRTC 收发音频；服务端把媒体流和业务逻辑彻底隔离。媒体前端与推理逻辑使用 Go，避免旧 Python asyncio 实现在连续帧调度上的抖动。
- **连续、全双工语音模型**：GPT‑Live 直接持续接收和生成音频，不再让独立 turn detector 决定“什么时候开始推理”，因此可以处理打断、重叠说话、附和等自然交互。
- **有状态推理层**：会话长期维持 KV cache。模型实例需要迁移或上下文超窗时，后台启动新实例，prefill 当前或压缩后的上下文，双实例并跑，准备好后无缝切流。
- **异步智能增强层**：语音模型负责即时交互；复杂推理、搜索和工具调用委派给 GPT‑5.5。委派在独立异步 RPC 路径上执行，慢工具不会卡住语音。
- **连续语音到离散消息的适配层**：应用服务根据 partial transcript、时序和说话人归属，维护“可变的 speculative view”和最终 authoritative transcript，兼容 UI、分析和安全系统。
- **连接与启动层**：WARP 将 WebRTC 媒体和数据启动从 6 RTT 压到 1 RTT；Instant Connect 预协商 SDP，把 signaling 移出关键路径。目标是用户点击后即可用一个 UDP 包启动实时媒体。

演进路线很清晰：

| 阶段 | 主架构 | 主要瓶颈 | GPT‑Live 的解决方式 |
|---|---|---|---|
| 第一代：级联式语音 | ASR → LLM → TTS | 串行延迟高，音色、停顿等语音信息丢失 | 改为原生 speech-to-speech |
| 第二代：端到端语音但仍回合制 | S2S model + turn detector | detector 早判会打断，晚判则响应慢；推理仍被“回合结束”门控 | 移除音频路径中的 turn detector |
| 第三代：GPT‑Live | 全双工流式 voice model + 状态推理 + 异步 frontier delegation | 长会话状态、工具延迟、网络启动、全球容量与尾延迟 | 双实例切换、预热和缓存 delegation、WARP/Instant Connect、生产 shadow test |

其中最重要的系统设计原则是：**voice must flow**。所有可能变慢的工作——上下文压缩、模型迁移、搜索、工具、持久化——都应移出 live path，并以可异步合并的方式回注结果。

后续路线大致是：

1. 从 ChatGPT Voice 扩展到 agentic coordination，例如桌面端控制电脑、协调多个 agent。
2. 将底层能力产品化为 GPT‑Live API。
3. 扩展到更多设备、应用与模态，同时保持低延迟和全双工交互。
4. 工程上持续强化多地域部署、会话容量模型、逐 engine 的可观测性、配置校验与分级灰度。

值得注意的是，它把容量度量从“单 GPU 能承受多少请求”改成了“系统能同时维持多少条会话，并让每一帧按时到达”。对于实时多模态系统，这是比吞吐更贴近用户体验的指标。

来源：[OpenAI：How we built a realtime system for responsive voice AI in six months](https://openai.com/index/continuous-voice-interaction-with-gpt-live/)

---

> **我**
>
> GPT‑Live voice model 的实现原理是怎样的，与当前的端到端 omni model 有什么区别？

> **AI**

先说结论：GPT‑Live 不是简单把当前端到端 omni model“加一个更好的 VAD”。它把优化目标从“单轮语音理解和生成”换成了“持续双向交互控制”，并把深度推理、搜索和工具调用解耦到后台 frontier model。

但要谨慎：OpenAI 没有公开 GPT‑Live 的具体网络结构、audio tokenizer、attention mask、训练目标或参数规模。因此下面的“实现原理”分为已公开事实和工程上高度合理、但未获官方确认的模型层解释。

```text
持续输入语音 ──► GPT‑Live voice model ──► 持续输出语音
                    │   ▲
                    │   │  同一会话状态 / KV cache
                    ▼
             决策：听、等、附和、插话、回应、委派
                    │
                    └── async delegation ──► GPT‑5.5 + search/tools/reasoning
                                               │
                                      结果异步回注入对话
```

### GPT‑Live voice model 的实现原理

已公开的关键点有三个。

第一，**全双工连续推理（full-duplex continuous inference）**。系统不再输入一个完整 user turn、输出一个完整 assistant turn。模型在持续消费输入音频的同时持续生成输出音频，并能每秒多次决定：继续听、短暂附和、保持沉默、插话、正式回答，或发起工具委派。[OpenAI：Introducing GPT‑Live](https://openai.com/index/introducing-gpt-live/)

第二，**把 turn-taking 收进模型控制环，而不是依赖外部 turn detector**。旧 Advanced Voice 的音频处理已经可以在一个模型内完成，但仍依赖“用户是否说完”的外部离散判定；GPT‑Live 将这一判定变成连续交互策略的一部分。因此它处理的不是二分类 `end_of_turn?`，而是一个在线策略：在当前重叠语音、停顿、语速和语义状态下，什么动作最合适。

第三，**语音交互与深度智能分工**。GPT‑Live 负责实时 conversation policy，GPT‑5.5 负责长链 reasoning、搜索和复杂工具使用。委派异步运行，voice model 用简短对话维持互动，待结果可用时再融合进对话；这也让后台模型可以随代际替换，而不必重训整个语音体验。

模型内部的合理推测是：它可能仍以音频离散表征或连续 speech embedding 为基础，维护一条或多条因果状态流；与传统单向生成不同，训练和 serving 必须让输入音频与输出音频在时间上重叠，并学习“何时发声”这一 interaction policy。**这只是技术推测，不是 OpenAI 已披露的实现。**

### 与端到端 omni model 的区别

以 GPT‑4o 类 omni model 为参照：GPT‑4o 官方定义是单个自回归网络，端到端联合处理 text、audio、image、video，并输出 text、audio、image。[GPT‑4o System Card](https://openai.com/index/gpt-4o-system-card/)

| 维度 | 端到端 omni（如 GPT‑4o） | GPT‑Live |
|---|---|---|
| 核心目标 | 跨模态理解与生成统一 | 人类式连续对话和低交互延迟 |
| 输入/输出范式 | 可端到端多模态，但通常按请求或回合组织 | 持续音频输入与输出，允许重叠 |
| 轮次控制 | 产品层或 turn detector 往往仍参与决定何时回复 | 模型在连续流中反复决定听、说、等、插话 |
| 推理能力来源 | 模型本体承担较多理解与生成 | voice model 保持交互；复杂问题委派给 GPT‑5.5 |
| 系统形态 | 单模型统一模态，但可作为回合制服务 | “快交互模型 + 强后台模型”的异步系统 |
| 长会话 | 主要是常规上下文管理问题 | 专门设计无缝实例迁移、并行 prefill、KV cache 重建和后台 compaction |
| 部署优化 | 更关注单请求首 token 延迟和模态能力 | 关注逐音频帧按时到达、尾延迟、网络握手和多地域会话容量 |

所以，二者并不是“端到端 vs. 非端到端”的简单对立：

- **omni** 强调模态在一个网络中统一建模；
- **GPT‑Live** 强调时间轴上的双向、连续、可打断交互；
- GPT‑Live 的整体产品能力又是一个**系统级组合**：voice model 本身不需要承担与 GPT‑5.5 同等级的深度推理，只要能准确管理交互、判断何时委派并自然接回结果即可。

从模型设计角度看，GPT‑Live 更接近把原来的目标：

```text
audio turn → response audio
```

升级为：

```text
history + incoming_audio(t) + own_audio(t)
  → {next_audio(t+Δ), interaction_action(t), delegation_action(t)}
```

这也是它能在用户停顿时不抢话、在用户插话时及时让出，同时又能在后台完成 agentic work 的根本原因。GPT‑Live System Card 也确认：在没有 delegation 时，GPT‑Live 的能力低于 GPT‑5.5 Thinking，复杂能力由 delegated model 承担。[GPT‑Live System Card](https://deploymentsafety.openai.com/gpt-live/gpt-live.pdf)

---

> **我**
>
> 有哪些公开论文 or 技术报告是介绍全双工连续推理 voice model 的，可以从中一窥 GPT‑Live 的实现方式？

> **AI**

有，而且到 2026 年已经形成了一条相当清晰的技术路线。要“反推”GPT‑Live，我建议优先读：

1. **Moshi**：理解 full-duplex voice backbone。
2. **DuplexOmni**：理解“实时交互模型 + 后台 thinking model”的双层架构，和 GPT‑Live 最像。
3. **SyncLLM**：理解如何给自回归 LLM 引入真实时间轴。
4. **DuplexSLA**：理解连续语音中如何并行产生 planning/tool action。
5. **SALMONN‑omni**：理解不用 audio codec token 的另一条路线。

| 优先级 | 论文 | 核心机制 | 对理解 GPT‑Live 的价值 |
|---|---|---|---|
| S | Moshi | 双音频流、audio codec token、Inner Monologue、流式 KV | 最可能接近 GPT‑Live voice backbone 的公开范式 |
| S | DuplexOmni | Interaction layer 与 Thinking layer 异步并行 | 与 GPT‑Live 的 voice model + GPT‑5.5 delegation 在系统抽象上接近 |
| S | SyncLLM | chunk timeline、同步 token、deadline-aware decoding | 解释模型如何获得“现实时间感” |
| A | DuplexSLA | user audio、assistant audio、action 三通道共享 160 ms 时钟 | 展示 delegation/tool call 如何进一步内生化 |
| A | SALMONN‑omni | codec-free embedding、dynamic thinking、说/听状态切换 | 展示 GPT‑Live 可能采用的另一种底座设计空间 |
| A | Neural FSM | LLM 输出 SPEAK/LISTEN 等控制 token | 最容易实现和验证的工程原型 |
| B | LSLM | listening encoder 与 speaking decoder 的多层融合 | 研究“边说边听”信息如何注入生成模型 |
| B | OmniFlatten | flatten 多流数据，不修改 GPT backbone | 研究如何低成本从 turn-based 模型升级到 full-duplex |

### 1. Moshi：目前最完整的开源 full-duplex 范式

**Moshi: a speech-text foundation model for real-time dialogue**（Défossez et al., arXiv 2024）[arXiv:2410.00037]  
[论文](https://arxiv.org/abs/2410.00037) · [项目与代码](https://github.com/kyutai-labs/moshi)

核心设计：

```text
时间 t:
User audio:      u_sem(t), u_acoustic_1(t)...u_acoustic_K(t)
Assistant audio: a_sem(t), a_acoustic_1(t)...a_acoustic_K(t)
Assistant text:  text(t)
                       │
              Temporal Transformer
                       │
                Depth Transformer
```

关键点：

- User 和 Assistant 语音是两条并行流，不需要先切成离散 turn。
- Mimi codec 用 residual vector quantization 同时表达 semantic 和 acoustic 信息。
- Temporal Transformer 沿时间建模，Depth Transformer 在同一时间步生成多级 codec token。
- `Inner Monologue` 先生成时间对齐的 text token，再生成 assistant audio token，维持语言能力。
- 理论延迟 160 ms，实际约 200 ms。

它最能解释 GPT‑Live 的几个行为：边听边说、backchannel、overlap、barge-in，以及为什么推理系统必须持续维护 state 和 KV cache。

但需要强调：不能据此认定两者采用相同的网络结构或 tokenizer。Moshi 只是目前最有参考价值的公开 full-duplex voice backbone。

### 2. DuplexOmni：系统形态和 GPT‑Live 最接近

**DuplexOmni: Real-Time Listening, Seeing, Thinking, and Speaking for Full-Duplex Interaction**（Huang et al., arXiv 2026）[arXiv:2606.09186]  
[论文](https://arxiv.org/abs/2606.09186)

```text
Streaming audio/video
        │
        ▼
Interaction Layer ───────► realtime speech
        │                       ▲
        └── async request ──► Thinking Layer
                               reasoning/tools
```

它明确拆分：

- Interaction layer：处理流式 audio/video，实时生成 text/speech。
- Thinking layer：可插拔 reasoning/tool-use 模型。
- 两层异步并行，thinking 不阻塞实时说话。
- Writer–Director pipeline 用于生成 continuous interaction 训练数据。

这与 GPT‑Live 官方披露的 `GPT‑Live voice model → asynchronous delegation → GPT‑5.5 + tools` 高度相似。它是目前公开论文里最适合用于推测 GPT‑Live 整体模型分工和训练数据组织的工作。

不过，两者只能说“架构同构”。没有公开证据表明 OpenAI 采用了 DuplexOmni 的具体实现。

### 3. SyncLLM：LLM 如何获得真实时间轴

**Beyond Turn-Based Interfaces: Synchronous LLMs as Full-Duplex Dialogue Agents**（Veluri et al., EMNLP 2024）[arXiv:2409.15594]  
[论文](https://arxiv.org/abs/2409.15594) · [项目页](https://syncllm.cs.washington.edu/)

SyncLLM 将两人语音编码成 HuBERT token，并按固定时间 chunk 交错排列：

```text
[SYNC_0]
user_chunk_0
agent_chunk_0
[SYNC_1]
user_chunk_1
agent_chunk_1
...
```

真正困难的是：到时间片 `N` 时，受网络传播和推理延迟影响，完整的 `user_chunk_N` 可能还没到，但 assistant 必须提前生成 `agent_chunk_N+1`。SyncLLM 为此先预测当前用户 chunk，再生成下一 assistant chunk。

这揭示了 GPT‑Live 很可能也要解决的通用问题：

- 模型推理不只受 token dependency 约束，还受 wall-clock deadline 约束。
- 输入流与生成流存在固定的网络和推理 lag。
- 训练时必须显式模拟 jitter、延迟和不完整输入。
- silence 也必须成为时序中的有效 observation，而不是简单丢弃。

这些是 full-duplex 系统的一般问题，不代表 GPT‑Live 已确认采用 SyncLLM 的具体 token 排列方式。

### 4. DuplexSLA：把 tool/action 加入连续时间轴

**DuplexSLA: A Full-Duplex Spoken Language Model with Synchronized Speech, Language, and Action**（Zhang et al., arXiv 2026）[arXiv:2605.20755]  
[论文](https://arxiv.org/abs/2605.20755)

它在共享的 160 ms chunk timeline 上联合建模三个 channel：

```text
User audio channel       continuous input
Assistant audio channel  discrete audio output
Action channel           planning text / structured tool call
```

同一个 backbone 可以一边输出 assistant speech，一边在 action channel：

- 生成 internal planning；
- 发起 tool call；
- 处理多次 tool action；
- 根据 backchannel 触发工具；
- 决定 pause、interrupt、continue listening。

它与 GPT‑Live 的差异很有意思：GPT‑Live 当前披露的是通过独立 async RPC 委派给 GPT‑5.5；DuplexSLA 则把 action 和 planning channel 内生到 full-duplex backbone。

因此可以把 DuplexSLA 看成一种可能的后续演进方向：保留外部强 reasoning model，但让 voice model 原生生成 delegation 或 action token。

### 5. SALMONN‑omni：codec-free 路线

**SALMONN-omni: A Standalone Speech LLM without Codec Injection for Full-duplex Conversation**（Yu et al., arXiv 2025）[arXiv:2505.17060]  
[论文](https://arxiv.org/abs/2505.17060)

它针对 Moshi 类方法的一个问题：大量 audio codec token 会占据 LLM token space 和推理预算，并可能削弱文本语言能力。

SALMONN‑omni 改为：

- 输入输出主要依赖连续 speech embedding，而不是把 codec token 全部注入 LLM vocabulary。
- 在 LLM 内部引入 dynamic thinking mechanism。
- 模型学习何时处于 listening、thinking、speaking 状态。
- 支持 context-dependent barge-in、backchannel、echo cancellation。
- 后续通过 RL 优化交互行为。

如果 GPT‑Live 追求大规模部署成本，codec-free 或低码率 semantic representation 是值得考虑的设计空间；不过官方没有披露它到底使用 codec token 还是连续 embedding。

### 6. Neural FSM：最容易复现的系统原型

**A Full-duplex Speech Dialogue Scheme Based On Large Language Models**（Wang et al., NeurIPS 2024）[arXiv:2405.19487]  
[论文](https://arxiv.org/abs/2405.19487)

它不是一个完全统一的 speech foundation model，而是：

```text
Streaming perception
        │
        ▼
LLM → text token + control token
        │
        ▼
Neural FSM: LISTEN / SPEAK
        │
        ▼
Streaming speech generation
```

LLM 对实时对话序列做 next-token prediction，并输出控制 token，决定等待、开始说、打断用户或响应用户。

它适合用来理解 GPT‑Live 中尚未公开的 `interaction action space`。GPT‑Live 官方说模型每秒多次决定 listen、speak、pause、interrupt 或 tool-use，本质上可以抽象为从二状态 FSM 扩展到多动作 policy。

### 7. LSLM：研究 listen stream 应该如何注入

**Language Model Can Listen While Speaking**（Ma et al., AAAI 2025）[arXiv:2408.02622]  
[论文](https://arxiv.org/abs/2408.02622)

模型包含：

- streaming SSL encoder：持续编码用户语音；
- decoder-only token TTS：持续生成 assistant 语音；
- early、middle、late fusion：研究用户流在哪一层注入 speaking model。

论文发现 middle fusion 在语音生成质量与实时交互能力之间较为均衡。

这对应一个很现实的问题：用户突然插话时，新的 user representation 应该怎样更新正在执行自回归解码的 assistant？是直接混入 input token，还是作为 cross-attention memory？

2026 年的 **How Should LLMs Listen While Speaking? A Study of User-Stream Routing in Full-Duplex Spoken Dialogue**（Lu et al., arXiv 2026）[arXiv:2605.10199] 又专门对比了 channel fusion 和 cross-attention routing，值得接着读。[论文](https://arxiv.org/abs/2605.10199)

### 8. OmniFlatten：研究 full-duplex post-training recipe

**OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation**（Zhang et al., arXiv 2024）[arXiv:2410.17799]  
[论文](https://arxiv.org/abs/2410.17799)

它最大的价值不在 backbone 创新，而在训练路线：

1. Modality alignment；
2. Half-duplex dialogue learning；
3. Full-duplex dialogue learning。

通过 flattening operation 把不同模态和双通道时间序列转成普通 GPT 可以处理的序列，不需要大幅修改 backbone。

这可能是复现 GPT‑Live 类模型成本较低的公开路线：先保证 turn-based speech-to-speech 质量，再使用包含 overlap、pause、backchannel、interruption 的双轨数据做 full-duplex post-training。

### 训练数据和评测

**DuplexChat: Constructing Speaker-Separated Full-Duplex Dialogue Speech at Scale for Spoken Dialogue Language Modeling**（Nakata et al., arXiv 2026）[arXiv:2607.04941]  
[论文](https://arxiv.org/abs/2607.04941)

它解决 full-duplex 最大的数据瓶颈：普通 podcast 或对话数据通常是单声道混音，无法形成独立的 user/assistant 时间流。其 pipeline 使用 diarization、speech separation 和 restoration 构造 speaker-separated 双通道语料。

GPT‑Live 类模型如果采用 learned turn-taking，训练数据至少需要保留：

- 双方独立 waveform；
- overlap；
- backchannel；
- pause length；
- interruption onset；
- 谁最终获得 floor；
- assistant 正在说话时接收到的用户语音。

**Full-Duplex-Bench: A Benchmark to Evaluate Full-duplex Spoken Dialogue Models on Turn-taking Capabilities**（Lin et al., arXiv 2025）[arXiv:2503.04721]  
[论文](https://arxiv.org/abs/2503.04721)

它覆盖 pause handling、backchanneling、turn-taking 和 interruption management。这是判断一个模型“真的 full-duplex”，还是仅仅“VAD 更快”的最低必要评测。

综述入口是 **From Turn-Taking to Synchronous Dialogue: A Survey of Full-Duplex Spoken Language Models**（Chen et al., arXiv 2025）[arXiv:2509.14515]：  
[论文](https://arxiv.org/abs/2509.14515) · [持续更新的论文清单](https://github.com/elpsykongloo/FD-SLMs)

它把现有方法分成：

- Engineered Synchronization：VAD、FSM、独立 interrupter 或 conversation predictor；
- Learned Synchronization：端到端学习时间、重叠和交互策略。

### 从这些论文反推 GPT‑Live

综合公开信息，GPT‑Live 的可能形态大致是：

```text
                    ┌─ user semantic/acoustic stream
Streaming encoder ──┤
                    └─ timing / silence / overlap state
                              │
                       Voice backbone
                    continuous state + KV
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
        assistant audio   interaction action   delegation action
                          listen/speak/pause         │
                          interrupt/backchannel      ▼
                                               GPT-5.5/tools
                                                    │
                                             async result injection
```

最值得验证的五个假设是：

1. **双流建模**：可能类似 Moshi，将 user audio 与 assistant self-audio 分开建模。
2. **固定时钟**：可能类似 SyncLLM 或 DuplexSLA，以几十到数百毫秒的 chunk 周期更新交互决策。
3. **隐式或显式 action head**：可能类似 Neural FSM，但 action space 更丰富。
4. **语音与深度思考解耦**：系统抽象类似 DuplexOmni，voice backbone 和 frontier reasoning model 异步并行。
5. **交互专项 post-training**：可能类似 OmniFlatten，并加入 RL 优化抢话率、响应延迟和 backchannel appropriateness。

这些都是公开论文与 OpenAI 系统披露之间的合理映射，不是 GPT‑Live 的已确认模型结构。

如果只读三篇，我会选 **Moshi → DuplexOmni → DuplexSLA**：分别覆盖 voice backbone、GPT‑Live 式系统分工，以及下一代 agentic full-duplex action channel。

---

## 参考资料索引

- [OpenAI：How we built a realtime system for responsive voice AI in six months](https://openai.com/index/continuous-voice-interaction-with-gpt-live/)
- [OpenAI：Introducing GPT‑Live](https://openai.com/index/introducing-gpt-live/)
- [OpenAI：GPT‑Live System Card](https://deploymentsafety.openai.com/gpt-live/gpt-live.pdf)
- [OpenAI：GPT‑4o System Card](https://openai.com/index/gpt-4o-system-card/)
- [Moshi](https://arxiv.org/abs/2410.00037)
- [SyncLLM](https://arxiv.org/abs/2409.15594)
- [A Full-duplex Speech Dialogue Scheme Based On Large Language Models](https://arxiv.org/abs/2405.19487)
- [OmniFlatten](https://arxiv.org/abs/2410.17799)
- [SALMONN‑omni](https://arxiv.org/abs/2505.17060)
- [DuplexOmni](https://arxiv.org/abs/2606.09186)
- [DuplexSLA](https://arxiv.org/abs/2605.20755)
- [Full-Duplex-Bench](https://arxiv.org/abs/2503.04721)
- [Full-Duplex Spoken Language Models Survey](https://arxiv.org/abs/2509.14515)
