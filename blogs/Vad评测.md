---
layout: post
title: 开源 VAD 模型评测笔记：5 个模型 × 5 个数据集 × 57 小时音频
description: WebRTC / Silero / FunASR-FSMN / FireRedVAD / pyannote 五种VAD模型的横评
date: 2026-09-18
tags: [语音识别, VAD, 模型评测, 工程实践]
---

## 使用的模型 & 评测数据集

VAD模型通常作为ASR模型的前置模型用来过滤非语音的音频，用来确保输入给ASR模型是有效的语音帧，虽然最近的LLM based的语音转写系统的上下文越来越长。但是在动辄长达数小时的会议转写系统中，和实际部署对于并发和资源的要求限制等，往往在实际生产中，会有选择的进行VAD分段。此次实验，是在早期选择VAD模型的一次简单的选型实验。本次实验主要从识别准确率和推理性能两个维度进行展开。

简单调研了一下VAD模型的开源情况，选取了五个比较主流的VAD模型，比较特殊的是webrtc，是基于传统方法做的。五个模型基本信息如下：

| 代号 | 类型 | 权重体积 | 一句话 | 仓库 |
|------|------|---------:|--------|------|
| `webrtc` | 传统能量/频谱 | 无 | Google WebRTC 的经典 VAD，纯 C，无独立权重 | [wiseman/py-webrtcvad](https://github.com/wiseman/py-webrtcvad) |
| `silero` | 轻量 DNN | 2.22 MB | Silero 的开源 DNN VAD，ONNX 推理 | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) |
| `funasr_fsmn` | FSMN | 1.65 MB | 达摩院 FunASR 的 FSMN-VAD，中文会议/对话常用 | [modelscope/FunASR](https://github.com/modelscope/FunASR) |
| `firered` | mel + 检测网络 | 2.28 MB | 小红书 FireRedVAD（非流式） | [FireRedTeam/FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) |
| `pyannote` | 端到端分割 | 5.64 MB | `segmentation-3.0` 骨干 + VoiceActivityDetection 后处理 | [pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio) |

统一接口的设计很朴素：**输入**是 wav 路径或 float32 单通道波形，**内部**统一重采样到 16 kHz 单通道，**输出**是 `list[SpeechSegment(start, end)]`，单位秒。

这里有一个刻意的取舍：**不做帧级概率 API**。pyannote 和 FireRed 内部其实都有帧级分数，但 webrtc 和 FunASR 的封装并不统一暴露，硬凑出来就是「有的模型用真概率、有的模型用段边界反推的伪概率」，算出来的 AUC 没有可比性。所以这次统一只比**段级输出栅格化到 10 ms 帧**之后的结果。

其中评测集还是选取几个会议场景的数据集规模不是很大，主要目标就是简单跑出一组对比的数据，有个直观的感受，数据集基本信息如下：

| 代号 | 语言 / 场景 | 规模 | 标注来源 | 来源 |
|------|-------------|------|----------|------|
| AliMeeting test far | 中 / 会议远场 | 20 场 · 10.78 h | RTTM | [OpenSLR 119](https://www.openslr.org/119/) |
| AliMeeting eval far | 中 / 会议远场 | 8 场 · 4.21 h | TextGrid → RTTM | 同上 |
| AISHELL-4 test | 中 / 会议（麦克风阵列） | 20 场 · 12.73 h | RTTM | [OpenSLR 111](https://www.openslr.org/111/) |
| MagicData-RAMC test | 中 / 手机对话近场 | 43 场 · 20.64 h | 语音活动时间戳 → RTTM | [OpenSLR 123](https://www.openslr.org/123/) |
| AMI IHM test | 英 / 会议近场 | 16 场 · 9.06 h | 段级时间戳 | [AMI Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) |

**GT 统一口径**：把各说话人的时间段**取并集**，得到 speech / non-speech 二分类。这样做的好处是所有数据集都能映射到同一个定义上，坏处是它丢失了「谁在说」和「重叠语音」的信息——一个 VAD 把重叠段判成语音，在这里是正确行为。


## 评测指标

### 段 → 帧

预测段和参考段都按 **10 ms** 栅格化成二值帧序列（1 = 语音，0 = 非语音），逐帧对比得到混淆计数：

| 符号 | 条件 | 含义 |
|------|------|------|
| TP | 参考=1 且 预测=1 | 正确检出的语音帧 |
| FP | 参考=0 且 预测=1 | 把非语音判成语音（虚警） |
| TN | 参考=0 且 预测=0 | 正确的非语音帧 |
| FN | 参考=1 且 预测=0 | 漏检的语音帧 |

```text
P     = TP / (TP + FP)          # Precision：预测为语音的帧中，真语音占比
R     = TP / (TP + FN)          # Recall：参考语音帧中，被检出占比
F1    = 2 * P * R / (P + R)     # 当 P+R=0 时 F1=0
      = 2*TP / (2*TP + FP + FN) # 与上一行等价

FAR   = FP / (FP + TN)          # 非语音帧上的虚警率
Miss  = FN / (TP + FN)          # 语音帧上的漏检率 = 1 − R
```

两个容易写错的地方：

1. **FAR 的分母是「全部非语音帧」`FP+TN`**，不是 `TP+FP`。它就是标准的 FPR。
2. **Miss 和 R 互补**，`Miss = 1 − R`，所以表里同时给这两个指标其实是冗余的，放在一起只是为了方便看漏检。

### 聚合口径：时长加权还是文件平均？

汇总时有两条路：

- **时长加权**：先把所有文件的 TP/FP/TN/FN 分别求和，再套公式（长文件权重大）；
- **文件平均**：每个文件先算 F1，再对文件取算术平均（每个会议等权）。

哪条更「对」取决于你的业务，但**它们会给出不同的结论**。看 AliMeeting test 这个例子：

| 引擎 | 时长加权 F1 | 文件平均 F1 | 中位数 F1 |
|------|-----------:|-----------:|---------:|
| pyannote | **0.993** | **0.993** | 0.994 |
| firered | 0.954 | 0.950 | 0.961 |
| funasr_fsmn | 0.948 | 0.939 | 0.970 |
| silero | **0.889** | 0.856 | **0.941** |
| webrtc | **0.880** | **0.858** | 0.928 |

注意 silero 和 webrtc 这一对：**时长加权下 silero 领先 0.9 个点，文件平均下 webrtc 反超 0.2 个点，中位数下 silero 又领先 1.3 个点**。同一批数据、同一个指标，聚合方式一换，谁赢就变了。

原因也不难猜：silero 在某个短文件上直接崩了（F1 = 0.074），文件平均被这个离群值拖死，而时长加权把它稀释掉了。所以我的建议是：**报 F1 的时候必须写清楚聚合口径，最好连中位数和最差文件一起报**，否则「F1 = 0.89」这个数字几乎无法解读。

本文下面所有主表都用**时长加权**，并额外补一节每文件分布。CSV 里 `*_unweighted` 列就是文件平均的结果。

## 评测结果

### 跨数据集一览（时长加权 F1）

| 引擎 | AliMeeting test | AliMeeting eval | AISHELL-4 | MagicData-RAMC | AMI |
|------|----------------:|----------------:|----------:|---------------:|----:|
| pyannote | **0.993** | **0.980** | **0.980** | 0.940 | **0.969** |
| firered | 0.954 | 0.950 | 0.933 | **0.947** | 0.917 |
| funasr_fsmn | 0.948 | 0.933 | 0.937 | 0.940 | 0.947 |
| silero | 0.889 | 0.867 | 0.868 | 0.941 | 0.918 |
| webrtc | 0.880 | 0.864 | 0.746 | 0.924 | 0.902 |

<div class="vad-chart" data-vad-chart="f1" role="img" aria-label="五款 VAD 在五个数据集上的时长加权 F1 对比"></div>

*图 1：五款 VAD 的跨数据集时长加权 F1。为看清差异，纵轴自 0.70 起，柱高差并不代表倍数差。*

### 各数据集明细

一行一个数据集（纵向堆叠，行间以横线分隔）。每行左侧自上而下是 **F1 / P / R / FAR / Miss** 五项指标，横向为数值 0–1；同一指标下的五根横条对应五款引擎，颜色与全文一致。FAR 与 Miss 越低越好，其余越高越好。

<div class="vad-chart" data-vad-chart="detail" role="img" aria-label="五个数据集上各引擎的 F1、P、R、FAR、Miss 明细对比"></div>

*图 2：各数据集明细（悬浮横条可看该数据集该引擎的完整指标与样本量）。*

### 均值掩盖了什么：每文件分布

上表全是汇总值。我自己又从 `results/eval_*.json` 里把每个文件的 F1 拉出来统计了一遍——这一步是我认为本次评测里**信息量最大**的部分：

一行一个数据集（纵向堆叠，行间以横线分隔），每行内五条横条对应五款引擎：**横条的左端是最差文件，右端是中位数，菱形是均值**，最右侧标出「F1 < 0.9 的文件数」。横条越长说明该引擎的文件间波动越大。

<div class="vad-chart" data-vad-chart="summary" role="img" aria-label="五个数据集上各引擎每文件 F1 的最差值、中位数、均值与低于 0.9 的文件数"></div>

*图 3：每文件 F1 汇总。看 silero 在 AliMeeting test 那条：中位数 0.941 看着不错，但左端一路拖到 **0.074**——这一条横条本身就是「均值掩盖了什么」的最好注脚。*

<div class="vad-chart" data-vad-chart="perfile" role="img" aria-label="五个数据集上各引擎的每文件 F1 分布箱线图"></div>

*图 4：每文件 F1 的分布。箱体是四分位区间，散点是单个文件——与上图可以对照着看。*

几个直接可读的结论：

- **pyannote 是唯一「没有差文件」的引擎**：除了 RAMC，它在所有数据集上最差的一个文件也没掉到 0.94 以下。这种「下限稳」的性质在工程上比均值高 0.01 有用得多。
- **silero / webrtc 的问题不是普遍差，而是个别文件直接崩。** AliMeeting test 里 silero 的中位数是 0.941（比 webrtc 高），但最差文件只有 **0.074**——几乎整场判成非语音（Miss 0.962）。同一批数据里，它既是「还不错」又是「完全不可用」，这取决于你抽到哪一场会议。
- **AISHELL-4 上 webrtc 有 17/20 个文件低于 0.9**，整体 Miss 0.400。这不是个别崩溃，是系统性失效。我倾向于把它归到「8 通道等权平均破坏了信号」这一条上（能量型 VAD 对相位抵消最敏感）。
- 顺带一个有意思的现象：AliMeeting 的两个集里，**五个引擎的最差文件是同一场会议**（test 里是 `R8008_M8015_MS808`，eval 里是 `R8008_M8013_MS807`）。所有模型同时在同一场会议上翻车，说明这是数据侧的困难样本（低信噪比/串音/标注边界），而不是模型之间的差异。

### FAR 到底有多「大」？

上表里 funasr_fsmn 在 RAMC 上 FAR = 0.600，pyannote 在 AliMeeting eval 上 FAR = 0.374——单看这两个数会很吓人，好像模型一多半时间在乱报。但这其实是个**分母陷阱**。

会议/对话音频里，语音帧占比极高（我按 `P/R/FAR` 反推了一下：AliMeeting 两集约 92%，AISHELL-4 约 91%，RAMC 约 83%，AMI 约 80%）。非语音帧是少数派，分母一小，FAR 就容易被放大。更有工程意义的量是**虚警帧占全部帧的比例 `FP / N`**：

| 数据集（语音帧占比） | 引擎 | FAR | FP / 全部帧 | 体感 |
|----------------------|------|----:|------------:|------|
| AliMeeting test（92.5%） | funasr_fsmn | 0.429 | **3.24%** | 每 30 帧有 1 帧是白送的 |
| | firered | 0.241 | 1.82% | |
| | webrtc | 0.149 | 1.13% | |
| | pyannote | 0.100 | 0.75% | |
| | silero | 0.094 | 0.71% | |
| MagicData-RAMC（82.8%） | funasr_fsmn | 0.600 | **10.30%** | 每 10 帧就有 1 帧非语音进 ASR |
| | webrtc | 0.404 | 6.94% | |
| | firered | 0.317 | 5.44% | |
| | pyannote | 0.290 | 4.97% | |
| | silero | 0.219 | 3.77% | |
| AISHELL-4（90.6%） | pyannote | 0.222 | 2.08% | |
| | funasr_fsmn | 0.160 | 1.50% | |
| | webrtc | 0.092 | 0.87% | |
| | firered | 0.071 | 0.66% | |
| | silero | 0.035 | 0.33% | |

<div class="vad-chart" data-vad-chart="far" role="img" aria-label="FAR 与虚警帧占全部帧比例的对比图"></div>

*图 5：同一批数值的两种读法。左边是 FAR（分母只有非语音帧），右边是虚警帧占全部帧的比例；funasr_fsmn 在 RAMC 上 FAR = 0.600，听起来像「六成时间在乱报」，实际只占全部帧的 10.3%。*

换成这个口径以后数字就很好理解了：**最差的情况下（RAMC 上的 funasr_fsmn）每 10 帧里有 1 帧非语音被送进 ASR**；而在 AISHELL-4 上 silero 的虚警只有 0.33%，几乎可以忽略。所以我建议报 VAD 结果时**同时给出 `FP/N`**，或者干脆只报 Precision——只报 FAR 很容易让人误判严重程度。

几点观察：

- **pyannote 在两个会议集上的 FAR 明显偏高**（AliMeeting eval 0.374、AISHELL-4 0.222）。这里有个很重要的干扰因素：我把 pyannote 的 `min_duration_on` / `min_duration_off` 都设成了 **0**，也就是**完全关掉了「过短语音段过滤」和「过短静音填充」这两步后处理**，纯靠 segmentation 的原始输出。这几乎必然推高虚警。换句话说，pyannote 的高 FAR 里有相当一部分是**我的配置选择**，不是模型的问题——这也正是「默认参数评测」的局限所在。
- **funasr_fsmn 是典型的「高召回 + 高虚警」**：RAMC 上 R = 0.996、Miss 只有 0.4%，但 P 只有 0.889。它的召回是靠大量误召堆出来的，所以**那张 R 特别好看的表不能单独看**。反过来说，如果你的下游能容忍噪声（比如后面还有一轮 ASR 置信度过滤），它的「几乎不漏」反而是优点。
- **RAMC 上所有引擎的 P 都偏低（0.89 ~ 0.95）**，和会议集比起来整体下了一个台阶。我怀疑这里面有 GT 口径的成分：RAMC 的 GT 来自官方给的语音活动时间戳，对于近场手机对话里的呼吸声、笑声、短暂停顿，标注约定大概率和会议集不一样，一部分「虚警」其实是**标注口径差异**而不是模型错。

## 性能结果（ONNX 端到端）

### 测试条件

- 固定 **10 s**、16 kHz、单通道测音（从 AMI IHM 里按 RMS 挑的一段有语音的片段）；
- warmup 5 次 + 正式 50 次，取均值 / p95 / p99；
- ONNXRuntime **CPU**，`intra_op_num_threads = inter_op_num_threads = 1 / 2 / 4`；
- 延迟含义是**端到端**：特征提取 + 模型推理 + 后处理成段，**不含模型加载**；
- WebRTC 没有 ONNX，标 `native`。

### 结果

| 引擎 | format | 体积 | mean@T1 | mean@T2 | mean@T4 | RTF@T2 | p95@T2 | p99@T2 |
|------|--------|-----:|--------:|--------:|--------:|-------:|-------:|-------:|
| webrtc | native | n/a | 2.3 ms | — | — | **0.00023** | 3.4 ms | 3.7 ms |
| firered | onnx | 2.28 MB | 33.2 ms | 26.8 ms | 25.5 ms | **0.00268** | 30.6 ms | 32.1 ms |
| pyannote | onnx | 5.64 MB | 37.4 ms | 27.3 ms | 25.4 ms | 0.00273 | 30.7 ms | 32.6 ms |
| silero | onnx | 2.22 MB | 64.8 ms | 59.7 ms | 54.9 ms | 0.00597 | 67.0 ms | 70.1 ms |
| funasr_fsmn | onnx | 1.65 MB | 121.2 ms | 118.4 ms | 118.8 ms | 0.01184 | 124.7 ms | 126.9 ms |

<div class="vad-chart" data-vad-chart="perf" role="img" aria-label="五款 VAD 的端到端延迟与模型体积对比"></div>

*图 6：左＝10 s 音频的端到端延迟（含特征提取与后处理），webrtc 是纯 C 实现、没有 Python 侧开销，比其他引擎快一到两个数量级；右＝体积与延迟的关系，silero 和 firered 体积几乎相同，延迟却差了 2.2 倍。*

复跑：

```powershell
python -m vadbench.perf --engines silero,firered,funasr_fsmn,pyannote,webrtc --threads 1,2,4 --clip-s 10 --runs 50
```

### 先证明「是同一个模型」，再比快慢

这一步我觉得是整个性能对比里最有价值的设计。

模型性能对比有个隐蔽的前提：**你跑的得是同一个模型**。把 PyTorch 模型导成 ONNX 之后输出会有微小漂移，如果漂移大到改变了段边界，那你比的其实是两个不同的东西。所以 `vadbench` 在跑性能之前会先做一次**对齐校验**（`vadbench/perf/align.py`）：

1. 用原始后端和 ONNX 后端分别对同一段音频推理，得到两份段级输出；
2. 都栅格化成 10 ms 帧，算帧级 F1；
3. **F1 = 1.0 才认为对齐通过**（阈值定在 0.99），否则这次性能数据直接标 `fail_align` 丢弃。

这次四个模型全部拿到了 **align F1 = 1.000**，也就是段级输出逐帧完全一致。所以上面那张延迟表比的其实是**同一个模型的不同运行时**，而不是「两个碰巧名字相同的模型」——踩过这个坑之后，我现在看任何跨运行时/跨后端的性能对比，都会先找找有没有类似的校验步骤。

## 选型建议

<div class="vad-chart" data-vad-chart="tradeoff" role="img" aria-label="各引擎在各数据集上的漏检率与虚警率权衡散点图"></div>

*图 7：25 个「引擎 × 数据集」运行点的取舍地图。横轴是漏检率，纵轴是虚警帧占比，左下角最理想；数据集之间的差异（点云整体左右漂移）比引擎之间的差异更大，这也是「必须在自己的数据上选型」的直观理由。*

把效果和性能合起来看，我自己的判断是：

| 场景 | 首选 | 备选 | 理由 |
|------|------|------|------|
| 离线会议转写（中/英，远场或近场） | pyannote | firered | pyannote 在所有会议集上都是 0.97+ 且没有差文件，代价是 FAR 稍高、模型 5.64 MB |
| 中文近场对话（手机/客服） | firered | silero | RAMC 上 firered 第一（0.947）且虚警适中，体积只有 2.28 MB，延迟和 pyannote 打平 |
| 实时流式 / 端侧 | firered 或 silero | webrtc | 延迟 25~60 ms 且可控；若设备算力极弱、只要粗切分，webrtc 的 2.3 ms 无解 |
| 下游是 LLM-based ASR，最怕幻觉 | silero 或 pyannote | — | 这两个的虚警帧占比最低（AISHELL-4 上 silero 只有 0.33%）；funasr 的高虚警在这种链路上要谨慎 |
| 绝不允许漏语音 | funasr_fsmn | — | R 最高（RAMC 0.996），代价是每 10 帧有 1 帧非语音——自己权衡 |


几个通用的结论：

1. **默认参数下 pyannote 和 firered 是第一梯队**，两者在延迟上也打平，选哪个主要看你的场景是会议还是近场对话、以及能不能接受 5.64 MB 的权重。
2. **VAD 的效果有很大的「配置成分」。** 我这次所有引擎都没调参，pyannote 的 FAR 偏高很可能就是关掉后处理导致的。**真要选型，应该在你自己的数据上做一轮阈值/后处理扫描**，而不是直接抄任何一篇评测的排名——包括这一篇。
3. **性能优化的第一刀砍向 wrapper。** 端到端延迟里 Python 侧开销的占比比模型本身大得多，这一点在 silero 和 funasr 上体现得特别明显。

<div class="callout callout-warn">
<div class="callout-title">⚠️ 长音频转写场景下，尤其要盯住 FAR</div>

<p>在长音频转写这个场景下，VAD 和 ASR 大概率是一个<strong>级联结构</strong>。如果下游用的是 LLM-based 的 ASR 模型（Qwen3-ASR 等），就需要特别留意 <strong>FAR</strong> 这个指标：在笔者的实际实践中，一旦 VAD 把各种类型的突发噪声、杂音送进 ASR 系统，而 ASR 转写时又是携带上下文的，就很容易诱发<strong>幻觉</strong>。这一点尤其要注意。</p>

</div>

<style>
/* Vadbench 图表容器：具体高度由 assets/js/vad-charts.js 按屏宽调整 */
.vad-chart { width: 100%; height: 440px; margin: 1.6em 0 .4em; }
.vad-chart[data-vad-chart="perfile"],
.vad-chart[data-vad-chart="tradeoff"] { height: 480px; }
/* 5 行明细横条图：一行一个数据集，指标在左、数值在右 */
.vad-chart[data-vad-chart="detail"] { height: 2000px; }
/* 每文件汇总图：一行一个数据集，五款引擎各一条 */
.vad-chart[data-vad-chart="summary"] { height: 1120px; }
@media (max-width: 720px) {
  .vad-chart[data-vad-chart="detail"] { height: 1750px; }
}
/* 图表渲染失败时的降级提示 */
.vad-chart-fallback {
  margin: 0;
  padding: 18px 20px;
  border: 1px dashed var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg-soft);
  color: var(--text-soft);
  font-size: 14.5px;
  text-align: center;
}
</style>
<script src="../assets/js/echarts.min.js"></script>
<script src="../assets/js/vad-charts.js"></script>

## 参考

- WebRTC VAD：[wiseman/py-webrtcvad](https://github.com/wiseman/py-webrtcvad)
- Silero VAD：[snakers4/silero-vad](https://github.com/snakers4/silero-vad) · [版本历史与可用模型](https://github.com/snakers4/silero-vad/wiki/Version-history-and-Available-Models)
- FunASR FSMN-VAD：[modelscope/FunASR](https://github.com/modelscope/FunASR) · [funasr/fsmn-vad](https://huggingface.co/funasr/fsmn-vad)
- FireRedVAD：[FireRedTeam/FireRedVAD](https://github.com/FireRedTeam/FireRedVAD)
- pyannote.audio：[pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio) · [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) · [pyannote/voice-activity-detection](https://huggingface.co/pyannote/voice-activity-detection)
- 数据集：[AliMeeting (OpenSLR 119)](https://www.openslr.org/119/) · [AISHELL-4 (OpenSLR 111)](https://www.openslr.org/111/) · [AISHELL-4 论文](https://arxiv.org/abs/2104.03603) · [MagicData-RAMC (OpenSLR 123)](https://www.openslr.org/123/) · [AMI Corpus](https://groups.inf.ed.ac.uk/ami/corpus/)
- 相关笔记：[Qwen3-ASR 技术报告阅读笔记](Qwen3ASR.html)

> 本文为个人评测记录，模型与数据版权归各自作者所有；MagicData-RAMC 的许可为 CC-BY-NC-ND，仅用于本次非商业评测。表中数据保留三位小数，均由 `results/` 下的原始 CSV / JSON 汇总而来。
