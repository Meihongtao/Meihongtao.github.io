
# Qwen3ASR 

最近笔者一直在基于Qwen3ASR这个模型做一些语音识别的系统的开发工作，Qwen3ASR这篇论文也是阅读了很多次，但是一直没有做什么整理性质的文字工作，为此简单记录一些我的理解和一些工程上的实践。如有疏漏，欢迎指出。

> 论文链接：[Qwen3-ASR（arXiv:2601.21337）](https://arxiv.org/abs/2601.21337)

## 说在前面

笔者不是一开始就从事语音识别系统，对早期的一些基于Tranducer，CTC结构的语音识别系统也只是有一些简单的了解。但是，最近几年，新开源出来的ASR系统绝大部分都是基于LLM的，在语义连贯，联想，知识这几个方向LLM还是有非常大的优势的，LLM中蕴含的大量知识对语音的转写质量有着极大的提升，大部分音频质量清晰的评测数据集的表现已经探底了，再往上提升已经十分有限。但是基于LLM的ASR系统在幻觉这个问题上还是有普遍的问题，无论是流式还是实时方向。回到本文，Qwen3ASR这个模型在结构上还是比较清晰的。简单而言，就是一个音频编码器(Aut Encoder) + 投影层 + LLM 的结构。

## 整体结构

Qwen3-ASR 这一族模型包含三个成员：

| 模型 | 构成 | 说明 |
| --- | --- | --- |
| Qwen3-ASR-1.7B | Qwen3-1.7B + Projector + AuT(300M, hidden 1024) | 追求效果，开源模型中 SOTA |
| Qwen3-ASR-0.6B | Qwen3-0.6B + Projector + AuT(180M, hidden 896) | 精度与效率的折中，可端侧部署 |
| Qwen3-ForcedAligner-0.6B | Qwen3-0.6B + AuT + 时间戳预测层 | 非自回归强制对齐，支持 11 种语言 |

其中笔者主要接触 Qwen3-ASR-1.7B 这个模型多一点。笔者实测下来 0.6B和1.7B的性能差异并不是很大。在某些场景下幻觉的发生概率0.6B甚至更好一点。当然这只是一个初步的观察，并没有相关的数据支持。


## 音频编码器

Qwen3-ASR 的编码器叫 **AuT**，是一个独立预训练的 attention-encoder-decoder（AED）结构模型。抛开 AED 这个不算新的骨架，有两个细节值得关注。

### 下采样率

AuT 对 128 维 Fbank 特征做 **8 倍下采样**，因此 1 秒音频大约产生 **12.5 个 token**，即每个 token 对应 80ms。换算起来很直观：10 秒音频约 125 个 token。

### 动态注意力窗口

AuT 内部使用动态的注意力窗口，论文里的范围是 **1s ~ 8s**。行为大致是：

- 音频长度**小于 8s** 时，注意力窗口就取整段音频长度；
- 音频长度**超过 8s** 时，AuT 会做强制切分。比如 12s 的音频，会被切成 `[0, 8]` 和 `[8, 12]` 两个窗口。

这个设计应该主要还是为了平衡推理效率。窗口小可以做流式（短 chunk），窗口大可以做离线长音频，同一个模型不用改结构就能兼顾两种形态。

### 论文原话

> **(1) AuT pretraining.** In this stage, we aim to obtain a pretrained encoder under the AED framework using large-scale labeled data. We leverage approximately 40 million hours of pseudo-labeled ASR data, where the majority is in Chinese and English. This pretrained encoder is shown to provide general and stable audio representations under dynamic attention window sizes.

从这段描述可以推测，AuT 在 8s 以内**任意**窗口长度下的表征质量都相当好——换句话说，窗口的"可变"能力是在预训练阶段就一起学进去的，而不是推理时临时拼凑出来的。

顺着这个特性，有两个可以自己动手改的方向：

**（1）把窗口上限调大。** 既然 8s 以内都稳，那配置文件里的这个限制或许可以试着放到 10s 甚至更大，让单窗口覆盖更长的上下文。理论上窗口越大，对语音的感知越完整，转写效果也可能更好——但注意力开销和显存会同步上涨，值得实测权衡。笔者这里有尝试调整窗口大小到12s左右，没有发现转写的性能出现明显的下降。当然这是直观的观察，并没有大量的实验做支撑。

**（2）自定义切分点，避免关键语音被截断。** 设想一个场景：一段 12s 的音频里，某个关键术语或名词的发音恰好落在 7~9s 这个区间。AuT 默认会切成 `[0, 8]` 和 `[8, 12]`，正好把这个词从中间劈开，两侧窗口各自只看到半截发音。一个可能的改法是绕开默认切分逻辑，自己指定注意力窗口的划分再推理，比如切成 `[0, 5]`、`[5, 9]`、`[9, 12]` 三段，让 7~9s 这个关键区间完整地落在第二个窗口内，从而避免默认行为带来的截断。


## 参考

- 论文：[Qwen3-ASR Technical Report (arXiv:2601.21337)](https://arxiv.org/abs/2601.21337) / [PDF](https://arxiv.org/pdf/2601.21337) / [HTML](https://arxiv.org/html/2601.21337v2)
- 模型权重：[Hugging Face](https://huggingface.co/collections/Qwen/qwen3-asr) / [ModelScope](https://modelscope.cn/collections/Qwen/Qwen3-ASR)
- 代码：[QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- 底座模型：[Qwen3-Omni Technical Report (arXiv:2509.17765)](https://arxiv.org/abs/2509.17765)
- 相关方法：[GLM-ASR-2512](https://docs.z.ai/guides/audio/glm-asr-2512) · [Fun-ASR Technical Report (arXiv:2509.12508)](https://arxiv.org/abs/2509.12508) · [LLM-ForcedAligner (arXiv:2601.18220)](https://arxiv.org/abs/2601.18220) · [Montreal Forced Aligner](https://doi.org/10.21437/Interspeech.2017-1386)

> 本文为个人阅读笔记，模型与数据版权归 Qwen Team 所有，模型以 Apache 2.0 协议开源。