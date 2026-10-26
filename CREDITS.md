# 致谢 · Credits

Segment Studio 能跑通，是站在这些作者和项目上的。下列插件与模型**不属于本仓库**，请按各自许可证安装与使用。没有他们，就没有时间轴一采、按窗二采和 16GB 上的 2.0。

## 本包已融合的上游

| 项目 | 作者 | 用途 | 链接 |
| --- | --- | --- | --- |
| ComfyUI-MiniMaxH3-TimelineDirector | 石兄松 / Songssx | 素材规划器、有限分段采样、Drift-Control、锁定原声、时间轴前端。本包是它的 GPL-3.0 发行版。 | https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector |
| comfyui-SelfLift | facok | 两阶段采样运行时（已打进 `selflift_runtime/`） | https://github.com/facok/comfyui-SelfLift |
| Comfyui_Minimax_h3_latent_Upscaler | LBH-123-AI | 3D latent 放大网络结构；按窗二采的 `h3_latent` 路径 | https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler |

**特别感谢石兄松。** 时间轴、分段续写和 Drift-Control 是长视频能接上的底座。Bilibili：https://space.bilibili.com/219572544 · YouTube：https://www.youtube.com/@shixiongsong

## 设计思路（未整包拷贝源码）

| 项目 | 作者 | 我们学了什么 | 链接 |
| --- | --- | --- | --- |
| ComfyUI_MiniMaxH3_Director | AIMixer | 按窗 0.4 编码 → 抬到 2.0 → 钉上一窗 AV 尾 → 再 T8，而不是整段 2.0 硬采 | https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director |
| ComfyUI-H3-Motion-Context | NikoDemon80 | 段间运动上下文 / 合法重叠 5·22·39·56（Director 亦致谢此项目） | https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context |
| selflift-Avatar | slmonker | SelfLift 渐进分辨率思路 | https://github.com/slmonker/selflift-Avatar |

**特别感谢 AIMixer 导演台。** 16GB 上能出 2.0，靠的是这套「按窗钉尾」而不是整段超分。

## 主工作流还需要安装的插件

示例图 `H3_SegmentStudio_8秒-多段_T8二采_2.0.json` 除本包外，还用到：

| 项目 | 作者 | 图里的节点 / 作用 | 链接 |
| --- | --- | --- | --- |
| ComfyUI（含官方 MiniMax H3） | comfyanonymous / Comfy-Org / MiniMax | UNET·CLIP·VAE 加载、KSampler、SaveVideo、官方 H3 条件与网格 | https://github.com/comfyanonymous/ComfyUI |
| ComfyUI-KJNodes | kijai | `PathchSageAttentionKJ`，16GB 上 T8 能跑进 141 帧；过程预览沿用其 tiny-VAE / OUTER_SAMPLE 思路 | https://github.com/kijai/ComfyUI-KJNodes |
| MiniMax H3 Audio T8 | T8mars / T8star | `MiniMaxH3AudioConditioningT8`（按窗二采在规划器条件失败时的回退） | https://github.com/T8mars/comfyui-minimax-h3-audio-T8 |
| ComfyUI-PT_H3ConcatAVLatent | PT | `PT_H3ConcatAVLatent`，把视频 latent 和音频 latent 拼成 H3 AV | 本地 `custom_nodes/ComfyUI-PT_H3ConcatAVLatent` |

没有 KJNodes 时，把 SageAttention 节点绕过即可，2.0 窗口可能要再缩短。没有 T8 / PT 节点时，一采主路径仍可用；按窗二采的回退条件和部分展开图会受影响。

## 模型与算法

| 项目 | 致谢 |
| --- | --- |
| MiniMax H3 | 视频 / 音频生成模型与官方 Comfy 节点 |
| SageAttention | thu-ml，https://github.com/thu-ml/SageAttention |
| NVIDIA RTX Video Super Resolution | 像素超分备选路径（`nvidia_rtx_vsr`） |

## 一句话

谢谢每一位把 MiniMax H3 从「能出 8 秒」推到「能接长片」的作者。本包只是把 Timeline 的速度和 Director 的按窗连续性接到 16GB 上；功劳在上游。
