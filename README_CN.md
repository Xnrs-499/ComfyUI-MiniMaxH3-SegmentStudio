# MiniMax H3 Segment Studio

独立的 ComfyUI 插件：在 **16GB 显卡**上做 MiniMax H3 长视频。

- Timeline turbo **一采 0.4**
- 导演台式 **按窗二采 2.0**（重叠钉尾，不是整段 T8）
- 每窗复用该段 **一采提示词 + 参考图**
- 窗与窗之间释放 Comfy 编译 CUDA graph，避免第 2 窗 3D 放大 OOM

这是 [ComfyUI-MiniMaxH3-TimelineDirector](https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector) 的 **GPL-3.0 独立发行版**，加上实际能跑通的 0.4→2.0 工作流。用了哪些插件、向谁致谢，见 **[CREDITS.md](CREDITS.md)** 与 [NOTICE](NOTICE)。

## 不要和原版 Timeline 同时开

节点 ID 与原版兼容，旧图还能用。安装后请：

1. 把 `custom_nodes/ComfyUI-MiniMaxH3-TimelineDirector` 改名为 `ComfyUI-MiniMaxH3-TimelineDirector.disabled`
2. 重启 ComfyUI
3. 打开 `example_workflows/` 里的示例图

## 主工作流

按**每段几秒**选图，段数在规划台改。

- `example_workflows/H3_SegmentStudio_8秒-多段_T8二采_2.0.json`：每段约 8 秒，按窗 T8
- `example_workflows/H3_SegmentStudio_15秒-多段_T8二采_2.0.json`：每段约 15 秒，按窗 T8
- `example_workflows/H3_SegmentStudio_8秒-多段_仅3D放大_2.0.json`：每段约 8 秒，只 3D 放大
- `example_workflows/H3_SegmentStudio_15秒-多段_仅3D放大_2.0.json`：每段约 15 秒，切窗 3D 放大

| 图 | 一采 | 二采 |
| --- | --- | --- |
| 8秒-多段 T8 | `output/H3_0.4MP_8秒-多段_T8/` | `output/H3_8秒-多段_T8/` |
| 15秒-多段 T8 | `output/H3_0.4MP_15秒-多段_T8/` | `output/H3_15秒-多段_T8/` |
| 8秒-多段 仅3D | `output/H3_0.4MP_8秒-多段_3D/` | `output/H3_8秒-多段_3D/` |
| 15秒-多段 仅3D | `output/H3_0.4MP_15秒-多段_3D/` | `output/H3_15秒-多段_3D/` |

RTX 5060 Ti 16GB + SageAttention，25 秒 / 600 帧大约 **一采 13 分钟 + 二采 42 分钟**。

### 16GB 硬规则

- 一采采样器 **Output Megapixels 必须是 0**，不要拧到 2.0。
- **15秒-多段**用带 15 秒的那两张：一采逐段落盘，二采按窗读 0.4 mp4，不要拆掉「一采成片」那根线。
- 按窗二采的 **窗口大小** 是下拉：约 5.9 / 6.6 / 7.3 / 8 / 10 秒，以及 **整段15秒**。默认 **约 5.9 秒（141）** 适合 **2.0 / 16GB**。分辨率调低后再选更长。重叠 **39**。
- OOM 之后必须完全退出 Comfy 再开，并从磁盘加载工作流，不要跑 Unsaved。
- 旧的「整段 VSR + 整段 T8」已经从示例里拿掉，16GB 上不要再接回去。

## 模型

放到 ComfyUI 常规目录：

- UNET：MiniMax H3 turbo T8
- CLIP：MiniMax H3 文本编码器
- 视频 VAE + 音频 VAE
- `latent_upscale_models/minimax_h3_latent_upscaler_3d_bf16.safetensors`（按窗二采也可选 `auto`）

**完整清单（文件名、放哪个目录、缺了会怎样、报错怎么查）见 [docs/INSTALL_CN.md](docs/INSTALL_CN.md)。**

可选：UNET 前接 `PathchSageAttentionKJ`（[KJNodes](https://github.com/kijai/ComfyUI-KJNodes)）。

过程预览直接画在一采采样器和按窗二采节点上：把 `MiniMax-H3-TAE.safetensors` 放到 `models/vae_approx/`（或 `taeh3.safetensors`）。保持「过程预览」打开即可，不用再接 KJ 的预览节点。

想接近「双采超清-最初版」的锐度：按窗二采把 **超分方式改成 `nvidia_rtx_vsr`**，二采步数 / denoise 保持 **2 / 0.2**，窗口 **141**。这就是最初版「VSR 拉到 2.0 再轻 T8」，只是一次只跑一窗，所以 16GB 不装整段。需要本机已装 nvidia-vfx；失败会退回 lanczos。

## 用了哪些插件

| 关系 | 插件 | 作者 |
| --- | --- | --- |
| 已打进本包 | [Timeline Director](https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector) | 石兄松 |
| 已打进本包 | [SelfLift](https://github.com/facok/comfyui-SelfLift) · [H3 3D Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) | facok · LBH-123-AI |
| 学了按窗钉尾，未拷源码 | [MiniMax H3 Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director) | AIMixer |
| 示例图需要另装 | [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 官方 H3 | Comfy-Org / MiniMax |
| 示例图需要另装 | [KJNodes](https://github.com/kijai/ComfyUI-KJNodes)（SageAttention） | kijai |
| 二采回退 / AV 拼接 | [H3 Audio T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8) · PT_H3ConcatAVLatent | T8mars · PT |

完整名单和感谢的话写在 [CREDITS.md](CREDITS.md)。**谢谢石兄松、AIMixer、kijai、T8mars、LBH-123-AI、facok，以及 MiniMax 与 ComfyUI。**

装哪些、必需还是可选、报“校验失败”怎么查：**[docs/INSTALL_CN.md](docs/INSTALL_CN.md)**。

## GitHub

https://github.com/Xnrs-499/ComfyUI-MiniMaxH3-SegmentStudio

这个文件夹就是仓库根目录。把它 clone 或复制到 `ComfyUI/custom_nodes/` 即可，不要把整个绘世 Comfy 目录当成仓库。

## 许可

GPL-3.0。时间轴规划器与有限分段采样器版权归原作者；按窗 2.0 与示例图是本发行版增加的部分。
