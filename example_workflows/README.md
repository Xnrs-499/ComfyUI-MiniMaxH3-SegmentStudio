# 示例工作流

按**每段时长**分类，段数都在规划台改（多段都能跑）。

| 文件 | 每段 | 二采 |
| --- | --- | --- |
| `H3_SegmentStudio_8秒-多段_T8二采_2.0.json` | 约 8 秒 | 按窗 T8 |
| `H3_SegmentStudio_15秒-多段_T8二采_2.0.json` | 约 15 秒 | 按窗 T8 |
| `H3_SegmentStudio_8秒-多段_仅3D放大_2.0.json` | 约 8 秒 | 只 3D 放大，不跑 T8 |
| `H3_SegmentStudio_15秒-多段_仅3D放大_2.0.json` | 约 15 秒 | 切窗 3D 放大，不跑 T8 |

需要 MiniMax H3 的 UNET / CLIP / VAE，以及 **KJNodes**（`PathchSageAttentionKJ`）。原样加载必须另装 [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)；可绕过该节点，但 16GB 上请把二采窗口缩短。规划台里的参考图路径是本机预设，换机器请重新指定。15 秒那两张不要拆「一采成片」那根线。一采采样器和按窗二采节点自带过程预览，权重放 `models/vae_approx/MiniMax-H3-TAE.safetensors`。完整依赖边界见 [docs/INSTALL_CN.md](../docs/INSTALL_CN.md)。

这些 JSON 的 `aux_id` 指向本仓库 `Xnrs-499/ComfyUI-MiniMaxH3-SegmentStudio`。Manager 若提示安装 Timeline Director，不要装；和本包节点 ID 冲突会在「保存 15秒-多段 0.4」上校验失败。

致谢见 [CREDITS.md](../CREDITS.md)。
