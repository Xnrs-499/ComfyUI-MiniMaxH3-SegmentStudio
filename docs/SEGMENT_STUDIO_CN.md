# Segment Studio：16GB 上一采 0.4、按窗二采 2.0

## 路径

1. 素材规划器分段（合法重叠 5/22/39/56）。
2. 有限分段采样器一采，**Output Megapixels = 0**，导出 `H3_0.4MP_3seg`。
3. 把 Merged Frames / Merged Audio / Finite Plan / **Final Video** 接到 **MiniMax H3 Windowed Refine**。按每段时长选图：8秒-多段或 15秒-多段（15 秒图二采按窗读一采 mp4）。
4. 窗口大小选「约5.9秒（2.0 / 16GB）」，重叠 39，超分方式 `h3_latent`，denoise 0.2，steps 2。
5. SaveVideo 接到 Windowed Refine 的 Final Video，前缀 `H3_双采超清`。

## 每窗在干什么

对每个重叠窗：

1. 0.4 像素编码成 latent
2. 3D latent 放大到 2.0 网格（CLIP/UNET 先卸掉；编译 CUDA graph 一并清掉）
3. 用上一窗 2.0 AV 尾钉住重叠区（Drift-Control）
4. 按该窗对应的一采段，复用提示词和参考图做条件
5. T8 采 2 步
6. 解码、裁重叠、落盘 mp4

16GB 上 2.0 SageAttention 实测：192 帧会炸，**141 帧能过**。窗口大小由你填，插件不再偷偷改；把输出百万像素调低，才比较容易开满整段。

## 致谢

用到的插件和感谢名单见仓库根目录 [CREDITS.md](../CREDITS.md)。

## 和整段二采的区别

整段 VSR 到 2.0 再整段 T8 会在 16GB 上 OOM。按窗路径从不把整条 2.0 latent 送进 UNET。
