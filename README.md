# MiniMax H3 Segment Studio

Independent ComfyUI pack for MiniMax H3 long video on a **16GB** card:

- Timeline turbo **first pass at 0.4 MP**
- Director-style **windowed 2.0 refine** (overlap re-pin, not whole-clip T8)
- Reuses each segment’s **first-pass prompt + reference images**
- Reclaims Comfy AIMDO CUDA graphs between windows so window 2+ does not OOM

[简体中文](README_CN.md) · English

This is a **GPL-3.0 fork** of [ComfyUI-MiniMaxH3-TimelineDirector](https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector) plus the 0.4→2.0 workflow we actually run. Plugin list and thanks: **[CREDITS.md](CREDITS.md)** · [NOTICE](NOTICE). Install checklist: **[docs/INSTALL_CN.md](docs/INSTALL_CN.md)**.

## Do not load both packs

Node IDs stay compatible with Timeline Director graphs. After copying this folder into `ComfyUI/custom_nodes/`:

1. Rename `ComfyUI-MiniMaxH3-TimelineDirector` → `ComfyUI-MiniMaxH3-TimelineDirector.disabled`
2. Restart ComfyUI
3. Open a graph in `example_workflows/`

## Flagship workflow

Pick by **seconds per segment**. Segment count is set on the planner.

- `example_workflows/H3_SegmentStudio_8秒-多段_T8二采_2.0.json` — ~8s boxes, windowed T8
- `example_workflows/H3_SegmentStudio_15秒-多段_T8二采_2.0.json` — ~15s boxes, windowed T8
- `example_workflows/H3_SegmentStudio_8秒-多段_仅3D放大_2.0.json` — ~8s boxes, 3D lift only
- `example_workflows/H3_SegmentStudio_15秒-多段_仅3D放大_2.0.json` — ~15s boxes, split 3D lift

| Graph | First pass | Second pass |
| --- | --- | --- |
| 8s T8 | `output/H3_0.4MP_8秒-多段_T8/` | `output/H3_8秒-多段_T8/` |
| 15s T8 | `output/H3_0.4MP_15秒-多段_T8/` | `output/H3_15秒-多段_T8/` |
| 8s 3D | `output/H3_0.4MP_8秒-多段_3D/` | `output/H3_8秒-多段_3D/` |
| 15s 3D | `output/H3_0.4MP_15秒-多段_3D/` | `output/H3_15秒-多段_3D/` |

On RTX 5060 Ti 16GB + SageAttention, a 25s / 600-frame clip is about **13 min first pass + 42 min refine**.

### 16GB rules

- First-pass sampler **Output Megapixels must stay 0**. Do not set it to 2.0.
- **15s-per-segment** graphs: first pass is sequential, refine reads the spilled 0.4 mp4 one window at a time. Keep **Final Video** wired into Windowed Refine.
- Windowed Refine **Window Frames** is a dropdown: 5.9s / 6.6s / 7.3s / 8s / 10s / **whole 15s**. Default **5.9s (141)** at **2.0** on 16GB. Drop Output Megapixels before picking a longer window. Overlap **39**.
- After an OOM: quit Comfy completely, restart, reload the **disk** workflow (not Unsaved).
- The old whole-clip VSR+T8 subgraph is **not** in this example. Do not rebuild it on 16GB.

## Models

Place as usual under `ComfyUI/models/`:

- UNET: MiniMax H3 turbo T8
- CLIP: MiniMax H3 text encoder
- Video VAE + Audio VAE
- `latent_upscale_models/minimax_h3_latent_upscaler_3d_bf16.safetensors` (or `auto` on Windowed Refine)

Optional: `PathchSageAttentionKJ` ([KJNodes](https://github.com/kijai/ComfyUI-KJNodes)) in front of the UNET.

Full install checklist (exact file names, target folders, optional vs required plugins, and the "validation failed" fix) is in **[docs/INSTALL_CN.md](docs/INSTALL_CN.md)** (Chinese).

Live preview on the sampler / Windowed Refine: put `MiniMax-H3-TAE.safetensors` in `models/vae_approx/` (or `taeh3.safetensors`). Leave **Live Preview** on; there is no extra KJ preview node to wire.

To match the original dual-pass look: set Windowed Refine **Upscale Method** to `nvidia_rtx_vsr`, keep Refine Steps / Denoise at **2 / 0.2**, and Window Frames at **141**. That is VSR-to-2.0 then light T8, one window at a time so 16GB never holds the whole clip. Needs nvidia-vfx.

## Plugins we built on

| Role | Plugin | Author |
| --- | --- | --- |
| Vendored here | [Timeline Director](https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector) | Shi Xiongsong / Songssx |
| Vendored here | [SelfLift](https://github.com/facok/comfyui-SelfLift) · [H3 3D Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) | facok · LBH-123-AI |
| Design only | [MiniMax H3 Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director) | AIMixer |
| Example graph | [ComfyUI](https://github.com/comfyanonymous/ComfyUI) official H3 | Comfy-Org / MiniMax |
| Example graph | [KJNodes](https://github.com/kijai/ComfyUI-KJNodes) (SageAttention) | kijai |
| Fallback / AV concat | [H3 Audio T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8) · PT_H3ConcatAVLatent | T8mars · PT |

Full thanks: [CREDITS.md](CREDITS.md).

## GitHub

https://github.com/Xnrs-499/ComfyUI-MiniMaxH3-SegmentStudio

This folder is the repository root. Clone it into `ComfyUI/custom_nodes/` (or copy the folder). Do not treat the whole ComfyUI tree as this repo.

## License

GPL-3.0. Upstream Timeline Director remains copyright of its authors; this pack adds the windowed 2.0 path and example graph.
