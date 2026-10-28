# 安装清单：模型、插件、放哪、缺了会怎样

这份文档回答三个问题：**要哪些模型**、**要哪些插件**、**分别放在哪个目录**。

示例工作流 `example_workflows/H3_SegmentStudio_*.json` 一共同到 11 种节点，其中只有 4 种来自本包以外。照着下面放好就能开图。

---

## 一、模型清单

示例图里文件名是**写死**的，放错目录或改了名字都会在下拉框里选不到。

| 用途 | 文件名 | 放置目录 | 典型大小 |
| --- | --- | --- | --- |
| UNET（MiniMax H3 turbo T8，融合 refdelta） | `minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors` | `ComfyUI/models/diffusion_models/` | 约 19.5 GB |
| CLIP 文本编码器（Qwen3-VL 32B，nvfp4 awq） | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `ComfyUI/models/text_encoders/` | 约 14.6 GB |
| 视频 VAE | `minimax_h3_video_vae_fp16.safetensors` | `ComfyUI/models/vae/` | 约 4.9 GB |
| 音频 VAE | `minimax_h3_audio_vae_fp32.safetensors` | `ComfyUI/models/vae/` | 约 0.6 GB |
| H3 3D latent 放大器（二采 0.4→2.0 的核心） | `minimax_h3_latent_upscaler_3d_bf16.safetensors` | `ComfyUI/models/latent_upscale_models/` | 约 0.64 GB |
| 过程预览小 VAE（可选） | `MiniMax-H3-TAE.safetensors` | `ComfyUI/models/vae_approx/` | 约 10 MB |

### 模型目录的两个坑

- **CLIP 必须放 `text_encoders`**。老版本 ComfyUI 里这个目录叫 `clip`，两个名字都能被 `CLIPLoader` 认到，但放错到 `checkpoints` 就一定找不到。
- **3D 放大器必须放 `latent_upscale_models`**。LTX 的 latent upscaler **不通用**，本包会显式拒绝非 H3 结构。
- 二采的「Latent 放大模型」下拉默认是 `auto`，会自动挑第一个名字带 `h3` 且结构兼容的模型：

```678:682:custom_nodes/ComfyUI-MiniMaxH3-SegmentStudio/sequential_segments.py
    if selected in {"", "none", "auto"}:
        preferred = [name for name in compatible if "h3" in name.lower()]
        return preferred[0] if preferred else compatible[0]
```

如果 `latent_upscale_models` 里一个 H3 兼容模型都没有，二采会直接失败：

```
Director-style 2.0 refine needs a MiniMax H3 latent upscaler under
ComfyUI/models/latent_upscale_models ... LTX upscalers are not compatible.
```

### 这些模型不需要

- **LoRA**：4 张示例图里没有 `LoraLoader`，UNET 本身就是 8 步 turbo。
- **SageAttention 不是模型**，它是个 pip 库，节点 `PathchSageAttentionKJ` 只是给 UNET 打补丁。

---

## 二、插件清单

### 1. 必须装

| 插件 | 提供的节点 | 作用 | 装在 |
| --- | --- | --- | --- |
| **ComfyUI 本体**（含官方 MiniMax H3 支持） | `UNETLoader` `CLIPLoader` `VAELoader` `BasicScheduler` `KSamplerSelect` `SaveVideo` | 加载模型、调度、保存视频 | 官方最新版 |
| **ComfyUI-KJNodes**（kijai） | `PathchSageAttentionKJ` | 4 张图都串在 UNET 前，16GB 上 T8 才跑得动 141 帧 | `custom_nodes/ComfyUI-KJNodes` |

KJNodes 是本包**唯一必须另装**的外部插件。没装的话，节点会变红并提示缺少 `PathchSageAttentionKJ`。

`PathchSageAttentionKJ` 的 `sage_attention` 默认 `auto`，会执行 `from sageattention import sageattn`，所以还要装 `sageattention` 这个 pip 库。没有的话可以把该节点设成 `disabled`，或者直接绕过它把 UNET 接到采样器——但要准备把二采窗口调短，否则 16GB 可能 OOM。

### 2. 可选（缺了自动回退，不中断）

| 插件 | 提供的节点 | 何时才会用到 | 缺了会怎样 |
| --- | --- | --- | --- |
| **MiniMax H3 Audio T8**（T8mars） | `MiniMaxH3AudioConditioningT8` | 按窗二采复用不了一采提示词/参考图时的回退条件 | 先试复用，失败才走它；纯回退路径 |
| **ComfyUI-PT_H3ConcatAVLatent**（PT） | `PT_H3ConcatAVLatent` | 把视频 latent 和音频 latent 拼成 H3 AV | 已用 `try/except` 包住，失败改用核心 `NestedTensor` |
| **ComfyUI_MiniMaxH3_Director**（AIMixer） | 无（只提供人像精修） | 只有用到 `MiniMaxH3TimelineFaceRefine` 节点才需要 | 4 张示例图都不含此节点 |
| **nvidia-vfx**（pip 包，提供 `nvvfx`） | 无 | 只有把二采超分方式改成 `nvidia_rtx_vsr` 才需要 | 自动退回 lanczos，不报错中断 |

AV 拼接的回退写法：

```616:625:custom_nodes/ComfyUI-MiniMaxH3-SegmentStudio/sequential_segments.py
            _node_class("PT_H3ConcatAVLatent").execute(video_latent, audio_latent)
        )
        if isinstance(packed, dict) and packed.get("samples") is not None:
            return packed
    except Exception:
        pass
    from comfy.nested_tensor import NestedTensor
```

### 3. 必须停用（不能同时开）

| 插件 | 原因 |
| --- | --- |
| `ComfyUI-MiniMaxH3-TimelineDirector` | 与本包注册的节点 ID 完全重合。同时启用时后加载的会覆盖先加载的，报“校验失败” |

改名成 `ComfyUI-MiniMaxH3-TimelineDirector.disabled` 后重启。本包在启动时会打印告警提醒这件事：

```20:26:custom_nodes/ComfyUI-MiniMaxH3-SegmentStudio/prestartup_script.py
        log.error(
            "MiniMax H3 Segment Studio: original Timeline Director is still enabled:\n"
            "  %s\n"
            "Rename that folder to ComfyUI-MiniMaxH3-TimelineDirector.disabled "
            "or node IDs will register twice and the last pack wins.",
            _CLASH,
        )
```

### 4. 不需要另装（已内置）

| 项目 | 说明 |
| --- | --- |
| SelfLift | 已打进 `selflift_runtime/`，不用再下载 |
| H3 3D latent Upscaler 插件 | 结构已内置，只需要模型文件 |
| Timeline Director | 功能已整包融合，不用再装上游 |

---

## 三、`nvidia_rtx_vsr` 到底依赖什么

**本包不打包 `nvidia-vfx`。** `requirements.txt` 里没有任何 pip 依赖：

```1:2:custom_nodes/ComfyUI-MiniMaxH3-SegmentStudio/requirements.txt
# No extra pip packages. Uses ComfyUI's torch / numpy / safetensors stack.
# Optional (already common in aki / kjnodes packs): SageAttention via PathchSageAttentionKJ.
```

`nvvfx` 是在函数内部**延迟导入**的，只有真正执行 VSR 时才加载：

```700:706:custom_nodes/ComfyUI-MiniMaxH3-SegmentStudio/segment_video_sampler.py
def _vsr_frames_one_by_one(frames: torch.Tensor, width: int, height: int, warmup=None) -> torch.Tensor:
    import nvvfx

    quality = getattr(getattr(nvvfx, "effects", None), "QualityLevel", None)
    level = getattr(quality, "ULTRA", None) if quality is not None else None
    ctx = nvvfx.VideoSuperRes(level) if level is not None else nvvfx.VideoSuperRes()
```

导入失败会被兜住，退回 lanczos 继续跑：

```777:782:custom_nodes/ComfyUI-MiniMaxH3-SegmentStudio/segment_video_sampler.py
        if method == "nvidia_rtx_vsr":
            try:
                out = _vsr_frames_one_by_one(frames, width, height, warmup)
            except Exception as exc:
                log.warning("nvidia_rtx_vsr failed (%s); falling back to lanczos", exc)
                out = _interpolate_frames(frames, width, height, "lanczos")
```

### 什么时候真的需要它

| 场景 | 需要 nvidia-vfx 吗 |
| --- | --- |
| 示例图原样跑（一采 Output MP = 0，二采 `h3_latent`） | **不需要** |
| `仅3D放大` 两张图 | **不需要** |
| 手动把二采超分方式改成 `nvidia_rtx_vsr`（追求最初版双采锐度） | **需要**，缺了退回 lanczos |

一采节点的 `upscale_method` 虽然写着 `nvidia_rtx_vsr`，但只有 `output_megapixels > 0` 时才会真正放大；4 张示例图的一采都是 `0`，所以那个值只是默认值，不会走到 VSR：

```1705:1712:custom_nodes/ComfyUI-MiniMaxH3-SegmentStudio/minimax_h3_finite_segments.py
                if float(output_megapixels or 0) > 0:
                    out_w, out_h = _megapixel_dims(src_w, src_h, output_megapixels)
                    pre_vsr = spill_images
                    pixel_method = str(upscale_method or "nvidia_rtx_vsr")
                    if pixel_method == "h3_latent":
                        pixel_method = "nvidia_rtx_vsr"
```

要装就装到 ComfyUI 那个 Python 里，并保持 NVIDIA 驱动足够新：

```bash
python -m pip install nvidia-vfx
```

---

## 四、报“校验失败”怎么查

典型表现（前端报错面板）：

- `由于节点校验检查意外失败，工作流无法通过校验` → 指向 `保存 15秒-多段 0.4`
- `工作流无法校验已连接的节点` → 指向 `按窗 T8 二采 · 15秒段 - video`

两处报错卡在**同一条线**上：一采节点 `MiniMaxH3FiniteSegmentSampler` 的第 5 个输出 `Final Video`（类型 `VIDEO`），分别接到 `SaveVideo` 和二采的 `一采成片`。

这不是缺模型、也不是缺插件，而是**节点定义对不上**。按顺序排查：

1. **旧版 Timeline Director 还开着。** 在 `custom_nodes` 里找不带 `.disabled` 的 `ComfyUI-MiniMaxH3-TimelineDirector`，整段改名加 `.disabled`，重启 ComfyUI。
2. **看启动控制台。** 搜 `register twice` 或 `Timeline Director is still enabled`，有这行就是第 1 条的原因。
3. **强刷前端。** 浏览器 `Ctrl+F5`，普通刷新不够。
4. **查接口。** 打开 `http://127.0.0.1:8188/api/object_info/MiniMaxH3FiniteSegmentSampler`，搜 `"Final Video"`。搜不到说明注册的仍是旧包。
5. **查 ComfyUI 本体版本。** 太旧的版本没有 `VIDEO` 类型，`SaveVideo` 也没有 `format` / `codec` 这两个控件（示例图用的是 `mp4` / `h264`）。

本机正常时的 schema 长这样，可以拿来对照：

```
MiniMaxH3WindowedRefine
  必需输入: model, clip, vae, audio_vae, sampler, seed, output_megapixels, ...
  可选输入: images, audio, finite_plan, source_video
MiniMaxH3FiniteSegmentSampler
  第 5 个输出: Final Video  VIDEO
```

注意报错里的 `- video` 是 `SaveVideo` 的**输入 socket 类型名**，不是某个节点的输入名，别顺着它去找 `video` 这个参数。

---

## 五、装完自检

1. 重启 ComfyUI，控制台没有 `register twice` / `Timeline Director is still enabled`。
2. `Ctrl+F5` 后从磁盘加载 `example_workflows/H3_SegmentStudio_8秒-多段_T8二采_2.0.json`。
3. 4 个加载器节点的下拉框里都能选到上面表格里的文件名，没有红框。
4. 一采 `Output Megapixels` 是 `0`；二采 `Final Video` 接到按窗二采。
5. 先跑 1 段短测，确认二采能读一采的 mp4 再上长片。
