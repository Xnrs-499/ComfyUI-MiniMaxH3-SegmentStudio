"""Live sampling preview on Finite Segment Sampler and Windowed Refine.

The KJNodes Model Preview Override widget is the reference UI, but it is a
separate graph node. This module attaches the same OUTER_SAMPLE wrapper to the
model while the preview DOM lives on our sampler nodes.
"""

from __future__ import annotations

import base64
import importlib.util
import io as pyio
import logging
import sys
from pathlib import Path

from comfy_api.latest import io

log = logging.getLogger("ComfyUI-MiniMaxH3-SegmentStudio.live_preview")

PREVIEW_EVENT = "h3ss_live_preview"
KJ_PREVIEW_EVENT = "kj_preview_override"
WRAPPER_KEY = "h3ss_live_preview"
JPEG_QUALITY = 80

try:
    import folder_paths
except Exception:
    folder_paths = None

try:
    from comfy.patcher_extension import WrappersMP
except Exception:
    WrappersMP = None


def tiny_vae_combo_options() -> list[str]:
    names = ["auto", "none"]
    try:
        if folder_paths is not None:
            names.extend(folder_paths.get_filename_list("vae_approx") or [])
    except Exception:
        pass
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def default_tiny_vae() -> str:
    try:
        files = list(folder_paths.get_filename_list("vae_approx") or []) if folder_paths else []
    except Exception:
        files = []
    preferred = (
        "MiniMax-H3-TAE.safetensors",
        "taeh3.safetensors",
        "taeh3.pth",
    )
    for name in preferred:
        if name in files:
            return name
    for name in files:
        lowered = name.lower()
        if "h3" in lowered and "tae" in lowered:
            return name
    return "none"


def resolve_tiny_vae(name: str | None) -> str:
    text = str(name or "auto").strip()
    if text in {"", "auto"}:
        return default_tiny_vae()
    if text == "none":
        return "none"
    return text


def node_unique_id(cls) -> str | None:
    value = getattr(getattr(cls, "hidden", None), "unique_id", None)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _kj_preview_override_module():
    # Prefer the already-loaded KJNodes module. Importing the file ourselves
    # fails because that module uses package-relative imports.
    for name, mod in list(sys.modules.items()):
        if getattr(mod, "_PreviewOverrideWrapper", None) is None:
            continue
        if "preview_override" in str(name):
            return mod
    return None


def _load_tiny_vae_decoder(name: str):
    if not name or name == "none":
        return None
    kj = _kj_preview_override_module()
    loader = getattr(kj, "load_tiny_vae_decoder", None) if kj is not None else None
    if loader is None:
        try:
            from comfyui_kjnodes.nodes.tiny_vae import load_tiny_vae_decoder as loader
        except Exception:
            loader = None
    if loader is None:
        tiny_path = (
            Path(__file__).resolve().parent.parent
            / "comfyui-kjnodes"
            / "nodes"
            / "tiny_vae.py"
        )
        if tiny_path.is_file():
            try:
                spec = importlib.util.spec_from_file_location("h3ss_kj_tiny_vae", tiny_path)
                if spec is not None and spec.loader is not None:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    loader = getattr(module, "load_tiny_vae_decoder", None)
            except Exception as exc:
                log.debug("tiny VAE loader import skipped: %s", exc)
    if loader is None:
        return None
    try:
        return loader(name)
    except Exception as exc:
        log.warning("Could not load tiny VAE %s: %s", name, exc)
        return None


def _send_preview(event: str, payload: dict) -> None:
    try:
        from server import PromptServer
    except Exception:
        return
    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return
    try:
        instance.send_sync(event, payload, getattr(instance, "client_id", None))
    except Exception:
        pass


class _StudioPreviewWrapper:
    """Minimal H3 preview when KJNodes is not importable."""

    def __init__(
        self, max_resolution, node_id, jpeg_quality, suppress_default,
        preview_frames=1, preview_fps=12, vae=None, tiny_vae="none",
    ):
        self.max_resolution = int(max_resolution or 0)
        self.node_id = str(node_id) if node_id is not None else None
        self.jpeg_quality = int(jpeg_quality or JPEG_QUALITY)
        self.suppress_default = bool(suppress_default)
        self.preview_frames = max(1, int(preview_frames or 1))
        self.preview_fps = max(1, int(preview_fps or 12))
        self.tiny_vae = tiny_vae
        self.vae = vae

    def __call__(
        self, executor, noise, latent_image, sampler, sigmas, denoise_mask,
        callback, disable_pbar, seed, latent_shapes=None,
    ):
        import time

        import numpy as np
        import torch
        from PIL import Image, ImageOps

        try:
            import latent_preview
        except Exception:
            latent_preview = None

        decoder = _load_tiny_vae_decoder(resolve_tiny_vae(self.tiny_vae))
        original_callback = callback
        node_id = self.node_id
        max_res = self.max_resolution
        quality = self.jpeg_quality
        anim_frames = self.preview_frames
        anim_fps = self.preview_fps
        sigmas_list = sigmas.detach().cpu().tolist() if sigmas is not None else []
        total_steps_init = max(0, len(sigmas_list) - 1)
        state = {"last_time": time.perf_counter()}

        if node_id is not None:
            _send_preview(PREVIEW_EVENT, {
                "node_id": node_id,
                "step": 0,
                "total": total_steps_init,
                "sigma": sigmas_list[0] if sigmas_list else None,
                "sigmas": sigmas_list,
            })

        def _to_pils(x0):
            frames = []
            if decoder is not None:
                try:
                    if x0.ndim == 5:
                        t_total = int(x0.shape[2])
                        if anim_frames < t_total:
                            indices = np.linspace(0, t_total - 1, anim_frames).round().astype(int).tolist()
                        else:
                            indices = list(range(t_total))
                        rgb = decoder.decode_video(x0[:1], frame_indices=indices)
                    elif x0.ndim == 4:
                        rgb = decoder.decode(x0[:1])[0].movedim(0, -1).unsqueeze(0)
                    else:
                        rgb = None
                    if rgb is not None:
                        u8 = rgb.clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
                        frames = [Image.fromarray(u8[i]) for i in range(u8.shape[0])]
                except Exception as exc:
                    log.warning("tiny VAE preview decode failed: %s", exc)
            if frames:
                return frames
            if latent_preview is None:
                return []
            try:
                previewer = latent_preview.get_previewer(
                    executor.class_obj.model_patcher.load_device,
                    executor.class_obj.model_patcher.model.latent_format,
                )
                out = previewer.decode_latent_to_preview(x0) if previewer is not None else None
                if isinstance(out, Image.Image):
                    return [out]
            except Exception:
                pass
            return []

        def _encode(pils):
            if not pils:
                return None, 0, 0, None
            first = pils[0].convert("RGB") if pils[0].mode != "RGB" else pils[0]
            sized = []
            for pil in pils:
                image = pil.convert("RGB") if pil.mode != "RGB" else pil
                if max_res and max_res > 0 and (image.width > max_res or image.height > max_res):
                    image = ImageOps.contain(image, (max_res, max_res), Image.LANCZOS)
                sized.append(image)
            buf = pyio.BytesIO()
            if len(sized) > 1:
                duration = max(1, int(1000 / anim_fps))
                sized[0].save(
                    buf, format="WEBP", save_all=True, append_images=sized[1:],
                    duration=duration, loop=0, quality=quality,
                )
                return base64.b64encode(buf.getvalue()).decode("ascii"), sized[0].width, sized[0].height, "image/webp"
            first = sized[0]
            first.save(buf, format="JPEG", quality=quality)
            return base64.b64encode(buf.getvalue()).decode("ascii"), first.width, first.height, "image/jpeg"

        def new_callback(step, x0, x, total_steps_):
            try:
                pils = _to_pils(x0)
                b64, width, height, mime = _encode(pils)
                if b64 and node_id is not None:
                    now = time.perf_counter()
                    step_ms = (now - state["last_time"]) * 1000.0
                    state["last_time"] = now
                    sigma_val = sigmas_list[step] if 0 <= step < len(sigmas_list) else None
                    _send_preview(PREVIEW_EVENT, {
                        "node_id": node_id,
                        "image": b64,
                        "mime": mime,
                        "w": width,
                        "h": height,
                        "step": step + 1,
                        "total": total_steps_,
                        "sigma": sigma_val,
                        "step_ms": step_ms,
                        "avg_step_ms": step_ms,
                        "fps": anim_fps if mime == "image/webp" else None,
                    })
            except Exception as exc:
                log.warning("Live preview callback failed: %s", exc)
            if original_callback is not None:
                original_callback(step, x0, x, total_steps_)

        return executor(
            noise, latent_image, sampler, sigmas, denoise_mask,
            new_callback, disable_pbar, seed, latent_shapes=latent_shapes,
        )


def attach_live_preview(
    model,
    node_id,
    *,
    enabled: bool = True,
    tiny_vae: str = "auto",
    max_resolution: int = 680,
    preview_frames: int = 60,
    preview_fps: int = 12,
):
    if not enabled or model is None:
        return model
    clone = getattr(model, "clone", None)
    if not callable(clone):
        return model
    try:
        patched = model.clone()
    except Exception:
        return model
    if not callable(getattr(patched, "add_wrapper_with_key", None)) or WrappersMP is None:
        return model
    vae_name = resolve_tiny_vae(tiny_vae)
    wrapper_cls = None
    kj = _kj_preview_override_module()
    if kj is not None:
        wrapper_cls = getattr(kj, "_PreviewOverrideWrapper", None)
    if wrapper_cls is None:
        wrapper_cls = _StudioPreviewWrapper
    try:
        wrapper = wrapper_cls(
            int(max_resolution or 0),
            node_id,
            JPEG_QUALITY,
            True,
            max(1, int(preview_frames or 1)),
            max(1, int(preview_fps or 12)),
            None,
            vae_name,
        )
        patched.add_wrapper_with_key(
            WrappersMP.OUTER_SAMPLE,
            WRAPPER_KEY,
            wrapper,
        )
    except Exception as exc:
        log.warning("Could not attach live preview wrapper: %s", exc)
        return model
    return patched


def live_preview_schema_inputs():
    return [
        io.Boolean.Input(
            "live_preview",
            default=True,
            display_name="Live Preview",
            tooltip=(
                "Show a KJ-style tiny-VAE preview on this node while sampling. "
                "Off skips the wrapper. Put MiniMax-H3-TAE.safetensors in models/vae_approx."
            ),
        ),
        io.Combo.Input(
            "preview_tiny_vae",
            options=tiny_vae_combo_options(),
            default="auto",
            display_name="Preview Tiny VAE",
            tooltip=(
                "auto picks MiniMax-H3-TAE.safetensors or taeh3.safetensors from "
                "models/vae_approx. none falls back to Latent2RGB."
            ),
        ),
        io.Int.Input(
            "preview_max_resolution",
            default=680,
            min=0,
            max=8192,
            step=8,
            display_name="Preview Max Resolution",
            tooltip="Max preview side in pixels. 0 = no downscale.",
        ),
        io.Int.Input(
            "preview_frames",
            default=60,
            min=1,
            max=1024,
            display_name="Preview Frames",
            tooltip=(
                "Animated preview frames decoded each step. 1 = still. "
                "Raise toward clip length (e.g. 241) for a full-clip loop; that costs more VRAM."
            ),
        ),
        io.Int.Input(
            "preview_fps",
            default=12,
            min=1,
            max=60,
            display_name="Preview FPS",
            tooltip="Playback FPS for the animated preview. Ignored when Preview Frames is 1.",
        ),
    ]


def graph_attach(graph, model, unique_id, *, enabled, tiny_vae, max_resolution, preview_frames, preview_fps):
    if not enabled:
        return model
    return graph.node(
        "MiniMaxH3LivePreviewAttach",
        id="live_preview_attach",
        model=model,
        target_node_id="" if unique_id is None else str(unique_id),
        enabled=True,
        tiny_vae=str(tiny_vae or "auto"),
        max_resolution=int(max_resolution or 680),
        preview_frames=int(preview_frames or 60),
        preview_fps=int(preview_fps or 12),
    ).out(0)


class MiniMaxH3LivePreviewAttach(io.ComfyNode):
    """Internal: wrap MODEL so SamplerCustomAdvanced streams previews to the parent node."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LivePreviewAttach",
            display_name="MiniMax H3 Live Preview Attach (Internal)",
            category="MiniMax H3/Internal",
            is_dev_only=True,
            inputs=[
                io.Model.Input("model"),
                io.String.Input("target_node_id", default=""),
                io.Boolean.Input("enabled", default=True),
                io.Combo.Input(
                    "tiny_vae",
                    options=tiny_vae_combo_options(),
                    default="auto",
                ),
                io.Int.Input("max_resolution", default=680, min=0, max=8192, step=8),
                io.Int.Input("preview_frames", default=60, min=1, max=1024),
                io.Int.Input("preview_fps", default=12, min=1, max=60),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls, model, target_node_id="", enabled=True, tiny_vae="auto",
        max_resolution=680, preview_frames=60, preview_fps=12,
    ):
        return io.NodeOutput(
            attach_live_preview(
                model,
                str(target_node_id or "").strip() or None,
                enabled=bool(enabled),
                tiny_vae=tiny_vae,
                max_resolution=int(max_resolution or 0),
                preview_frames=int(preview_frames or 1),
                preview_fps=int(preview_fps or 12),
            )
        )
