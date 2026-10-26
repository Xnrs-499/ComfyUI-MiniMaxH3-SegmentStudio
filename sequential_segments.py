"""Director-style sequential sampling for Timeline finite segments.

The expanded GraphBuilder path keeps every segment's AV latent linked as
previous_latent, so extra-long multi-segment jobs OOM even at 0.4 MP. This runner samples one
segment, spills mp4, moves the continuation latent to CPU, then optionally
unloads models — the same loop Director uses, still on Timeline's T8/turbo path.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch

import folder_paths

log = logging.getLogger("ComfyUI-MiniMaxH3-TimelineDirector.sequential")

SEQUENTIAL_AUTO_MIN_SEGMENTS = 8
CACHE_ROOT_DIRNAME = "minimax_seg_cache"
EXECUTION_MODES = ("auto", "sequential", "graph")


def should_run_sequential(execution_mode: str, segment_count: int) -> bool:
    mode = str(execution_mode or "auto").strip().lower()
    if mode == "graph":
        return False
    if mode == "sequential":
        return True
    return int(segment_count or 0) >= SEQUENTIAL_AUTO_MIN_SEGMENTS


def normalize_cache_name(raw: Any) -> str:
    text = str(raw or "").strip()
    text = re.sub(r'[<>:"/\\\\|?*]', "_", text)
    if text.upper() in {"CON", "PRN", "AUX", "NUL"} or re.match(r"^(COM|LPT)\d+$", text.upper()):
        text = f"_{text}"
    return text[:64]


def sequential_fingerprint(
    finite: dict,
    *,
    seed: int,
    continue_audio_latent: bool,
    ref_image_size: str,
    export_mode: str,
    output_megapixels: float = 0.0,
    upscale_method: str = "nvidia_rtx_vsr",
    refine_denoise: float = 0.0,
    refine_steps: int = 0,
    face_refine=None,
) -> str:
    source = finite.get("source_plan") or {}
    payload = {
        "prompts": list(finite.get("prompts") or []),
        "lengths": list(finite.get("segment_lengths") or []),
        "overlaps": list(finite.get("segment_overlaps") or []),
        "target_output_frames": int(finite.get("target_output_frames") or 0),
        "second_pass": bool(finite.get("second_pass")),
        "second_pass_model": str(finite.get("second_pass_model") or ""),
        "second_pass_high_steps": finite.get("second_pass_high_steps"),
        "width": int(source.get("width") or 0),
        "height": int(source.get("height") or 0),
        "seed": int(seed),
        "continue_audio_latent": bool(continue_audio_latent),
        "ref_image_size": str(ref_image_size),
        "export_mode": str(export_mode),
        "output_megapixels": float(output_megapixels or 0),
        "upscale_method": str(upscale_method or ""),
        "refine_denoise": float(refine_denoise or 0),
        "refine_steps": int(refine_steps or 0),
        "face_refine": face_refine is not None,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def unpack_node_output(out):
    if out is None:
        return ()
    result = getattr(out, "result", None)
    if result is not None:
        return tuple(result) if isinstance(result, (tuple, list)) else (result,)
    args = getattr(out, "args", None)
    if args is not None:
        return tuple(args) if isinstance(args, (tuple, list)) else (args,)
    if isinstance(out, (tuple, list)):
        return tuple(out)
    return (out,)


def cleanup_segment_vram(*, enabled: bool = True, unload_models: bool = True) -> None:
    """Release GPU memory between segments. Adapted from Director's vram_cleanup."""
    if not enabled:
        return
    gc.collect()
    try:
        import comfy.model_management as mm

        mm.cleanup_models_gc()
        _evict_dead_loaded_models()
        if unload_models:
            mm.unload_all_models()
            mm.cleanup_models()
        _evict_dead_loaded_models()
        gc.collect()
        mm.soft_empty_cache()
    except Exception as exc:
        log.warning("Segment VRAM cleanup failed: %s", exc)


def _release_comfy_compiler_graphs() -> None:
    """Drop AIMDO CUDA graphs that Comfy only resets when a node returns.

    Windowed refine is one long node. After window-1 T8, graphs keep ~10GB
    reserved, so window-2 LatentResizer3D conv3d sees only ~4GB free.
    """
    try:
        from comfy.model_prefetch import GRAPH_WARMED_MODULES, _drop_graph, cleanup_prefetch_queues

        for module in list(GRAPH_WARMED_MODULES):
            try:
                _drop_graph(module)
            except Exception:
                continue
        cleanup_prefetch_queues()
    except Exception as exc:
        log.warning("Compiler graph cleanup failed: %s", exc)
    try:
        import comfy.model_management as mm

        if hasattr(mm, "reset_cast_buffers"):
            mm.reset_cast_buffers()
    except Exception:
        pass
    try:
        import comfy_aimdo.model_vbar as vbar

        if hasattr(vbar, "vbars_reset_watermark_limits"):
            vbar.vbars_reset_watermark_limits()
    except Exception:
        pass
    try:
        torch.compiler.reset()
    except Exception:
        try:
            torch._dynamo.reset()
        except Exception:
            pass


def _free_gpu_for_phase(reason: str = "") -> None:
    """Evict compiler graphs plus dynamic UNET/CLIP before 3D lift."""
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    _release_comfy_compiler_graphs()
    cleanup_segment_vram(enabled=True, unload_models=True)
    try:
        import comfy.model_management as mm

        device = mm.get_torch_device()
        mm.free_memory(1e30, device)
        mm.soft_empty_cache(True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        free = mm.get_free_memory(device)
        log.info(
            "GPU free before %s: %.1f GB",
            reason or "next phase",
            float(free) / (1024 ** 3),
        )
    except Exception as exc:
        log.warning("GPU phase free failed: %s", exc)


def _evict_dead_loaded_models() -> int:
    try:
        import comfy.model_management as mm
    except Exception:
        return 0
    models = getattr(mm, "current_loaded_models", None)
    if not models:
        return 0
    evicted = 0
    for i in range(len(models) - 1, -1, -1):
        cur = models[i]
        try:
            if not cur.is_dead():
                continue
            models.pop(i)
            evicted += 1
        except Exception:
            continue
    return evicted


def _tensorish_to_cpu(value):
    if getattr(value, "is_nested", False):
        from comfy.nested_tensor import NestedTensor

        parts = tuple(p.detach().to("cpu").contiguous() for p in value.unbind())
        return NestedTensor(parts)
    if torch.is_tensor(value):
        return value.detach().to("cpu").contiguous()
    return value


def _deep_to_cpu(obj, memo=None):
    """Clone tensors in nested cond/latent trees onto CPU so caches cannot pin VRAM."""
    if memo is None:
        memo = {}
    obj_id = id(obj)
    if obj_id in memo:
        return memo[obj_id]
    if getattr(obj, "is_nested", False):
        from comfy.nested_tensor import NestedTensor

        out = NestedTensor(
            tuple(part.detach().to("cpu").contiguous().clone() for part in obj.unbind())
        )
        memo[obj_id] = out
        return out
    if torch.is_tensor(obj):
        out = obj.detach().to("cpu").contiguous().clone()
        memo[obj_id] = out
        return out
    if isinstance(obj, dict):
        out = {}
        memo[obj_id] = out
        for key, value in obj.items():
            out[key] = _deep_to_cpu(value, memo)
        return out
    if isinstance(obj, list):
        out = []
        memo[obj_id] = out
        out.extend(_deep_to_cpu(value, memo) for value in obj)
        return out
    if isinstance(obj, tuple):
        out = tuple(_deep_to_cpu(value, memo) for value in obj)
        memo[obj_id] = out
        return out
    return obj


def latent_to_cpu(latent):
    if not isinstance(latent, dict):
        return latent
    out = dict(latent)
    if "samples" in out:
        out["samples"] = _tensorish_to_cpu(out["samples"])
    if out.get("noise_mask") is not None:
        out["noise_mask"] = _tensorish_to_cpu(out["noise_mask"])
    carry = out.get("selflift_low_resolution_carry")
    if torch.is_tensor(carry):
        out["selflift_low_resolution_carry"] = carry.detach().to("cpu")
    elif isinstance(carry, dict):
        out["selflift_low_resolution_carry"] = latent_to_cpu(carry)
    return out


def _cache_dir(cache_name: str, fingerprint: str) -> Path:
    root = Path(folder_paths.get_output_directory()) / CACHE_ROOT_DIRNAME
    name = normalize_cache_name(cache_name) or fingerprint
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_resume(
    cache_dir: Path, fingerprint: str, state: dict, previous_latent, previous_images,
    previous_refine_latent=None,
) -> None:
    payload = {
        "fingerprint": fingerprint,
        "next_iteration": int(state.get("next_iteration") or 0),
        "video_files": list(state.get("video_files") or []),
        "written_frames": int(state.get("written_frames") or 0),
        "spill_directory": state.get("spill_directory"),
        "export_directory": state.get("export_directory"),
        "export_mode": state.get("export_mode"),
        "continue_audio_latent": bool(state.get("continue_audio_latent")),
    }
    (cache_dir / "progress.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    if previous_latent is not None:
        torch.save(latent_to_cpu(previous_latent), cache_dir / "previous_latent.pt")
    if previous_images is not None:
        torch.save(_tensorish_to_cpu(previous_images), cache_dir / "previous_images.pt")
    if previous_refine_latent is not None:
        torch.save(latent_to_cpu(previous_refine_latent), cache_dir / "previous_refine_latent.pt")
    merged_audio = state.get("merged_audio")
    if merged_audio is not None:
        audio = dict(merged_audio)
        waveform = audio.get("waveform")
        if torch.is_tensor(waveform):
            audio["waveform"] = waveform.detach().to("cpu")
        torch.save(audio, cache_dir / "merged_audio.pt")


def _load_resume(cache_dir: Path, fingerprint: str, finite: dict) -> dict | None:
    progress_path = cache_dir / "progress.json"
    if not progress_path.is_file():
        return None
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if str(payload.get("fingerprint") or "") != fingerprint:
        return None
    completed = int(payload.get("next_iteration") or 0)
    if completed < 1:
        return None
    files = [Path(path) for path in payload.get("video_files") or []]
    if len(files) != completed or any(not path.is_file() for path in files):
        log.warning("Sequential resume skipped: segment mp4 files are missing")
        return None
    latent_path = cache_dir / "previous_latent.pt"
    if not latent_path.is_file():
        return None
    try:
        previous_latent = torch.load(latent_path, map_location="cpu", weights_only=False)
    except TypeError:
        previous_latent = torch.load(latent_path, map_location="cpu")
    previous_images = None
    images_path = cache_dir / "previous_images.pt"
    if images_path.is_file():
        try:
            previous_images = torch.load(images_path, map_location="cpu", weights_only=False)
        except TypeError:
            previous_images = torch.load(images_path, map_location="cpu")
    merged_audio = None
    audio_path = cache_dir / "merged_audio.pt"
    if audio_path.is_file():
        try:
            merged_audio = torch.load(audio_path, map_location="cpu", weights_only=False)
        except TypeError:
            merged_audio = torch.load(audio_path, map_location="cpu")
    previous_refine_latent = None
    refine_path = cache_dir / "previous_refine_latent.pt"
    if refine_path.is_file():
        try:
            previous_refine_latent = torch.load(refine_path, map_location="cpu", weights_only=False)
        except TypeError:
            previous_refine_latent = torch.load(refine_path, map_location="cpu")
    return {
        "finite_plan": finite,
        "next_iteration": completed,
        "previous_latent": previous_latent,
        "previous_images": previous_images,
        "previous_refine_latent": previous_refine_latent,
        "video_files": [str(path) for path in files],
        "merged_audio": merged_audio,
        "written_frames": int(payload.get("written_frames") or 0),
        "spill_directory": payload.get("spill_directory"),
        "continue_audio_latent": bool(payload.get("continue_audio_latent")),
        "export_mode": payload.get("export_mode"),
        "export_directory": payload.get("export_directory"),
    }


def _interrupt() -> None:
    try:
        import comfy.model_management as mm

        mm.throw_exception_if_processing_interrupted()
    except Exception:
        pass


def _decode_av(sampled, vae, audio_vae, *, decode_audio: bool):
    from comfy_extras.nodes_audio import vae_decode_audio

    latent = sampled["samples"] if isinstance(sampled, dict) else sampled
    video_latent = latent.unbind()[0] if getattr(latent, "is_nested", False) else latent
    images = vae.decode(video_latent)
    if len(images.shape) == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    images = images.detach().to(device="cpu", dtype=torch.float32).contiguous()
    audio = vae_decode_audio(audio_vae, sampled) if decode_audio else None
    return images, audio


def _sample_latent(model, positive, sampling_latent, sampler, sigmas, seed, vae, selflift_settings):
    if selflift_settings:
        from .selflift_runtime.nodes import SelfLiftH3Sampler

        sampled, = SelfLiftH3Sampler().sample(
            model, positive, positive, vae, sampling_latent, sampler, sigmas, int(seed), 1.0,
            selflift_settings["transition_step"], selflift_settings["lowres_scale"],
            selflift_settings["rho"], selflift_settings["w_min"], selflift_settings["w_max"],
            selflift_settings["upscaler_model"],
        )
        return sampled
    from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

    noise_obj = unpack_node_output(RandomNoise.execute(int(seed)))[0]
    guider_obj = unpack_node_output(BasicGuider.execute(model, positive))[0]
    try:
        sampled = unpack_node_output(
            SamplerCustomAdvanced.execute(noise_obj, guider_obj, sampler, sigmas, sampling_latent)
        )[0]
    finally:
        guider_obj = None
        noise_obj = None
    return sampled


def h3_grid_length(frames: int) -> int:
    count = max(int(frames), 5)
    if (count - 5) % 17 == 0:
        return count
    return 5 + 17 * (((count - 5) + 16) // 17)


def owning_gen_window(gen_windows: list[dict], start: int, end: int, trim: int = 0) -> dict | None:
    """Pick the first-pass GEN box that covers most of this refine hop's new frames."""
    kept_start = int(start) + max(int(trim), 0)
    kept_end = int(end)
    best = None
    best_overlap = 0
    for item in gen_windows or []:
        lo = max(kept_start, int(item["start"]))
        hi = min(kept_end, int(item["end"]))
        overlap = hi - lo
        if overlap > best_overlap:
            best_overlap = overlap
            best = item
    return best


def windows_from_plan(finite: dict, total_frames: int) -> list[dict]:
    """Map overlap-trimmed concat frames back to overlapping refine windows."""
    from .experimental_latent_guide import _valid_guide_frames

    count = int(finite.get("segment_count") or 0)
    lengths = list(finite.get("segment_lengths") or [])
    overlaps = list(finite.get("segment_overlaps") or [0] * count)
    if count < 1 or len(lengths) != count:
        raise ValueError("finite_plan is missing segment lengths")
    total = max(int(total_frames), 0)
    kept = 0
    windows = []
    for index in range(count):
        length = int(lengths[index])
        overlap = int(overlaps[index]) if index else 0
        trim = _valid_guide_frames(overlap) if overlap else 0
        start = max(0, kept - trim)
        end = min(total, start + length)
        if end - start <= trim:
            raise ValueError(
                f"Refine window {index + 1} ({end - start} frames) is shorter than its {trim}-frame overlap"
            )
        windows.append({"index": index, "start": start, "end": end, "trim": trim})
        kept += end - start - trim
    return windows


REFINE_MAX_FRAMES_DEFAULT = 141  # 5.875s at 24fps (5+17*8). 16GB-safe at 2.0; not a clamp.
WINDOW_FRAME_PRESETS = (141, 158, 175, 192, 243, 362)
WINDOW_FRAME_CHOICES = (
    "141 (~5.9s)",
    "158 (~6.6s)",
    "175 (~7.3s)",
    "192 (~8s)",
    "243 (~10s)",
    "362 (whole 15s)",
)
WINDOW_FRAME_CHOICE_DEFAULT = WINDOW_FRAME_CHOICES[0]


def t8_window_frame_cap(requested: int | None = None) -> int:
    """Snap Window Frames to an H3-legal length (5+17n). Uses the widget as-is."""
    raw = int(requested if requested is not None else REFINE_MAX_FRAMES_DEFAULT)
    return h3_grid_length(max(raw, 5))


def parse_window_frames(value) -> int:
    """Read the Window Frames combo (or a legacy int) into an H3-legal length."""
    if value is None or isinstance(value, bool):
        return REFINE_MAX_FRAMES_DEFAULT
    if isinstance(value, (int, float)):
        return t8_window_frame_cap(int(value))
    text = str(value).strip().lower()
    if not text:
        return REFINE_MAX_FRAMES_DEFAULT
    if "15" in text and any(token in text for token in ("whole", "full", "segment")):
        return 362
    match = re.search(r"(\d+)", text)
    if match:
        return t8_window_frame_cap(int(match.group(1)))
    return REFINE_MAX_FRAMES_DEFAULT


def windows_capped_for_refine(
    windows: list[dict],
    total_frames: int,
    max_frames: int = REFINE_MAX_FRAMES_DEFAULT,
    overlap_frames: int = 39,
) -> tuple[list[dict], bool]:
    """Keep plan windows when they fit the 2.0 T8 budget; otherwise slide capped hops."""
    cap = h3_grid_length(max(int(max_frames), 5))
    items = list(windows or [])
    if items and all(int(item["end"]) - int(item["start"]) <= cap for item in items):
        return items, False
    return windows_sliding(total_frames, cap, overlap_frames), True


def windows_sliding(total_frames: int, window_frames: int = 192, overlap_frames: int = 39) -> list[dict]:
    from .experimental_latent_guide import _valid_guide_frames

    total = max(int(total_frames), 0)
    overlap = _valid_guide_frames(int(overlap_frames)) if int(overlap_frames) >= 5 else 0
    window = h3_grid_length(max(int(window_frames), overlap + 5))
    if total <= 0:
        return []
    if total <= window:
        return [{"index": 0, "start": 0, "end": total, "trim": 0}]
    windows = []
    start = 0
    while start < total:
        end = min(total, start + window)
        trim = overlap if start else 0
        if end - start <= trim:
            break
        windows.append({"start": start, "end": end, "trim": trim})
        if end >= total:
            break
        start = end - overlap
    for index, item in enumerate(windows):
        item["index"] = index
    return windows


def is_placeholder_images(images) -> bool:
    """True when Merged Frames were omitted (8x8 stub) or the IMAGE input is empty."""
    if images is None or not torch.is_tensor(images):
        return True
    frames = images
    if frames.ndim == 5:
        frames = frames.reshape(-1, *frames.shape[-3:])
    if frames.ndim != 4:
        return True
    return int(frames.shape[0]) < 5 or min(int(frames.shape[1]), int(frames.shape[2])) <= 16


def decode_video_window(video, start: int, count: int, fps: float = 24.0):
    """Load one H3 window from a file-backed 0.4 mp4. Never materializes the whole clip."""
    from comfy_api.latest._input_impl.video_types import VideoFromFile

    if video is None:
        raise ValueError("Windowed refine has no first-pass video to read")
    src = video.get_stream_source() if hasattr(video, "get_stream_source") else video
    if not isinstance(src, (str, Path)):
        raise ValueError(
            "Windowed refine from video needs the first-pass Final Video as a file-backed mp4. "
            "Keep sampler Output Megapixels at 0 so the 0.4 clip is spilled to disk."
        )
    start = max(0, int(start))
    count = max(1, int(count))
    rate = float(fps or 24.0)
    try:
        raw_rate = video.get_frame_rate() if hasattr(video, "get_frame_rate") else None
        if raw_rate:
            rate = float(raw_rate)
    except Exception:
        pass
    clip = VideoFromFile(str(src), start_time=start / rate, duration=count / rate)
    images = clip.get_components().images
    if images is None or not torch.is_tensor(images) or int(images.shape[0]) == 0:
        raise ValueError(
            f"Could not decode frames {start}:{start + count} from first-pass mp4 {src}"
        )
    frames = images.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if frames.ndim == 5:
        frames = frames.reshape(-1, *frames.shape[-3:])
    have = int(frames.shape[0])
    if have < count:
        pad = count - have
        frames = torch.cat([frames, frames[-1:].expand(pad, -1, -1, -1)], dim=0)
    return frames[:count].contiguous()


def _node_class(name: str):
    from nodes import NODE_CLASS_MAPPINGS

    cls = NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError(f"{name} is not installed")
    return cls


def _encode_pixels(vae, images):
    from nodes import VAEEncode

    packed = VAEEncode().encode(vae, images)[0]
    return packed


def _concat_av_latent(video_latent, audio_latent):
    try:
        packed = unpack_node_output(
            _node_class("PT_H3ConcatAVLatent").execute(video_latent, audio_latent)
        )[0]
        if isinstance(packed, dict) and packed.get("samples") is not None:
            return packed
    except Exception:
        pass
    from comfy.nested_tensor import NestedTensor

    video = video_latent["samples"] if isinstance(video_latent, dict) else video_latent
    audio = audio_latent["samples"] if isinstance(audio_latent, dict) else audio_latent
    return {"samples": NestedTensor((video, audio))}


def _silent_audio(frame_count: int, sample_rate: int = 44100):
    from .segment_video_sampler import H3_FPS

    samples = max(1, round(int(frame_count) / H3_FPS * int(sample_rate)))
    return {"waveform": torch.zeros(1, 2, samples), "sample_rate": int(sample_rate)}


def _pad_frames(images, length: int):
    have = int(images.shape[0])
    need = int(length)
    if have == need:
        return images
    if have > need:
        return images[:need].contiguous()
    last = images[-1:].expand(need - have, -1, -1, -1)
    return torch.cat([images, last], dim=0).contiguous()


VAE_SPATIAL = 16
H3_LATENT_UPSCALE = "h3_latent"


def pixel_to_latent_hw(width: int, height: int) -> tuple[int, int]:
    """Pixel canvas W×H → video-latent (H, W). MiniMax VAE spatial ÷16."""
    return max(1, int(height) // VAE_SPATIAL), max(1, int(width) // VAE_SPATIAL)


def h3_upscaler_combo_options() -> list[str]:
    try:
        from .selflift_runtime.h3_upscaler import checkpoint_is_h3_compatible, list_upscaler_models

        names = [name for name in list_upscaler_models() if checkpoint_is_h3_compatible(name)]
    except Exception:
        names = []
    return ["auto"] + names if names else ["auto"]


def resolve_h3_upscaler_name(requested: str = "") -> str:
    from .selflift_runtime.h3_upscaler import checkpoint_is_h3_compatible, list_upscaler_models

    models = list_upscaler_models()
    compatible = [name for name in models if checkpoint_is_h3_compatible(name)]
    if not compatible:
        raise ValueError(
            "Director-style 2.0 refine needs a MiniMax H3 latent upscaler under "
            "ComfyUI/models/latent_upscale_models (for example "
            "minimax_h3_latent_upscaler_3d_bf16.safetensors). LTX upscalers are not compatible. "
            "Or set Upscale Method to nvidia_rtx_vsr / lanczos (encodes 2.0 pixels; heavier)."
        )
    selected = str(requested or "").strip()
    if selected in {"", "none", "auto"}:
        preferred = [name for name in compatible if "h3" in name.lower()]
        return preferred[0] if preferred else compatible[0]
    if selected not in models:
        raise ValueError(f"The selected H3 latent upscaler is unavailable: {selected}")
    if not checkpoint_is_h3_compatible(selected):
        raise ValueError(f"The selected latent upscaler is not H3-compatible: {selected}")
    return selected


def _latent_without_mask(latent):
    if not isinstance(latent, dict):
        return latent
    out = dict(latent)
    out.pop("noise_mask", None)
    return out


def _av_video_audio(latent, label: str = "refine_latent"):
    from .experimental_latent_guide import _h3_streams

    try:
        return _h3_streams(latent, label)
    except ValueError:
        samples = latent.get("samples") if isinstance(latent, dict) else latent
        if getattr(samples, "is_nested", False):
            parts = tuple(samples.unbind())
            video = parts[0]
            audio = parts[1] if len(parts) > 1 else None
        else:
            video, audio = samples, None
        if torch.is_tensor(video) and video.ndim == 4:
            video = video.unsqueeze(0)
        if not torch.is_tensor(video) or video.ndim != 5 or int(video.shape[1]) != 24:
            raise ValueError(f"{label} is not a valid MiniMax H3 video latent")
        if audio is None or not torch.is_tensor(audio):
            raise ValueError(f"{label} is missing the H3 audio stream")
        return video, audio


def _slice_av_to_frames(latent, frame_count: int):
    """Drop H3-grid padding from a nested AV latent. Keeps the overlap tail."""
    from comfy.nested_tensor import NestedTensor

    from .experimental_latent_guide import _video_latent_t

    video, audio = _av_video_audio(latent, "refine_latent")
    frames = max(int(frame_count), 5)
    tokens = min(_video_latent_t(frames), int(video.shape[2]))
    audio_tokens = min(max(1, round(int(frame_count) * 40 / 24)), int(audio.shape[-1]))
    out = dict(latent) if isinstance(latent, dict) else {}
    out["samples"] = NestedTensor((
        video[:, :, :tokens].contiguous(),
        audio[..., :audio_tokens].contiguous(),
    ))
    out.pop("noise_mask", None)
    return out


def _lift_av_latent(av_latent, out_w: int, out_h: int, model_name: str):
    from comfy.nested_tensor import NestedTensor

    from .selflift_runtime.h3_upscaler import learned_latent_lift

    video, audio = _av_video_audio(av_latent, "refine_latent")
    dst_hw = pixel_to_latent_hw(out_w, out_h)
    if tuple(video.shape[-2:]) == dst_hw:
        return _latent_without_mask(av_latent)
    # 16GB: sibling dynamic UNET/CLIP are not evicted by load_models_gpu.
    # After window-1 T8, Comfy CUDA graphs can leave only ~4GB until node end.
    _free_gpu_for_phase("3D latent lift")
    max_chunk = None
    if int(dst_hw[0]) * int(dst_hw[1]) > 40 * 70:
        free_gb = 0.0
        try:
            import comfy.model_management as mm

            free_gb = float(mm.get_free_memory(mm.get_torch_device())) / (1024 ** 3)
        except Exception:
            free_gb = 0.0
        if free_gb < 6.0:
            max_chunk = 2
        elif free_gb < 10.0:
            max_chunk = 4
        else:
            max_chunk = 8
        log.info("3D lift max_chunk=%s (free %.1f GB)", max_chunk, free_gb)
    lifted = learned_latent_lift(video, dst_hw, model_name, max_chunk=max_chunk)
    lifted = lifted.detach().to(device="cpu", dtype=torch.float32).contiguous()
    audio = audio.detach().to(device="cpu").contiguous()
    out = dict(av_latent) if isinstance(av_latent, dict) else {}
    out["samples"] = NestedTensor((lifted, audio))
    out.pop("noise_mask", None)
    return out


def lift_decode_window(
    *,
    images,
    audio,
    vae, audio_vae,
    src_w, src_h, out_w, out_h,
    trim=0,
    upscale_method=H3_LATENT_UPSCALE,
    latent_upscale_model="",
):
    """Encode 0.4, 3D-lift to the output grid, decode. No T8 sample."""
    from comfy_extras.nodes_audio import VAEEncodeAudio

    from .segment_video_sampler import MiniMaxH3MatchFrameCount, _pin_audio_to_frames

    frames = images.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if frames.ndim == 5:
        frames = frames.reshape(-1, *frames.shape[-3:])
    if int(frames.shape[1]) <= 16 and int(frames.shape[2]) <= 16:
        raise ValueError(
            "Lift-only refine needs the 0.4-MP frame batch. Sampler Output Megapixels "
            "must stay 0 or Merged Frames is the 8x8 placeholder."
        )
    source_count = int(frames.shape[0])
    length = h3_grid_length(source_count)
    frames_in = _pad_frames(frames, length)
    if audio is None:
        audio_in = _silent_audio(length)
    else:
        audio_in = _pin_audio_to_frames(dict(audio), length)

    method = str(upscale_method or H3_LATENT_UPSCALE).strip().lower()
    same_canvas = int(out_w) == int(src_w) and int(out_h) == int(src_h)
    audio_latent = unpack_node_output(VAEEncodeAudio.execute(audio_vae, audio_in))[0]
    if same_canvas:
        video_latent = _encode_pixels(vae, frames_in)
        av_latent = _concat_av_latent(video_latent, audio_latent)
        del video_latent
    elif method == H3_LATENT_UPSCALE:
        video_latent = _encode_pixels(vae, frames_in)
        av_latent = _lift_av_latent(
            _concat_av_latent(video_latent, audio_latent),
            int(out_w), int(out_h),
            resolve_h3_upscaler_name(latent_upscale_model),
        )
        del video_latent
        cleanup_segment_vram(enabled=True, unload_models=True)
    else:
        frames_hi = _pixel_upscale_frames(frames_in, int(out_w), int(out_h), method)
        video_latent = _encode_pixels(vae, frames_hi)
        av_latent = _concat_av_latent(video_latent, audio_latent)
        del frames_hi, video_latent

    sampled = latent_to_cpu(_slice_av_to_frames(av_latent, source_count))
    cleanup_segment_vram(enabled=True, unload_models=True)
    refined, _unused_audio = _decode_av(sampled, vae, audio_vae, decode_audio=False)
    matched = unpack_node_output(MiniMaxH3MatchFrameCount.execute(refined, frames))[0]
    pin = int(trim or 0)
    kept = matched[pin:] if pin else matched
    del frames_in, audio_latent, av_latent, sampled, refined
    _free_gpu_for_phase("after 3D lift decode")
    return kept.contiguous()


def _encode_refine_positive(
    *,
    clip, vae, audio_vae,
    length: int,
    src_w: int, src_h: int,
    out_w: int, out_h: int,
    refine_prompt: str = "",
    refine_plan=None,
):
    """Reuse first-pass prompt+refs (original dual-pass style). Fall back to the generic T8 refine prompt."""
    from .minimax_h3_finite_segments import _REFINE_PROMPT
    from .minimax_h3_timeline_director import MiniMaxH3TimelineEncoder

    prompt = str(refine_prompt or "").strip()
    plan = refine_plan if isinstance(refine_plan, dict) else None
    if plan is not None:
        timeline = plan.get("timeline") if isinstance(plan.get("timeline"), dict) else {}
        if not prompt:
            prompt = str(timeline.get("globalPrompt") or "").strip()
        enc_plan = copy.deepcopy(plan)
        enc_plan["width"] = int(src_w)
        enc_plan["height"] = int(src_h)
        enc_plan["length"] = int(length)
        enc_plan["generation_seconds"] = int(length) / 24.0
        try:
            positive = unpack_node_output(
                MiniMaxH3TimelineEncoder.execute(
                    clip, vae, audio_vae, enc_plan, prompt, "match",
                )
            )[0]
            log.info(
                "Refine reused first-pass prompt+refs at %dx%d length %d",
                int(src_w), int(src_h), int(length),
            )
            return positive
        except Exception as exc:
            log.warning("First-pass refine conditioning failed (%s); using T8 refine prompt", exc)

    t8_cls = _node_class("MiniMaxH3AudioConditioningT8")
    return unpack_node_output(
        t8_cls.execute(
            clip, vae, audio_vae, prompt or _REFINE_PROMPT,
            int(out_w), int(out_h), int(length),
            "auto", "native", 1.0,
            False, 0, True, "match", "official_2_to_15s",
            allow_above_reference_area=True,
        )
    )[0]


def _pixel_upscale_frames(frames, out_w: int, out_h: int, method: str, warmup=None):
    from .segment_video_sampler import MiniMaxH3SegmentFrameUpscale

    pixel_method = str(method or "nvidia_rtx_vsr")
    if pixel_method == H3_LATENT_UPSCALE:
        pixel_method = "nvidia_rtx_vsr"
    return unpack_node_output(
        MiniMaxH3SegmentFrameUpscale.execute(
            frames, int(out_w), int(out_h), pixel_method, warmup,
        )
    )[0]


def refine_window_director_style(
    *,
    images,
    audio,
    model, clip, vae, audio_vae, sampler,
    seed, src_w, src_h, out_w, out_h,
    refine_steps, refine_denoise,
    trim=0, previous_refine_latent=None,
    upscale_method=H3_LATENT_UPSCALE,
    latent_upscale_model="",
    refine_prompt="",
    refine_plan=None,
    positive=None,
    positive_cache=None,
    positive_cache_key=None,
    vsr_warmup=None,
):
    """Director 2.0 refine on Timeline's T8 path.

    Encode the 0.4 window, lift to the output latent grid, re-pin the previous
    2.0 AV tail, sample 2 T8 steps at the output canvas, decode, then trim the
    pin. Never holds a whole-clip 2.0 tensor.

    Returns ``(kept_images, full_sampled_latent_on_cpu)``. The latent still
    includes the overlap tail so the next window can re-pin.
    """
    from comfy_extras.nodes_audio import VAEEncodeAudio
    from comfy_extras.nodes_custom_sampler import BasicScheduler

    from .minimax_h3_finite_segments import MiniMaxH3FiniteLatentContinuation
    from .segment_video_sampler import MiniMaxH3MatchFrameCount, _pin_audio_to_frames

    frames = images.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if frames.ndim == 5:
        frames = frames.reshape(-1, *frames.shape[-3:])
    if int(frames.shape[1]) <= 16 and int(frames.shape[2]) <= 16:
        raise ValueError(
            "Refine needs the 0.4-MP frame batch. Sampler Output Megapixels must stay 0 "
            "or Merged Frames is the 8x8 placeholder."
        )
    source_count = int(frames.shape[0])
    length = h3_grid_length(source_count)
    frames_in = _pad_frames(frames, length)
    if audio is None:
        audio_in = _silent_audio(length)
    else:
        audio_in = _pin_audio_to_frames(dict(audio), length)

    method = str(upscale_method or H3_LATENT_UPSCALE).strip().lower()
    same_canvas = int(out_w) == int(src_w) and int(out_h) == int(src_h)
    pixel_vsr = (not same_canvas) and method != H3_LATENT_UPSCALE
    audio_latent = unpack_node_output(VAEEncodeAudio.execute(audio_vae, audio_in))[0]

    def _ensure_positive():
        nonlocal positive
        if positive is not None:
            return
        if positive_cache is not None and positive_cache_key in positive_cache:
            positive = _deep_to_cpu(positive_cache[positive_cache_key])
            return
        positive = _encode_refine_positive(
            clip=clip, vae=vae, audio_vae=audio_vae,
            length=int(length), src_w=int(src_w), src_h=int(src_h),
            out_w=int(out_w), out_h=int(out_h),
            refine_prompt=refine_prompt, refine_plan=refine_plan,
        )
        if positive_cache is not None and positive_cache_key is not None:
            positive_cache[positive_cache_key] = _deep_to_cpu(positive)
            positive = _deep_to_cpu(positive_cache[positive_cache_key])
        else:
            positive = _deep_to_cpu(positive)

    if pixel_vsr:
        # Original dual-pass: CLIP/refs first, then VSR pixels, then 2.0 encode.
        # Keep CLIP (~15GB) off the GPU while 2.0 RGB and T8 UNET run.
        _ensure_positive()
        cleanup_segment_vram(enabled=True, unload_models=True)
        _free_gpu_for_phase("before pixel upscale")
        frames_hi = _pixel_upscale_frames(
            frames_in, int(out_w), int(out_h), method, warmup=vsr_warmup,
        )
        video_latent = _encode_pixels(vae, frames_hi)
        av_latent = _concat_av_latent(video_latent, audio_latent)
        del frames_hi, video_latent
        cleanup_segment_vram(enabled=True, unload_models=True)
    elif same_canvas:
        video_latent = _encode_pixels(vae, frames_in)
        av_latent = _concat_av_latent(video_latent, audio_latent)
        _ensure_positive()
    else:
        video_latent = _encode_pixels(vae, frames_in)
        # CLIP T8 is ~15GB. Lift first on a clean GPU, then encode T8.
        av_latent = _lift_av_latent(
            _concat_av_latent(video_latent, audio_latent),
            int(out_w), int(out_h),
            resolve_h3_upscaler_name(latent_upscale_model),
        )
        del video_latent
        cleanup_segment_vram(enabled=True, unload_models=True)
        _ensure_positive()
    cleanup_segment_vram(enabled=True, unload_models=True)
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    sigmas = unpack_node_output(
        BasicScheduler.execute(model, "beta", int(refine_steps), float(refine_denoise))
    )[0]
    pin = int(trim or 0)
    sampling_model = model
    sampling_latent = _latent_without_mask(av_latent)
    if pin > 0 and previous_refine_latent is not None:
        continuation = unpack_node_output(
            MiniMaxH3FiniteLatentContinuation.execute(
                positive, sampling_latent, 1, pin,
                False, model, sigmas, previous_refine_latent,
            )
        )
        positive, sampling_latent, _actual_overlap, sampling_model = continuation[:4]

    sampled = _sample_latent(sampling_model, positive, sampling_latent, sampler, sigmas, seed, vae, None)
    sampled = latent_to_cpu(_slice_av_to_frames(sampled, source_count))
    cleanup_segment_vram(enabled=True, unload_models=True)
    refined, _unused_audio = _decode_av(sampled, vae, audio_vae, decode_audio=False)
    matched = unpack_node_output(MiniMaxH3MatchFrameCount.execute(refined, frames))[0]
    kept = matched[pin:] if pin else matched
    del frames_in, audio_latent, av_latent, sampling_latent, refined, positive, sigmas
    _free_gpu_for_phase("after 2.0 window decode")
    return kept.contiguous(), sampled


def refine_images_director_capped(
    *,
    images,
    audio,
    model, clip, vae, audio_vae, sampler,
    seed, src_w, src_h, out_w, out_h,
    refine_steps, refine_denoise,
    trim=0, previous_refine_latent=None,
    upscale_method=H3_LATENT_UPSCALE,
    latent_upscale_model="",
    max_frames=REFINE_MAX_FRAMES_DEFAULT,
    sub_overlap=39,
    clear_vram_between_windows=True,
    refine_prompt="",
    refine_plan=None,
):
    """Director-style 2.0 refine, split so no T8 window exceeds ``max_frames``."""
    frames = images.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if frames.ndim == 5:
        frames = frames.reshape(-1, *frames.shape[-3:])
    cap = h3_grid_length(max(int(max_frames), 5))
    if int(frames.shape[0]) <= cap:
        return refine_window_director_style(
            images=frames, audio=audio,
            model=model, clip=clip, vae=vae, audio_vae=audio_vae, sampler=sampler,
            seed=seed, src_w=src_w, src_h=src_h, out_w=out_w, out_h=out_h,
            refine_steps=refine_steps, refine_denoise=refine_denoise,
            trim=trim, previous_refine_latent=previous_refine_latent,
            upscale_method=upscale_method, latent_upscale_model=latent_upscale_model,
            refine_prompt=refine_prompt, refine_plan=refine_plan,
        )

    from .segment_video_sampler import _audio_slice

    windows = windows_sliding(int(frames.shape[0]), cap, sub_overlap)
    parts = []
    prev = previous_refine_latent
    positive_cache = {}
    vsr_warmup = None
    pixel_vsr = str(upscale_method or "").strip().lower() not in {"", H3_LATENT_UPSCALE}
    for item in windows:
        sl = frames[int(item["start"]):int(item["end"])]
        sl_audio = (
            _audio_slice(audio, int(item["start"]), int(item["end"]) - int(item["start"]))
            if audio is not None else None
        )
        pin = int(trim) if int(item["index"]) == 0 else int(item["trim"])
        hop_len = h3_grid_length(int(item["end"]) - int(item["start"]))
        kept, prev = refine_window_director_style(
            images=sl, audio=sl_audio,
            model=model, clip=clip, vae=vae, audio_vae=audio_vae, sampler=sampler,
            seed=seed, src_w=src_w, src_h=src_h, out_w=out_w, out_h=out_h,
            refine_steps=refine_steps, refine_denoise=refine_denoise,
            trim=pin, previous_refine_latent=prev,
            upscale_method=upscale_method, latent_upscale_model=latent_upscale_model,
            refine_prompt=refine_prompt, refine_plan=refine_plan,
            positive_cache=positive_cache, positive_cache_key=hop_len,
            vsr_warmup=vsr_warmup if pixel_vsr else None,
        )
        if pixel_vsr and int(sl.shape[0]) > 0:
            warm = min(8, int(sl.shape[0]))
            vsr_warmup = sl[-warm:].to(dtype=torch.float32).contiguous()
        parts.append(kept)
        del sl, kept
        if clear_vram_between_windows and int(item["index"]) + 1 < len(windows):
            _free_gpu_for_phase("next refine window")
    return torch.cat(parts, dim=0).contiguous(), prev


def refine_frames_t8(
    *,
    images,
    audio,
    model, clip, vae, audio_vae, sampler,
    seed, width, height, refine_steps, refine_denoise,
):
    """Same-canvas T8 resample. Prefer refine_window_director_style for 2.0."""
    kept, _latent = refine_window_director_style(
        images=images, audio=audio,
        model=model, clip=clip, vae=vae, audio_vae=audio_vae, sampler=sampler,
        seed=seed, src_w=width, src_h=height, out_w=width, out_h=height,
        refine_steps=refine_steps, refine_denoise=refine_denoise,
        trim=0, previous_refine_latent=None,
        upscale_method=H3_LATENT_UPSCALE,
    )
    return kept


def run_finite_sampler_sequential(
    *,
    model, clip, vae, audio_vae, finite, sampler, sigmas, seed,
    continue_audio_latent, ref_image_size, export_mode,
    output_megapixels=0.0, upscale_method="nvidia_rtx_vsr",
    refine_denoise=0.0, refine_steps=0, face_refine=None,
    execution_mode="auto", clear_vram_between_segments=True, cache_name="",
    archive_decode: bool = False,
    live_preview=True, preview_tiny_vae="auto", preview_max_resolution=680,
    preview_frames=60, preview_fps=12, preview_node_id=None,
):
    from comfy_api.latest import io

    from .minimax_h3_finite_segments import (
        MiniMaxH3FiniteLatentContinuation,
        MiniMaxH3LockAudioLatent,
        MiniMaxH3LockedAudioMaster,
        MiniMaxH3LockedAudioSlice,
        MiniMaxH3SilentAudioMaster,
        MiniMaxH3SilentAudioSlice,
        _finite_locked_audio_asset,
        _finite_plan_for_segment,
        _finite_video_audio_muted,
        _megapixel_dims,
        _omit_merged_frames,
        _selflift_settings,
    )
    from .minimax_h3_timeline_director import MiniMaxH3TimelineEncoder
    from .segment_video_sampler import (
        MiniMaxH3DecodedOverlapTrim,
        MiniMaxH3FiniteDecodedSpill,
        MiniMaxH3SegmentFrameUpscale,
        MiniMaxH3SegmentVideoArchive,
        MiniMaxH3SegmentVideoInitialize,
        MiniMaxH3SegmentVideoOutput,
    )
    from .drift_control_av import drift_control_step_count
    from .live_preview import attach_live_preview

    model = attach_live_preview(
        model, preview_node_id,
        enabled=bool(live_preview),
        tiny_vae=preview_tiny_vae,
        max_resolution=preview_max_resolution,
        preview_frames=preview_frames,
        preview_fps=preview_fps,
    )

    count = int(finite["segment_count"])
    fingerprint = sequential_fingerprint(
        finite, seed=seed, continue_audio_latent=continue_audio_latent,
        ref_image_size=ref_image_size, export_mode=export_mode,
        output_megapixels=output_megapixels, upscale_method=upscale_method,
        refine_denoise=refine_denoise, refine_steps=refine_steps,
        face_refine=face_refine,
    )
    cache_dir = _cache_dir(cache_name, fingerprint)
    resumed = _load_resume(cache_dir, fingerprint, finite)
    skipped = 0
    if resumed is not None and int(resumed.get("next_iteration") or 0) < count:
        state = resumed
        skipped = int(state["next_iteration"])
        log.info("Sequential resume: skipping %d finished segment(s) from %s", skipped, cache_dir)
    elif resumed is not None and int(resumed.get("next_iteration") or 0) >= count:
        state = resumed
        skipped = count
        log.info("Sequential resume: all %d segments already on disk, assembling", count)
    else:
        state = unpack_node_output(
            MiniMaxH3SegmentVideoInitialize.execute(
                finite, bool(continue_audio_latent), str(export_mode or "segments"),
            )
        )[0]

    locked_audio = _finite_locked_audio_asset(finite)
    muted_video_audio = _finite_video_audio_muted(finite)
    fixed_audio = locked_audio is not None or muted_video_audio
    steps = drift_control_step_count(sigmas)
    if steps < 1:
        raise ValueError("Drift-Control AV requires a sigma schedule with at least one sampling step")
    selflift_settings = (
        _selflift_settings(
            steps, finite.get("second_pass_model", ""), finite.get("second_pass_high_steps"),
        )
        if finite.get("second_pass") else None
    )
    omit_frames = _omit_merged_frames(finite, output_megapixels)
    source = finite.get("source_plan") or {}
    src_w = int(source.get("width") or 864)
    src_h = int(source.get("height") or 480)
    do_refine = int(refine_steps or 0) > 0 and float(refine_denoise or 0) > 0

    try:
        import comfy.utils

        pbar = comfy.utils.ProgressBar(count) if getattr(comfy.utils, "PROGRESS_BAR_ENABLED", True) else None
    except Exception:
        pbar = None

    previous_latent = state.get("previous_latent")
    previous_images = state.get("previous_images")
    prev_refine_latent = state.get("previous_refine_latent")
    prev_pre_vsr = None
    start_index = int(state.get("next_iteration") or 0)

    for index in range(start_index, count):
        _interrupt()
        number = index + 1
        prompt = finite["prompts"][index]
        overlap = int(finite.get("segment_overlaps", [finite.get("overlap_frames", 0)] * count)[index])
        segment_plan = _finite_plan_for_segment(finite, number)
        encoded = unpack_node_output(
            MiniMaxH3TimelineEncoder.execute(
                clip, vae, audio_vae, segment_plan, prompt, ref_image_size,
            )
        )
        positive, target_latent = encoded[0], encoded[1]
        continuation = unpack_node_output(
            MiniMaxH3FiniteLatentContinuation.execute(
                positive, target_latent, index, overlap,
                bool(continue_audio_latent) and not fixed_audio,
                model, sigmas,
                previous_latent if previous_latent is not None and overlap > 0 else None,
                previous_images if overlap == 1 else None,
                vae if overlap == 1 else None,
                audio_vae if overlap == 1 else None,
            )
        )
        positive, sampling_latent, _actual_overlap, sampling_model = continuation[:4]
        if fixed_audio:
            slice_cls = MiniMaxH3LockedAudioSlice if locked_audio is not None else MiniMaxH3SilentAudioSlice
            source_audio = unpack_node_output(slice_cls.execute(segment_plan))[0]
            from comfy_extras.nodes_audio import VAEEncodeAudio

            encoded_audio = unpack_node_output(
                VAEEncodeAudio.execute(audio_vae, source_audio)
            )[0]
            sampling_latent = unpack_node_output(
                MiniMaxH3LockAudioLatent.execute(sampling_latent, encoded_audio)
            )[0]

        sampled = _sample_latent(
            sampling_model, positive, sampling_latent, sampler, sigmas, seed, vae, selflift_settings,
        )

        if archive_decode:
            state = unpack_node_output(
                MiniMaxH3SegmentVideoArchive.execute(
                    finite, state, sampled, vae, audio_vae, index,
                )
            )[0]
            previous_latent = latent_to_cpu(state.get("previous_latent"))
            previous_images = state.get("previous_images")
        else:
            decode_audio = locked_audio is None and not muted_video_audio
            images, audio = _decode_av(sampled, vae, audio_vae, decode_audio=decode_audio)
            spill_images = images
            spill_audio = audio
            spill_overlap = overlap
            spill_trim_head = not (bool(continue_audio_latent) and not fixed_audio)
            if do_refine:
                out_w, out_h = (
                    _megapixel_dims(src_w, src_h, output_megapixels)
                    if float(output_megapixels or 0) > 0
                    else (src_w, src_h)
                )
                pin = overlap if index else 0
                spill_images, prev_refine_latent = refine_images_director_capped(
                    images=images, audio=audio,
                    model=model, clip=clip, vae=vae, audio_vae=audio_vae, sampler=sampler,
                    seed=int(seed), src_w=src_w, src_h=src_h, out_w=out_w, out_h=out_h,
                    refine_steps=int(refine_steps), refine_denoise=float(refine_denoise),
                    trim=pin, previous_refine_latent=prev_refine_latent,
                    upscale_method=str(upscale_method or H3_LATENT_UPSCALE),
                    latent_upscale_model=str(finite.get("second_pass_model") or ""),
                    max_frames=t8_window_frame_cap(),
                    sub_overlap=39,
                    clear_vram_between_windows=bool(clear_vram_between_segments),
                    refine_prompt=prompt, refine_plan=segment_plan,
                )
                trimmed = unpack_node_output(
                    MiniMaxH3DecodedOverlapTrim.execute(
                        images, index, overlap, True, audio,
                    )
                )
                spill_audio = trimmed[1]
                spill_overlap = 0
                spill_trim_head = True
                if face_refine is not None:
                    from .timeline_face_refine import MiniMaxH3TimelineFaceRefine

                    spill_images = unpack_node_output(
                        MiniMaxH3TimelineFaceRefine.execute(
                            spill_images, face_refine, model, clip, vae, audio_vae,
                            int(seed), str(prompt or ""), index,
                        )
                    )[0]
            elif float(output_megapixels or 0) > 0 or face_refine is not None:
                trimmed = unpack_node_output(
                    MiniMaxH3DecodedOverlapTrim.execute(
                        images, index, overlap, True, audio,
                    )
                )
                spill_images, spill_audio = trimmed[0], trimmed[1]
                spill_overlap = 0
                spill_trim_head = True
                if face_refine is not None:
                    from .timeline_face_refine import MiniMaxH3TimelineFaceRefine

                    spill_images = unpack_node_output(
                        MiniMaxH3TimelineFaceRefine.execute(
                            spill_images, face_refine, model, clip, vae, audio_vae,
                            int(seed), str(prompt or ""), index,
                        )
                    )[0]
                if float(output_megapixels or 0) > 0:
                    out_w, out_h = _megapixel_dims(src_w, src_h, output_megapixels)
                    upscaled = unpack_node_output(
                        MiniMaxH3SegmentFrameUpscale.execute(
                            spill_images, out_w, out_h, str(upscale_method or "nvidia_rtx_vsr"),
                            prev_pre_vsr,
                        )
                    )[0]
                    prev_pre_vsr = spill_images
                    spill_images = upscaled
            state = unpack_node_output(
                MiniMaxH3FiniteDecodedSpill.execute(
                    state, spill_images, index, spill_overlap, spill_trim_head, spill_audio,
                )
            )[0]
            previous_latent = latent_to_cpu(sampled)
            overlaps = finite.get("segment_overlaps", [finite.get("overlap_frames", 0)] * count)
            right_overlap = int(overlaps[index + 1]) if index + 1 < len(overlaps) else 0
            previous_images = images[-1:].clone() if right_overlap == 1 else None
            del images, spill_images, sampled, sampling_latent, positive, target_latent

        state["previous_latent"] = previous_latent
        state["previous_images"] = previous_images
        state["previous_refine_latent"] = prev_refine_latent
        _save_resume(
            cache_dir, fingerprint, state, previous_latent, previous_images,
            previous_refine_latent=prev_refine_latent,
        )
        if pbar is not None:
            pbar.update_absolute(index + 1)
        log.info("Sequential segment %d/%d written", number, count)
        if clear_vram_between_segments and index + 1 < count:
            cleanup_segment_vram(enabled=True, unload_models=True)

    spilled = unpack_node_output(MiniMaxH3SegmentVideoOutput.execute(state))
    last_sampled, merged_video, merged_audio, status = spilled[:4]
    if locked_audio is not None:
        merged_audio = unpack_node_output(MiniMaxH3LockedAudioMaster.execute(finite))[0]
    elif muted_video_audio:
        merged_audio = unpack_node_output(MiniMaxH3SilentAudioMaster.execute(finite))[0]

    if omit_frames:
        merged_images = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
    else:
        from comfy_extras.nodes_video import GetVideoComponents

        try:
            components = unpack_node_output(GetVideoComponents.execute(merged_video))
            merged_images = components[0]
        except Exception:
            merged_images = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
            omit_frames = True

    if clear_vram_between_segments:
        cleanup_segment_vram(enabled=True, unload_models=False)

    status = (
        f"Sequential sampling (Director-style loop) ran {count} segments"
        f"{f'; resumed after {skipped}' if skipped else ''}. "
        f"{status} Continuation latents stay on CPU between segments"
        f"{'; models unload between segments' if clear_vram_between_segments else ''}."
        f" Resume cache: {cache_dir}."
    )
    if do_refine:
        out_label = (
            f"{float(output_megapixels):.1f} MP"
            if float(output_megapixels or 0) > 0
            else "planner resolution"
        )
        status += (
            f" Director-style refine used {int(refine_steps)} T8 beta step(s) at {out_label}, "
            f"denoise {float(refine_denoise):g}, method {upscale_method}; "
            "segment 2+ re-pins the previous refined AV tail before sampling."
        )
    if omit_frames:
        status += " Merged Frames were omitted; connect Final Video to SaveVideo."

    return io.NodeOutput(last_sampled, merged_images, merged_audio, status, merged_video)


def run_video_sampler_sequential(**kwargs):
    """Low-memory VIDEO sampler: sequential archive/decode, no in-graph expansion."""
    from comfy_api.latest import io

    output = run_finite_sampler_sequential(archive_decode=True, **kwargs)
    last_sampled, _frames, merged_audio, status, merged_video = unpack_node_output(output)
    return io.NodeOutput(last_sampled, merged_video, merged_audio, status)
