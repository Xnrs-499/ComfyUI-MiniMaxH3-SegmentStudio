"""Memory-bounded finite sampling that spills decoded segments to temporary video files."""

from __future__ import annotations

import copy
import gc
import logging
import shutil
import time
import uuid
import weakref
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import torch

import folder_paths
from comfy_api.latest import InputImpl, Types, io
from comfy_execution.graph_utils import GraphBuilder
from comfy_extras.nodes_audio import vae_decode_audio

from .drift_control_av import drift_control_step_count
from .experimental_latent_guide import _valid_guide_frames
from .minimax_h3_finite_segments import (
    H3_FPS,
    FiniteLoopState,
    FiniteSegmentPlan,
    _finite_locked_audio_asset,
    _finite_plan_for_segment,
    _finite_video_audio_muted,
    _locked_audio_interval,
    _require_finite_plan,
    _selflift_settings,
    _silent_audio_interval,
)


SPILL_ROOT = "minimax_h3_segment_spill"
EXPORT_ROOT = "minimax_seg_export"
EXPORT_MODE_ALL = "all"
EXPORT_MODE_SEGMENTS = "segments"
log = logging.getLogger("ComfyUI-MiniMaxH3-TimelineDirector.segment_export")


def _normalize_export_mode(mode) -> str:
    text = str(mode or EXPORT_MODE_SEGMENTS).strip().lower()
    if text in {EXPORT_MODE_ALL, "merged"}:
        return EXPORT_MODE_ALL
    return EXPORT_MODE_SEGMENTS


def _new_export_dir() -> Path:
    base = Path(folder_paths.get_output_directory()).resolve() / EXPORT_ROOT
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = base / stamp
    if root.exists():
        for index in range(1, 1000):
            candidate = base / f"{stamp}_{index:03d}"
            if not candidate.exists():
                root = candidate
                break
    root.mkdir(parents=True, exist_ok=False)
    log.info("Timeline segment mp4 run dir: %s", root)
    return root


def _spill_root() -> Path:
    root = Path(folder_paths.get_temp_directory()).resolve() / SPILL_ROOT
    root.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - 24 * 60 * 60
    for path in root.iterdir():
        if path.is_dir() and path.stat().st_mtime < cutoff:
            shutil.rmtree(path, ignore_errors=True)
    return root


def _inside_spill(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(_spill_root())
    except ValueError as error:
        raise ValueError("Segment spill path is outside ComfyUI's temporary directory") from error
    return resolved


def _inside_export(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    root = Path(folder_paths.get_output_directory()).resolve() / EXPORT_ROOT
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("Segment export path is outside ComfyUI's output directory") from error
    return resolved


def _keep_segment_files(state: dict) -> bool:
    return (
        str(state.get("export_mode") or "") == EXPORT_MODE_SEGMENTS
        and bool(state.get("export_directory"))
    )


def _video_dir(state: dict) -> Path:
    if _keep_segment_files(state):
        return _inside_export(state["export_directory"])
    return _inside_spill(state["spill_directory"])


def _segment_mp4_path(state: dict, index: int) -> Path:
    directory = _video_dir(state)
    if _keep_segment_files(state):
        return directory / f"seg_{index + 1:04d}.mp4"
    return directory / f"segment_{index + 1:03d}.mp4"


def _audio_slice(audio: dict | None, start_frame: int, frame_count: int) -> dict | None:
    waveform = audio.get("waveform") if isinstance(audio, dict) else None
    sample_rate = int(audio.get("sample_rate", 0)) if isinstance(audio, dict) else 0
    if waveform is None or sample_rate <= 0 or int(frame_count) <= 0:
        return None
    start = round(max(0, int(start_frame)) / H3_FPS * sample_rate)
    count = max(1, round(int(frame_count) / H3_FPS * sample_rate))
    end = min(int(waveform.shape[-1]), start + count)
    start = min(max(0, start), end)
    sliced = waveform[..., start:end]
    if sliced.shape[-1] <= 0:
        return None
    result = dict(audio)
    result["waveform"] = sliced.clone()
    return result


def _locked_audio_for_export(finite: dict, start_frame: int, frame_count: int) -> dict | None:
    locked = _finite_locked_audio_asset(finite)
    if locked is None or int(frame_count) <= 0:
        return None
    first = _finite_plan_for_segment(finite, 1)
    plan = copy.deepcopy(first)
    if locked.get("lockKind") != "timeline_video_audio":
        plan["timeline"]["audios"] = [locked]
    full = _locked_audio_interval(plan, output_frames=max(1, int(start_frame) + int(frame_count)))
    return _audio_slice(full, start_frame, frame_count)


def _cpu_audio(audio):
    if not isinstance(audio, dict) or audio.get("waveform") is None:
        return audio
    result = dict(audio)
    waveform = audio["waveform"]
    if hasattr(waveform, "detach"):
        result["waveform"] = waveform.detach().to(device="cpu").contiguous()
    return result


def _write_segment_mp4(path: Path, images, audio=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio is not None:
        audio = _cpu_audio(audio)
    components = Types.VideoComponents(
        images=images,
        frame_rate=Fraction(H3_FPS),
        audio=audio,
    )
    video = InputImpl.VideoFromComponents(
        components, bit_depth=8, color_space="sRGB",
    )
    try:
        video.save_to(
            str(path),
            format=Types.VideoContainer.MP4,
            codec=Types.VideoCodec.H264,
            crf=18.0,
        )
    except Exception:
        if audio is None:
            raise
        InputImpl.VideoFromComponents(
            Types.VideoComponents(images=images, frame_rate=Fraction(H3_FPS)),
            bit_depth=8,
            color_space="sRGB",
        ).save_to(
            str(path),
            format=Types.VideoContainer.MP4,
            codec=Types.VideoCodec.H264,
            crf=18.0,
        )


def _atomic_write_segment_mp4(path: Path, images, audio=None) -> None:
    partial = path.with_name(path.name + ".partial")
    try:
        _write_segment_mp4(partial, images, audio)
        partial.replace(path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _export_audio_for_clip(finite: dict, generated_audio, start_frame: int, frame_count: int):
    if _finite_video_audio_muted(finite):
        return None
    if _finite_locked_audio_asset(finite) is not None:
        try:
            return _locked_audio_for_export(finite, start_frame, frame_count)
        except Exception as exc:
            log.warning("Locked audio slice for segment export failed: %s", exc)
            return None
    return _audio_slice(generated_audio, 0, frame_count)


def _pin_audio_to_frames(audio, frame_count: int):
    """Force waveform length to frame_count / 24s so joined clips do not drift."""
    if not isinstance(audio, dict) or audio.get("waveform") is None:
        return audio
    sample_rate = int(audio.get("sample_rate") or 0)
    if sample_rate <= 0 or int(frame_count) <= 0:
        return audio
    waveform = audio["waveform"]
    need = max(1, round(int(frame_count) / H3_FPS * sample_rate))
    have = int(waveform.shape[-1])
    if have == need:
        return audio
    out = dict(audio)
    if have > need:
        out["waveform"] = waveform[..., :need].contiguous()
        return out
    pad = waveform.new_zeros(*waveform.shape[:-1], need - have)
    out["waveform"] = torch.cat([waveform, pad], dim=-1)
    return out


def _trim_audio_head(audio: dict, frames: int) -> dict:
    waveform = audio.get("waveform") if isinstance(audio, dict) else None
    sample_rate = int(audio.get("sample_rate", 0)) if isinstance(audio, dict) else 0
    if waveform is None or sample_rate <= 0:
        raise ValueError("Decoded segment audio must contain waveform and sample_rate")
    count = round(frames / H3_FPS * sample_rate)
    if count and waveform.shape[-1] <= count:
        raise ValueError("Decoded segment audio is shorter than its overlap")
    result = dict(audio)
    result["waveform"] = waveform[..., count:].clone() if count else waveform
    return result


def _trim_audio_tail(audio: dict, frames: int) -> dict:
    waveform = audio["waveform"]
    count = round(frames / H3_FPS * int(audio["sample_rate"]))
    if count and waveform.shape[-1] <= count:
        raise ValueError("Accumulated audio is shorter than its overlap")
    result = dict(audio)
    result["waveform"] = waveform[..., :-count].clone() if count else waveform
    return result


def _concat_audio(left: dict | None, right: dict) -> dict:
    if left is None:
        return right
    if int(left["sample_rate"]) != int(right["sample_rate"]):
        raise ValueError("Decoded segment audio sample rates do not match")
    return {
        "waveform": torch.cat((left["waveform"], right["waveform"]), dim=-1),
        "sample_rate": int(right["sample_rate"]),
    }


# Temporal VSR context carried from the previous segment's planner-res tail.
VSR_WARMUP_FRAMES = 8


class MiniMaxH3SegmentVideoInitialize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SegmentVideoInitialize",
            display_name="MiniMax H3 Initialize Temporary Segment Video",
            category="MiniMax H3/Internal",
            is_dev_only=True,
            inputs=[
                FiniteSegmentPlan.Input("finite_plan"),
                io.Boolean.Input("continue_audio_latent", default=True),
                io.Combo.Input(
                    "export_mode",
                    options=["segments", "all"],
                    default="segments",
                ),
            ],
            outputs=[FiniteLoopState.Output()],
        )

    @classmethod
    def execute(cls, finite_plan, continue_audio_latent, export_mode="segments"):
        finite = _require_finite_plan(finite_plan)
        mode = _normalize_export_mode(export_mode)
        export_directory = None
        if mode == EXPORT_MODE_SEGMENTS:
            try:
                directory = _new_export_dir()
                export_directory = str(directory)
            except OSError as exc:
                log.warning("Segment export dir unavailable (%s); using temp spill", exc)
                mode = EXPORT_MODE_ALL
                directory = _spill_root() / uuid.uuid4().hex
                directory.mkdir()
        else:
            directory = _spill_root() / uuid.uuid4().hex
            directory.mkdir()
        return io.NodeOutput({
            "finite_plan": finite,
            "next_iteration": 0,
            "previous_latent": None,
            "previous_images": None,
            "video_files": [],
            "merged_audio": None,
            "written_frames": 0,
            "spill_directory": str(directory),
            "continue_audio_latent": bool(continue_audio_latent),
            "export_mode": mode,
            "export_directory": export_directory,
        })


class MiniMaxH3SegmentVideoArchive(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SegmentVideoArchive",
            display_name="MiniMax H3 Archive Temporary Segment Video",
            category="MiniMax H3/Internal",
            is_dev_only=True,
            inputs=[
                FiniteSegmentPlan.Input("finite_plan"),
                FiniteLoopState.Input("state"),
                io.Latent.Input("sampled_latent"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.Int.Input("iteration_index", force_input=True),
            ],
            outputs=[FiniteLoopState.Output()],
        )

    @classmethod
    def execute(cls, finite_plan, state, sampled_latent, vae, audio_vae, iteration_index):
        finite = _require_finite_plan(finite_plan)
        index = int(iteration_index)
        if int(state.get("next_iteration", -1)) != index:
            raise ValueError("Temporary segment-video state is out of sequence")
        latent = sampled_latent["samples"]
        video_latent = latent.unbind()[0] if getattr(latent, "is_nested", False) else latent
        images = vae.decode(video_latent)
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        images = images.detach().to(device="cpu", dtype=torch.float32).contiguous()
        locked = _finite_locked_audio_asset(finite)
        muted = _finite_video_audio_muted(finite)
        audio = None if locked is not None or muted else vae_decode_audio(audio_vae, sampled_latent)
        overlaps = finite.get("segment_overlaps", [finite.get("overlap_frames", 0)] * int(finite["segment_count"]))
        overlap = int(overlaps[index]) if index else 0
        trim_frames = _valid_guide_frames(overlap) if overlap else 0
        if int(images.shape[0]) <= trim_frames:
            raise ValueError(f"Segment {index + 1} is shorter than its opening overlap")
        current = images[trim_frames:]
        target = int(finite.get("target_output_frames") or 0)
        written = int(state.get("written_frames") or 0)
        if target and written + int(current.shape[0]) > target:
            current = current[:target - written]
        path = _segment_mp4_path(state, index)
        clip_audio = None
        if locked is None and not muted:
            clip_audio = _trim_audio_head(audio, trim_frames)
        clip_audio = _export_audio_for_clip(finite, clip_audio, written, int(current.shape[0]))
        clip_audio = _pin_audio_to_frames(clip_audio, int(current.shape[0]))
        _atomic_write_segment_mp4(path, current, clip_audio)

        merged_audio = state.get("merged_audio")
        if locked is None and not muted:
            soft = bool(state.get("continue_audio_latent"))
            current_audio = audio if soft else _trim_audio_head(audio, trim_frames)
            if merged_audio is not None and soft and trim_frames:
                merged_audio = _trim_audio_tail(merged_audio, trim_frames)
            merged_audio = _concat_audio(merged_audio, current_audio)

        files = list(state.get("video_files", []))
        files.append(str(path))
        right_overlap = int(overlaps[index + 1]) if index + 1 < len(overlaps) else 0
        next_state = dict(state)
        next_state.update(
            next_iteration=index + 1,
            previous_latent=sampled_latent,
            previous_images=images[-1:].clone() if right_overlap == 1 else None,
            video_files=files,
            merged_audio=merged_audio,
            written_frames=written + int(current.shape[0]),
        )
        del current, images
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return io.NodeOutput(next_state)


class MiniMaxH3SegmentVideoOutput(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SegmentVideoOutput",
            display_name="MiniMax H3 Finish Temporary Segment Video",
            category="MiniMax H3/Internal",
            is_dev_only=True,
            inputs=[FiniteLoopState.Input("state")],
            outputs=[
                io.Latent.Output(display_name="Last Sampled Latent"),
                io.Video.Output(display_name="Final Video"),
                io.Audio.Output(display_name="Final Audio"),
                io.String.Output(display_name="Status"),
            ],
        )

    @classmethod
    def execute(cls, state):
        finite = _require_finite_plan(state.get("finite_plan"))
        if int(state.get("next_iteration", 0)) != int(finite["segment_count"]):
            raise ValueError("Not every segment was archived before final video assembly")
        directory = _video_dir(state)
        keep_segments = _keep_segment_files(state)
        locked = _finite_locked_audio_asset(finite)
        if locked is not None:
            first = _finite_plan_for_segment(finite, 1)
            plan = copy.deepcopy(first)
            if locked.get("lockKind") != "timeline_video_audio":
                plan["timeline"]["audios"] = [locked]
            audio = _locked_audio_interval(plan, output_frames=int(finite.get("target_output_frames") or 0))
        elif _finite_video_audio_muted(finite):
            audio = _silent_audio_interval(
                _finite_plan_for_segment(finite, 1),
                output_frames=int(finite.get("target_output_frames") or state.get("written_frames") or 0),
            )
        else:
            audio = state.get("merged_audio")
            if audio is None:
                raise ValueError("Generated segment audio is missing")
            target = int(finite.get("target_output_frames") or state.get("written_frames") or 0)
            audio = _pin_audio_to_frames(audio, target)
        sources = []
        for path in state["video_files"]:
            resolved = Path(path).resolve()
            resolved.relative_to(directory)
            sources.append(InputImpl.VideoFromFile(str(resolved)))
        final_path = directory / ("full.mp4" if keep_segments else "assembled.mp4")
        InputImpl.VideoFromList(sources, audio, Types.VideoCodec.H264).save_to(
            str(final_path), format=Types.VideoContainer.MP4, codec=Types.VideoCodec.AUTO,
        )
        if not keep_segments:
            for path in state["video_files"]:
                Path(path).unlink(missing_ok=True)
        result = InputImpl.VideoFromFile(str(final_path))
        if not keep_segments:
            try:
                weakref.finalize(result, shutil.rmtree, directory, True)
            except TypeError:
                pass
        status = (
            f"Decoded and encoded {finite['segment_count']} segments one at a time, then joined them without "
            "holding the full RGB timeline in memory. No latent checkpoint was written."
        )
        if keep_segments:
            status += (
                f" Segment export kept crash-safe clips in {directory} "
                "(seg_XXXX.mp4 plus full.mp4)."
            )
        return io.NodeOutput(state.get("previous_latent"), result, audio, status)


class MiniMaxH3FiniteDecodedSpill(io.ComfyNode):
    """Write one already-decoded segment to a temp mp4 so RGB is not kept in the graph."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FiniteDecodedSpill",
            display_name="MiniMax H3 Spill Decoded Segment (Internal)",
            category="MiniMax H3/Internal",
            is_dev_only=True,
            inputs=[
                FiniteLoopState.Input("state"),
                io.Image.Input("images"),
                io.Int.Input("iteration_index", force_input=True),
                io.Int.Input("overlap_frames", default=0, min=0, max=3592),
                io.Boolean.Input("trim_audio_head", default=True),
                io.Audio.Input("audio", optional=True),
            ],
            outputs=[FiniteLoopState.Output()],
        )

    @classmethod
    def execute(
        cls, state, images, iteration_index, overlap_frames,
        trim_audio_head=True, audio=None,
    ):
        finite = _require_finite_plan(state.get("finite_plan"))
        index = int(iteration_index)
        if int(state.get("next_iteration", -1)) != index:
            raise ValueError("Temporary segment-video state is out of sequence")
        frames = images.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if len(frames.shape) == 5:
            frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
        locked = _finite_locked_audio_asset(finite)
        muted = _finite_video_audio_muted(finite)
        overlap = int(overlap_frames) if index else 0
        trim_frames = _valid_guide_frames(overlap) if overlap else 0
        if int(frames.shape[0]) <= trim_frames:
            raise ValueError(f"Segment {index + 1} is shorter than its opening overlap")
        current = frames[trim_frames:]
        target = int(finite.get("target_output_frames") or 0)
        written = int(state.get("written_frames") or 0)
        if target and written + int(current.shape[0]) > target:
            current = current[:target - written]
        path = _segment_mp4_path(state, index)
        clip_audio = None
        if locked is None and not muted and audio is not None:
            clip_audio = _trim_audio_head(audio, trim_frames)
        clip_audio = _export_audio_for_clip(finite, clip_audio, written, int(current.shape[0]))
        clip_audio = _pin_audio_to_frames(clip_audio, int(current.shape[0]))
        _atomic_write_segment_mp4(path, current, clip_audio)

        merged_audio = state.get("merged_audio")
        if locked is None and not muted and audio is not None:
            if int(trim_frames) == 0:
                current_audio = clip_audio
            else:
                soft = not bool(trim_audio_head)
                current_audio = audio if soft else _trim_audio_head(audio, trim_frames)
                if merged_audio is not None and soft and trim_frames:
                    merged_audio = _trim_audio_tail(merged_audio, trim_frames)
            merged_audio = _concat_audio(merged_audio, current_audio)

        files = list(state.get("video_files", []))
        files.append(str(path))
        next_state = dict(state)
        next_state.update(
            next_iteration=index + 1,
            video_files=files,
            merged_audio=merged_audio,
            written_frames=written + int(current.shape[0]),
        )
        del current, frames, images
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return io.NodeOutput(next_state)


class MiniMaxH3DecodedOverlapTrim(io.ComfyNode):
    """Drop the opening overlap before a per-segment 2.0 upscale."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3DecodedOverlapTrim",
            display_name="MiniMax H3 Trim Decoded Overlap (Internal)",
            category="MiniMax H3/Internal",
            is_dev_only=True,
            inputs=[
                io.Image.Input("images"),
                io.Int.Input("iteration_index", force_input=True),
                io.Int.Input("overlap_frames", default=0, min=0, max=3592),
                io.Boolean.Input("trim_audio_head", default=True),
                io.Audio.Input("audio", optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="Trimmed Frames"),
                io.Audio.Output(display_name="Trimmed Audio"),
            ],
        )

    @classmethod
    def execute(cls, images, iteration_index, overlap_frames, trim_audio_head=True, audio=None):
        frames = images.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if len(frames.shape) == 5:
            frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
        overlap = int(overlap_frames) if int(iteration_index) else 0
        trim_frames = _valid_guide_frames(overlap) if overlap else 0
        if int(frames.shape[0]) <= trim_frames:
            raise ValueError(f"Segment {int(iteration_index) + 1} is shorter than its opening overlap")
        current = frames[trim_frames:]
        trimmed_audio = audio
        if audio is not None and bool(trim_audio_head):
            if trim_frames:
                trimmed_audio = _trim_audio_head(audio, trim_frames)
            trimmed_audio = _pin_audio_to_frames(trimmed_audio, int(current.shape[0]))
        del frames, images
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return io.NodeOutput(current, trimmed_audio)


class MiniMaxH3MatchFrameCount(io.ComfyNode):
    """Crop or pad decoded frames so a second H3 pass cannot change clip length."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MatchFrameCount",
            display_name="MiniMax H3 Match Frame Count (Internal)",
            category="MiniMax H3/Internal",
            is_dev_only=True,
            inputs=[
                io.Image.Input("images"),
                io.Image.Input("reference"),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, images, reference):
        frames = images
        if len(frames.shape) == 5:
            frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
        count = int(reference.shape[0])
        have = int(frames.shape[0])
        if have == count:
            return io.NodeOutput(frames)
        if have > count:
            return io.NodeOutput(frames[:count].contiguous())
        last = frames[-1:].expand(count - have, -1, -1, -1)
        return io.NodeOutput(torch.cat([frames, last], dim=0).contiguous())


def _center_crop_to_aspect(frames: torch.Tensor, width: int, height: int) -> torch.Tensor:
    _n, src_h, src_w, _c = frames.shape
    src_aspect = max(int(src_w), 1) / max(int(src_h), 1)
    dst_aspect = max(int(width), 1) / max(int(height), 1)
    if src_aspect > dst_aspect:
        crop_w = max(1, int(round(int(src_h) * dst_aspect)))
        crop_h = int(src_h)
        x = max(0, (int(src_w) - crop_w) // 2)
        y = 0
    else:
        crop_w = int(src_w)
        crop_h = max(1, int(round(int(src_w) / dst_aspect)))
        x = 0
        y = max(0, (int(src_h) - crop_h) // 2)
    return frames[:, y:y + crop_h, x:x + crop_w]


def _interpolate_frames(frames: torch.Tensor, width: int, height: int, method: str) -> torch.Tensor:
    from comfy.utils import common_upscale

    chw = frames.movedim(-1, 1)
    out = common_upscale(chw, int(width), int(height), method, crop="disabled")
    return _comfy_image_nhwc(out.movedim(1, -1))


def _comfy_image_nhwc(images: torch.Tensor) -> torch.Tensor:
    """Force ComfyUI IMAGE layout: NHWC float32 in [0, 1]."""
    tensor = images.detach()
    if tensor.ndim == 5:
        tensor = tensor.reshape(-1, *tensor.shape[-3:])
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"Expected image batch, got shape {tuple(tensor.shape)}")
    if int(tensor.shape[1]) == 3 and int(tensor.shape[-1]) != 3:
        tensor = tensor.movedim(1, -1)
    elif int(tensor.shape[-1]) == 1:
        tensor = tensor.repeat(1, 1, 1, 3)
    tensor = tensor[..., :3].contiguous().float()
    finite = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)
    peak = float(finite.max().clamp(min=0.0))
    if peak > 2.0:
        finite = finite / 255.0
        peak = float(finite.max().clamp(min=0.0))
    if peak > 1.0:
        finite = finite / peak
    return finite.clamp(0.0, 1.0).contiguous()


def _hwc_from_vsr(packed) -> torch.Tensor:
    tensor = torch.from_dlpack(packed).clone()
    while tensor.ndim > 3 and 1 in tuple(tensor.shape):
        tensor = tensor.squeeze()
    if tensor.ndim == 3 and int(tensor.shape[0]) == 3:
        tensor = tensor.movedim(0, -1)
    elif tensor.ndim != 3 or int(tensor.shape[-1]) not in {3, 4}:
        raise ValueError(f"Unexpected VSR output shape {tuple(tensor.shape)}")
    return tensor[..., :3].contiguous().float().cpu()


def _vsr_gpu_frame(frame: torch.Tensor) -> torch.Tensor:
    return (
        frame[..., :3]
        .movedim(-1, 0)
        .contiguous()
        .to(device="cuda", dtype=torch.float32)
        .clamp(0.0, 1.0)
    )


def _vsr_frames_one_by_one(frames: torch.Tensor, width: int, height: int, warmup=None) -> torch.Tensor:
    import nvvfx

    quality = getattr(getattr(nvvfx, "effects", None), "QualityLevel", None)
    level = getattr(quality, "ULTRA", None) if quality is not None else None
    ctx = nvvfx.VideoSuperRes(level) if level is not None else nvvfx.VideoSuperRes()
    sr = ctx.__enter__()
    parts = []
    try:
        sr.output_width = max(8, round(int(width) / 8) * 8)
        sr.output_height = max(8, round(int(height) / 8) * 8)
        if hasattr(sr, "load"):
            sr.load()
        if warmup is not None and int(warmup.shape[0]) > 0:
            start = max(0, int(warmup.shape[0]) - int(VSR_WARMUP_FRAMES))
            for index in range(start, int(warmup.shape[0])):
                gpu = _vsr_gpu_frame(warmup[index])
                packed = sr.run(gpu).image
                del gpu, packed
        for index in range(int(frames.shape[0])):
            gpu = _vsr_gpu_frame(frames[index])
            packed = sr.run(gpu).image
            parts.append(_hwc_from_vsr(packed))
            del gpu, packed
            if index % 4 == 3 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        return _comfy_image_nhwc(torch.stack(parts, dim=0))
    finally:
        try:
            ctx.__exit__(None, None, None)
        except Exception:
            pass
        del parts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class MiniMaxH3SegmentFrameUpscale(io.ComfyNode):
    """Upscale one spilled segment on CPU, one VSR frame at a time, so the H3 UNET stays loaded."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SegmentFrameUpscale",
            display_name="MiniMax H3 Segment Frame Upscale (Internal)",
            category="MiniMax H3/Internal",
            is_dev_only=True,
            inputs=[
                io.Image.Input("images"),
                io.Int.Input("width", default=1920, min=64, max=4096),
                io.Int.Input("height", default=1088, min=64, max=4096),
                io.Combo.Input(
                    "upscale_method",
                    options=["nvidia_rtx_vsr", "lanczos", "bicubic"],
                    default="nvidia_rtx_vsr",
                ),
                io.Image.Input("warmup_images", optional=True),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, images, width, height, upscale_method="nvidia_rtx_vsr", warmup_images=None):
        frames = _comfy_image_nhwc(images.detach().to(device="cpu"))
        width = int(width)
        height = int(height)
        frames = _center_crop_to_aspect(frames, width, height)
        warmup = None
        if warmup_images is not None:
            warmup = _center_crop_to_aspect(
                _comfy_image_nhwc(warmup_images.detach().to(device="cpu")),
                width,
                height,
            )
        method = str(upscale_method or "lanczos").strip().lower()
        if method == "nvidia_rtx_vsr":
            try:
                out = _vsr_frames_one_by_one(frames, width, height, warmup)
            except Exception as exc:
                log.warning("nvidia_rtx_vsr failed (%s); falling back to lanczos", exc)
                out = _interpolate_frames(frames, width, height, "lanczos")
        else:
            out = _interpolate_frames(frames, width, height, method if method else "lanczos")
        out = _comfy_image_nhwc(out)
        del frames, images, warmup, warmup_images
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return io.NodeOutput(out)


class MiniMaxH3FiniteSegmentVideoSampler(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FiniteSegmentVideoSampler",
            display_name="MiniMax H3 有限分段采样(低内存版本)",
            category="MiniMax H3/Long Video",
            description=(
                "Decode and encode one segment at a time. Eight or more segments run a Director-style "
                "Python loop instead of expanding the full graph. export_mode=segments writes each clip to "
                "output/minimax_seg_export as soon as it finishes, then joins them as VIDEO."
            ),
            enable_expand=True,
            inputs=[
                io.Model.Input("model"), io.Clip.Input("clip"), io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"), FiniteSegmentPlan.Input("finite_plan"),
                io.Sampler.Input("sampler"), io.Sigmas.Input("sigmas"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                io.Boolean.Input("continue_audio_latent", default=True),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match"),
                io.Combo.Input(
                    "export_mode",
                    options=["segments", "all"],
                    default="segments",
                    display_name="Export Mode",
                    tooltip=(
                        "segments: write each clip to output/minimax_seg_export as soon as it finishes "
                        "(crash-safe). all: keep clips in temp and only assemble the final video."
                    ),
                ),
                io.Combo.Input(
                    "execution_mode",
                    options=["auto", "sequential", "graph"],
                    default="auto",
                    display_name="Execution Mode",
                    tooltip=(
                        "auto: sequential loop from 8 segments. sequential: always loop. "
                        "graph: expand every segment (OOMs on long timelines)."
                    ),
                ),
                io.Boolean.Input(
                    "clear_vram_between_segments",
                    display_name="Clear VRAM Between Segments",
                    default=True,
                ),
                io.String.Input(
                    "cache_name",
                    default="",
                    display_name="Cache Name",
                    tooltip="Optional resume folder under output/minimax_seg_cache.",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="Last Sampled Latent"),
                io.Video.Output(display_name="Final Video"),
                io.Audio.Output(display_name="Final Audio"),
                io.String.Output(display_name="Status"),
            ],
        )

    @classmethod
    def execute(
        cls, model, clip, vae, audio_vae, finite_plan, sampler, sigmas, seed,
        continue_audio_latent, ref_image_size="match", export_mode="segments",
        execution_mode="auto", clear_vram_between_segments=True, cache_name="",
    ):
        finite = _require_finite_plan(finite_plan)
        from .sequential_segments import should_run_sequential, run_video_sampler_sequential
        if should_run_sequential(execution_mode, finite.get("segment_count", 0)):
            return run_video_sampler_sequential(
                model=model, clip=clip, vae=vae, audio_vae=audio_vae, finite=finite,
                sampler=sampler, sigmas=sigmas, seed=seed,
                continue_audio_latent=continue_audio_latent,
                ref_image_size=ref_image_size, export_mode=export_mode,
                execution_mode=execution_mode,
                clear_vram_between_segments=clear_vram_between_segments,
                cache_name=cache_name,
            )
        graph = GraphBuilder()
        initialized = graph.node(
            "MiniMaxH3SegmentVideoInitialize", id="segment_video_initialize",
            finite_plan=finite, continue_audio_latent=bool(continue_audio_latent),
            export_mode=_normalize_export_mode(export_mode),
        )
        state = initialized.out(0)
        steps = drift_control_step_count(sigmas)
        if steps < 1:
            raise ValueError("Sampling requires a sigma schedule with at least one step")
        lift = _selflift_settings(
            steps, finite.get("second_pass_model", ""), finite.get("second_pass_high_steps"),
        ) if finite.get("second_pass") else None
        for index, prompt in enumerate(finite["prompts"]):
            number = index + 1
            segment_plan = _finite_plan_for_segment(finite, number)
            encoder = graph.node(
                "MiniMaxH3TimelineEncoder", id=f"video_encode_{number}", clip=clip,
                vae=vae, audio_vae=audio_vae, plan=segment_plan, prompt=prompt,
                ref_image_size=ref_image_size,
            )
            prepared = graph.node(
                "MiniMaxH3FiniteLoopPrepare", id=f"video_prepare_{number}", finite_plan=finite,
                loop_state=state, segment_plan=segment_plan, positive=encoder.out(0),
                target_latent=encoder.out(1), model=model, sigmas=sigmas, vae=vae,
                audio_vae=audio_vae, iteration_index=index,
                continue_audio_latent=bool(continue_audio_latent),
            )
            if lift is not None:
                sampled = graph.node(
                    "MiniMaxH3TimelineSelfLiftSampler", id=f"video_sample_{number}",
                    model=prepared.out(2), positive=prepared.out(0), negative=prepared.out(0),
                    vae=vae, latent_image=prepared.out(1), sampler=sampler, sigmas=sigmas,
                    seed=int(seed), cfg=1.0, **lift,
                )
            else:
                noise = graph.node("RandomNoise", id=f"video_noise_{number}", noise_seed=int(seed))
                guider = graph.node(
                    "BasicGuider", id=f"video_guider_{number}",
                    model=prepared.out(2), conditioning=prepared.out(0),
                )
                sampled = graph.node(
                    "SamplerCustomAdvanced", id=f"video_sample_{number}", noise=noise.out(0),
                    guider=guider.out(0), sampler=sampler, sigmas=sigmas,
                    latent_image=prepared.out(1),
                )
            archived = graph.node(
                "MiniMaxH3SegmentVideoArchive", id=f"video_archive_{number}", finite_plan=finite,
                state=state, sampled_latent=sampled.out(0), vae=vae, audio_vae=audio_vae,
                iteration_index=index,
            )
            state = archived.out(0)
        output = graph.node("MiniMaxH3SegmentVideoOutput", id="segment_video_output", state=state)
        return io.NodeOutput(
            output.out(0), output.out(1), output.out(2), output.out(3), expand=graph.finalize(),
        )
