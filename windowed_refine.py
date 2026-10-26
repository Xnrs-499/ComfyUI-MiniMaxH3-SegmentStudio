"""Windowed 0.4 -> 2.0 refine: Director re-pin on Timeline's T8 path.

Whole-clip VSR-to-2.0 followed by T8 on 600 frames OOMs a 16 GB card. This node
walks overlapping H3 windows, lifts each 0.4 latent with the H3 3D upscaler,
re-pins the previous 2.0 AV tail, samples T8 at the output canvas, then spills mp4.
"""

from __future__ import annotations

import logging

import torch
from comfy_api.latest import io

from .live_preview import attach_live_preview, live_preview_schema_inputs, node_unique_id
from .minimax_h3_finite_segments import (
    FiniteSegmentPlan,
    _finite_plan_for_segment,
    _megapixel_dims,
    _require_finite_plan,
)
from .sequential_segments import (
    H3_LATENT_UPSCALE,
    WINDOW_FRAME_CHOICE_DEFAULT,
    WINDOW_FRAME_CHOICES,
    _free_gpu_for_phase,
    cleanup_segment_vram,
    decode_video_window,
    h3_grid_length,
    h3_upscaler_combo_options,
    is_placeholder_images,
    lift_decode_window,
    owning_gen_window,
    parse_window_frames,
    refine_window_director_style,
    t8_window_frame_cap,
    unpack_node_output,
    windows_capped_for_refine,
    windows_from_plan,
    windows_sliding,
)

log = logging.getLogger("ComfyUI-MiniMaxH3-TimelineDirector.windowed_refine")


def _as_nhwc(images):
    frames = images.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if frames.ndim == 5:
        frames = frames.reshape(-1, *frames.shape[-3:])
    return frames


class MiniMaxH3WindowedRefine(io.ComfyNode):
    """Second pass for an already-generated 0.4 clip. One overlapping window at a time."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3WindowedRefine",
            display_name="MiniMax H3 Windowed Refine",
            category="MiniMax H3/Long Video",
            description=(
                "Refine a concatenated 0.4-MP clip to 2.0 MP without encoding the whole "
                "2.0 timeline through H3. Each overlapping window is lifted with the H3 "
                "3D latent upscaler, the previous 2.0 AV tail is re-pinned, then T8 "
                "samples at the output canvas. Short clips can use Merged Frames. "
                "12+ segments omit those pixels: connect first-pass Final Video instead "
                "and this node reads one window from the 0.4 mp4 at a time."
            ),
            inputs=[
                io.Image.Input("images", optional=True),
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.Sampler.Input("sampler"),
                io.Audio.Input("audio", optional=True),
                FiniteSegmentPlan.Input(
                    "finite_plan",
                    optional=True,
                    display_name="Finite Segment Plan",
                    tooltip=(
                        "Optional. Width/height and the 3D upscaler come from the first-pass plan. "
                        "First-pass boxes longer than Window Frames are split. Raise Window Frames "
                        "to the segment length to run a whole box if VRAM allows."
                    ),
                ),
                io.Video.Input(
                    "source_video",
                    optional=True,
                    display_name="First-pass Video",
                    tooltip=(
                        "Connect the first-pass Final Video. Used automatically when Merged "
                        "Frames is the 8x8 omit stub (12+ segments or long 0.4 RGB)."
                    ),
                ),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                io.Float.Input(
                    "output_megapixels",
                    default=2.0,
                    min=0.0,
                    max=4.0,
                    step=0.1,
                    display_name="Output Megapixels",
                    tooltip=(
                        "Refine canvas. 2.0 is sharpest and heaviest. Drop to 1.5 / 1.0 before "
                        "picking a longer window, including whole 15s."
                    ),
                ),
                io.Combo.Input(
                    "upscale_method",
                    options=[H3_LATENT_UPSCALE, "nvidia_rtx_vsr", "lanczos", "bicubic"],
                    default=H3_LATENT_UPSCALE,
                    display_name="Upscale Method",
                    tooltip=(
                        "h3_latent: 3D lift (softer, 16GB-safe default). "
                        "nvidia_rtx_vsr: original dual-pass — RTX VSR pixels then light T8. "
                        "Window size is the Window Frames dropdown. Needs nvidia-vfx."
                    ),
                ),
                io.Float.Input(
                    "refine_denoise",
                    default=0.2,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    display_name="Refine Denoise",
                ),
                io.Int.Input(
                    "refine_steps",
                    default=2,
                    min=0,
                    max=20,
                    display_name="Refine Steps",
                    tooltip="0 with h3_latent = 3D lift then decode, no T8 sample.",
                ),
                io.Boolean.Input(
                    "clear_vram_between_windows",
                    default=True,
                    display_name="Clear VRAM Between Windows",
                ),
                io.Combo.Input(
                    "window_frames",
                    options=list(WINDOW_FRAME_CHOICES),
                    default=WINDOW_FRAME_CHOICE_DEFAULT,
                    display_name="Window Frames",
                    tooltip=(
                        "Dropdown: 5.9s is typical at 2.0 on 16GB. 8s is a whole 8s box. "
                        "Pick whole 15s after dropping Output Megapixels if VRAM allows."
                    ),
                ),
                io.Int.Input(
                    "overlap_frames",
                    default=39,
                    min=0,
                    max=56,
                    display_name="Overlap Frames",
                    tooltip=(
                        "H3-legal overlap: 5 / 22 / 39 / 56. 39 ≈ 1.6s, 56 ≈ 2.3s. "
                        "48 frames (exactly 2s) is not a valid MiniMax overlap."
                    ),
                ),
                io.Combo.Input(
                    "latent_upscale_model",
                    options=h3_upscaler_combo_options(),
                    default="auto",
                    display_name="Latent Upscale Model",
                    tooltip="Used when Upscale Method is h3_latent. auto picks a compatible H3 checkpoint.",
                ),
                io.Boolean.Input(
                    "split_long_windows",
                    default=False,
                    display_name="Split Long Lift Windows",
                    tooltip=(
                        "Lift-only (Refine Steps 0): split first-pass boxes longer than Window "
                        "Frames. Off keeps one lift per first-pass box. T8 already splits to "
                        "Window Frames."
                    ),
                ),
                *live_preview_schema_inputs(),
            ],
            hidden=[io.Hidden.unique_id],
            outputs=[
                io.Video.Output(display_name="Final Video"),
                io.Audio.Output(display_name="Audio"),
                io.String.Output(display_name="Status"),
            ],
        )

    @classmethod
    def execute(
        cls, images=None, model=None, clip=None, vae=None, audio_vae=None, sampler=None,
        audio=None, finite_plan=None, source_video=None, seed=0, output_megapixels=2.0,
        upscale_method=H3_LATENT_UPSCALE, refine_denoise=0.2, refine_steps=2,
        clear_vram_between_windows=True, window_frames=WINDOW_FRAME_CHOICE_DEFAULT, overlap_frames=39,
        latent_upscale_model="auto", split_long_windows=False,
        live_preview=True, preview_tiny_vae="auto", preview_max_resolution=680,
        preview_frames=60, preview_fps=12, unique_id=None,
    ):
        from .segment_video_sampler import (
            MiniMaxH3FiniteDecodedSpill,
            MiniMaxH3SegmentFrameUpscale,
            MiniMaxH3SegmentVideoInitialize,
            MiniMaxH3SegmentVideoOutput,
            _audio_slice,
            _pin_audio_to_frames,
        )

        window_frames = parse_window_frames(window_frames)

        model = attach_live_preview(
            model, unique_id if unique_id is not None else node_unique_id(cls),
            enabled=bool(live_preview),
            tiny_vae=preview_tiny_vae,
            max_resolution=preview_max_resolution,
            preview_frames=preview_frames,
            preview_fps=preview_fps,
        )

        use_video = is_placeholder_images(images)
        frames = None
        src_w = 864
        src_h = 480
        if not use_video:
            frames = _as_nhwc(images)
            if frames.dtype == torch.float32:
                frames = frames.to(dtype=torch.float16).contiguous()
            total = int(frames.shape[0])
            src_w = int(frames.shape[2])
            src_h = int(frames.shape[1])
        else:
            if source_video is None:
                raise ValueError(
                    "Windowed refine needs 0.4-MP Merged Frames, or the first-pass Final Video "
                    "when those frames were omitted (12+ segments). Keep sampler Output "
                    "Megapixels at 0."
                )
            try:
                total = int(source_video.get_frame_count())
            except Exception as exc:
                raise ValueError("Could not read frame count from first-pass Final Video") from exc
            try:
                src_w, src_h = source_video.get_dimensions()
            except Exception:
                pass
            if total < 5:
                raise ValueError("First-pass Final Video is shorter than one H3 window")

        finite = None
        if finite_plan is not None:
            finite = _require_finite_plan(finite_plan)
            if use_video:
                plan_total = int(finite.get("target_output_frames") or 0)
                if plan_total >= 5:
                    total = plan_total
            windows = windows_from_plan(finite, total)
        else:
            windows = windows_sliding(total, window_frames, overlap_frames)
        if use_video:
            log.info("Windowed refine reading %d frames from first-pass mp4, %d window(s)", total, len(windows))

        origin_finite = finite
        origin_windows = list(windows)
        source = (finite or {}).get("source_plan") or {}
        src_w = int(source.get("width") or src_w)
        src_h = int(source.get("height") or src_h)
        do_refine = int(refine_steps or 0) > 0 and float(refine_denoise or 0) > 0
        do_upscale = float(output_megapixels or 0) > 0
        method = str(upscale_method or H3_LATENT_UPSCALE).strip().lower()
        do_lift_only = (not do_refine) and do_upscale and method == H3_LATENT_UPSCALE
        out_w, out_h = _megapixel_dims(src_w, src_h, output_megapixels) if do_upscale else (src_w, src_h)
        upscaler_name = str(latent_upscale_model or "").strip()
        if upscaler_name in {"", "auto"}:
            upscaler_name = str((finite or {}).get("second_pass_model") or "")

        capped = False
        longest = max((int(item["end"]) - int(item["start"]) for item in windows), default=0)
        cap_lift = do_lift_only and bool(split_long_windows)
        if do_upscale and (do_refine or cap_lift):
            windows, capped = windows_capped_for_refine(
                windows, total, t8_window_frame_cap(window_frames),
                int(overlap_frames or 39),
            )
        if finite is None or capped:
            source_plan = source or {"width": src_w, "height": src_h, "length": total}
            second_pass_model = str((finite or {}).get("second_pass_model") or "")
            finite = _synthetic_plan(windows, total, src_w, src_h)
            finite["source_plan"] = source_plan
            if second_pass_model:
                finite["second_pass_model"] = second_pass_model
            if capped:
                hop = t8_window_frame_cap(window_frames)
                log.info(
                    "Refine split to %d-frame (~%.1fs) windows from Window Frames; first-pass had a %d-frame window",
                    hop, hop / 24.0, longest,
                )

        state = unpack_node_output(
            MiniMaxH3SegmentVideoInitialize.execute(finite, False, "segments")
        )[0]
        prev_refine_latent = None
        prev_pre_vsr = None
        prev_vsr_warmup = None
        pinned = 0
        reused_refs = 0
        positive_cache = {}
        try:
            import comfy.utils

            pbar = comfy.utils.ProgressBar(len(windows)) if getattr(comfy.utils, "PROGRESS_BAR_ENABLED", True) else None
        except Exception:
            pbar = None

        for item in windows:
            index = int(item["index"])
            start = int(item["start"])
            end = int(item["end"])
            trim = int(item["trim"]) if index else 0
            if frames is not None:
                window = frames[start:end].to(dtype=torch.float32).contiguous()
            else:
                window = decode_video_window(source_video, start, end - start)
            window_audio = _audio_slice(audio, start, end - start) if audio is not None else None
            if do_refine:
                refine_prompt = ""
                refine_plan = None
                gen = owning_gen_window(origin_windows, start, end, trim)
                if origin_finite is not None and gen is not None:
                    gen_index = int(gen["index"])
                    prompts = list(origin_finite.get("prompts") or [])
                    if gen_index < len(prompts):
                        refine_prompt = str(prompts[gen_index] or "")
                    try:
                        refine_plan = _finite_plan_for_segment(origin_finite, gen_index + 1)
                    except Exception:
                        refine_plan = origin_finite.get("source_plan")
                    if refine_plan is not None:
                        reused_refs += 1
                hop_len = h3_grid_length(end - start)
                cache_key = (int(gen["index"]) if gen else -1, hop_len)
                kept, prev_refine_latent = refine_window_director_style(
                    images=window, audio=window_audio,
                    model=model, clip=clip, vae=vae, audio_vae=audio_vae, sampler=sampler,
                    seed=int(seed), src_w=src_w, src_h=src_h, out_w=out_w, out_h=out_h,
                    refine_steps=int(refine_steps), refine_denoise=float(refine_denoise),
                    trim=trim, previous_refine_latent=prev_refine_latent,
                    upscale_method=str(upscale_method or H3_LATENT_UPSCALE),
                    latent_upscale_model=upscaler_name,
                    refine_prompt=refine_prompt, refine_plan=refine_plan,
                    positive_cache=positive_cache, positive_cache_key=cache_key,
                    vsr_warmup=prev_vsr_warmup if method == "nvidia_rtx_vsr" else None,
                )
                if method == "nvidia_rtx_vsr" and int(window.shape[0]) > 0:
                    warm = min(8, int(window.shape[0]))
                    prev_vsr_warmup = window[-warm:].to(dtype=torch.float32).contiguous()
                if trim:
                    pinned += 1
                spill_images = kept
            elif do_lift_only:
                kept = lift_decode_window(
                    images=window, audio=window_audio,
                    vae=vae, audio_vae=audio_vae,
                    src_w=src_w, src_h=src_h, out_w=out_w, out_h=out_h,
                    trim=trim,
                    upscale_method=H3_LATENT_UPSCALE,
                    latent_upscale_model=upscaler_name,
                )
                spill_images = kept
            else:
                kept = window[trim:] if trim else window
                spill_images = kept
                if do_upscale:
                    spill_images = unpack_node_output(
                        MiniMaxH3SegmentFrameUpscale.execute(
                            kept, out_w, out_h, str(upscale_method or "nvidia_rtx_vsr"), prev_pre_vsr,
                        )
                    )[0]
                    prev_pre_vsr = kept
            kept_audio = _audio_slice(audio, start + trim, int(kept.shape[0])) if audio is not None else None
            if kept_audio is not None:
                kept_audio = _pin_audio_to_frames(kept_audio, int(kept.shape[0]))
            state = unpack_node_output(
                MiniMaxH3FiniteDecodedSpill.execute(
                    state, spill_images, index, 0, True, kept_audio,
                )
            )[0]
            log.info(
                "Windowed refine %d/%d frames %d-%d kept %d canvas %dx%d",
                index + 1, len(windows), start, end, int(kept.shape[0]),
                int(spill_images.shape[2]), int(spill_images.shape[1]),
            )
            del window, kept, spill_images
            if pbar is not None:
                pbar.update_absolute(index + 1)
            if clear_vram_between_windows and index + 1 < len(windows):
                # Window 2 3D lift OOMs if window 1's dynamic UNET still holds
                # the 15GB CUDA cache. Empty-cache-only is not enough.
                _free_gpu_for_phase("next refine window")

        if audio is not None:
            state["merged_audio"] = _pin_audio_to_frames(dict(audio), int(state.get("written_frames") or total))
        spilled = unpack_node_output(MiniMaxH3SegmentVideoOutput.execute(state))
        _last_latent, video, spilled_audio, spill_status = spilled[:4]
        final_audio = audio if audio is not None else spilled_audio
        if clear_vram_between_windows:
            cleanup_segment_vram(enabled=True, unload_models=False)

        status = (
            f"Windowed refine ran {len(windows)} overlapping window(s) from {src_w}x{src_h} "
            f"to {out_w}x{out_h}; {spill_status} H3 never encoded the full "
            f"{float(output_megapixels or 0):.1f} MP clip."
        )
        if use_video:
            status += " First-pass Merged Frames were omitted; each window was decoded from the 0.4 mp4."
        if capped:
            cap_frames = t8_window_frame_cap(window_frames)
            kind = "T8" if do_refine else "3D lift"
            status += (
                f" {kind} windows were split to {cap_frames} frames "
                f"(~{cap_frames / 24:.1f}s) as set by Window Frames; first-pass box was "
                f"{longest} frames (~{longest / 24:.1f}s)."
            )
        if do_refine:
            status += (
                f" Director-style T8 used {int(refine_steps)} beta step(s), "
                f"denoise {float(refine_denoise):g}, method {upscale_method}; "
                f"re-pinned {pinned} window tail(s)."
            )
            if reused_refs:
                status += (
                    f" Reused first-pass prompt+reference images on {reused_refs} window(s) "
                    "(same as the original dual-pass Get_positive reuse)."
                )
        elif do_lift_only:
            status += (
                f" 3D latent lift then decode only (no T8) at {out_w}x{out_h}; "
                f"{len(windows)} window(s)."
            )
        elif do_upscale:
            status += f" Each kept window was upscaled to {out_w}x{out_h} with {upscale_method}."
        return io.NodeOutput(video, final_audio, status)


def _synthetic_plan(windows: list[dict], total_frames: int, width: int, height: int) -> dict:
    lengths = [int(item["end"] - item["start"]) for item in windows]
    overlaps = [int(item["trim"]) for item in windows]
    kept = sum(length - trim for length, trim in zip(lengths, overlaps))
    return {
        "type": "minimax_h3_finite_segment_plan",
        "version": 4,
        "mode": "timeline_segments",
        "source_plan": {"width": int(width), "height": int(height), "length": int(total_frames)},
        "segment_count": len(windows),
        "prompts": [""] * len(windows),
        "overlap_frames": overlaps[1] if len(overlaps) > 1 else 0,
        "segment_overlaps": overlaps,
        "segment_lengths": lengths,
        "target_output_frames": kept,
    }
