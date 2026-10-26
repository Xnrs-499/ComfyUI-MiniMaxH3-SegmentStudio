"""Structure smoke test for pure finite planning plus finite sampling."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
import tempfile
import types
from fractions import Fraction
from pathlib import Path

import torch
from aiohttp import web
from server import PromptServer


def _load_package(plugin_dir: Path):
    if not hasattr(PromptServer, "instance"):
        PromptServer.instance = types.SimpleNamespace(routes=web.RouteTableDef())
    package_name = "minimax_h3_finite_smoke"
    spec = importlib.util.spec_from_file_location(
        package_name, plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return sys.modules[f"{package_name}.minimax_h3_finite_segments"]


def _plan():
    return {
        "type": "MINIMAX_H3_TIMELINE_PLAN", "version": 1,
        "width": 640, "height": 352, "generation_seconds": 8.0, "length": 192,
        "prompt_index": None, "segment_count": 3,
        "timeline": {
            "selection": {"start": 0.0, "duration": 8.0}, "clips": [],
            "images": [{"id": "p1", "file": "one.png"}, {"id": "p2", "file": "two.png"}],
            "audios": [{"id": "a1", "file": "one.wav"}, {"id": "a2", "file": "two.wav"}],
            "segmentConfig": {
                "count": 3,
                "segments": [
                    {"images": ["p1", "p2"], "audios": ["a1"]},
                    {"images": ["p2"], "audios": ["a2"]},
                    {"images": ["p1"], "audios": []},
                ],
            },
        },
    }


def _prompt(label: str):
    return (
        "integrated_multimodal_description:\n"
        f"[Shot 1] {label}.\n"
        "overall_soundscape:\nRoom.\nnon_diegetic_music:\nNone."
    )


def main():
    finite = _load_package(Path(__file__).resolve().parents[1])

    # In two-stage mode, enabling an embedded reference-video soundtrack turns
    # it into the same zero-denoise/final-master path used by explicit locked
    # audio. Disabling video audio leaves sound generation to H3.
    video_plan = _plan()
    video_plan["timeline"]["videoAudioEnabled"] = True
    video_plan["timeline"]["videoClips"] = [{
        "id": "v1", "file": "reference.mp4", "start": 0.0,
        "trimStart": 1.25, "duration": 8.0, "hasAudio": True,
    }]
    video_finite = {
        "type": "minimax_h3_finite_segment_plan", "version": 4,
        "mode": "single_segment", "source_plan": video_plan,
        "segment_count": 1, "segment_plans": [video_plan],
        "prompts": [_prompt("Video")], "second_pass": True,
        "overlap_frames": 0, "segment_overlaps": [0],
        "segment_lengths": [192], "target_output_frames": 192,
    }
    automatic_video_lock = finite._finite_locked_audio_asset(video_finite)
    assert automatic_video_lock["lockKind"] == "timeline_video_audio"
    video_finite["second_pass"] = False
    assert finite._finite_locked_audio_asset(video_finite)["lockKind"] == "timeline_video_audio"
    one_pass_video_output = finite.MiniMaxH3FiniteSegmentSampler.execute(
        model=object(), clip=object(), vae=object(), audio_vae=object(),
        finite_plan=video_finite, sampler=object(),
        sigmas=torch.linspace(1.0, 0.0, 5), seed=100,
        continue_audio_latent=True, ref_image_size="match",
    )
    one_pass_video_types = [
        node["class_type"] for node in one_pass_video_output.expand.values()
    ]
    assert "SamplerCustomAdvanced" in one_pass_video_types
    assert "MiniMaxH3LockedAudioSlice" in one_pass_video_types
    assert "MiniMaxH3LockAudioLatent" in one_pass_video_types
    assert "MiniMaxH3LockedAudioMaster" in one_pass_video_types
    assert "AudioConcat" not in one_pass_video_types
    video_plan["timeline"]["videoAudioEnabled"] = False
    assert finite._finite_locked_audio_asset(video_finite) is None
    assert finite._finite_video_audio_muted(video_finite) is True
    muted_output = finite.MiniMaxH3FiniteSegmentSampler.execute(
        model=object(), clip=object(), vae=object(), audio_vae=object(),
        finite_plan=video_finite, sampler=object(),
        sigmas=torch.linspace(1.0, 0.0, 5), seed=100,
        continue_audio_latent=True, ref_image_size="match",
    )
    muted_types = [node["class_type"] for node in muted_output.expand.values()]
    assert "MiniMaxH3SilentAudioSlice" in muted_types
    assert "MiniMaxH3LockAudioLatent" in muted_types
    assert "MiniMaxH3SilentAudioMaster" in muted_types
    assert "AudioConcat" not in muted_types
    assert "final output is silent" in muted_output[3]

    video_finite["second_pass"] = True
    video_plan["timeline"]["videoAudioEnabled"] = True
    assert finite._finite_video_audio_muted(video_finite) is False

    original_selflift_settings = finite._selflift_settings
    finite._selflift_settings = lambda *_args, **_kwargs: {
        "transition_step": 2, "lowres_scale": 0.5, "rho": 0.0,
        "w_min": 0.5, "w_max": 1.0, "upscaler_model": "test.safetensors",
    }
    try:
        video_output = finite.MiniMaxH3FiniteSegmentSampler.execute(
            model=object(), clip=object(), vae=object(), audio_vae=object(),
            finite_plan=video_finite, sampler=object(),
            sigmas=torch.linspace(1.0, 0.0, 5), seed=100,
            continue_audio_latent=True, ref_image_size="match",
        )
    finally:
        finite._selflift_settings = original_selflift_settings
    video_types = [node["class_type"] for node in video_output.expand.values()]
    assert "MiniMaxH3LockedAudioSlice" in video_types
    assert "MiniMaxH3LockAudioLatent" in video_types
    assert "MiniMaxH3LockedAudioMaster" in video_types
    assert "AudioConcat" not in video_types
    assert "exact edited timeline waveform" in video_output[3]

    planner_schema = finite.MiniMaxH3FiniteSegmentExpansion.define_schema()
    sampler_schema = finite.MiniMaxH3FiniteSegmentSampler.define_schema()
    planner_inputs = {item.id for item in planner_schema.inputs}
    assert not planner_inputs.intersection({"model", "clip", "vae", "audio_vae", "sampler", "sigmas", "seed"})
    assert sampler_schema.enable_expand is True
    sampler_input_ids = {item.id for item in sampler_schema.inputs}
    assert "increment_seed" not in sampler_input_ids
    assert "gradient_temporal_mask" not in sampler_input_ids
    assert "continuation_mode" not in sampler_input_ids
    assert "export_mode" in sampler_input_ids
    assert "face_refine" in sampler_input_ids
    assert "execution_mode" in sampler_input_ids
    assert "clear_vram_between_segments" in sampler_input_ids
    assert "cache_name" in sampler_input_ids
    assert "live_preview" in sampler_input_ids
    assert "preview_tiny_vae" in sampler_input_ids
    assert "preview_frames" in sampler_input_ids
    upscale = next(item for item in sampler_schema.inputs if item.id == "upscale_method")
    assert "h3_latent" in list(getattr(upscale, "options", []) or [])
    wr = importlib.import_module("minimax_h3_finite_smoke.windowed_refine")
    wr_schema = wr.MiniMaxH3WindowedRefine.define_schema()
    wr_ids = {item.id for item in wr_schema.inputs}
    assert "latent_upscale_model" in wr_ids
    assert "source_video" in wr_ids
    assert "split_long_windows" in wr_ids
    assert "live_preview" in wr_ids
    assert "preview_tiny_vae" in wr_ids
    lp = importlib.import_module("minimax_h3_finite_smoke.live_preview")
    sentinel = object()
    assert lp.attach_live_preview(sentinel, "8", enabled=True) is sentinel
    assert lp.attach_live_preview(sentinel, "8", enabled=False) is sentinel
    assert "auto" in lp.tiny_vae_combo_options()
    attach_ids = {item.id for item in lp.MiniMaxH3LivePreviewAttach.define_schema().inputs}
    assert "target_node_id" in attach_ids
    wr_upscale = next(item for item in wr_schema.inputs if item.id == "upscale_method")
    assert list(getattr(wr_upscale, "options", []) or [])[0] == "h3_latent"
    assert "nvidia_rtx_vsr" in list(getattr(wr_upscale, "options", []) or [])
    seq = importlib.import_module("minimax_h3_finite_smoke.sequential_segments")
    wr_window = next(item for item in wr_schema.inputs if item.id == "window_frames")
    assert list(getattr(wr_window, "options", []) or []) == list(seq.WINDOW_FRAME_CHOICES)
    assert getattr(wr_window, "default", None) == seq.WINDOW_FRAME_CHOICE_DEFAULT
    refine_src = inspect.getsource(seq.refine_window_director_style)
    assert "before pixel upscale" in refine_src
    assert "vsr_warmup" in refine_src
    cond = [[torch.ones(2, 3), {"pooled_output": torch.zeros(4), "list": [torch.tensor(3.0)]}]]
    cpu_cond = seq._deep_to_cpu(cond)
    assert cpu_cond[0][0].device.type == "cpu"
    cpu_cond[0][0].zero_()
    assert float(cond[0][0].sum()) == 6.0
    free_src = inspect.getsource(seq._free_gpu_for_phase) + inspect.getsource(seq._release_comfy_compiler_graphs)
    assert "_release_comfy_compiler_graphs" in inspect.getsource(seq._free_gpu_for_phase)
    assert "cleanup_prefetch_queues" in free_src
    assert "reset_cast_buffers" in free_src
    assert seq.should_run_sequential("auto", 3) is False
    assert seq.should_run_sequential("auto", 8) is True
    assert seq.should_run_sequential("auto", 20) is True
    assert seq.should_run_sequential("sequential", 1) is True
    assert seq.should_run_sequential("graph", 40) is False
    assert seq.h3_grid_length(204) == 209
    assert seq.h3_grid_length(192) == 192
    sliding = seq.windows_sliding(600, 192, 39)
    assert sliding[0] == {"index": 0, "start": 0, "end": 192, "trim": 0}
    assert sliding[-1]["end"] == 600
    assert sum(item["end"] - item["start"] - item["trim"] for item in sliding) == 600
    planned_windows = seq.windows_from_plan(
        {
            "segment_count": 3,
            "segment_lengths": [192, 243, 243],
            "segment_overlaps": [0, 39, 39],
        },
        600,
    )
    assert [(w["start"], w["end"], w["trim"]) for w in planned_windows] == [
        (0, 192, 0), (153, 396, 39), (357, 600, 39),
    ]
    assert sum(w["end"] - w["start"] - w["trim"] for w in planned_windows) == 600
    capped_windows, did_cap = seq.windows_capped_for_refine(planned_windows, 600, 141, 39)
    assert did_cap is True
    assert all(w["end"] - w["start"] <= 141 for w in capped_windows)
    assert sum(w["end"] - w["start"] - w["trim"] for w in capped_windows) == 600
    full_windows, did_full = seq.windows_capped_for_refine(planned_windows, 600, 243, 39)
    assert did_full is False
    assert full_windows == planned_windows
    assert seq.t8_window_frame_cap(141) == 141
    assert seq.t8_window_frame_cap(158) == 158
    assert seq.t8_window_frame_cap(175) == 175
    assert seq.t8_window_frame_cap(192) == 192
    assert seq.t8_window_frame_cap(150) == 158
    assert seq.t8_window_frame_cap(362) == 362
    assert seq.parse_window_frames(141) == 141
    assert seq.parse_window_frames("141 (~5.9s)") == 141
    assert seq.parse_window_frames("192 (~8s)") == 192
    assert seq.parse_window_frames("362 (whole 15s)") == 362
    assert seq.parse_window_frames("whole 15s") == 362
    assert seq.WINDOW_FRAME_CHOICES[-1] == "362 (whole 15s)"
    assert seq.is_placeholder_images(None) is True
    assert seq.is_placeholder_images(torch.zeros(1, 8, 8, 3)) is True
    assert seq.is_placeholder_images(torch.zeros(10, 480, 864, 3)) is False
    capped_src = inspect.getsource(seq.refine_images_director_capped)
    assert "_free_gpu_for_phase" in capped_src
    from comfy_api.latest import InputImpl, Types

    clip = torch.zeros(16, 32, 32, 3)
    for index in range(16):
        clip[index, :, :, 0] = index / 15.0
    path = Path(tempfile.mkdtemp()) / "window.mp4"
    InputImpl.VideoFromComponents(
        Types.VideoComponents(images=clip, frame_rate=Fraction(24, 1)),
        bit_depth=8,
        color_space="sRGB",
    ).save_to(str(path), format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264)

    class _FileVideo:
        def get_stream_source(self):
            return str(path)

        def get_frame_rate(self):
            return 24

    decoded = seq.decode_video_window(_FileVideo(), 2, 6)
    assert int(decoded.shape[0]) == 6
    assert int(decoded.shape[-1]) == 3
    assert seq.owning_gen_window(planned_windows, 0, 141, 0)["index"] == 0
    assert seq.owning_gen_window(planned_windows, 102, 243, 39)["index"] == 1
    assert seq.owning_gen_window(planned_windows, 510, 600, 39)["index"] == 2
    hop_finite = {
        "type": "minimax_h3_finite_segment_plan",
        "version": 4,
        "mode": "timeline_segments",
        "source_plan": _plan(),
        "segment_count": 6,
        "prompts": [""] * 6,
        "overlap_frames": 39,
        "segment_overlaps": [0, 39, 39, 39, 39, 39],
        "segment_lengths": [141, 141, 141, 141, 141, 90],
        "target_output_frames": 600,
    }
    assert finite._finite_audio_segment_count(hop_finite) == 3
    assert finite._finite_locked_audio_asset(hop_finite) is None
    assert finite._finite_video_audio_muted(hop_finite) is False
    short_keep, did_keep = seq.windows_capped_for_refine(
        [{"index": 0, "start": 0, "end": 141, "trim": 0}], 141, 141, 39,
    )
    assert did_keep is False
    assert short_keep[0]["end"] == 141
    assert seq.pixel_to_latent_hw(864, 480) == (30, 54)
    assert seq.pixel_to_latent_hw(1920, 1088) == (68, 120)
    assert seq.h3_upscaler_combo_options()[0] == "auto"
    from comfy.nested_tensor import NestedTensor
    packed = {"samples": NestedTensor((
        torch.zeros(1, 24, 12, 30, 54),
        torch.zeros(1, 32, 1, 80),
    )), "noise_mask": torch.ones(1)}
    sliced = seq._slice_av_to_frames(packed, 5)
    video, audio = sliced["samples"].unbind()
    assert tuple(video.shape) == (1, 24, 2, 30, 54)
    assert int(audio.shape[-1]) == round(5 * 40 / 24)
    assert "noise_mask" not in sliced
    assert getattr(seq.refine_window_director_style, "__name__", "") == "refine_window_director_style"

    prompts = "\n--- SEGMENT ---\n".join(
        [_prompt("First"), _prompt("Second"), _prompt("Third")]
    )
    planned = finite.MiniMaxH3FiniteSegmentExpansion.execute(
        plan=_plan(), segment_prompts=prompts, segment_count=3,
        overlap_frames=48, inject_continuity_instruction=True,
    )
    finite_plan = planned[0]
    assert planned[1] == 39
    assert "performs no sampling" in planned[2]
    assert "carried latent continuation" not in finite_plan["prompts"][0]
    assert "carried latent continuation" in finite_plan["prompts"][1]

    output = finite.MiniMaxH3FiniteSegmentSampler.execute(
        model=object(), clip=object(), vae=object(), audio_vae=object(),
        finite_plan=finite_plan, sampler=object(),
        sigmas=torch.linspace(1.0, 0.0, 5), seed=100,
        continue_audio_latent=True, ref_image_size="match",
    )
    graph = output.expand
    by_type = {}
    for node_id, node in graph.items():
        by_type.setdefault(node["class_type"], []).append((node_id, node["inputs"]))
    assert len(by_type["MiniMaxH3TimelineEncoder"]) == 3
    assert len(by_type["MiniMaxH3FiniteLatentContinuation"]) == 3
    assert len(by_type["SamplerCustomAdvanced"]) == 3
    assert len(by_type["MiniMaxH3FiniteDecodedSpill"]) == 3
    assert "MiniMaxH3LivePreviewAttach" in by_type
    attach_inputs = by_type["MiniMaxH3LivePreviewAttach"][0][1]
    assert attach_inputs.get("tiny_vae") == "auto"
    assert attach_inputs.get("preview_frames") == 60
    off_preview = finite.MiniMaxH3FiniteSegmentSampler.execute(
        model=object(), clip=object(), vae=object(), audio_vae=object(),
        finite_plan=finite_plan, sampler=object(),
        sigmas=torch.linspace(1.0, 0.0, 5), seed=100,
        continue_audio_latent=True, ref_image_size="match",
        live_preview=False,
    )
    off_types = {node["class_type"] for node in off_preview.expand.values()}
    assert "MiniMaxH3LivePreviewAttach" not in off_types
    assert "MiniMaxH3SegmentVideoInitialize" in by_type
    assert "MiniMaxH3SegmentVideoOutput" in by_type
    assert by_type["MiniMaxH3SegmentVideoInitialize"][0][1]["export_mode"] == "segments"
    assert "Export mode is segments" in output[3]
    assert "ImageResizeKJv2" not in by_type
    assert "MiniMaxH3AudioConditioningT8" not in by_type
    assert "MiniMaxH3TimelineFaceRefine" not in by_type

    two_mp = finite.MiniMaxH3FiniteSegmentSampler.execute(
        model=object(), clip=object(), vae=object(), audio_vae=object(),
        finite_plan=finite_plan, sampler=object(),
        sigmas=torch.linspace(1.0, 0.0, 5), seed=100,
        continue_audio_latent=True, ref_image_size="match",
        output_megapixels=2.0, upscale_method="nvidia_rtx_vsr",
        refine_denoise=0.2, refine_steps=2,
    )
    two_types = {node["class_type"] for node in two_mp.expand.values()}
    assert "MiniMaxH3DecodedOverlapTrim" in two_types
    assert "MiniMaxH3SegmentFrameUpscale" in two_types
    assert "ImageResizeKJv2" not in two_types
    assert "MiniMaxH3AudioConditioningT8" in two_types
    assert "MiniMaxH3MatchFrameCount" in two_types
    assert "PT_H3ConcatAVLatent" in two_types
    assert "upscaled to 2.0 MP" in two_mp[3]
    assert "planner resolution" in two_mp[3]
    upscale_ids = {
        node_id for node_id, node in two_mp.expand.items()
        if node["class_type"] == "MiniMaxH3SegmentFrameUpscale"
    }
    for node in two_mp.expand.values():
        if node["class_type"] != "VAEEncode":
            continue
        pixels = node["inputs"]["pixels"]
        if isinstance(pixels, list):
            assert pixels[0] not in upscale_ids
    two_spills = [
        node["inputs"] for node in two_mp.expand.values()
        if node["class_type"] == "MiniMaxH3FiniteDecodedSpill"
    ]
    assert all(item["overlap_frames"] == 0 for item in two_spills)
    assert all(item["trim_audio_head"] is True for item in two_spills)
    two_trims = [
        node["inputs"] for node in two_mp.expand.values()
        if node["class_type"] == "MiniMaxH3DecodedOverlapTrim"
    ]
    assert all(item["trim_audio_head"] is True for item in two_trims)
    assert "Audio is trimmed to the same kept frames" in two_mp[3]
    assert "warm VSR with the previous planner-res tail" in two_mp[3]
    upscales = sorted(
        (node_id, node["inputs"])
        for node_id, node in two_mp.expand.items()
        if node["class_type"] == "MiniMaxH3SegmentFrameUpscale"
    )
    assert "warmup_images" not in upscales[0][1]
    assert str(upscales[1][1]["warmup_images"][0]).endswith("refine_match_1")
    assert str(upscales[2][1]["warmup_images"][0]).endswith("refine_match_2")
    assert "MiniMaxH3TimelineFaceRefine" not in two_types

    faced = finite.MiniMaxH3FiniteSegmentSampler.execute(
        model=object(), clip=object(), vae=object(), audio_vae=object(),
        finite_plan=finite_plan, sampler=object(),
        sigmas=torch.linspace(1.0, 0.0, 5), seed=100,
        continue_audio_latent=True, ref_image_size="match",
        output_megapixels=2.0, upscale_method="nvidia_rtx_vsr",
        refine_denoise=0.2, refine_steps=2,
        face_refine={"enabled": True, "steps": 8, "denoise": 0.4},
    )
    faced_types = [node["class_type"] for node in faced.expand.values()]
    assert faced_types.count("MiniMaxH3TimelineFaceRefine") == 3
    assert "MiniMaxH3SegmentFrameUpscale" in faced_types
    assert "ImageResizeKJv2" not in faced_types
    assert "FaceRefine ran at planner resolution before 2.0 VSR" in faced[3]
    assert "close-ups that would paste as a sticker are skipped" in faced[3]
    assert "Keep Material Planner secondPass off" in faced[3]
    face_ids = {
        node_id for node_id, node in faced.expand.items()
        if node["class_type"] == "MiniMaxH3TimelineFaceRefine"
    }
    for node in faced.expand.values():
        if node["class_type"] != "MiniMaxH3SegmentFrameUpscale":
            continue
        src = node["inputs"]["images"]
        assert isinstance(src, list) and src[0] in face_ids

    assert "MiniMaxH3FiniteSegmentFinalize" not in by_type
    assert "ImageBatch" not in by_type
    assert "AudioConcat" not in by_type

    encoders = sorted(by_type["MiniMaxH3TimelineEncoder"])
    assert [node[1]["plan"]["prompt_index"] for node in encoders] == [1, 2, 3]
    assert [len(node[1]["plan"]["timeline"]["images"]) for node in encoders] == [2, 1, 1]
    assert [len(node[1]["plan"]["timeline"]["audios"]) for node in encoders] == [1, 1, 0]
    noises = sorted(by_type["RandomNoise"])
    assert [node[1]["noise_seed"] for node in noises] == [100, 100, 100]
    continuations = sorted(by_type["MiniMaxH3FiniteLatentContinuation"])
    assert "previous_latent" not in continuations[0][1]
    assert "previous_latent" in continuations[1][1]
    assert all("gradient_temporal_mask" not in item[1] for item in continuations)
    assert all("continuation_mode" not in item[1] for item in continuations)
    spills = sorted(by_type["MiniMaxH3FiniteDecodedSpill"])
    assert all(item[1]["trim_audio_head"] is False for item in spills)
    assert [item[1]["iteration_index"] for item in spills] == [0, 1, 2]
    assert "Drift-Control AV 39-frame" in output[3]
    assert "Soft AV half-cosine release" in output[3]
    assert "all segments use seed 100" in output[3]
    assert all(
        node["class_type"] not in {
            "Loop", "LoopVariable", "CloseLoop", "MiniMaxH3LoopPromptSelector",
            "MiniMaxH3LoopLatentGuide", "MiniMaxH3LoopSegmentFinalize",
        }
        for node in graph.values()
    )

    for steps in (8, 20):
        drift_output = finite.MiniMaxH3FiniteSegmentSampler.execute(
            model=object(), clip=object(), vae=object(), audio_vae=object(),
            finite_plan=finite_plan, sampler=object(),
            sigmas=torch.linspace(1.0, 0.0, steps + 1),
            seed=100, continue_audio_latent=True, ref_image_size="match",
        )
        drift_continuations = [
            node["inputs"] for node in drift_output.expand.values()
            if node["class_type"] == "MiniMaxH3FiniteLatentContinuation"
        ]
        assert len(drift_continuations) == 3
        assert all("continuation_mode" not in item for item in drift_continuations)
        drift_spills = [
            node["inputs"] for node in drift_output.expand.values()
            if node["class_type"] == "MiniMaxH3FiniteDecodedSpill"
        ]
        assert all(item["trim_audio_head"] is False for item in drift_spills)
        assert sum(
            node["class_type"] == "MiniMaxH3FiniteAudioTrimTail"
            for node in drift_output.expand.values()
        ) == 0
        assert f"adapted to {steps} sampling steps" in drift_output[3]
        assert "Soft AV half-cosine release" in drift_output[3]

    short_planned = finite.MiniMaxH3FiniteSegmentExpansion.execute(
        plan=_plan(), segment_prompts=prompts, segment_count=3,
        overlap_frames=24, inject_continuity_instruction=True,
    )
    assert short_planned[1] == 22
    short_output = finite.MiniMaxH3FiniteSegmentSampler.execute(
        model=object(), clip=object(), vae=object(), audio_vae=object(),
        finite_plan=short_planned[0], sampler=object(),
        sigmas=torch.linspace(1.0, 0.0, 5), seed=100,
        continue_audio_latent=True, ref_image_size="match",
    )
    assert "Drift-Control AV 22-frame" in short_output[3]
    short_spills = [
        node["inputs"]["overlap_frames"]
        for node in short_output.expand.values()
        if node["class_type"] == "MiniMaxH3FiniteDecodedSpill"
    ]
    assert short_spills == [22, 22, 22]

    sample_rate = 16000
    accumulated = {
        "waveform": torch.arange(sample_rate * 4, dtype=torch.float32).reshape(1, 1, -1),
        "sample_rate": sample_rate,
    }
    tail_trimmed = finite.MiniMaxH3FiniteAudioTrimTail.execute(
        accumulated, overlap_frames=48,
    )[0]
    assert tail_trimmed["waveform"].shape[-1] == sample_rate * 4 - round(39 / 24 * sample_rate)

    # The preceding segment owns the visual overlap: keep its complete tail
    # and remove the matching opening frames from the incoming segment.
    preceding = torch.ones((8, 1, 1, 1), dtype=torch.float32)
    incoming = torch.full((8, 1, 1, 1), 2.0, dtype=torch.float32)
    joined = finite.MiniMaxH3FiniteSegmentFinalize.execute(
        {"samples": torch.zeros(1)}, incoming, iteration=1,
        overlap_frames=5, accumulated_images=preceding,
    )[1]
    assert joined.shape[0] == 11
    assert torch.all(joined[:8] == 1)
    assert torch.all(joined[8:] == 2)

    images = torch.arange(1467, dtype=torch.float32).reshape(1467, 1, 1, 1)
    long_audio = {
        "waveform": torch.zeros((1, 2, sample_rate * 63)),
        "sample_rate": sample_rate,
    }
    exact_images, exact_audio = finite.MiniMaxH3FiniteOutputTrim.execute(
        images, long_audio, output_frames=1440,
    )[:2]
    assert exact_images.shape[0] == 1440
    assert exact_images[-1].item() == 1439
    assert exact_audio["waveform"].shape[-1] == sample_rate * 60

    # Native Start/End Loop integration exposes one segment at a time and
    # carries only the state required by the following iteration.
    loop_classes = {
        "MiniMaxH3FiniteLoopInitialize",
        "MiniMaxH3FiniteLoopSegment",
        "MiniMaxH3FiniteLoopPrepare",
        "MiniMaxH3FiniteLoopAccumulate",
        "MiniMaxH3FiniteLoopOutput",
    }
    assert loop_classes.issubset(sys.modules["minimax_h3_finite_smoke"].NODE_CLASS_MAPPINGS)
    loop_state, loop_count = finite.MiniMaxH3FiniteLoopInitialize.execute(finite_plan)
    assert loop_count == 3 and loop_state["next_iteration"] == 0
    selected = finite.MiniMaxH3FiniteLoopSegment.execute(finite_plan, 1)
    assert selected[1] == finite_plan["prompts"][1]
    assert selected[2] == 39

    prepared = finite.MiniMaxH3FiniteLoopPrepare.execute(
        finite_plan=finite_plan, loop_state=loop_state,
        segment_plan=finite._finite_plan_for_segment(finite_plan, 1),
        positive=object(), target_latent=object(), model=object(),
        sigmas=torch.linspace(1.0, 0.0, 5), vae=object(), audio_vae=object(),
        iteration_index=0, continue_audio_latent=True,
    )
    assert any(
        node["class_type"] == "MiniMaxH3FiniteLatentContinuation"
        for node in prepared.expand.values()
    )

    segment_images = torch.ones((60, 1, 1, 1), dtype=torch.float32)
    segment_audio = {
        "waveform": torch.zeros((1, 2, sample_rate * 3)),
        "sample_rate": sample_rate,
    }
    for index in range(3):
        loop_state, merged_loop_images, merged_loop_audio = finite.MiniMaxH3FiniteLoopAccumulate.execute(
            finite_plan=finite_plan, loop_state=loop_state,
            sampled_latent={"samples": torch.tensor(float(index))},
            images=segment_images * (index + 1), audio=segment_audio,
            iteration_index=index, continue_audio_latent=True,
        )
    assert loop_state["next_iteration"] == 3
    assert merged_loop_images.shape[0] == 60 + 2 * (60 - 39)
    assert merged_loop_audio["waveform"].shape[-1] > 0
    final_loop = finite.MiniMaxH3FiniteLoopOutput.execute(loop_state)
    assert final_loop[1].shape[0] == merged_loop_images.shape[0]
    assert "Native ComfyUI Loop completed 3" in final_loop[3]

    svs = sys.modules["minimax_h3_finite_smoke.segment_video_sampler"]
    pinned = svs._pin_audio_to_frames(
        {"waveform": torch.zeros((1, 1, 48000)), "sample_rate": 24000},
        frame_count=24,
    )
    assert int(pinned["waveform"].shape[-1]) == 24000
    sr = 24000
    overlap_audio = {
        "waveform": torch.zeros((1, 1, round(20 / 24 * sr))),
        "sample_rate": sr,
    }
    trimmed = svs.MiniMaxH3DecodedOverlapTrim.execute(
        images=torch.zeros((20, 8, 8, 3)),
        iteration_index=1,
        overlap_frames=5,
        trim_audio_head=True,
        audio=overlap_audio,
    )
    assert int(trimmed[0].shape[0]) == 15
    assert int(trimmed[1]["waveform"].shape[-1]) == round(15 / 24 * sr)
    matched = svs.MiniMaxH3MatchFrameCount.execute(
        images=torch.zeros((10, 8, 8, 3)),
        reference=torch.zeros((6, 8, 8, 3)),
    )
    assert int(matched[0].shape[0]) == 6
    cropped = svs._center_crop_to_aspect(torch.zeros((2, 480, 864, 3)), 1920, 1088)
    assert cropped.shape[0] == 2
    assert abs((cropped.shape[2] / cropped.shape[1]) - (1920 / 1088)) < 0.02
    upscaled = svs.MiniMaxH3SegmentFrameUpscale.execute(
        images=torch.zeros((2, 48, 80, 3)), width=64, height=32, upscale_method="lanczos",
    )
    assert tuple(upscaled[0].shape) == (2, 32, 64, 3)
    washed = svs._comfy_image_nhwc(torch.full((1, 8, 8, 3), 180.0))
    assert float(washed.max()) <= 1.0
    assert abs(float(washed.mean()) - 180.0 / 255.0) < 0.02
    overshoot = torch.zeros((1, 2, 2, 3))
    overshoot[0, 0, 0] = 0.5
    overshoot[0, 0, 1] = 1.2
    mapped = svs._comfy_image_nhwc(overshoot)
    assert float(mapped.max()) <= 1.0 + 1e-5
    assert abs(float(mapped[0, 0, 0, 0]) - (0.5 / 1.2)) < 0.02
    nchw = svs._comfy_image_nhwc(torch.rand(1, 3, 16, 24))
    assert tuple(nchw.shape) == (1, 16, 24, 3)
    custom_nodes = Path(__file__).resolve().parents[2]
    if str(custom_nodes) not in sys.path:
        sys.path.insert(0, str(custom_nodes))
    from ComfyUI_MiniMaxH3_Director.director.face_refine.track import closeup_would_plate
    assert closeup_would_plate([(0.0, 0.0, 480.0, 480.0)], 864, 480, [360.0])
    assert not closeup_would_plate([(400.0, 180.0, 96.0, 96.0)], 864, 480, [48.0])
    assert svs._normalize_export_mode("all") == "all"
    assert svs._normalize_export_mode("merged") == "all"
    assert svs._normalize_export_mode("segments") == "segments"
    assert svs._normalize_export_mode("SEGMENTS") == "segments"
    print("finite planning/sampling smoke test: PASS")


if __name__ == "__main__":
    main()
