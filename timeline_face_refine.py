"""Apply Director FaceRefine to one Timeline segment. Imported only when wired."""

from __future__ import annotations

import gc
from types import SimpleNamespace

import torch
from comfy_api.latest import io

FaceRefinePack = io.Custom("MMX_DIR_FACE_REFINE")


class MiniMaxH3TimelineFaceRefine(io.ComfyNode):
    """YOLO-crop the face, H3-resample the crop, stitch back. Internal expand node."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TimelineFaceRefine",
            display_name="MiniMax H3 Face Refine (Internal)",
            category="MiniMax H3/Internal",
            is_dev_only=True,
            inputs=[
                io.Image.Input("images"),
                FaceRefinePack.Input("face_refine"),
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.String.Input("prompt", default="", multiline=True),
                io.Int.Input("segment_index", default=0, min=0, max=100000),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(
        cls, images, face_refine, model, clip, vae, audio_vae, seed, prompt="", segment_index=0,
    ):
        try:
            from ComfyUI_MiniMaxH3_Director.director.face_refine.pack import (
                normalize_face_refine_pack,
            )
            from ComfyUI_MiniMaxH3_Director.director.face_refine.runtime import (
                apply_segment_face_refine,
            )
        except ImportError as exc:
            raise ValueError(
                "FaceRefine requires ComfyUI_MiniMaxH3_Director in custom_nodes"
            ) from exc

        pack = normalize_face_refine_pack(face_refine)
        if not pack:
            return io.NodeOutput(images)
        frames = images
        if isinstance(images, torch.Tensor):
            frames = images.detach().to(device="cpu", dtype=torch.float32)
            frames = torch.nan_to_num(frames, nan=0.0, posinf=1.0, neginf=0.0)
            peak = float(frames.max().clamp(min=0.0)) if frames.numel() else 1.0
            if peak > 1.5:
                frames = frames / (255.0 if peak > 2.0 else peak)
            frames = frames[..., :3].contiguous().clamp(0.0, 1.0)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        seg = SimpleNamespace(
            prompt=str(prompt or "a person, face close-up"),
            task_key="t2v",
            refs=[],
            index=int(segment_index),
            ref_audios=[],
            ref_image_size="match",
        )
        plan = SimpleNamespace(global_prompt="", raw={}, audio_decode_cache=None)
        stitched = frames
        try:
            stitched, face_note = apply_segment_face_refine(
                frames=frames,
                plan=plan,
                seg=seg,
                pack=pack,
                model=model,
                vae=vae,
                audio_vae=audio_vae,
                clip=clip,
                seed=int(seed),
                cfg=1.0,
                shift_video=12.0,
                shift_audio=3.0,
            )
            skipped = str(face_note or "").startswith("FaceRefine skipped")
            if not skipped and isinstance(stitched, torch.Tensor) and isinstance(frames, torch.Tensor):
                from ComfyUI_MiniMaxH3_Director.director.face_refine.stitch import (
                    FACE_REFINE_SEAM_FADE_FRAMES,
                    fade_stitch_at_seams,
                )
                fade_n = int(FACE_REFINE_SEAM_FADE_FRAMES)
                stitched = fade_stitch_at_seams(
                    stitched,
                    frames,
                    head_frames=fade_n if int(segment_index) > 0 else 0,
                    tail_frames=fade_n,
                )
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if isinstance(stitched, torch.Tensor):
            stitched = torch.nan_to_num(stitched, nan=0.0, posinf=1.0, neginf=0.0)
            stitched = stitched[..., :3].contiguous().float().clamp(0.0, 1.0)
        return io.NodeOutput(stitched)
