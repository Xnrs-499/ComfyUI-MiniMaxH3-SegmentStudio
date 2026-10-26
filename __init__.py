"""MiniMax H3 Segment Studio for ComfyUI.

GPL-3.0 fork of Timeline Director plus windowed 2.0 refine for 16GB cards.
Do not load ComfyUI-MiniMaxH3-TimelineDirector at the same time.
"""

from .minimax_h3_timeline_director import (
    MiniMaxH3TimelineDirector,
    MiniMaxH3TimelineEncoder,
    MiniMaxH3OmniPromptBridge,
    MiniMaxH3TimelinePlanner,
)
from .minimax_h3_finite_segments import (
    MiniMaxH3FiniteLatentContinuation,
    MiniMaxH3FiniteAudioTrimTail,
    MiniMaxH3FiniteLoopAccumulate,
    MiniMaxH3FiniteLoopInitialize,
    MiniMaxH3FiniteLoopOutput,
    MiniMaxH3FiniteLoopPrepare,
    MiniMaxH3FiniteLoopSegment,
    MiniMaxH3FiniteOutputTrim,
    MiniMaxH3FiniteSegmentFinalize,
    MiniMaxH3FiniteSegmentSampler,
    MiniMaxH3LockedAudioSlice,
    MiniMaxH3SilentAudioSlice,
    MiniMaxH3LockAudioLatent,
    MiniMaxH3LockedAudioMaster,
    MiniMaxH3SilentAudioMaster,
)
from .selflift_runtime import SelfLiftH3Sampler
from .preset_nodes import MiniMaxH3PresetExporter, MiniMaxH3PresetLoader
from .segment_video_sampler import (
    MiniMaxH3DecodedOverlapTrim,
    MiniMaxH3MatchFrameCount,
    MiniMaxH3FiniteDecodedSpill,
    MiniMaxH3FiniteSegmentVideoSampler,
    MiniMaxH3SegmentFrameUpscale,
    MiniMaxH3SegmentVideoArchive,
    MiniMaxH3SegmentVideoInitialize,
    MiniMaxH3SegmentVideoOutput,
)
from .timeline_face_refine import MiniMaxH3TimelineFaceRefine
from .windowed_refine import MiniMaxH3WindowedRefine
from .live_preview import MiniMaxH3LivePreviewAttach

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3TimelineDirector": MiniMaxH3TimelineDirector,
    "MiniMaxH3TimelinePlanner": MiniMaxH3TimelinePlanner,
    "MiniMaxH3TimelineEncoder": MiniMaxH3TimelineEncoder,
    "MiniMaxH3OmniPromptBridge": MiniMaxH3OmniPromptBridge,
    "MiniMaxH3FiniteSegmentSampler": MiniMaxH3FiniteSegmentSampler,
    "MiniMaxH3FiniteLoopInitialize": MiniMaxH3FiniteLoopInitialize,
    "MiniMaxH3FiniteLoopSegment": MiniMaxH3FiniteLoopSegment,
    "MiniMaxH3FiniteLoopPrepare": MiniMaxH3FiniteLoopPrepare,
    "MiniMaxH3FiniteLoopAccumulate": MiniMaxH3FiniteLoopAccumulate,
    "MiniMaxH3FiniteLoopOutput": MiniMaxH3FiniteLoopOutput,
    "MiniMaxH3FiniteAudioTrimTail": MiniMaxH3FiniteAudioTrimTail,
    "MiniMaxH3FiniteOutputTrim": MiniMaxH3FiniteOutputTrim,
    "MiniMaxH3FiniteLatentContinuation": MiniMaxH3FiniteLatentContinuation,
    "MiniMaxH3FiniteSegmentFinalize": MiniMaxH3FiniteSegmentFinalize,
    "MiniMaxH3LockedAudioSlice": MiniMaxH3LockedAudioSlice,
    "MiniMaxH3SilentAudioSlice": MiniMaxH3SilentAudioSlice,
    "MiniMaxH3LockAudioLatent": MiniMaxH3LockAudioLatent,
    "MiniMaxH3LockedAudioMaster": MiniMaxH3LockedAudioMaster,
    "MiniMaxH3SilentAudioMaster": MiniMaxH3SilentAudioMaster,
    "MiniMaxH3TimelineSelfLiftSampler": SelfLiftH3Sampler,
    "MiniMaxH3PresetLoader": MiniMaxH3PresetLoader,
    "MiniMaxH3PresetExporter": MiniMaxH3PresetExporter,
    "MiniMaxH3FiniteSegmentVideoSampler": MiniMaxH3FiniteSegmentVideoSampler,
    "MiniMaxH3FiniteDecodedSpill": MiniMaxH3FiniteDecodedSpill,
    "MiniMaxH3DecodedOverlapTrim": MiniMaxH3DecodedOverlapTrim,
    "MiniMaxH3MatchFrameCount": MiniMaxH3MatchFrameCount,
    "MiniMaxH3SegmentFrameUpscale": MiniMaxH3SegmentFrameUpscale,
    "MiniMaxH3SegmentVideoInitialize": MiniMaxH3SegmentVideoInitialize,
    "MiniMaxH3SegmentVideoArchive": MiniMaxH3SegmentVideoArchive,
    "MiniMaxH3SegmentVideoOutput": MiniMaxH3SegmentVideoOutput,
    "MiniMaxH3TimelineFaceRefine": MiniMaxH3TimelineFaceRefine,
    "MiniMaxH3WindowedRefine": MiniMaxH3WindowedRefine,
    "MiniMaxH3LivePreviewAttach": MiniMaxH3LivePreviewAttach,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3TimelineDirector": "MiniMax H3 Timeline Director",
    "MiniMaxH3TimelinePlanner": "MiniMax H3 Material Planner",
    "MiniMaxH3TimelineEncoder": "MiniMax H3 Plan Encoder",
    "MiniMaxH3OmniPromptBridge": "MiniMax H3 Omni Media Prompt Bridge",
    "MiniMaxH3FiniteSegmentSampler": "MiniMax H3 Finite Segment Sampler",
    "MiniMaxH3FiniteLoopInitialize": "MiniMax H3 Initialize Segment Loop",
    "MiniMaxH3FiniteLoopSegment": "MiniMax H3 Select Loop Segment",
    "MiniMaxH3FiniteLoopPrepare": "MiniMax H3 Prepare Loop Segment",
    "MiniMaxH3FiniteLoopAccumulate": "MiniMax H3 Accumulate Loop Segment",
    "MiniMaxH3FiniteLoopOutput": "MiniMax H3 Finish Segment Loop",
    "MiniMaxH3FiniteAudioTrimTail": "MiniMax H3 Finite Audio Tail Trim (Internal)",
    "MiniMaxH3FiniteOutputTrim": "MiniMax H3 Finite Output Trim (Internal)",
    "MiniMaxH3FiniteLatentContinuation": "MiniMax H3 Finite Latent Continuation (Internal)",
    "MiniMaxH3FiniteSegmentFinalize": "MiniMax H3 Finite Segment Finalize (Internal)",
    "MiniMaxH3LockedAudioSlice": "MiniMax H3 Locked Audio Slice (Internal)",
    "MiniMaxH3SilentAudioSlice": "MiniMax H3 Silent Audio Slice (Internal)",
    "MiniMaxH3LockAudioLatent": "MiniMax H3 Lock Audio Latent (Internal)",
    "MiniMaxH3LockedAudioMaster": "MiniMax H3 Locked Audio Master (Internal)",
    "MiniMaxH3SilentAudioMaster": "MiniMax H3 Silent Audio Master (Internal)",
    "MiniMaxH3TimelineSelfLiftSampler": "MiniMax H3 Two-Stage Sampler",
    "MiniMaxH3PresetLoader": "MiniMax H3 Local Preset Loader",
    "MiniMaxH3PresetExporter": "MiniMax H3 Preset Export",
    "MiniMaxH3FiniteSegmentVideoSampler": "MiniMax H3 有限分段采样(低内存版本)",
    "MiniMaxH3FiniteDecodedSpill": "MiniMax H3 Spill Decoded Segment (Internal)",
    "MiniMaxH3DecodedOverlapTrim": "MiniMax H3 Trim Decoded Overlap (Internal)",
    "MiniMaxH3MatchFrameCount": "MiniMax H3 Match Frame Count (Internal)",
    "MiniMaxH3SegmentFrameUpscale": "MiniMax H3 Segment Frame Upscale (Internal)",
    "MiniMaxH3SegmentVideoInitialize": "MiniMax H3 Initialize Temporary Segment Video",
    "MiniMaxH3SegmentVideoArchive": "MiniMax H3 Archive Temporary Segment Video",
    "MiniMaxH3SegmentVideoOutput": "MiniMax H3 Finish Temporary Segment Video",
    "MiniMaxH3TimelineFaceRefine": "MiniMax H3 Face Refine (Internal)",
    "MiniMaxH3WindowedRefine": "MiniMax H3 Windowed Refine",
    "MiniMaxH3LivePreviewAttach": "MiniMax H3 Live Preview Attach (Internal)",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
