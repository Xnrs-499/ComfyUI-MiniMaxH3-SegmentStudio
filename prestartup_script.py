"""Refuse to silently collide with the original Timeline Director pack."""

from pathlib import Path
import logging

log = logging.getLogger("ComfyUI-MiniMaxH3-SegmentStudio")

_CUSTOM_NODES = Path(__file__).resolve().parent.parent
_CLASH = _CUSTOM_NODES / "ComfyUI-MiniMaxH3-TimelineDirector"

if _CLASH.is_dir():
    if (_CLASH / ".stand_aside").is_file():
        log.info(
            "Original Timeline Director is present but standing aside (%s). "
            "Segment Studio will register the MiniMax H3 nodes.",
            _CLASH,
        )
    else:
        log.error(
            "MiniMax H3 Segment Studio: original Timeline Director is still enabled:\n"
            "  %s\n"
            "Rename that folder to ComfyUI-MiniMaxH3-TimelineDirector.disabled "
            "or node IDs will register twice and the last pack wins.",
            _CLASH,
        )
