"""Segment Studio pack identity and flagship graph."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "example_workflows"


def _load(name: str) -> dict:
    return json.loads((EX / name).read_text(encoding="utf-8"))


def _node(graph: dict, type_name: str) -> dict:
    return next(node for node in graph["nodes"] if node["type"] == type_name)


def main() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "comfyui-minimax-h3-segment-studio"' in pyproject
    assert 'requires-comfyui = ">=0.30.0"' in pyproject
    assert (ROOT / "NOTICE").is_file()
    assert (ROOT / "CREDITS.md").is_file()
    install = (ROOT / "docs" / "INSTALL_CN.md").read_text(encoding="utf-8")
    assert "唯一缺了会红框的外部插件是 KJNodes" in install
    assert "MiniMaxH3AudioConditioningT8 is not installed" in install
    readme_cn = (ROOT / "README_CN.md").read_text(encoding="utf-8")
    assert "示例图原样打开必须另装" in readme_cn
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "Not on the example canvas" in notice
    credits = (ROOT / "CREDITS.md").read_text(encoding="utf-8")
    for name in (
        "TimelineDirector",
        "AIMixer",
        "KJNodes",
        "comfyui-minimax-h3-audio-T8",
        "SelfLift",
        "latent_Upscaler",
    ):
        assert name in credits, name
    assert (ROOT / "LICENSE").is_file()
    assert (ROOT / "prestartup_script.py").is_file()

    t8_6 = _load("H3_SegmentStudio_8秒-多段_T8二采_2.0.json")
    types = {node["type"] for node in t8_6["nodes"]}
    assert "MiniMaxH3WindowedRefine" in types
    assert "MiniMaxH3FiniteSegmentSampler" in types
    assert "MiniMaxH3TimelinePlanner" in types
    assert "MiniMaxH3AudioConditioningT8" not in types
    assert "ImageResizeKJv2" not in types
    sampler = _node(t8_6, "MiniMaxH3FiniteSegmentSampler")
    assert sampler["widgets_values"][4] == "segments"
    assert sampler["widgets_values"][5] == 0
    refine = _node(t8_6, "MiniMaxH3WindowedRefine")
    assert refine["widgets_values"][3] == "h3_latent"
    assert refine["widgets_values"][4] == 0.2
    assert refine["widgets_values"][5] == 2
    assert refine["widgets_values"][7] == "141 (~5.9s)"
    assert refine["widgets_values"][8] == 39
    assert any(inp.get("name") == "source_video" for inp in refine["inputs"])
    assert any(link[0] == 233 and link[1] == 8 and link[3] == 350 for link in t8_6["links"])
    note = next(node for node in t8_6["nodes"] if node["id"] == 114)["widgets_values"][0]
    assert "8秒-多段" in note
    assert "20/40" not in note and "20–40" not in note

    lift_6 = _load("H3_SegmentStudio_8秒-多段_仅3D放大_2.0.json")
    lift_refine = _node(lift_6, "MiniMaxH3WindowedRefine")
    assert lift_refine["widgets_values"][3] == "h3_latent"
    assert lift_refine["widgets_values"][4] == 0
    assert lift_refine["widgets_values"][5] == 0
    assert lift_refine["widgets_values_named"].get("split_long_windows", False) is False

    t8_15 = _load("H3_SegmentStudio_15秒-多段_T8二采_2.0.json")
    t8_15_sampler = _node(t8_15, "MiniMaxH3FiniteSegmentSampler")
    t8_15_refine = _node(t8_15, "MiniMaxH3WindowedRefine")
    assert t8_15_sampler["widgets_values"][5] == 0
    assert t8_15_sampler["widgets_values_named"]["execution_mode"] == "sequential"
    assert t8_15_sampler["widgets_values_named"]["cache_name"] == "15秒-多段T8"
    assert t8_15_refine["widgets_values"][3] == "h3_latent"
    assert t8_15_refine["widgets_values"][4] == 0.2
    assert t8_15_refine["widgets_values"][5] == 2
    assert t8_15_refine["widgets_values"][7] == "141 (~5.9s)"
    assert t8_15_refine["widgets_values"][8] == 39
    saves = {node["widgets_values"][0] for node in t8_15["nodes"] if node["type"] == "SaveVideo"}
    assert "H3_15秒-多段_T8/CASE_ID" in saves
    assert "H3_0.4MP_15秒-多段_T8/CASE_ID" in saves
    t8_15_note = next(node for node in t8_15["nodes"] if node["id"] == 114)["widgets_values"][0]
    assert "15秒-多段" in t8_15_note
    assert "20/40" not in t8_15_note

    lift_15 = _load("H3_SegmentStudio_15秒-多段_仅3D放大_2.0.json")
    lift_15_refine = _node(lift_15, "MiniMaxH3WindowedRefine")
    lift_15_sampler = _node(lift_15, "MiniMaxH3FiniteSegmentSampler")
    assert lift_15_refine["widgets_values"][4] == 0
    assert lift_15_refine["widgets_values"][5] == 0
    assert lift_15_refine["widgets_values_named"]["split_long_windows"] is True
    assert lift_15_sampler["widgets_values_named"]["execution_mode"] == "sequential"
    ids = {t8_6["id"], lift_6["id"], t8_15["id"], lift_15["id"]}
    assert len(ids) == 4
    pack_aux = "Xnrs-499/ComfyUI-MiniMaxH3-SegmentStudio"
    for graph_path in EX.glob("H3_SegmentStudio_*.json"):
        graph = _load(graph_path.name)
        for node in graph["nodes"]:
            aux = (node.get("properties") or {}).get("aux_id", "")
            assert "TimelineDirector" not in str(aux), (graph_path.name, node.get("type"), aux)
            if str(node.get("type", "")).startswith("MiniMaxH3"):
                assert aux == pack_aux, (graph_path.name, node.get("type"), aux)
    print("segment studio smoke test: PASS")


if __name__ == "__main__":
    main()
