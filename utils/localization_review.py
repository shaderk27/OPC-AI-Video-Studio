"""Shared localization-review payload and decision application helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


def build_localization_review(clips: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build an audio-only review payload; subtitles render automatically."""
    items = []
    for clip_index, clip in enumerate(clips, start=1):
        dubbing_plans = clip.get("dubbing_plans", {})
        for language, plan in dubbing_plans.items():
            dubbing = deepcopy(dubbing_plans.get(language))
            items.append({
                "clip_index": clip_index,
                "clip_id": clip.get("filename", f"clip_{clip_index}"),
                "language": language,
                "approved": False,
                "subtitle_plan": None,
                "dubbing_plan": dubbing,
                "warnings": _warnings(None, dubbing),
            })
    return {"items": items, "count": len(items)}


def approve_all(payload: Dict[str, Any]) -> Dict[str, Any]:
    approved = deepcopy(payload)
    for item in approved.get("items", []):
        item["approved"] = True
    return approved


def apply_localization_review(
    clips: List[Dict[str, Any]],
    payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Apply audio approvals while retaining immutable cue timing."""
    for clip in clips:
        for plan in clip.get("subtitle_plans", {}).values():
            plan["approved"] = True
            plan["approval_mode"] = "automatic"
        for plan in clip.get("dubbing_plans", {}).values():
            plan["approved"] = False

    for item in payload.get("items", []):
        index = item.get("clip_index")
        language = str(item.get("language", ""))
        if not isinstance(index, int) or not (1 <= index <= len(clips)) or not language:
            raise ValueError("Localization review contains an invalid clip/language key")
        clip = clips[index - 1]
        approved = bool(item.get("approved", False))
        dubbing_plan = clip.get("dubbing_plans", {}).get(language)
        if dubbing_plan is not None:
            _apply_dubbing_edits(dubbing_plan, item.get("dubbing_plan"))
            dubbing_plan["approved"] = approved
            dubbing_plan["approval_mode"] = "automatic"
    return clips


def _apply_dubbing_edits(plan: Dict[str, Any], edited: Any) -> None:
    if not isinstance(edited, dict):
        return
    edits = {cue.get("cue_id"): cue for cue in edited.get("cues", []) if isinstance(cue, dict)}
    for cue in plan.get("cues", []):
        update = edits.get(cue.get("cue_id"))
        if not update:
            continue
        if "spoken_text" in update and str(update["spoken_text"]).strip():
            cue["spoken_text"] = str(update["spoken_text"]).strip()


def _warnings(subtitle: Any, dubbing: Any) -> List[str]:
    warnings = []
    for plan in (subtitle, dubbing):
        if not isinstance(plan, dict):
            continue
        for cue in plan.get("cues", []):
            warning = str(cue.get("warning", "")).strip()
            if warning and warning not in warnings:
                warnings.append(warning)
    return warnings
