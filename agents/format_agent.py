"""
Format Adapter Agent (Enhancement)
====================================
Role: Multi-Platform Video Formatter

Adapts clips for different social media platforms by resizing and
aspect-ratio adjusting. Generates vertical (9:16) for TikTok/Reels/Shorts,
horizontal (16:9) for YouTube, and square (1:1) for Instagram Feed.

Platforms that share the same target resolution are rendered ONCE and the
same file is shared across them (e.g. TikTok / YouTube Shorts / Instagram
Reels are all 1080x1920, so only ONE vertical render is produced instead of
three identical ones).
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from agents.base_agent import BaseAgent
from utils.video_utils import resize_video


def parse_aspect_ratio(value: Any) -> Tuple[int, int]:
    """Parse common ratio forms without guessing ambiguous numeric values."""
    if isinstance(value, (tuple, list)) and len(value) == 2:
        parts = value
    elif isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+)\s*[:xX/]\s*(\d+)\s*", value)
        if not match:
            raise ValueError(f"Invalid aspect ratio: {value!r}")
        parts = match.groups()
    else:
        raise ValueError(f"Invalid aspect ratio: {value!r}")
    try:
        width_ratio, height_ratio = (int(parts[0]), int(parts[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid aspect ratio: {value!r}") from exc
    if width_ratio <= 0 or height_ratio <= 0:
        raise ValueError(f"Aspect-ratio values must be positive: {value!r}")
    return width_ratio, height_ratio


class FormatAgent(BaseAgent):
    """Adapts clips for different social media platforms."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="FormatAdapter",
            role="Multi-Platform Video Formatter",
            config=config,
        )
        self.f_config = config.get("format_agent", {})
        self.paths = config.get("paths", {})
        self.formats = self.f_config.get("formats", [])

    def _group_formats(self) -> Dict[Tuple[int, int], List[Dict]]:
        """Group configured formats by unique (width, height).

        Platforms with identical target dimensions share one rendered file,
        so the encoder only runs once per unique resolution.
        """
        groups: Dict[Tuple[int, int], List[Dict]] = {}
        for fmt in self.formats:
            if not fmt.get("enabled", True):
                continue
            key = (int(fmt.get("width", 0)), int(fmt.get("height", 0)))
            if key[0] <= 0 or key[1] <= 0:
                self.log(f"  SKIP invalid format {fmt.get('platform')}: bad dimensions", "warning")
                continue
            try:
                ratio = parse_aspect_ratio(
                    fmt.get("aspect_ratio") or (key[0], key[1])
                )
            except ValueError as exc:
                self.log(f"  SKIP invalid format {fmt.get('platform')}: {exc}", "warning")
                continue
            if abs((key[0] / key[1]) - (ratio[0] / ratio[1])) > 0.02:
                self.log(
                    f"  SKIP invalid format {fmt.get('platform')}: "
                    "dimensions do not match aspect ratio",
                    "warning",
                )
                continue
            normalized = dict(fmt)
            normalized["aspect_ratio"] = f"{ratio[0]}:{ratio[1]}"
            groups.setdefault(key, []).append(normalized)
        return groups

    def _render_group(
        self,
        source_path: str,
        clip_name: str,
        output_dir: str,
        group_key: Tuple[int, int],
        fmts: List[Dict],
        lang: str = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Render ONE file per unique resolution, then map every platform to it.

        Returns a dict {platform: version_info} where platforms in the same
        resolution group all point at the same rendered file.
        """
        width, height = group_key
        prefix = clip_name if lang is None else f"{clip_name}_{lang}"
        formatted_filename = f"{prefix}_{width}x{height}.mp4"
        formatted_dir = os.path.join(output_dir, "formatted")
        self.ensure_dir(formatted_dir)
        formatted_path = os.path.join(formatted_dir, formatted_filename)

        platforms = [fmt["platform"] for fmt in fmts]
        versions: Dict[str, Dict[str, Any]] = {}

        try:
            resize_video(source_path, formatted_path, width, height)
            for fmt in fmts:
                versions[fmt["platform"]] = {
                    "path": formatted_path,
                    "filename": formatted_filename,
                    "width": width,
                    "height": height,
                    "aspect_ratio": fmt.get("aspect_ratio", ""),
                }
            self.log(f"  {', '.join(platforms)} -> {width}x{height} OK (1 shared render)")
        except Exception as e:
            self.log(f"  {', '.join(platforms)} -> {width}x{height}: FAILED - {e}", "error")
            for fmt in fmts:
                versions[fmt["platform"]] = {"error": str(e)}

        return versions

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format clips for all configured platforms."""
        clips = input_data.get("clips", [])
        output_dir = self.paths.get("output_dir", "output")
        self.ensure_dir(output_dir)

        groups = self._group_formats()
        total_targets = sum(len(fmts) for fmts in groups.values())
        total_renders = len(groups)
        self.log(
            f"{total_targets} platform targets -> {total_renders} unique render(s) per clip"
        )

        updated_clips = []

        for i, clip in enumerate(clips):
            clip_name = Path(clip["filename"]).stem
            self.log(f"Formatting {clip_name}")

            platform_versions: Dict[str, Dict[str, Any]] = {}
            for group_key, fmts in groups.items():
                platform_versions.update(
                    self._render_group(
                        clip["path"], clip_name, output_dir, group_key, fmts
                    )
                )

            # Also format translated audio clips if they exist (same dedup)
            audio_translations = clip.get("audio_translations", {})
            for lang, lang_data in audio_translations.items():
                video_path = lang_data.get("video_path")
                if video_path and os.path.exists(video_path):
                    lang_platform_versions: Dict[str, Dict[str, Any]] = {}
                    for group_key, fmts in groups.items():
                        lang_platform_versions.update(
                            self._render_group(
                                video_path, clip_name, output_dir, group_key, fmts, lang=lang
                            )
                        )
                    if lang_platform_versions:
                        lang_data["formatted_versions"] = lang_platform_versions

            clip["platform_versions"] = platform_versions
            updated_clips.append(clip)

        self.log(
            f"Formatted {len(updated_clips)} clips: {total_targets} platform targets, "
            f"{total_renders} unique render(s) per clip"
        )
        return {"clips": updated_clips, **{k: v for k, v in input_data.items() if k != "clips"}}
