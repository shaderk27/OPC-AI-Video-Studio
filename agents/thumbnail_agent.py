"""
Thumbnail Agent (Enhancement)
==============================
Role: Eye-Catching Thumbnail Generator

Extracts the most visually striking frames from each clip and generates
thumbnail images optimized for social media. Can add text overlays with
the clip title.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from utils.video_utils import extract_frame, get_video_info


class ThumbnailAgent(BaseAgent):
    """Generates thumbnails from video clips."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="ThumbnailGenerator",
            role="Eye-Catching Thumbnail Generator",
            config=config,
        )
        self.thumb_config = config.get("thumbnail_agent", {})
        self.paths = config.get("paths", {})
        self.width = self.thumb_config.get("width", 1280)
        self.height = self.thumb_config.get("height", 720)
        self.candidates = self.thumb_config.get("candidates_per_clip", 3)
        self.add_overlay = self.thumb_config.get("add_text_overlay", True)
        self.font_size = self.thumb_config.get("font_size", 48)

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate thumbnail candidates for each clip."""
        clips = input_data.get("clips", [])
        thumb_dir = self.paths.get("thumbnails_dir", "output/thumbnails")
        self.ensure_dir(thumb_dir)

        updated_clips = []

        for i, clip in enumerate(clips):
            clip_path = clip["path"]
            clip_name = Path(clip["filename"]).stem
            self.log(f"Generating thumbnails for {clip_name}")

            # Get clip duration
            video_info = get_video_info(clip_path)
            duration = video_info["duration"]

            # Extract frames at different timestamps
            # Use 25%, 50%, 75% of clip duration for candidates
            timestamps = [
                duration * 0.25,
                duration * 0.50,
                duration * 0.75,
            ][:self.candidates]

            thumbnails = []
            for j, ts in enumerate(timestamps):
                thumb_filename = f"{clip_name}_thumb_{j+1}.png"
                thumb_path = os.path.join(thumb_dir, thumb_filename)
                extract_frame(clip_path, ts, thumb_path)
                thumbnails.append({
                    "path": thumb_path,
                    "filename": thumb_filename,
                    "timestamp": ts,
                })

            # Add text overlay if enabled
            if self.add_overlay and thumbnails:
                best_thumb = thumbnails[0]
                try:
                    overlay_path = self._add_text_overlay(
                        best_thumb["path"],
                        clip.get("title", clip_name),
                        best_thumb["path"].replace("_thumb_1", "_thumb_overlay"),
                    )
                    best_thumb["overlay_path"] = overlay_path
                except Exception as e:
                    self.log(f"  Text overlay failed: {e}", "warning")

            clip["thumbnails"] = thumbnails
            updated_clips.append(clip)

        self.log(f"Generated thumbnails for {len(updated_clips)} clips")
        return {"clips": updated_clips, **{k: v for k, v in input_data.items() if k != "clips"}}

    def _add_text_overlay(self, image_path: str, text: str, output_path: str) -> str:
        """Add text overlay to thumbnail image using Pillow."""
        from PIL import Image, ImageDraw, ImageFont
        import textwrap

        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        # Try to load a font, fall back to default
        font = None
        font_paths = [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    font = ImageFont.truetype(fp, self.font_size)
                    break
                except Exception:
                    continue

        if font is None:
            font = ImageFont.load_default()

        # Wrap text
        wrapped = textwrap.fill(text, width=25)
        bbox = draw.textbbox((0, 0), wrapped, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position at bottom-center
        x = (img.width - text_width) // 2
        y = img.height - text_height - 30

        # Draw semi-transparent background bar
        padding = 10
        draw.rectangle(
            [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
            fill=(0, 0, 0, 180),
        )

        # Draw text with outline
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            draw.text((x + dx, y + dy), wrapped, fill="black", font=font)
        draw.text((x, y), wrapped, fill="white", font=font)

        img.save(output_path)
        return output_path
