"""
Quality Reviewer Agent (Enhancement)
======================================
Role: Quality Assurance Specialist

Reviews all generated clips, subtitles, and audio translations for quality
issues. Checks audio/video sync, subtitle alignment, silence gaps, and
audio levels. Generates a quality report.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from utils.video_utils import get_video_info, detect_silence, get_audio_energy_segments


class QualityAgent(BaseAgent):
    """Reviews clips for quality issues."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="QualityReviewer",
            role="Quality Assurance Specialist",
            config=config,
        )
        self.q_config = config.get("quality_agent", {})
        self.paths = config.get("paths", {})
        self.check_av_sync = self.q_config.get("check_av_sync", True)
        self.check_subtitle_sync = self.q_config.get("check_subtitle_sync", True)
        self.check_silence_gaps = self.q_config.get("check_silence_gaps", True)
        self.min_audio_level = self.q_config.get("min_audio_level", -30)
        self.generate_report = self.q_config.get("generate_report", True)

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run quality checks on all clips and generate report."""
        clips = input_data.get("clips", [])
        self.log(f"Reviewing {len(clips)} clips...")

        report = {
            "total_clips": len(clips),
            "total_subtitles": 0,
            "total_audio_translations": 0,
            "issues": [],
            "summary": {},
            "clip_reports": [],
        }

        for i, clip in enumerate(clips):
            clip_report = {
                "clip": clip["filename"],
                "duration": clip.get("duration", 0),
                "checks": {},
                "issues": [],
                "passed": True,
            }

            # Check clip video
            clip_issues = self._check_clip(clip)
            clip_report["checks"]["clip"] = clip_issues
            if self._has_errors(clip_issues["issues"]):
                clip_report["issues"].extend(clip_issues["issues"])
                clip_report["passed"] = False

            # Check subtitles
            subtitles = clip.get("subtitles", {})
            report["total_subtitles"] += len(subtitles)
            for lang, lang_data in subtitles.items():
                sub_issues = self._check_subtitles(clip, lang, lang_data)
                clip_report["checks"][f"subtitles_{lang}"] = sub_issues
                if self._has_errors(sub_issues["issues"]):
                    clip_report["issues"].extend(sub_issues["issues"])
                    clip_report["passed"] = False

            # Check audio translations
            audio_translations = clip.get("audio_translations", {})
            report["total_audio_translations"] += len(audio_translations)
            for lang, lang_data in audio_translations.items():
                audio_issues = self._check_audio_translation(clip, lang, lang_data)
                clip_report["checks"][f"audio_{lang}"] = audio_issues
                if self._has_errors(audio_issues["issues"]):
                    clip_report["issues"].extend(audio_issues["issues"])
                    clip_report["passed"] = False

            report["clip_reports"].append(clip_report)
            report["issues"].extend(clip_report["issues"])

        # Generate summary
        total_issues = len(report["issues"])
        passed_clips = sum(1 for cr in report["clip_reports"] if cr["passed"])
        report["summary"] = {
            "total_issues": total_issues,
            "clips_passed": passed_clips,
            "clips_failed": len(clips) - passed_clips,
            "pass_rate": f"{(passed_clips / len(clips) * 100):.0f}%" if clips else "N/A",
        }

        # Save report
        if self.generate_report:
            report_path = os.path.join(
                self.paths.get("output_dir", "output"),
                "quality_report.json"
            )
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            self.log(f"Quality report saved to: {report_path}")

        self.log(f"Quality review: {passed_clips}/{len(clips)} clips passed, "
                 f"{total_issues} issues found")

        return {"clips": clips, "quality_report": report,
                **{k: v for k, v in input_data.items() if k != "clips"}}

    @staticmethod
    def _has_errors(issues: List[Dict]) -> bool:
        """A clip only FAILS on error-severity issues; warnings/info are noted
        but don't count as failures (they shouldn't zero out the pass rate)."""
        return any(iss.get("severity") == "error" for iss in issues)

    def _check_clip(self, clip: Dict) -> Dict:
        """Check individual clip for quality issues."""
        issues = []
        clip_path = clip["path"]

        try:
            info = get_video_info(clip_path)

            # Check duration
            duration = info["duration"]
            if duration < 5:
                issues.append({"severity": "warning",
                              "message": f"Clip very short: {duration:.1f}s"})
            if duration > 90:
                issues.append({"severity": "warning",
                              "message": f"Clip longer than 90s: {duration:.1f}s"})

            # Check audio
            if not info["has_audio"]:
                issues.append({"severity": "error",
                              "message": "No audio track in clip"})

            # Check silence gaps
            if self.check_silence_gaps and info["has_audio"]:
                silences = detect_silence(clip_path, min_silence_duration=1.5)
                if silences:
                    for s_start, s_end in silences:
                        issues.append({
                            "severity": "info",
                            "message": f"Silence gap: {s_start:.1f}s - {s_end:.1f}s "
                                      f"({s_end - s_start:.1f}s)"
                        })

            # Check audio level
            if info["has_audio"]:
                energy_segs = get_audio_energy_segments(clip_path, segment_duration=1.0)
                if energy_segs:
                    avg_energy = sum(s["energy"] for s in energy_segs) / len(energy_segs)
                    if avg_energy < self.min_audio_level:
                        issues.append({
                            "severity": "warning",
                            "message": f"Audio may be too quiet: avg {avg_energy:.1f}dB "
                                      f"(threshold: {self.min_audio_level}dB)"
                        })

        except Exception as e:
            issues.append({"severity": "error",
                          "message": f"Failed to check clip: {e}"})

        return {"issues": issues, "info": info if 'info' in locals() else None}

    def _check_subtitles(self, clip: Dict, lang: str, lang_data: Dict) -> Dict:
        """Check subtitle quality."""
        issues = []
        segments = lang_data.get("segments", [])

        if not segments:
            issues.append({"severity": "warning",
                          "message": f"No subtitle segments for {lang}"})
            return {"issues": issues}

        # Check for overlapping timestamps
        for i in range(len(segments) - 1):
            if segments[i]["end"] > segments[i + 1]["start"]:
                issues.append({
                    "severity": "info",
                    "message": f"Subtitle overlap at segment {i+1}-{i+2} ({lang})"
                })

        # Check for very short segments
        for i, seg in enumerate(segments):
            seg_duration = seg["end"] - seg["start"]
            if seg_duration < 0.3:
                issues.append({
                    "severity": "info",
                    "message": f"Very short subtitle at segment {i+1} ({lang}): "
                              f"{seg_duration:.1f}s"
                })

        # Check for empty text
        empty_count = sum(1 for seg in segments if not seg["text"].strip())
        if empty_count > 0:
            issues.append({
                "severity": "warning",
                "message": f"{empty_count} empty subtitle segments ({lang})"
            })

        return {"issues": issues, "segment_count": len(segments)}

    def _check_audio_translation(self, clip: Dict, lang: str, lang_data: Dict) -> Dict:
        """Check audio translation quality."""
        issues = []

        video_path = lang_data.get("video_path")
        if not video_path or not os.path.exists(video_path):
            issues.append({"severity": "error",
                          "message": f"Translated video not found for {lang}"})
            return {"issues": issues}

        try:
            info = get_video_info(video_path)

            # Check if video has audio
            if not info["has_audio"]:
                issues.append({"severity": "error",
                              "message": f"Translated video has no audio ({lang})"})

            # Check duration matches original. Compare against the ACTUAL clip
            # file on disk — clip["duration"] is the PRE-PADDING intended length
            # (cut_clip adds padding_before + padding_after), so it is always
            # ~2.5s shorter than the real file and would false-alarm.
            actual_clip_duration = None
            try:
                actual_clip_duration = get_video_info(clip["path"])["duration"]
            except Exception:
                pass
            orig_duration = actual_clip_duration or clip.get("duration", 0)
            trans_duration = info["duration"]
            duration_diff = abs(trans_duration - orig_duration)

            if duration_diff > 2.0:
                issues.append({
                    "severity": "warning",
                    "message": f"Duration mismatch ({lang}): "
                              f"original={orig_duration:.1f}s, "
                              f"translated={trans_duration:.1f}s "
                              f"(diff: {duration_diff:.1f}s)"
                })

        except Exception as e:
            issues.append({"severity": "error",
                          "message": f"Failed to check audio translation: {e}"})

        return {"issues": issues}
