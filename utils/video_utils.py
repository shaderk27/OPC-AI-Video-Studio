"""
Video Utilities
================
Core video processing functions used across agents.
Handles ffmpeg path resolution, video info extraction, cutting clips,
audio extraction, and silence detection.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Resolve ffmpeg binary from imageio-ffmpeg (bundled, no system install needed).
# NOTE: imageio-ffmpeg ships ONLY the ffmpeg executable - there is no ffprobe.
# All probing below parses `ffmpeg -i` output instead, with an optional fallback
# to a real system ffprobe if one happens to be installed on PATH.
import imageio_ffmpeg

FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()


def get_ffmpeg_path() -> str:
    """Return path to the bundled ffmpeg executable."""
    return FFMPEG_PATH


def _find_ffprobe() -> Optional[str]:
    """Locate a real ffprobe on the system PATH, if one exists (optional)."""
    return shutil.which("ffprobe")


def _run_media_command(cmd: List[str]) -> subprocess.CompletedProcess:
    """Run ffmpeg/ffprobe without relying on the Windows ANSI code page.

    FFmpeg emits UTF-8 diagnostics. On Windows, ``text=True`` otherwise uses
    the active locale (commonly cp1252), whose decoder can crash the subprocess
    reader thread and leave stdout/stderr as ``None``. Replacement decoding
    preserves all metadata-bearing ASCII text.
    """
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _parse_ffmpeg_info(stderr: Optional[str]) -> Dict:
    """
    Parse `ffmpeg -i <file>` stderr output into a media-info dict.
    ffmpeg always prints stream info to stderr when given only `-i` (no output).
    Returns the same shape as the ffprobe branch of get_video_info().
    """
    stderr = stderr or ""
    info = {
        "duration": 0.0,
        "width": 0,
        "height": 0,
        "fps": 0,
        "codec": "none",
        "has_audio": False,
        "audio_codec": "none",
    }

    # Duration: 00:12:34.56, start: 0.000000, bitrate: 4567 kb/s
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    if m:
        hours, minutes, seconds = m.groups()
        info["duration"] = int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    for line in stderr.splitlines():
        if "Stream #" not in line:
            continue
        if "Video:" in line:
            vm = re.search(r"Video:\s*(\w+)", line)
            if vm and vm.group(1).lower() != "none":
                info["codec"] = vm.group(1)
            rm = re.search(r"(\d{2,5})x(\d{2,5})", line)
            if rm:
                info["width"] = int(rm.group(1))
                info["height"] = int(rm.group(2))
            fm = re.search(r"(\d+(?:\.\d+)?)\s*fps", line)
            if fm:
                info["fps"] = round(float(fm.group(1)), 2)
        elif "Audio:" in line:
            info["has_audio"] = True
            am = re.search(r"Audio:\s*(\w+)", line)
            if am:
                info["audio_codec"] = am.group(1)

    return info


def get_video_info(video_path: str) -> Dict:
    """
    Get video metadata.
    Uses a real ffprobe if one is on the system PATH; otherwise falls back to
    parsing `ffmpeg -i` output (ffmpeg is always available via imageio-ffmpeg).
    Returns dict with: duration, width, height, fps, codec, audio_codec, has_audio
    """
    probe = _find_ffprobe()
    if probe:
        cmd = [
            probe, "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            video_path,
        ]
        result = _run_media_command(cmd)
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                video_stream = None
                audio_stream = None
                for stream in data.get("streams", []):
                    if stream["codec_type"] == "video" and video_stream is None:
                        video_stream = stream
                    elif stream["codec_type"] == "audio" and audio_stream is None:
                        audio_stream = stream

                info = {
                    "duration": float(data.get("format", {}).get("duration", 0)),
                    "width": int(video_stream["width"]) if video_stream else 0,
                    "height": int(video_stream["height"]) if video_stream else 0,
                    "fps": 0,
                    "codec": video_stream.get("codec_name", "unknown") if video_stream else "none",
                    "has_audio": audio_stream is not None,
                    "audio_codec": audio_stream.get("codec_name", "none") if audio_stream else "none",
                }

                if video_stream and "r_frame_rate" in video_stream:
                    num, den = video_stream["r_frame_rate"].split("/")
                    info["fps"] = round(int(num) / int(den), 2) if int(den) != 0 else 0
                return info
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                pass  # probe output was unusable -> fall back to ffmpeg parsing

    # Fallback: ffmpeg prints media info to stderr when given only `-i`.
    cmd = [FFMPEG_PATH, "-hide_banner", "-i", video_path]
    result = _run_media_command(cmd)
    return _parse_ffmpeg_info(result.stderr)


def cut_clip(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: str,
    padding_before: float = 0,
    padding_after: float = 0,
) -> str:
    """
    Cut a segment from a video file using ffmpeg.
    Uses stream copy for speed (no re-encoding) with accurate seeking.

    Args:
        video_path: Path to source video.
        start_time: Start time in seconds.
        end_time: End time in seconds.
        output_path: Output file path.
        padding_before: Extra seconds before start.
        padding_after: Extra seconds after end.
    """
    start = max(0, start_time - padding_before)
    end = end_time + padding_after
    duration = end - start

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        FFMPEG_PATH, "-y",
        "-ss", f"{start:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac",
        "-movflags", "+faststart",
        output_path,
    ]

    result = _run_media_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg cut failed: {result.stderr[:500]}")

    return output_path


def create_review_proxy(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: str,
    max_width: int = 640,
) -> str:
    """Create a small, accurately-seeked MP4 for multimodal review.

    Review proxies preserve audio and visual timing while reducing upload size.
    They are temporary analysis artifacts, not final deliverables.
    """
    start = max(0.0, float(start_time))
    end = max(start, float(end_time))
    duration = end - start
    if duration <= 0:
        raise ValueError("Review proxy duration must be greater than zero")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG_PATH, "-y",
        "-ss", f"{start:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-map", "0:v:0",
        "-map", "0:a:0?",
        "-vf", f"scale={max(160, int(max_width))}:-2:force_original_aspect_ratio=decrease",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "32",
        "-c:a", "aac", "-b:a", "64k", "-ac", "1",
        "-movflags", "+faststart",
        output_path,
    ]
    result = _run_media_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg review proxy failed: {result.stderr[:500]}")
    return output_path


def extract_audio(video_path: str, output_path: str, sample_rate: int = 16000) -> str:
    """
    Extract audio from video as WAV file (16kHz mono for whisper).
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        FFMPEG_PATH, "-y",
        "-i", video_path,
        "-vn",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-f", "wav",
        output_path,
    ]

    result = _run_media_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {result.stderr[:500]}")

    return output_path


def detect_silence(
    video_path: str,
    min_silence_duration: float = 0.5,
    noise_threshold: str = "-30dB",
) -> List[Tuple[float, float]]:
    """
    Detect silence segments in the audio track.
    Returns list of (silence_start, silence_end) tuples.
    """
    cmd = [
        FFMPEG_PATH, "-i", video_path,
        "-af", f"silencedetect=n={noise_threshold}:d={min_silence_duration}",
        "-f", "null", "-",
    ]

    result = _run_media_command(cmd)
    silences = []
    current_start = None

    for line in result.stderr.split("\n"):
        if "silence_start:" in line:
            current_start = float(line.split("silence_start:")[1].strip())
        elif "silence_end:" in line and current_start is not None:
            end_part = line.split("silence_end:")[1].strip()
            end_val = float(end_part.split("|")[0].strip())
            silences.append((current_start, end_val))
            current_start = None

    return silences


def get_audio_energy_segments(
    video_path: str,
    segment_duration: float = 2.0,
) -> List[Dict]:
    """
    Analyze audio energy across the video to find high-energy moments.
    Returns list of {start, end, energy} dicts sorted by energy (highest first).
    """
    # Extract audio and analyze with ffmpeg's astats filter
    cmd = [
        FFMPEG_PATH, "-i", video_path,
        "-af", f"astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-",
        "-f", "null", "-",
    ]

    result = _run_media_command(cmd)
    segments = []
    current_time = 0.0

    # ametadata's ``file=-`` output is written to stdout by ffmpeg, while
    # diagnostics remain on stderr. Parse both for compatibility with ffmpeg
    # builds that route filter metadata differently.
    metadata_output = f"{result.stdout}\n{result.stderr}"
    for line in metadata_output.splitlines():
        if "pts_time:" in line:
            try:
                current_time = float(line.split("pts_time:")[1].strip())
            except (ValueError, IndexError):
                pass
        elif "RMS_level=" in line:
            try:
                rms_str = line.split("RMS_level=")[1].strip()
                rms = float(rms_str) if rms_str != "-inf" else -100.0
                segments.append({"time": current_time, "energy": rms})
            except (ValueError, IndexError):
                pass

    # Aggregate into segments
    if not segments:
        return []

    video_info = get_video_info(video_path)
    total_duration = video_info["duration"]
    aggregated = []
    num_segments = int(total_duration / segment_duration)

    for i in range(num_segments):
        seg_start = i * segment_duration
        seg_end = (i + 1) * segment_duration
        seg_energies = [s["energy"] for s in segments if seg_start <= s["time"] < seg_end]
        avg_energy = sum(seg_energies) / len(seg_energies) if seg_energies else -100.0
        aggregated.append({
            "start": seg_start,
            "end": seg_end,
            "energy": avg_energy,
        })

    aggregated.sort(key=lambda x: x["energy"], reverse=True)
    return aggregated


def merge_audio_into_video(
    video_path: str,
    audio_path: str,
    output_path: str,
    original_volume: float = 0.15,
    voiceover_volume: float = 1.0,
) -> str:
    """
    Merge a translated voiceover audio with the original video.
    Original audio is lowered (ducked) and voiceover is layered on top.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        FFMPEG_PATH, "-y",
        "-i", video_path,
        "-i", audio_path,
        "-filter_complex",
        f"[0:a]volume={original_volume}[orig];"
        f"[1:a]volume={voiceover_volume}[vo];"
        f"[orig][vo]amix=inputs=2:duration=first[aout]",
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]

    result = _run_media_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Audio merge failed: {result.stderr[:500]}")

    return output_path


def resize_video(
    video_path: str,
    output_path: str,
    width: int,
    height: int,
) -> str:
    """
    Resize/aspect-ratio adapt a video for a target platform.
    Uses scale + crop to fill the target dimensions exactly.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Scale to cover target, then center-crop
    cmd = [
        FFMPEG_PATH, "-y",
        "-i", video_path,
        "-vf",
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height}",
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac",
        "-movflags", "+faststart",
        output_path,
    ]

    result = _run_media_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Resize failed: {result.stderr[:500]}")

    return output_path


def burn_subtitles_into_video(
    video_path: str,
    subtitle_path: str,
    output_path: str,
) -> str:
    """
    Burn subtitle file (SRT) directly into video (hardcoded subtitles).
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Escape Windows path separators for ffmpeg subtitles filter
    sub_path_escaped = subtitle_path.replace("\\", "/").replace(":", "\\:")

    cmd = [
        FFMPEG_PATH, "-y",
        "-i", video_path,
        "-vf", f"subtitles='{sub_path_escaped}'",
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "copy",
        "-movflags", "+faststart",
        output_path,
    ]

    result = _run_media_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Subtitle burn failed: {result.stderr[:500]}")

    return output_path


def extract_frame(video_path: str, timestamp: float, output_path: str) -> str:
    """Extract a single frame from video at given timestamp as PNG."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        FFMPEG_PATH, "-y",
        "-ss", f"{timestamp:.3f}",
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        output_path,
    ]

    result = _run_media_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Frame extraction failed: {result.stderr[:500]}")

    return output_path


def get_audio_duration(audio_path: str) -> float:
    """Get duration of an audio file in seconds (also works for video files)."""
    return get_video_info(audio_path)["duration"]
