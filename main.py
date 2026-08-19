"""
OPC AI Video Studio - Main Orchestrator
=========================================
One-Person Company AI Video Studio Project
Based on the project by Prof. Yong Zhang, Shenzhen University

One Human + Multiple AI Agents = One-Person Video Studio

This orchestrator coordinates a team of AI agents that:
1. Cut highlights from long videos into short clips
2. Generate multi-language subtitles
3. Translate audio into multiple languages
4. Generate thumbnails, social media copy, format for platforms, and review quality

Usage:
    python main.py <video_path> [options]

    Options:
        --config <path>      Path to config file (default: config.yaml)
        --languages <codes>  Comma-separated target language codes (e.g., en,zh,es)
        --max-clips <n>      Maximum number of clips to generate
        --skip <stages>      Comma-separated stages to skip (e.g., thumbnail,quality)
        --output <dir>       Output directory (default: output/)

Example:
    python main.py input/my_video.mp4 --languages en,zh,ja --max-clips 5
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import yaml
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn


def _make_console() -> Console:
    """Build a Console that cannot crash the pipeline.

    When the pipeline runs inside a detached/background server process on
    Windows, stdout may have no valid console handle; rich's legacy Windows
    renderer then raises OSError [Errno 22] on flush. Probe once and fall
    back to a null console so runs keep working headless.
    """
    try:
        c = Console()
        c.print("")  # probe — exercises the same write/flush path
        return c
    except OSError:
        return Console(file=io.StringIO(), force_terminal=False, force_interactive=False)


console = _make_console()


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_banner():
    """Print the studio banner."""
    banner = """
[bold cyan]╔══════════════════════════════════════════════════════════════╗[/bold cyan]
[bold cyan]║[/bold cyan]          [bold white]OPC AI Video Studio[/bold white]                              [bold cyan]║[/bold cyan]
[bold cyan]║[/bold cyan]   [italic]One Human + Multiple AI Agents = One-Person Studio[/italic]    [bold cyan]║[/bold cyan]
[bold cyan]║[/bold cyan]                                                              [bold cyan]║[/bold cyan]
[bold cyan]║[/bold cyan]   [dim]Based on the project by Prof. Yong Zhang, SZU[/dim]          [bold cyan]║[/bold cyan]
[bold cyan]╚══════════════════════════════════════════════════════════════╝[/bold cyan]
    """
    console.print(banner)


def print_pipeline_diagram(stages: list):
    """Print a visual pipeline diagram."""
    table = Table(title="AI Agent Pipeline", show_header=True, header_style="bold cyan")
    table.add_column("Stage", style="cyan", width=6)
    table.add_column("Agent", style="white")
    table.add_column("Role", style="dim")
    table.add_column("Status", width=10)

    for i, stage in enumerate(stages, 1):
        table.add_row(str(i), stage, get_agent_role(stage), "[dim]pending[/dim]")

    console.print(table)


def get_agent_role(stage: str) -> str:
    roles = {
        "content_analysis": "Video Content Intelligence Analyst",
        "clip": "Video Highlight Cutter",
        "subtitle": "Multi-Language Subtitle Generator",
        "audio": "Audio Translation & Voiceover",
        "thumbnail": "Thumbnail Generator",
        "social_media": "Social Media Copywriter",
        "format": "Multi-Platform Formatter",
        "quality": "Quality Assurance Reviewer",
    }
    return roles.get(stage, stage)


def run_studio(video_path: str, config: dict, skip_stages: list = None,
               override_languages: str = None, override_max_clips: int = None,
               on_stage: callable = None, on_review: callable = None):
    """Run the full AI Video Studio pipeline.

    on_stage: optional callback(stage, status, message) invoked with
              status in ("started", "completed", "failed"). Used by the
              web backend (api/server.py) to stream live progress.
    on_review: optional callback(clips) invoked immediately after Clip Expert.
               It must return only the clips approved for downstream agents.
    """

    skip_stages = skip_stages or []
    start_time = time.time()

    # Apply overrides
    if override_languages:
        langs = [l.strip() for l in override_languages.split(",")]
        config.setdefault("subtitle_agent", {})["target_languages"] = langs
        config.setdefault("audio_agent", {})["target_languages"] = langs

    if override_max_clips:
        config.setdefault("clip_agent", {})["max_clips"] = override_max_clips

    # Get pipeline stages
    pipeline = config.get("orchestrator", {}).get("pipeline", [
        "content_analysis", "clip", "subtitle", "audio", "thumbnail", "social_media", "format", "quality"
    ])
    active_stages = [s for s in pipeline if s not in skip_stages]

    print_banner()
    console.print(f"\n[bold]Input:[/bold] {escape(video_path)}")
    console.print(f"[bold]Pipeline:[/bold] {escape(' -> '.join(active_stages))}")
    console.print(f"[bold]Languages:[/bold] {escape(str(config.get('subtitle_agent', {}).get('target_languages', ['en'])))}\n")

    print_pipeline_diagram(active_stages)

    # Verify input video exists
    if not os.path.exists(video_path):
        console.print(f"[red]ERROR: Video file not found: {escape(video_path)}[/red]")
        sys.exit(1)

    # Initialize pipeline data
    pipeline_data = {
        "video_path": os.path.abspath(video_path),
        "video_name": Path(video_path).stem,
    }

    results = {}

    # Run each stage
    for stage in active_stages:
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold cyan]Stage: {stage.upper()}[/bold cyan]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]")

        if on_stage:
            on_stage(stage, "started", None)

        try:
            if stage == "content_analysis":
                from agents.content_analyzer_agent import ContentAnalyzerAgent
                agent = ContentAnalyzerAgent(config)
                result = agent.run(pipeline_data)
                pipeline_data["content_analysis"] = result["content_analysis"]
                pipeline_data["full_transcript"] = result["full_transcript"]
                pipeline_data["detected_language"] = result["detected_language"]
                pipeline_data["video_info"] = result["video_info"]
                results["content_analysis"] = result

            elif stage == "clip":
                from agents.clip_agent import ClipAgent
                agent = ClipAgent(config)
                result = agent.run(pipeline_data)
                pipeline_data["clips"] = result["clips"]
                pipeline_data["video_info"] = result["video_info"]
                results["clip"] = result

            elif stage == "subtitle":
                from agents.subtitle_agent import SubtitleAgent
                agent = SubtitleAgent(config)
                result = agent.run(pipeline_data)
                pipeline_data["clips"] = result["clips"]
                results["subtitle"] = result

            elif stage == "audio":
                from agents.audio_agent import AudioAgent
                agent = AudioAgent(config)
                result = agent.run(pipeline_data)
                pipeline_data["clips"] = result["clips"]
                results["audio"] = result

            elif stage == "thumbnail":
                from agents.thumbnail_agent import ThumbnailAgent
                agent = ThumbnailAgent(config)
                result = agent.run(pipeline_data)
                pipeline_data["clips"] = result["clips"]
                results["thumbnail"] = result

            elif stage == "social_media":
                from agents.social_media_agent import SocialMediaAgent
                agent = SocialMediaAgent(config)
                result = agent.run(pipeline_data)
                pipeline_data["clips"] = result["clips"]
                results["social_media"] = result

            elif stage == "format":
                from agents.format_agent import FormatAgent
                agent = FormatAgent(config)
                result = agent.run(pipeline_data)
                pipeline_data["clips"] = result["clips"]
                results["format"] = result

            elif stage == "quality":
                from agents.quality_agent import QualityAgent
                agent = QualityAgent(config)
                result = agent.run(pipeline_data)
                results["quality"] = result

            if on_stage:
                summary = {}
                if stage == "clip":
                    summary = {"clips": len(result.get("clips", []))}
                on_stage(stage, "completed", summary)

            if stage == "clip" and on_review:
                approved_clips = on_review(list(pipeline_data.get("clips", [])))
                pipeline_data["clips"] = list(approved_clips or [])
                pipeline_data["review"] = {
                    "generated_count": len(result.get("clips", [])),
                    "approved_count": len(pipeline_data["clips"]),
                    "rejected_count": (
                        len(result.get("clips", [])) - len(pipeline_data["clips"])
                    ),
                }
                if not pipeline_data["clips"]:
                    console.print(
                        "[yellow]No clips approved. Skipping all downstream stages.[/yellow]"
                    )
                    break

        except Exception as e:
            console.print(f"[red]Stage '{escape(stage)}' failed: {escape(str(e))}[/red]")
            import traceback
            try:
                traceback.print_exc()
            except OSError:
                pass  # stderr unavailable in detached server process
            if on_stage:
                on_stage(stage, "failed", str(e))
            if stage in ("content_analysis", "clip", "subtitle"):  # Core stages - stop on failure
                console.print(f"[red]Core stage '{escape(stage)}' failed. Stopping pipeline.[/red]")
                break
            console.print(f"[yellow]Continuing to next stage...[/yellow]")

    # Generate final summary
    elapsed = time.time() - start_time
    print_summary(pipeline_data, results, elapsed)

    # Save manifest
    save_manifest(pipeline_data, results, elapsed, config)

    return pipeline_data


def print_summary(pipeline_data: dict, results: dict, elapsed: float):
    """Print final summary of all outputs."""
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold green]OPC AI Video Studio - Complete![/bold green]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

    clips = pipeline_data.get("clips", [])

    table = Table(title="Generated Content Summary", show_header=True, header_style="bold green")
    table.add_column("Clip", style="cyan")
    table.add_column("Duration", justify="right")
    table.add_column("Subtitles")
    table.add_column("Audio")
    table.add_column("Thumbnail")
    table.add_column("Social")
    table.add_column("Formats")

    for clip in clips:
        sub_count = len(clip.get("subtitles", {}))
        audio_count = len(clip.get("audio_translations", {}))
        thumb_count = len(clip.get("thumbnails", []))
        has_social = "yes" if clip.get("social_media") else "no"
        fmt_count = len(clip.get("platform_versions", {}))

        table.add_row(
            escape(str(clip.get("filename", "?"))),
            f"{clip.get('duration', 0):.1f}s",
            str(sub_count) + " langs" if sub_count else "-",
            str(audio_count) + " langs" if audio_count else "-",
            str(thumb_count) if thumb_count else "-",
            has_social,
            str(fmt_count) if fmt_count else "-",
        )

    console.print(table)
    console.print(f"\n[bold]Total time:[/bold] {elapsed:.1f}s")
    console.print(f"[bold]Clips generated:[/bold] {len(clips)}")

    # Content analysis summary
    ca = pipeline_data.get("content_analysis")
    if ca:
        console.print(f"[bold]Content analysis:[/bold] {ca.get('total_segments', 0)} segments, "
                     f"{len(ca.get('topics', []))} topics, "
                     f"{len(ca.get('chapters', []))} chapters, "
                     f"{len(ca.get('key_moments', []))} key moments")
        console.print(f"[bold]Summary:[/bold] {escape(str(ca.get('summary', 'N/A'))[:120])}")

    # Quality report if available
    if "quality" in results:
        qr = results["quality"].get("quality_report", {}).get("summary", {})
        if qr:
            console.print(f"[bold]Quality:[/bold] {qr.get('clips_passed', '?')}/"
                         f"{qr.get('clips_passed', 0) + qr.get('clips_failed', 0)} clips passed, "
                         f"{qr.get('total_issues', 0)} issues")


def save_manifest(pipeline_data: dict, results: dict, elapsed: float, config: dict):
    """Save a manifest file with all generated content paths."""
    manifest = {
        "project": "OPC AI Video Studio",
        "video": pipeline_data.get("video_name", ""),
        "video_path": pipeline_data.get("video_path", ""),
        "processing_time_seconds": round(elapsed, 1),
        "config": {
            "target_languages": config.get("subtitle_agent", {}).get("target_languages", []),
            "max_clips": config.get("clip_agent", {}).get("max_clips", 5),
            "whisper_model": config.get("subtitle_agent", {}).get("whisper_model", "base"),
        },
        "content_analysis": {},
        "clips": [],
    }

    # Add content analysis summary
    ca = pipeline_data.get("content_analysis")
    if ca:
        manifest["content_analysis"] = {
            "detected_language": ca.get("detected_language", ""),
            "total_segments": ca.get("total_segments", 0),
            "total_words": ca.get("total_words", 0),
            "topics": [t["term"] for t in ca.get("topics", [])],
            "chapters": len(ca.get("chapters", [])),
            "key_moments": len(ca.get("key_moments", [])),
            "summary": ca.get("summary", ""),
            "overall_sentiment": ca.get("sentiment", {}).get("overall", ""),
        }

    for clip in pipeline_data.get("clips", []):
        clip_manifest = {
            "filename": clip.get("filename", ""),
            "path": clip.get("path", ""),
            "duration": clip.get("duration", 0),
            "start": clip.get("start", 0),
            "end": clip.get("end", 0),
            "title": clip.get("title", ""),
            "reason": clip.get("reason", ""),
            "detected_language": clip.get("detected_language", ""),
            "hook_score": clip.get("hook_score", 0),
            "hook_score_breakdown": clip.get("hook_score_breakdown", {}),
            "topic_relevance": clip.get("topic_relevance", 0),
            "matched_topic": clip.get("matched_topic", ""),
            "is_topic_relevant": clip.get("is_topic_relevant", False),
            "selection_score": clip.get("selection_score", 0),
        }

        if clip.get("subtitles"):
            clip_manifest["subtitles"] = {}
            for lang, lang_data in clip["subtitles"].items():
                clip_manifest["subtitles"][lang] = {
                    "formats": {k: v.get("path", "") for k, v in
                               lang_data.get("formats", {}).items()},
                    "segment_count": len(lang_data.get("segments", [])),
                }

        if clip.get("audio_translations"):
            clip_manifest["audio_translations"] = {}
            for lang, lang_data in clip["audio_translations"].items():
                clip_manifest["audio_translations"][lang] = {
                    "video_path": lang_data.get("video_path", ""),
                    "voiceover_path": lang_data.get("voiceover_path", ""),
                }

        if clip.get("thumbnails"):
            clip_manifest["thumbnails"] = [
                t.get("path", "") for t in clip["thumbnails"]
            ]

        if clip.get("social_media"):
            clip_manifest["social_media"] = clip["social_media"].get("copy_file", "")

        if clip.get("platform_versions"):
            clip_manifest["platform_versions"] = {
                k: v.get("path", "") for k, v in clip["platform_versions"].items()
                if "path" in v
            }

        manifest["clips"].append(clip_manifest)

    manifest_path = os.path.join(
        config.get("paths", {}).get("output_dir", "output"),
        "manifest.json"
    )
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    console.print(f"\n[dim]Manifest saved to: {escape(manifest_path)}[/dim]")


def main():
    parser = argparse.ArgumentParser(
        description="OPC AI Video Studio - One-Person AI Video Studio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py input/my_video.mp4
  python main.py input/travel_vlog.mp4 --languages en,zh,ja --max-clips 5
  python main.py input/lecture.mp4 --skip thumbnail,quality
        """,
    )
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--languages", help="Comma-separated target languages (e.g., en,zh,es)")
    parser.add_argument("--max-clips", type=int, help="Maximum number of clips to generate")
    parser.add_argument("--skip", help="Comma-separated stages to skip")
    parser.add_argument("--output", help="Output directory override")

    args = parser.parse_args()

    # Load config
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)

    config = load_config(config_path)

    # Apply output override
    if args.output:
        config.setdefault("paths", {})["output_dir"] = args.output

    # Parse skip stages
    skip_stages = args.skip.split(",") if args.skip else []

    # Run the studio
    run_studio(
        video_path=args.video_path,
        config=config,
        skip_stages=skip_stages,
        override_languages=args.languages,
        override_max_clips=args.max_clips,
    )


if __name__ == "__main__":
    main()
