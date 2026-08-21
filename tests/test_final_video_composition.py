import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from api import webapp


class FinalVideoCompositionTests(unittest.TestCase):
    def test_composes_from_formatted_dub_and_matching_subtitle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            formatted_source = root / "source_youtube.mp4"
            dubbed = root / "dub_ko.mp4"
            formatted_dub = root / "dub_ko_youtube.mp4"
            subtitle = root / "subtitle_ko.srt"
            for path in (source, formatted_source, dubbed, formatted_dub, subtitle):
                path.write_bytes(b"fixture")

            manifest = {
                "clips": [{
                    "filename": "clip.mp4",
                    "path": str(source),
                    "duration": 20,
                    "platform_versions": {"YouTube": str(formatted_source)},
                    "subtitles": {"ko": {"formats": {"srt": str(subtitle)}}},
                    "audio_translations": {"ko": {
                        "video_path": str(dubbed),
                        "formatted_versions": {"YouTube": str(formatted_dub)},
                    }},
                }],
            }
            run = webapp._new_run()
            run.update({
                "active_stages": ["subtitle", "audio", "format"],
                "manifest": manifest,
            })
            cfg = {
                "paths": {"output_dir": str(root)},
                "format_agent": {"formats": [{
                    "platform": "YouTube",
                    "aspect_ratio": "16:9",
                    "enabled": True,
                }]},
            }
            request = webapp.WebRunRequest(
                video=str(source),
                subtitle_languages=["ko"],
                audio_languages=["ko"],
                subtitle_mode="both",
                formats=["YouTube"],
            )

            def fake_burn(video, srt, output):
                Path(output).write_bytes(b"final")
                return output

            with patch(
                "utils.video_utils.burn_subtitles_into_video",
                side_effect=fake_burn,
            ) as burn:
                webapp._compose_final_videos(run, cfg, request)

            burn.assert_called_once()
            self.assertEqual(str(formatted_dub), burn.call_args.args[0])
            self.assertEqual(str(subtitle), burn.call_args.args[1])
            artifact = manifest["final_videos"][0]
            self.assertEqual("YouTube", artifact["platform"])
            self.assertEqual("16:9", artifact["aspect_ratio"])
            self.assertEqual("ko", artifact["audio_language"])
            self.assertEqual("ko", artifact["subtitle_language"])
            self.assertEqual("burned", artifact["subtitle_delivery"])
            self.assertTrue(Path(artifact["path"]).is_file())
            saved = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(artifact["filename"], saved["final_videos"][0]["filename"])

    def test_final_downloads_are_exposed_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            final_path = Path(tmp) / "final.mp4"
            final_path.write_bytes(b"final")
            run_id = "final-download-test"
            webapp.RUNS[run_id] = {
                "manifest": {"clips": [], "final_videos": [{
                    "clip_index": 1,
                    "filename": final_path.name,
                    "path": str(final_path),
                    "platform": "YouTube",
                    "audio_language": "ko",
                    "subtitle_language": "ko",
                }]},
                "output_dir": tmp,
            }
            try:
                downloads = webapp.api_run_downloads(run_id)
            finally:
                webapp.RUNS.pop(run_id, None)

            self.assertEqual(1, len(downloads["final"]))
            self.assertEqual("final.mp4", downloads["final"][0]["filename"])
            self.assertEqual("ko", downloads["final"][0]["audio_language"])


if __name__ == "__main__":
    unittest.main()
