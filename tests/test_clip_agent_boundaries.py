import unittest
from unittest.mock import patch

from agents.clip_agent import ClipAgent


class ClipAgentBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.agent = ClipAgent({
            "clip_agent": {
                "min_clip_duration": 15,
                "max_clip_duration": 60,
                "padding_before": 1.5,
                "padding_after": 1.0,
                "completion_padding_after": 6.0,
                "transcript_boundary_max_extension": 2.0,
            },
        })

    def test_completion_tail_includes_shot_and_finishes_commentary(self):
        highlights = [{
            "start": 40.0,
            "end": 50.0,
            "reason": "penalty buildup",
        }]
        transcript = [{
            "start": 53.0,
            "end": 58.0,
            "text": "He scores, and the crowd erupts!",
        }]

        result = self.agent._extend_for_completion(
            highlights, total_duration=100.0,
            transcript_segments=transcript,
        )

        self.assertEqual(result[0]["start"], 40.0)
        self.assertEqual(result[0]["end"], 58.0)

    def test_long_transcript_segment_does_not_overextend_clip(self):
        highlights = [{
            "start": 40.0,
            "end": 50.0,
            "reason": "penalty buildup",
        }]
        transcript = [{
            "start": 53.0,
            "end": 80.0,
            "text": "An unusually long transcription segment.",
        }]

        result = self.agent._extend_for_completion(
            highlights, total_duration=100.0,
            transcript_segments=transcript,
        )

        self.assertEqual(result[0]["end"], 56.0)

    def test_completion_tail_preserves_payoff_within_maximum_duration(self):
        highlights = [{
            "start": 10.0,
            "end": 70.0,
            "reason": "high audio energy",
        }]

        result = self.agent._extend_for_completion(
            highlights, total_duration=100.0
        )

        self.assertEqual(result[0]["start"], 16.0)
        self.assertEqual(result[0]["end"], 76.0)
        self.assertEqual(result[0]["end"] - result[0]["start"], 60.0)

    @patch("agents.clip_agent.detect_silence")
    def test_silence_snapping_never_shortens_highlight(self, mock_silence):
        mock_silence.return_value = [
            (8.0, 9.0),       # Safe start boundary before the clip.
            (18.5, 19.0),    # Must not pull the endpoint backward.
            (21.0, 21.5),    # Safe end boundary after the clip.
        ]
        highlights = [{"start": 10.0, "end": 20.0}]

        result = self.agent._snap_to_silence("match.mp4", highlights)

        self.assertEqual(result[0]["start"], 9.0)
        self.assertEqual(result[0]["end"], 21.0)

    def test_final_boundaries_apply_padding_once_and_report_true_duration(self):
        highlights = [{"start": 40.0, "end": 58.0}]

        result = self.agent._finalize_clip_boundaries(
            highlights, total_duration=100.0
        )

        self.assertEqual(result[0]["start"], 38.5)
        self.assertEqual(result[0]["end"], 59.0)
        self.assertEqual(result[0]["end"] - result[0]["start"], 20.5)


if __name__ == "__main__":
    unittest.main()
