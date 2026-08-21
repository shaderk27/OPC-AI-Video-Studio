import unittest

from agents.clip_agent import ClipAgent


class ClipAgentFusionTests(unittest.TestCase):
    def setUp(self):
        self.agent = ClipAgent({
            "clip_agent": {
                "emotion_audio_fusion_bonus": 15.0,
            },
        })

    def test_overlapping_emotion_and_audio_receive_combined_bonus(self):
        emotional = {
            "start": 10.0,
            "end": 25.0,
            "score": 30,
            "reason": "emotional intensity",
            "source": "content_key_moment",
            "sentiment_intensity": 2,
        }
        energy = {
            "start": 12.0,
            "end": 27.0,
            "score": 25,
            "reason": "high audio energy",
            "source": "audio_energy",
            "audio_peak_start": 18.0,
            "audio_peak_end": 20.0,
            "audio_energy_db": -12.0,
            "audio_energy_score": 90.0,
        }

        fused = self.agent._fuse_emotion_and_energy([emotional, energy])

        self.assertEqual(fused[0]["fusion_bonus"], 9.0)
        self.assertEqual(fused[0]["score"], 39.0)
        self.assertEqual(fused[0]["audio_energy_score"], 90.0)
        self.assertEqual(fused[0]["audio_energy_db"], -12.0)
        self.assertIn("emotional + audio peak", fused[0]["reason"])

    def test_non_overlapping_evidence_is_not_fused(self):
        emotional = {
            "start": 10.0,
            "end": 25.0,
            "score": 30,
            "reason": "emotional intensity",
            "source": "content_key_moment",
            "sentiment_intensity": 2,
        }
        energy = {
            "start": 30.0,
            "end": 45.0,
            "score": 25,
            "source": "audio_energy",
            "audio_peak_start": 35.0,
            "audio_peak_end": 37.0,
            "audio_energy_score": 90.0,
        }

        fused = self.agent._fuse_emotion_and_energy([emotional, energy])

        self.assertEqual(fused[0]["score"], 30)
        self.assertNotIn("fusion_bonus", fused[0])

    def test_deduplication_keeps_fused_semantic_candidate(self):
        emotional = {
            "start": 10.0,
            "end": 25.0,
            "score": 20,
            "reason": "emotional intensity",
            "source": "content_key_moment",
            "sentiment_intensity": 3,
        }
        energy = {
            "start": 10.0,
            "end": 25.0,
            "score": 30,
            "source": "audio_energy",
            "audio_peak_start": 18.0,
            "audio_peak_end": 20.0,
            "audio_energy_score": 100.0,
        }

        fused = self.agent._fuse_emotion_and_energy([emotional, energy])
        kept = self.agent._deduplicate_highlights(fused)

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["source"], "content_key_moment")
        self.assertEqual(kept[0]["score"], 45.0)


if __name__ == "__main__":
    unittest.main()
