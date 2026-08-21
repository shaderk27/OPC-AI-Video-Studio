import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from utils.subtitle_utils import _load_whisper_model, transcribe_audio


class _WhisperModel:
    def __init__(self):
        self.kwargs = None

    def transcribe(self, audio_path, **kwargs):
        self.kwargs = kwargs
        return ([SimpleNamespace(
            start=0.0, end=2.0, text="  Lionel   Messi scores.  "
        )], None)


class TranscriptionOptionsTests(unittest.TestCase):
    @patch("utils.subtitle_utils._load_whisper_model")
    def test_passes_accuracy_and_vocabulary_options(self, load_model):
        model = _WhisperModel()
        load_model.return_value = model

        result = transcribe_audio(
            "speech.wav",
            model_size="small",
            language="en",
            beam_size=7,
            vad_filter=True,
            vad_min_silence_ms=450,
            vocabulary_hints=["Lionel Messi", "Argentina"],
        )

        load_model.assert_called_once_with("small")
        self.assertEqual(model.kwargs["beam_size"], 7)
        self.assertEqual(
            model.kwargs["vad_parameters"]["min_silence_duration_ms"], 450
        )
        self.assertIn("Lionel Messi", model.kwargs["initial_prompt"])
        self.assertEqual(result[0]["text"], "Lionel Messi scores.")

    @patch.dict(os.environ, {"HF_ENDPOINT": "https://unavailable.example"})
    @patch("utils.subtitle_utils._activate_hf_endpoint")
    @patch("utils.subtitle_utils._download_whisper_model_direct")
    @patch("utils.subtitle_utils._cached_whisper_model_path")
    @patch("faster_whisper.WhisperModel")
    def test_failed_small_download_uses_cached_base(
        self, whisper_model, cached_path, direct_download, activate_endpoint
    ):
        base_path = Path("models/whisper/base-snapshot")
        cached_path.side_effect = [None, base_path]
        fallback = object()
        whisper_model.side_effect = [
            RuntimeError("configured endpoint failed"),
            RuntimeError("mirror failed"),
            RuntimeError("official endpoint failed"),
            fallback,
        ]
        direct_download.side_effect = RuntimeError("direct download failed")

        with self.assertWarnsRegex(RuntimeWarning, "cached 'base'"):
            result = _load_whisper_model("small")

        self.assertIs(result, fallback)
        self.assertEqual(whisper_model.call_args.args[0], str(base_path))
        attempted = [call.args[0] for call in activate_endpoint.call_args_list]
        self.assertIn("https://hf-mirror.com", attempted)
        self.assertIn("https://huggingface.co", attempted)


if __name__ == "__main__":
    unittest.main()
