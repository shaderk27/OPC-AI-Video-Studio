import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException

import main
from api import webapp
from api.models import RunRequest
from api.webapp import WebRunRequest
from agents.manager_agent import ManagerAgent
from utils.gemini_clip_reviewer import GeminiRequiredError


class GeminiRequiredPipelineTests(unittest.TestCase):
    def test_api_requests_require_gemini_by_default(self):
        request = RunRequest(video_id="uploaded-video")

        self.assertTrue(request.gemini_required)

    def test_web_override_enables_and_requires_gemini(self):
        config = {
            "clip_agent": {"gemini_review": {"enabled": False, "required": False}},
            "format_agent": {"formats": []},
        }
        request = WebRunRequest(video="input/example.mp4", gemini_required=True)
        run = {}

        webapp._apply_overrides(config, request, run)

        review = config["clip_agent"]["gemini_review"]
        self.assertTrue(review["enabled"])
        self.assertTrue(review["required"])
        content_review = config["content_analyzer"]["gemini_analysis"]
        self.assertTrue(content_review["enabled"])
        self.assertTrue(content_review["required"])

    def test_web_run_rejects_required_mode_when_setup_is_invalid(self):
        request = WebRunRequest(video="input/example.mp4", gemini_required=True)
        webapp.ACTIVE_RUN.clear()

        with patch(
            "utils.gemini_clip_reviewer.GeminiClipReviewer.configuration_error",
            return_value="GEMINI_API_KEY is not set",
        ):
            with self.assertRaises(HTTPException) as caught:
                webapp.api_run(request)

        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("GEMINI_API_KEY", caught.exception.detail)
        self.assertFalse(webapp.ACTIVE_RUN.get("busy", False))

    def test_web_run_rejects_required_mode_without_clip_stage(self):
        request = WebRunRequest(
            video="input/example.mp4",
            gemini_required=True,
            stages=["content_analysis", "subtitle"],
        )
        webapp.ACTIVE_RUN.clear()

        with self.assertRaises(HTTPException) as caught:
            webapp.api_run(request)

        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("Clip Expert", caught.exception.detail)

    def test_cli_pipeline_propagates_required_gemini_failure(self):
        config = {
            "orchestrator": {"pipeline": ["clip"]},
            "clip_agent": {"gemini_review": {"enabled": True, "required": True}},
            "subtitle_agent": {"target_languages": ["en"]},
        }
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video:
            video_path = video.name
        try:
            with patch.object(main.console, "print"), patch(
                    "traceback.print_exc"), patch(
                    "agents.clip_agent.ClipAgent.run",
                    side_effect=GeminiRequiredError("Gemini API unreachable")):
                with self.assertRaisesRegex(GeminiRequiredError, "unreachable"):
                    main.run_studio(video_path=video_path, config=config)
        finally:
            os.remove(video_path)

    def test_manager_pipeline_propagates_required_gemini_failure(self):
        manager = ManagerAgent({
            "manager_agent": {"max_retries": 0},
            "clip_agent": {"gemini_review": {"enabled": True, "required": True}},
        })
        failing_agent = Mock()
        failing_agent.run.side_effect = GeminiRequiredError("Gemini API unreachable")

        with patch.object(manager, "_instantiate_agent", return_value=failing_agent):
            with self.assertRaisesRegex(RuntimeError, "Gemini API unreachable"):
                manager._run_stage_with_retry("clip", {}, {})


if __name__ == "__main__":
    unittest.main()
