"""Shared structured Gemini client for the Subtitle and Audio experts."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


class GeminiAgentError(RuntimeError):
    """Raised when a required Gemini expert call cannot produce valid output."""


class GeminiAgentClient:
    """Stateless, schema-constrained Gemini caller with per-request audit traces."""

    def __init__(
        self,
        config: Dict[str, Any],
        section: str,
        agent_id: str,
        role: str,
        system_instruction: str,
        schema: Dict[str, Any],
        client: Any = None,
    ):
        self.config = config
        self.section = section
        self.agent_id = agent_id
        self.role = role
        self.system_instruction = system_instruction
        self.schema = schema
        settings = config.get(section, {}).get("gemini", {})
        self.enabled = bool(settings.get("enabled", True))
        self.required = bool(settings.get("required", True))
        self.model = settings.get("model", "gemini-3.5-flash-lite")
        self.prompt_version = str(settings.get("prompt_version", "1.0"))
        self.timeout_ms = max(1000, int(settings.get("timeout_seconds", 60) * 1000))
        self.transport_retries = max(0, int(settings.get("transport_retries", 2)))
        self.api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self._client = client

    @property
    def available(self) -> bool:
        return self.enabled and bool(self._client or self.api_key)

    def configuration_error(self) -> str:
        if not self.enabled:
            return f"{self.role} Gemini expert is disabled"
        if not self._client and not self.api_key:
            return "GEMINI_API_KEY is not set"
        try:
            self._get_client()
        except GeminiAgentError as exc:
            return str(exc)
        return ""

    def generate(
        self,
        payload: Dict[str, Any],
        *,
        clip_id: str,
        language: str,
        request_kind: str = "plan",
        repair_attempt: int = 0,
        media_path: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> tuple[Dict[str, Any], str]:
        if not self.available:
            raise GeminiAgentError(
                f"{self.role} is unavailable: {self.configuration_error()}"
            )

        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        hash_input = serialized.encode("utf-8")
        if media_path:
            media = Path(media_path)
            if not media.is_file():
                raise GeminiAgentError(f"Gemini media file not found: {media}")
            hash_input += media.read_bytes()
        input_hash = hashlib.sha256(hash_input).hexdigest()
        request_model = str(model_override or self.model)
        started = time.monotonic()
        trace: Dict[str, Any] = {
            "agent_id": self.agent_id,
            "role": self.role,
            "model": request_model,
            "prompt_version": self.prompt_version,
            "request_kind": request_kind,
            "clip_id": clip_id,
            "language": language,
            "input_hash": input_hash,
            "repair_attempt": repair_attempt,
            "started_at": datetime.now().astimezone().isoformat(),
            "status": "started",
        }
        uploaded = None
        contents: Any = serialized
        if media_path:
            try:
                client = self._get_client()
                uploaded = client.files.upload(file=media_path)
                uploaded = self._wait_until_active(client, uploaded)
                contents = [serialized, uploaded]
                trace["media"] = {
                    "filename": Path(media_path).name,
                    "sha256_included_in_input_hash": True,
                }
            except Exception as exc:
                raise GeminiAgentError(
                    f"Could not upload subtitle audio to Gemini: {exc}"
                ) from exc

        trace["transport_attempts"] = []
        max_attempts = self.transport_retries + 1
        last_error: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            attempt_started = time.monotonic()
            try:
                response = self._get_client().models.generate_content(
                    model=request_model,
                    contents=contents,
                    config={
                        "system_instruction": self.system_instruction,
                        "response_mime_type": "application/json",
                        "response_json_schema": self.schema,
                    },
                )
                parsed = getattr(response, "parsed", None)
                if not isinstance(parsed, dict):
                    try:
                        parsed = json.loads(getattr(response, "text", "") or "")
                    except (TypeError, json.JSONDecodeError) as exc:
                        raise GeminiAgentError("Gemini returned invalid JSON") from exc
                if not isinstance(parsed, dict):
                    raise GeminiAgentError("Gemini returned a non-object response")
                trace["transport_attempts"].append({
                    "attempt": attempt,
                    "status": "complete",
                    "latency_seconds": round(time.monotonic() - attempt_started, 3),
                })
                trace.update({
                    "status": "complete",
                    "result": parsed,
                    "latency_seconds": round(time.monotonic() - started, 3),
                })
                usage = getattr(response, "usage_metadata", None)
                if usage is not None:
                    trace["usage"] = {
                        name: getattr(usage, name, None)
                        for name in (
                            "prompt_token_count", "candidates_token_count",
                            "total_token_count",
                        )
                        if getattr(usage, name, None) is not None
                    }
                trace_path = self._write_trace(trace)
                self._delete_uploaded(uploaded)
                return parsed, trace_path
            except Exception as exc:
                last_error = exc
                trace["transport_attempts"].append({
                    "attempt": attempt,
                    "status": "failed",
                    "error": str(exc),
                    "latency_seconds": round(time.monotonic() - attempt_started, 3),
                })
                if attempt < max_attempts:
                    time.sleep(min(4.0, float(2 ** (attempt - 1))))

        wrapped = (
            last_error
            if isinstance(last_error, GeminiAgentError)
            else GeminiAgentError(f"Gemini request failed: {last_error}")
        )
        trace.update({
            "status": "failed",
            "error": str(wrapped),
            "latency_seconds": round(time.monotonic() - started, 3),
        })
        self._write_trace(trace)
        self._delete_uploaded(uploaded)
        if wrapped is last_error:
            raise wrapped
        raise wrapped from last_error

    def _wait_until_active(self, client: Any, uploaded: Any) -> Any:
        deadline = time.monotonic() + (self.timeout_ms / 1000)
        current = uploaded
        while True:
            state = getattr(current, "state", None)
            state_name = getattr(state, "name", str(state or "")).upper()
            if state_name in ("", "ACTIVE") or state_name.endswith("ACTIVE"):
                return current
            if state_name == "FAILED" or state_name.endswith("FAILED"):
                raise GeminiAgentError("Gemini could not process subtitle audio")
            if time.monotonic() >= deadline:
                raise GeminiAgentError("Gemini subtitle audio processing timed out")
            time.sleep(1)
            current = client.files.get(name=current.name)

    def _delete_uploaded(self, uploaded: Any) -> None:
        name = getattr(uploaded, "name", "") if uploaded is not None else ""
        if not name:
            return
        try:
            self._get_client().files.delete(name=name)
        except Exception:
            pass

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from google import genai

            self._client = genai.Client(
                api_key=self.api_key,
                http_options={"timeout": self.timeout_ms},
            )
        except Exception as exc:
            raise GeminiAgentError(f"Could not initialize Gemini client: {exc}") from exc
        return self._client

    def _write_trace(self, trace: Dict[str, Any]) -> str:
        paths = self.config.get("paths", {})
        trace_dir = Path(
            paths.get(
                "agent_traces_dir",
                Path(paths.get("output_dir", "output")) / "agent_traces",
            )
        )
        trace_dir.mkdir(parents=True, exist_ok=True)
        clip = _safe_name(str(trace.get("clip_id", "clip")))
        language = _safe_name(str(trace.get("language", "unknown")))
        filename = (
            f"{self.agent_id}_{clip}_{language}_"
            f"{trace.get('request_kind', 'plan')}_{uuid.uuid4().hex[:8]}.json"
        )
        trace_path = trace_dir / filename
        with trace_path.open("w", encoding="utf-8") as stream:
            json.dump(trace, stream, indent=2, ensure_ascii=False)
        return str(trace_path)

    def record_tool_call(
        self,
        *,
        tool: str,
        clip_id: str,
        language: str,
        payload: Dict[str, Any],
        status: str,
        latency_seconds: float,
        model: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: str = "",
        fallback_used: bool = False,
    ) -> str:
        """Persist a tool execution trace alongside Gemini reasoning traces."""
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        trace: Dict[str, Any] = {
            "agent_id": self.agent_id,
            "role": self.role,
            "model": model or self.model,
            "prompt_version": self.prompt_version,
            "request_kind": "tool_call",
            "tool": tool,
            "clip_id": clip_id,
            "language": language,
            "input_hash": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            "started_at": datetime.now().astimezone().isoformat(),
            "status": status,
            "latency_seconds": round(latency_seconds, 3),
            "fallback_used": fallback_used,
        }
        if result is not None:
            trace["result"] = result
        if error:
            trace["error"] = error
        return self._write_trace(trace)


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value)[:80]
