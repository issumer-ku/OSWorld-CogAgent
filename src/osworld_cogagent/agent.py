"""Enhanced CogAgent-9B adapter for OSWorld screenshot observations."""

from __future__ import annotations

import base64
import logging
import os
from io import BytesIO
from typing import Any, Optional

from PIL import Image

from .client import call_openai_compatible
from .images import image_perceptual_hash, perceptual_hash_distance
from .parser import ParsedCogAgentAction, parse_cogagent_response
from .protocol import model_error_action


logger = logging.getLogger("osworld_cogagent.agent")


class CogAgent:
    """Translate native CogAgent operations into OSWorld pyautogui actions."""

    def __init__(
        self,
        platform: str = "WIN",
        model: str = "cogagent-9b-20241220",
        max_tokens: int = 1024,
        top_p: float = 1.0,
        temperature: float = 0.0,
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        history_n: int = 15,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        loop_action_limit: int = 3,
        format_failure_limit: int = 3,
        completion_verification: bool = True,
    ):
        if action_space != "pyautogui":
            raise ValueError("CogAgent supports only pyautogui action space")
        if observation_type != "screenshot":
            raise ValueError("CogAgent-9B supports only screenshot observation")
        if platform not in {"WIN", "Mac", "Mobile"}:
            raise ValueError("CogAgent platform must be WIN, Mac, or Mobile")
        self.platform = platform
        self.model = model
        self.max_tokens = int(max_tokens)
        self.top_p = float(top_p)
        self.temperature = float(temperature)
        self.action_space = action_space
        self.observation_type = observation_type
        self.history_n = max(0, int(history_n))
        self.base_url = base_url
        self.api_key = api_key
        self.loop_action_limit = max(2, int(loop_action_limit))
        self.format_failure_limit = max(1, int(format_failure_limit))
        self.completion_verification = bool(completion_verification)
        self.history_steps: list[str] = []
        self.history_actions: list[str] = []
        self.action_records: list[tuple[str, int]] = []
        self.pending_feedback = ""
        self.consecutive_format_failures = 0
        self.consecutive_completion_audit_failures = 0

    @staticmethod
    def _image_dimensions(image_bytes: bytes) -> tuple[int, int]:
        with Image.open(BytesIO(image_bytes)) as image:
            return image.size

    @staticmethod
    def _image_url(image_bytes: bytes) -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _history_text(self) -> str:
        if not self.history_steps:
            return "\nHistory steps: "
        start = max(0, len(self.history_steps) - self.history_n) if self.history_n else len(self.history_steps)
        entries = ["\nHistory steps: "]
        for index, (step, action) in enumerate(
            zip(self.history_steps[start:], self.history_actions[start:]),
            start=start,
        ):
            entries.append(f"\n{index}. {step}\t{action}")
        return "".join(entries)

    def _query(self, instruction: str, extra_feedback: str = "") -> str:
        query = (
            f"Task: {instruction}{self._history_text()}\n"
            f"(Platform: {self.platform})\n"
            "(Answer in Status-Action-Operation format.)"
        )
        guidance = (
            "\nUse exactly one Grounded Operation. Supported operations are CLICK, "
            "DOUBLE_CLICK, RIGHT_CLICK, HOVER, TYPE, SCROLL_UP, SCROLL_DOWN, "
            "SCROLL_LEFT, SCROLL_RIGHT, KEY_PRESS, GESTURE, LAUNCH, and END. "
            "Do not use QUOTE_TEXT, QUOTE_CLIPBOARD, or LLM. Use END only when "
            "every requested condition is complete."
        )
        if self.platform == "WIN":
            guidance += (
                " The observed desktop is Ubuntu, but WIN is CogAgent's closest "
                "desktop vocabulary; use Ctrl shortcuts and visible controls."
            )
        if self.pending_feedback:
            extra_feedback = "\n".join(
                part for part in (self.pending_feedback, extra_feedback) if part
            )
            self.pending_feedback = ""
        if extra_feedback:
            guidance += f"\nExecution feedback: {extra_feedback}"
        return query + guidance

    def _messages(
        self,
        instruction: str,
        image_bytes: bytes,
        extra_feedback: str = "",
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._query(instruction, extra_feedback)},
                    {"type": "image_url", "image_url": {"url": self._image_url(image_bytes)}},
                ],
            }
        ]

    def _call(self, messages: list[dict[str, Any]], *, temperature: Optional[float] = None) -> str:
        return call_openai_compatible(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p,
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def _is_repeated(self, operation: str, screenshot_hash: int) -> bool:
        tolerance = int(os.getenv("OSWORLD_COGAGENT_STATIC_HASH_DISTANCE", "8"))
        recent = self.action_records[-self.loop_action_limit :]
        if len(recent) < self.loop_action_limit:
            return False
        return all(
            previous_operation == operation
            and perceptual_hash_distance(previous_hash, screenshot_hash) <= tolerance
            for previous_operation, previous_hash in recent
        )

    def _completion_audit(
        self,
        instruction: str,
        image_bytes: bytes,
        width: int,
        height: int,
    ) -> tuple[str, ParsedCogAgentAction]:
        feedback = (
            "COMPLETION AUDIT: inspect the current screenshot and verify every "
            "condition in the exact task. If visibly complete, return Grounded "
            "Operation: END. Otherwise return the single next executable operation."
        )
        response = self._call(self._messages(instruction, image_bytes, feedback), temperature=0.0)
        return response, parse_cogagent_response(response, width, height)

    def predict(self, instruction: str, obs: dict[str, Any]) -> tuple[str, list[str]]:
        image_bytes = obs.get("screenshot")
        if not image_bytes:
            raise ValueError("CogAgent requires obs['screenshot']")
        width, height = self._image_dimensions(image_bytes)
        screenshot_hash = image_perceptual_hash(image_bytes)
        retries = max(0, int(os.getenv("OSWORLD_COGAGENT_FORMAT_RETRIES", "2")))
        response = ""
        parsed: Optional[ParsedCogAgentAction] = None
        feedback = ""

        for attempt in range(retries + 1):
            response = self._call(
                self._messages(instruction, image_bytes, feedback),
                temperature=(
                    self.temperature
                    if attempt == 0
                    else float(os.getenv("OSWORLD_COGAGENT_RETRY_TEMPERATURE", "0.2"))
                ),
            )
            logger.info("CogAgent output (attempt %d/%d): %s", attempt + 1, retries + 1, response)
            try:
                candidate = parse_cogagent_response(response, width, height)
            except ValueError as exc:
                feedback = (
                    f"The preceding response could not be executed: {exc}. Return "
                    "one complete `Grounded Operation:` in the required format."
                )
                continue
            if self._is_repeated(candidate.grounded_operation, screenshot_hash):
                feedback = (
                    "The same operation has already been executed repeatedly while "
                    "the screen remained unchanged. Use a materially different operation."
                )
                continue
            parsed = candidate
            break

        if parsed is None:
            self.pending_feedback = feedback
            self.consecutive_format_failures += 1
            if self.consecutive_format_failures >= self.format_failure_limit:
                return response, [model_error_action("cogagent_format_recovery_exhausted")]
            return response, ["WAIT"]
        self.consecutive_format_failures = 0

        if parsed.pyautogui_code == ["DONE"] and self.completion_verification:
            try:
                response, parsed = self._completion_audit(instruction, image_bytes, width, height)
            except Exception as exc:
                logger.warning("CogAgent completion audit failed: %s", exc)
                self.consecutive_completion_audit_failures += 1
                self.pending_feedback = (
                    "The completion claim was not executed because its independent "
                    "audit failed. Inspect the current result and continue."
                )
                if self.consecutive_completion_audit_failures >= self.format_failure_limit:
                    return response, [model_error_action("cogagent_completion_audit_failed")]
                return response, ["WAIT"]
        self.consecutive_completion_audit_failures = 0

        self.history_steps.append(parsed.grounded_operation)
        self.history_actions.append(parsed.action)
        self.action_records.append((parsed.grounded_operation, screenshot_hash))
        return response, parsed.pyautogui_code

    def reset(self, runtime_logger=None, *args, **kwargs) -> None:
        global logger
        if runtime_logger is not None:
            logger = runtime_logger
        self.history_steps = []
        self.history_actions = []
        self.action_records = []
        self.pending_feedback = ""
        self.consecutive_format_failures = 0
        self.consecutive_completion_audit_failures = 0
