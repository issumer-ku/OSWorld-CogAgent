"""Control signals shared by the agent and standalone runner."""

from typing import Optional


MODEL_ERROR_PREFIX = "MODEL_ERROR:"


def model_error_action(reason: str) -> str:
    normalized = "_".join(str(reason or "unknown").strip().lower().split())
    return f"{MODEL_ERROR_PREFIX}{normalized or 'unknown'}"


def decode_model_error_action(action) -> Optional[str]:
    if not isinstance(action, str) or not action.startswith(MODEL_ERROR_PREFIX):
        return None
    return action[len(MODEL_ERROR_PREFIX) :].strip() or "unknown"
