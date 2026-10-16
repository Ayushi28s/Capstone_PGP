
import logging
import os
from dataclasses import dataclass

from nemoguardrails import LLMRails, RailsConfig

from app.config import settings
from app.db import log_guardrail_event


logger = logging.getLogger(__name__)

_rails: LLMRails | None = None

REFUSAL_MESSAGE = "I'm sorry, I can't respond to that."


@dataclass
class InputRailResult:
    allowed: bool
    message: str
    triggered_flow: str | None = None
    blocked_reason: str | None = None


def _get_rails() -> LLMRails:
    global _rails

    if _rails is None:
        if not settings.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not configured."
            )

        # Configure NeMo Guardrails to use Groq's
        # OpenAI-compatible API.
        os.environ["OPENAI_API_KEY"] = settings.GROQ_API_KEY
        os.environ["OPENAI_BASE_URL"] = settings.GROQ_BASE_URL

        config = RailsConfig.from_path(
            settings.NEMO_CONFIG_DIR
        )

        _rails = LLMRails(config)

        logger.info(
            "NeMo Guardrails initialized successfully."
        )

    return _rails


def check_input(
    text: str,
    session_id: str | None = None,
) -> InputRailResult:
    """
    Validate an incoming user request using NeMo Guardrails.

    Legitimate retail operations requests, including refund
    processing, are allowed when the configured input safety
    rules permit them.

    Malicious or disallowed requests remain blocked.

    Validation failures and execution errors are handled
    using fail-closed behaviour.
    """

    try:
        rails = _get_rails()

        logger.info(
            "Running NeMo input validation for session %s",
            session_id,
        )

        response = rails.generate(
            messages=[
                {
                    "role": "user",
                    "content": text,
                }
            ]
        )

        # Extract the response safely.
        content = (
            response.get("content", "")
            if isinstance(response, dict)
            else ""
        )

        if not isinstance(content, str):
            content = ""

        content = content.strip()

        # -------------------------------------------------
        # Block requests rejected by NeMo Guardrails
        # -------------------------------------------------

        if content == REFUSAL_MESSAGE:

            logger.warning(
                "NeMo input guardrail blocked session %s",
                session_id,
            )

            log_guardrail_event(
                session_id,
                "input_rail:self_check_input",
                "blocked",
                content,
            )

            return InputRailResult(
                allowed=False,
                message=content,
                triggered_flow="self_check_input",
                blocked_reason="input_validation_failed",
            )

        # -------------------------------------------------
        # Handle invalid guardrail responses
        # -------------------------------------------------

        if (
            not content
            or "internal error" in content.lower()
        ):

            logger.error(
                "NeMo returned an invalid validation response "
                "for session %s: %r",
                session_id,
                content[:300],
            )

            log_guardrail_event(
                session_id,
                "input_rail:validation_error",
                "blocked",
                "Guardrail validation returned an invalid response.",
            )

            return InputRailResult(
                allowed=False,
                message=(
                    "Unable to validate this request. "
                    "Please try again."
                ),
                triggered_flow="validation_error",
                blocked_reason="guardrail_validation_failed",
            )

        # -------------------------------------------------
        # Allow requests accepted by NeMo Guardrails
        # -------------------------------------------------

        logger.info(
            "NeMo input validation passed for session %s",
            session_id,
        )

        return InputRailResult(
            allowed=True,
            message=text,
            triggered_flow=None,
            blocked_reason=None,
        )

    # -----------------------------------------------------
    # Handle NeMo execution failures
    # -----------------------------------------------------

    except Exception as exc:

        logger.exception(
            "NeMo input validation failed for session %s",
            session_id,
        )

        log_guardrail_event(
            session_id,
            "input_rail:execution_error",
            "blocked",
            (
                "Guardrail execution failed: "
                f"{type(exc).__name__}"
            ),
        )

        return InputRailResult(
            allowed=False,
            message=(
                "Unable to validate this request. "
                "Please try again."
            ),
            triggered_flow="execution_error",
            blocked_reason="guardrail_execution_failed",
        )
