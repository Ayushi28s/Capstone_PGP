from crewai import LLM as CrewAILLM
from langchain_openai import ChatOpenAI

from app.config import settings


class GroqCrewAILLM(CrewAILLM):
    """Adapt CrewAI completion requests for Groq."""

    def _prepare_completion_params(
        self,
        messages,
        tools=None,
        skip_file_processing=False,
    ):
        params = super()._prepare_completion_params(
            messages,
            tools=tools,
            skip_file_processing=skip_file_processing,
        )

        # Groq does not accept CrewAI's cache breakpoint markers.
        params["messages"] = [
            {
                key: value
                for key, value in message.items()
                if key != "cache_breakpoint"
            }
            for message in params["messages"]
        ]

        # Do not force an unregistered output-formatting tool.
        if tools:
            params["tool_choice"] = "auto"
            params["parallel_tool_calls"] = False
        else:
            params.pop("tool_choice", None)
            params.pop("parallel_tool_calls", None)

        return params


def get_llm(
    temperature: float = 0.0,
    max_tokens: int = 1000,
) -> ChatOpenAI:

    if not settings.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add your Groq API key to the .env file."
        )

    return ChatOpenAI(
        model=settings.MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=settings.GROQ_API_KEY,
        base_url=settings.GROQ_BASE_URL,
    )


def router_llm() -> ChatOpenAI:
    """Fallback LLM router for low-confidence intent classification."""
    return get_llm(
        temperature=0.0,
        max_tokens=settings.MAX_TOKENS_ROUTER,
    )


def supervisor_llm() -> ChatOpenAI:
    return get_llm(
        temperature=0.0,
        max_tokens=settings.MAX_TOKENS_SUPERVISOR,
    )


def agent_llm() -> ChatOpenAI:
    return get_llm(
        temperature=0.1,
        max_tokens=settings.MAX_TOKENS_AGENT,
    )


def crew_agent_llm() -> CrewAILLM:
    """Create a Groq-compatible CrewAI LLM."""

    if not settings.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add your Groq API key to the .env file."
        )

    return GroqCrewAILLM(
        model=f"groq/{settings.MODEL}",
        api_key=settings.GROQ_API_KEY,
        max_tokens=settings.MAX_TOKENS_CREW_AGENT,
        temperature=0.0,
    )


def deep_agent_llm() -> ChatOpenAI:
    return get_llm(
        temperature=0.2,
        max_tokens=settings.MAX_TOKENS_DEEP_AGENT,
    )


def summary_llm() -> ChatOpenAI:
    return get_llm(
        temperature=0.2,
        max_tokens=settings.MAX_TOKENS_SUMMARY,
    )