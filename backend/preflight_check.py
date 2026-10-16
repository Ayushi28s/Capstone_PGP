
import os
import sys

from app.config import settings


def main() -> int:
    problems = []

    # Verify Groq API key
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        problems.append(
            "GROQ_API_KEY is not set. "
            "Copy .env.example to .env and add your Groq API key."
        )
    else:
        try:
            from app.llm_client import get_llm

            llm = get_llm(max_tokens=16)

            response = llm.invoke(
                "Reply with the single word: ready"
            )

            print("Groq API connection successful.")
            print("Model response:", response.content)

        except Exception as exc:
            problems.append(
                f"Groq API key is present, but the test call failed: {exc}"
            )

    # Verify SQLite database
    if not os.path.exists(settings.SQLITE_DB_PATH):
        problems.append(
            f"No database at {settings.SQLITE_DB_PATH}. "
            "Run `python scripts/seed_db.py` first."
        )

    # Verify ChromaDB index
    if not os.path.exists(settings.CHROMA_DIR):
        problems.append(
            f"No Chroma index at {settings.CHROMA_DIR}. "
            "Run `python scripts/seed_db.py` first."
        )

    # Verify knowledge graph
    if not os.path.exists(settings.KNOWLEDGE_GRAPH_PATH):
        problems.append(
            f"No knowledge graph at {settings.KNOWLEDGE_GRAPH_PATH}. "
            "Run `python scripts/seed_db.py` first."
        )

    # Verify intent classifier
    if not os.path.exists(settings.INTENT_ROUTER_MODEL_PATH):
        problems.append(
            f"No trained intent classifier at "
            f"{settings.INTENT_ROUTER_MODEL_PATH}. "
            "Run `python scripts/seed_db.py` first."
        )

    # Report problems
    if problems:
        print("Preflight check FAILED:\n")

        for problem in problems:
            print(f"  - {problem}")

        return 1

    print(
        "Preflight check passed. "
        "Groq API connection verified, database seeded, "
        "and index/graph/classifier all ready."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
