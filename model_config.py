"""Central model IDs — override via environment variables on Hugging Face or locally."""
import os

# OpenAI: gpt-4 is often legacy/restricted; gpt-4o is widely available
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Anthropic: Claude Sonnet 4 (override e.g. claude-sonnet-4-6 for newer snapshots)
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

# Google AI: Gemini 3 Flash (preview id; set GEMINI_MODEL if Google renames the stable alias)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

# App routing: UI labels + results dict keys (single source of truth)
MODEL_KEY_GPT = "GPT-4"
MODEL_KEY_CLAUDE = "Claude Sonnet 4"
MODEL_KEY_GEMINI = "Gemini 3 Flash"
MODEL_KEYS_ORDERED = [MODEL_KEY_GPT, MODEL_KEY_CLAUDE, MODEL_KEY_GEMINI]
