from .translator import OpenAICompatibleTranslator


class AsyncGroqTranslator(OpenAICompatibleTranslator):
    """Groq's AsyncGroq client is OpenAI-compatible (chat.completions.create)."""
    pass
