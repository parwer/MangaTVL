from .translator import OpenAICompatibleTranslator


class AsyncOpenRouterTranslator(OpenAICompatibleTranslator):
    """OpenRouter uses an AsyncOpenAI client — fully OpenAI-compatible."""
    pass
