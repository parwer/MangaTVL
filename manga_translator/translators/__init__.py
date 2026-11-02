from .translator import AsyncTranslatorBase
from .openrouter import AsyncOpenRouterTranslator
from .groq import AsyncGroqTranslator
from .context_translator import ContextTranslator


def __getattr__(name):  # google-genai only when Gemini is used
    if name == "AsyncGeminiTranslator":
        from .gemini import AsyncGeminiTranslator
        return AsyncGeminiTranslator
    raise AttributeError(name)
