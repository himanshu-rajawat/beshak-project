import os

from openai import OpenAI

from app.llm_providers.base import BaseLLMProvider, parse_json_response, register

_OPENAI_GPT4O = "gpt-4o"
_OPENAI_MINI = "gpt-4o-mini"

_JSON_FORMAT = {"type": "json_object"}


@register("openai")
class OpenAIProvider(BaseLLMProvider):

    def __init__(self):
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY is not set in environment")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def _messages_with_system(self, system: str, messages: list[dict]) -> list[dict]:
        return [{"role": "system", "content": system}] + messages

    def chat(self, messages: list[dict], system: str, max_tokens: int = 2048) -> str:
        response = self._get_client().chat.completions.create(
            model=_OPENAI_GPT4O,
            max_tokens=max_tokens,
            messages=self._messages_with_system(system, messages),
        )
        return response.choices[0].message.content or ""

    def chat_json(self, messages: list[dict], system: str, max_tokens: int = 2048) -> dict:
        response = self._get_client().chat.completions.create(
            model=_OPENAI_GPT4O,
            max_tokens=max_tokens,
            response_format=_JSON_FORMAT,
            messages=self._messages_with_system(system, messages),
        )
        return parse_json_response(response.choices[0].message.content or "", fallback={})

    def chat_lightweight(self, messages: list[dict], system: str, max_tokens: int = 600) -> str:
        response = self._get_client().chat.completions.create(
            model=_OPENAI_MINI,
            max_tokens=max_tokens,
            messages=self._messages_with_system(system, messages),
        )
        return response.choices[0].message.content or ""
