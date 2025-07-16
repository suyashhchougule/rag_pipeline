from __future__ import annotations
from abc import ABC, abstractmethod
import base64
import time
import httpx
import warnings
from pathlib import Path
from typing import Optional, List, Any, Protocol, runtime_checkable

from langchain_core.messages import HumanMessage

@runtime_checkable
class RateLimiter(Protocol):
    def get_status(self) -> dict: ...
    def acquire(self, tokens: int = 0, timeout: float = None) -> bool: ...

@runtime_checkable
class TokenLogger(Protocol):
    def log(self, prompt: int, completion: int) -> None: ...

class BaseResponseGenerator(ABC):
    """Abstract base class defining the response-generation workflow."""
    def __init__(
        self,
        rate_limiter: Optional[RateLimiter] = None,
        token_logger: Optional[TokenLogger] = None,
    ):
        self.rate_limiter = rate_limiter
        self.token_logger = token_logger
        self._default_model: Optional[str] = None

    def generate_response(
        self,
        text: str,
        image_url: Optional[str] = None,
        image_data: Optional[bytes] = None,
        response_format: Optional[dict] = None,
        model_name: Optional[str] = None,
    ) -> Optional[str]:
        self._validate_input(text, image_url, image_data)
        if model_name and model_name != self._default_model:
            self._configure_model(model_name)
        messages = self._build_messages(text, image_url, image_data)
        raw = self._invoke(messages, response_format)
        self._record_tokens(raw)
        return self._extract_text(raw)

    def _validate_input(self, text: str, image_url: Optional[str], image_data: Optional[bytes]) -> None:
        if not text:
            raise ValueError("Input text must not be empty.")
        if image_url and image_data:
            raise ValueError("Provide either image_url or image_data, not both.")

    def _enforce_rate_limit(self, tokens) -> None:
        if self.rate_limiter:
            status = self.rate_limiter.get_status()
            # could log status here
            if not self.rate_limiter.acquire(tokens=tokens):
                raise RuntimeError("Unable to acquire rate limit slot.")
        else:
            warnings.warn("No rate limiter provided; skipping enforcement.")

    def _build_messages(
        self, text: str, image_url: Optional[str], image_data: Optional[bytes]
    ) -> List[HumanMessage]:
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        if image_url or image_data:
            print("=====================")
            encoded = self._encode_image(image_url, image_data)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            })
        return [HumanMessage(content=content)]

    def _encode_image(self, image_url: Optional[str], image_data: Optional[bytes]) -> str:
        if image_url:
            path = Path(image_url)
            if path.exists():
                data = path.read_bytes()
            else:
                data = httpx.get(image_url).content
        elif image_data:
            data = image_data
        else:
            raise ValueError("No image data to encode.")
        return base64.b64encode(data).decode("utf-8")

    def _record_tokens(self, response: Any) -> None:
        if not self.token_logger or not response:
            return
        usage = getattr(response, "response_metadata", {}).get("token_usage", {})
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        self.token_logger.log(prompt, completion)
        self._enforce_rate_limit(prompt)

    def _extract_text(self, response: Any) -> str:
        return getattr(response, "content", "")

    @abstractmethod
    def _configure_model(self, model_name: str) -> None: ...
    @abstractmethod
    def _invoke(self, messages: List[HumanMessage], response_format: Optional[dict]) -> Any: ...