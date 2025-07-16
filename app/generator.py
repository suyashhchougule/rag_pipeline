from __future__ import annotations
from typing import Optional, List, Any
from langchain_openai import ChatOpenAI
from openai import BadRequestError
from core.generator import BaseResponseGenerator, RateLimiter, TokenLogger
from langchain_core.messages import HumanMessage
import time

class OpenAIChatGenerator(BaseResponseGenerator):
    """Concrete implementation using OpenAI via LangChain's ChatOpenAI client."""
    def __init__(
        self,
        config: dict,
        rate_limiter: Optional[RateLimiter] = None,
        token_logger: Optional[TokenLogger] = None,
    ):
        super().__init__(rate_limiter, token_logger)
        self._config = config
        self._init_chat_client(config)

    def _init_chat_client(self, cfg: dict) -> None:
        platform = cfg["PLATFORM"]
        gen = cfg[platform]
        model_id = gen["MODEL_ID"]
        self._default_model = model_id

        api_key = str(cfg.KEYS[model_id])
        endpoint = gen["ENDPOINT"]
        temp = gen["TEMPERATURE"]
        max_tk = gen["MAX_TOKENS"]

        self.client = ChatOpenAI(
            openai_api_key=api_key,
            base_url=endpoint,
            model_name=model_id,
            temperature=temp,
            max_tokens=max_tk,
        )

    def _configure_model(self, model_name: str) -> None:
        gen = self._config.get("CEREBRAS", {})
        if model_name not in gen:
            raise ValueError(f"Unknown model: {model_name}")
        new_cfg = {"PLATFORM": self._config["PLATFORM"], model_name: gen[model_name]}
        self._init_chat_client({**self._config, **new_cfg})
        self._default_model = model_name

    def _invoke(
        self, messages: List[HumanMessage], response_format: Optional[dict]
    ) -> Any:
        start = time.time()
        client = (
            self.client.bind(response_format={"type": "json_object"})
            if response_format
            else self.client
        )
        try:
            response = client.invoke(messages)
            elapsed = time.time() - start
            print(f"Model response ({self._default_model}) in {elapsed:.2f}s")
            return response
        except BadRequestError as e:
            raise RuntimeError(f"Model invocation failed: {e}")
    
    def _build_messages(
        self, text: str, image_url: Optional[str], image_data: Optional[bytes]
    ) -> List[HumanMessage]:
        # Only support text for this model
        return [HumanMessage(content=text)]