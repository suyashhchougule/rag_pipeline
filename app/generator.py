from __future__ import annotations
from typing import Optional, List, Any
from langchain_openai import ChatOpenAI
from openai import BadRequestError
from core.generator import BaseResponseGenerator, RateLimiter, TokenLogger
from langchain_core.messages import HumanMessage
import time
import logging
log = logging.getLogger(__name__)

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
        default_section = config.DEFAULT.GENERATOR_MODEL_SECTION
        self._init_chat_client(default_section)

    def _init_chat_client(self, section: str) -> None:
        cfg = self._get_model_config(section)
        self._default_model = section
        self.client = ChatOpenAI(
            openai_api_key=cfg["api_key"],
            base_url=cfg["endpoint"],
            model_name=cfg["model_id"],
            temperature=cfg["temp"],
            max_tokens=cfg["max_tokens"],
        )

    def _get_model_config(self, section: str):
        gen_cfg = self._config.get(section)
        if not gen_cfg:
            raise ValueError(f"Model config section '{section}' not found in config.")
        model_id = gen_cfg["MODEL_ID"]
        api_key = self._config.KEYS[gen_cfg["API_KEY"]]
        log.info(f"setting generation model ({model_id})")
        return {
            "model_id": model_id,
            "endpoint": gen_cfg["ENDPOINT"],
            "temp": gen_cfg.get("TEMPERATURE", 0.1),
            "max_tokens": gen_cfg.get("MAX_TOKENS"),
            "api_key": api_key,
        }

    def _configure_model(self, model_section: str) -> None:
        self._init_chat_client(model_section)

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
            log.info(f"Model response ({self._default_model}) in {elapsed:.2f}s")
            return response
        except BadRequestError as e:
            raise RuntimeError(f"Model invocation failed: {e}")
    
    def _build_messages(
        self, text: str, image_url: Optional[str], image_data: Optional[bytes]
    ) -> List[HumanMessage]:
        # Only support text for this model
        return [HumanMessage(content=text)]