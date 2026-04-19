"""MCP (Model Context Protocol) Hub/Proxy for LLM routing.

Provides a unified interface for calling LLMs across providers,
with support for fallback chains, proxy routing, and request logging.

Optional dependency: openai SDK (used for OpenRouter and compatible endpoints).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI  # type: ignore
    _OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore
    _OPENAI_AVAILABLE = False


class MCPHub:
    """Hub/proxy for LLM calls across providers.

    Reads defaults from system.ini [llm] section passed via ``system_config``.
    Override at construction or per-call via kwargs.

    Parameters
    ----------
    api_key:
        LLM provider API key (falls back to env OPENROUTER_API_KEY / OPENAI_API_KEY).
    base_url:
        Provider base URL.
    model:
        Default model identifier.
    system_config:
        Dict parsed from system.ini (keyed by section name).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        system_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = (system_config or {}).get("llm", {})
        self.api_key = (
            api_key
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENAI_API_KEY", "")
        )
        self.base_url = base_url or cfg.get("base_url", "https://openrouter.ai/api/v1")
        self.model = model or cfg.get("model", "openai/gpt-4o-mini")
        self.base_prompt = cfg.get(
            "base_prompt", "You are a trading research assistant."
        )
        self._client: Any = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            if not _OPENAI_AVAILABLE:
                raise RuntimeError(
                    "openai package not installed; cannot make LLM calls."
                )
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Call the LLM with the given messages.

        Parameters
        ----------
        messages:
            List of message dicts with 'role' and 'content'.
        model:
            Override model for this call.
        system_prompt:
            Optional system message prepended to messages.
        **kwargs:
            Additional kwargs forwarded to the chat completion API.

        Returns
        -------
        str
            The assistant's response text.
        """
        full_messages: List[Dict[str, str]] = []
        sp = system_prompt if system_prompt is not None else self.base_prompt
        if sp:
            full_messages.append({"role": "system", "content": sp})
        full_messages.extend(messages)

        use_model = model or self.model
        logger.debug("MCPHub.call model=%s messages=%d", use_model, len(full_messages))

        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=use_model,
                messages=full_messages,
                **kwargs,
            )
            content: str = resp.choices[0].message.content or ""
            return content
        except Exception as exc:
            logger.warning("MCPHub.call failed: %s", exc)
            return f"[MCPHub error: {exc}]"

    def call_json(
        self,
        messages: List[Dict[str, str]],
        schema_hint: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Call the LLM and parse the response as JSON.

        Falls back to ``{"raw": response_text}`` on parse failure.
        """
        hint = schema_hint or "Respond with valid JSON only."
        augmented = list(messages)
        augmented.append({"role": "user", "content": hint})
        text = self.call(augmented, **kwargs)
        try:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                # Strip markdown code fences
                parts = cleaned.split("```", 2)
                if len(parts) >= 2:
                    cleaned = parts[1]
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:]
            return json.loads(cleaned.strip())
        except Exception:
            return {"raw": text}
