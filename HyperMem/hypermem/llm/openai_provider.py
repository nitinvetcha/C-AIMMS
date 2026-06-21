"""
OpenAI-style LLM provider implementation using NVIDIA’s OpenAI-compatible API.

This version defaults to a Llama model hosted on NVIDIA,
but still reads model / key / base URL from environment variables
so you can override them without changing code.
"""

import os
import time
import json
import aiohttp
from typing import Optional
import asyncio
import random

from .protocol import LLMProvider, LLMError
from hypermem.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """
    OpenAI-style LLM provider using NVIDIA’s OpenAI-compatible endpoint.

    Defaults:
        - model: NVIDIA Llama-3.1-8B Instruct (override via HYPERMEM_LLM_MODEL)
        - api_key: NVIDIA API key (HYPERMEM_NVIDIA_API_KEY)
        - base_url: NVIDIA integrate endpoint (HYPERMEM_LLM_BASE_URL to override)
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = 100 * 1024,
        enable_stats: bool = False,
        **kwargs,
    ):
        # Defaults can be overridden by env or constructor args
        default_model = os.getenv(
            "HYPERMEM_LLM_MODEL",
            "nvidia/llama-3.1-8b-instruct",   # change to the exact NVIDIA model name
        )
        default_base_url = os.getenv(
            "HYPERMEM_LLM_BASE_URL",
            "https://integrate.api.nvidia.com/v1",  # NVIDIA OpenAI-compatible endpoint
        )
        default_api_key = os.getenv("HYPERMEM_NVIDIA_API_KEY")

        self.model = model or default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_stats = enable_stats

        self.api_key = api_key or default_api_key
        self.base_url = base_url or default_base_url

        if not self.api_key:
            raise LLMError(
                "NVIDIA API key not set. Please set HYPERMEM_NVIDIA_API_KEY."
            )

        if self.enable_stats:
            self.current_call_stats = None
            self.accumulated_stats = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "call_count": 0,
                "total_duration": 0.0,
            }

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
        response_format: dict | None = None,
    ) -> str:
        start_time = time.perf_counter()

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None else self.temperature,
            "response_format": response_format,
        }

        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        elif self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        max_retries = 3
        for retry_num in range(max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=600)
                async with aiohttp.ClientSession(
                    timeout=timeout, trust_env=True
                ) as session:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        json=data,
                        headers=headers,
                    ) as response:
                        chunks = []
                        async for chunk in response.content.iter_any():
                            chunks.append(chunk)
                        raw = b"".join(chunks).decode()
                        response_data = json.loads(raw)

                        if response.status != 200:
                            error_msg = response_data.get("error", {}).get(
                                "message", f"HTTP {response.status}"
                            )
                            logger.error(
                                f"[ERROR] [Llama-{self.model}] HTTP Error {response.status}: {error_msg}"
                            )
                            if response.status == 429:
                                logger.warning(
                                    "429 Too Many Requests, waiting before retry"
                                )
                                await asyncio.sleep(random.randint(5, 20))
                            raise LLMError(
                                f"HTTP Error {response.status}: {error_msg}"
                            )

                        end_time = time.perf_counter()

                        finish_reason = (
                            response_data.get("choices", [{}])[0]
                            .get("finish_reason", "")
                        )
                        if finish_reason != "stop":
                            logger.warning(
                                f"[Llama-{self.model}] Finish reason: {finish_reason}"
                            )

                        usage = response_data.get("usage", {})
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        total_tokens = usage.get("total_tokens", 0)

                        logger.debug(
                            f"[Llama-{self.model}] Elapsed: {end_time - start_time:.2f}s, "
                            f"Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}"
                        )

                        if self.enable_stats:
                            self.current_call_stats = {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": total_tokens,
                                "duration": end_time - start_time,
                                "timestamp": time.time(),
                            }
                            self.accumulated_stats["prompt_tokens"] += prompt_tokens
                            self.accumulated_stats["completion_tokens"] += (
                                completion_tokens
                            )
                            self.accumulated_stats["total_tokens"] += total_tokens
                            self.accumulated_stats["call_count"] += 1
                            self.accumulated_stats["total_duration"] += (
                                end_time - start_time
                            )

                        return response_data["choices"][0]["message"]["content"]

            except aiohttp.ClientError as e:
                error_time = time.perf_counter()
                logger.error(
                    f"[Llama-{self.model}] aiohttp.ClientError after {error_time - start_time:.2f}s: {e}"
                )
                if retry_num == max_retries - 1:
                    raise LLMError(f"Request failed: {str(e)}")
            except Exception as e:
                error_time = time.perf_counter()
                logger.error(
                    f"[Llama-{self.model}] Exception after {error_time - start_time:.2f}s: {e}"
                )
                if retry_num == max_retries - 1:
                    raise LLMError(f"Request failed: {str(e)}")

    async def test_connection(self) -> bool:
        try:
            logger.info(f"[INFO] [Llama-{self.model}] Testing API connection...")
            test_response = await self.generate("Hello", temperature=0.1)
            success = len(test_response) > 0
            if success:
                logger.info(
                    f"[SUCCESS] [Llama-{self.model}] API connection test successful"
                )
            else:
                logger.error(
                    f"[ERROR] [Llama-{self.model}] API connection test failed: Empty response"
                )
            return success
        except Exception as e:
            logger.error(
                f"[ERROR] [Llama-{self.model}] API connection test failed: {e}"
            )
            return False

    def get_current_call_stats(self) -> Optional[dict]:
        if self.enable_stats:
            return self.current_call_stats
        return None

    def get_accumulated_stats(self) -> Optional[dict]:
        if self.enable_stats:
            return self.accumulated_stats.copy()
        return None

    def reset_accumulated_stats(self) -> None:
        if self.enable_stats:
            self.accumulated_stats = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "call_count": 0,
                "total_duration": 0.0,
            }

    def __repr__(self) -> str:
        return f"OpenAIProvider(model={self.model}, base_url={self.base_url})"