"""
vLLM adapter for AIR.

This module provides an adapter for using vLLM models with the Adaptive
Inference Runtime. vLLM is optimized for high-throughput serving and
is particularly useful for production deployments.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch

from air.adapters.base import ModelAdapter
from air.types import GenerationConfig, Token

if TYPE_CHECKING:
    from air.types import KVCache, Logits


class VLLMAdapter(ModelAdapter):
    """
    Adapter for vLLM inference engine.

    This adapter wraps vLLM's API to provide a unified interface compatible
    with AIR's routing and speculation systems. vLLM is designed for
    high-throughput serving with features like PagedAttention.

    Example:
        >>> adapter = VLLMAdapter(
        ...     model_id="meta-llama/Llama-2-7b-hf",
        ...     tensor_parallel_size=1,
        ...     gpu_memory_utilization=0.9
        ... )
        >>> config = GenerationConfig(max_tokens=100)
        >>> for token in adapter.generate("Explain quantum computing", config):
        ...     print(token.text, end="")
    """

    def __init__(
        self,
        model_id: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        dtype: str = "auto",
        trust_remote_code: bool = False,
        logprobs: int = 1,
    ) -> None:
        """
        Initialize the vLLM adapter.

        Args:
            model_id: Model identifier (HuggingFace model name).
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory to use.
            max_model_len: Maximum sequence length for the model.
            dtype: Data type for model weights ("auto", "float16", "bfloat16").
            trust_remote_code: Allow execution of remote code from model repo.
            logprobs: Number of logprobs to request (for token-level scoring).
        """
        super().__init__(model_id)
        self._tensor_parallel_size = tensor_parallel_size
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._dtype = dtype
        self._trust_remote_code = trust_remote_code
        self._logprobs = logprobs

        # Lazy load vLLM engine when first used
        self._llm: Any | None = None
        self._tokenizer: Any | None = None
        self._sampling_params_cls: Any | None = None
        self._is_loaded = False
        self._vocab_size = 0
        self._context_length = 0
        self._last_prompt_tokens: list[int] | None = None

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size if available."""
        return self._vocab_size

    @property
    def context_length(self) -> int:
        """Return context length if available."""
        return self._context_length

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if not self._is_loaded or self._llm is None:
            raise RuntimeError(
                f"Model '{self._model_id}' is not loaded. "
                "Set model_path or call load() method first."
            )

    def load(self, model_path: str | None = None) -> None:
        """
        Load the vLLM model.

        Args:
            model_path: Optional path to load model from. If None, uses model_id.
        """
        if self._is_loaded:
            return

        model_name = model_path or self._model_id
        try:
            from vllm import LLM, SamplingParams
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "vLLM is required for VLLMAdapter. Install with `pip install vllm`."
            ) from exc

        llm_kwargs: dict[str, Any] = {
            "model": model_name,
            "tensor_parallel_size": self._tensor_parallel_size,
            "gpu_memory_utilization": self._gpu_memory_utilization,
            "dtype": self._dtype,
            "trust_remote_code": self._trust_remote_code,
        }
        if self._max_model_len is not None:
            llm_kwargs["max_model_len"] = self._max_model_len

        self._llm = LLM(**llm_kwargs)
        self._sampling_params_cls = SamplingParams
        self._tokenizer = self._resolve_tokenizer()
        self._vocab_size = self._resolve_vocab_size()
        self._context_length = self._resolve_context_length()
        self._is_loaded = True

    def load_model(self, path: str) -> None:
        """Compatibility wrapper for interface load_model."""
        self.load(model_path=path)

    def unload(self) -> None:
        """Unload the model to free resources."""
        self._llm = None
        self._tokenizer = None
        self._sampling_params_cls = None
        self._is_loaded = False
        self._vocab_size = 0
        self._context_length = 0
        self._last_prompt_tokens = None

    def unload_model(self) -> None:
        """Compatibility wrapper for interface unload_model."""
        self.unload()

    def generate(self, prompt: str, config: GenerationConfig) -> Iterator[Token]:
        """
        Generate tokens from the vLLM model.

        Args:
            prompt: Input text prompt.
            config: Generation configuration.

        Yields:
            Generated tokens with logprobs.
        """
        self._ensure_loaded()
        if not prompt:
            raise ValueError("prompt must be non-empty")
        assert self._llm is not None

        self._last_prompt_tokens = self.tokenize(prompt)
        sampling_params = self._build_sampling_params(
            config=config,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            logprobs=self._logprobs,
        )
        outputs = self._llm.generate(prompt, sampling_params)
        completion = self._first_completion(outputs)
        if completion is None:
            return iter(())
        return self._yield_tokens(completion)

    def get_logits(self, tokens: list[int]) -> Logits:
        """
        Get logits for next token prediction.

        Args:
            tokens: Token IDs for the prompt context.

        Returns:
            Logits tensor for next token.
        """
        self._ensure_loaded()
        if not tokens:
            raise ValueError("tokens must be non-empty")
        assert self._llm is not None
        prompt = self.detokenize(tokens)
        if not prompt:
            raise ValueError("tokens must decode to a non-empty prompt")

        if self._vocab_size <= 0:
            raise RuntimeError("Unable to determine vocab size for vLLM adapter.")

        self._last_prompt_tokens = list(tokens)
        sampling_params = self._build_sampling_params(
            config=GenerationConfig(max_tokens=1, temperature=0.0),
            max_tokens=1,
            temperature=0.0,
            logprobs=self._logprobs,
        )
        outputs = self._llm.generate(prompt, sampling_params)
        completion = self._first_completion(outputs)
        if completion is None:
            raise RuntimeError("vLLM returned no output for logits request.")

        logprobs = self._extract_logprobs(completion)
        if not logprobs:
            raise RuntimeError("vLLM did not return logprobs; increase logprobs parameter.")

        logits = torch.full((self._vocab_size,), float("-inf"), dtype=torch.float32)
        token_logprobs = logprobs[0] if logprobs else {}
        for token_id, value in token_logprobs.items():
            logits[int(token_id)] = float(value)
        return logits

    def verify(self, draft_tokens: list[Token]) -> tuple[list[Token], int]:
        """
        Verify draft tokens for speculative decoding.

        Args:
            draft_tokens: Draft tokens to verify.

        Returns:
            Accepted tokens and count.
        """
        self._ensure_loaded()
        assert self._llm is not None

        if not draft_tokens:
            return [], 0

        if self._last_prompt_tokens is None:
            raise RuntimeError(
                "No prompt context available for verification. "
                "Call generate() or get_logits() first."
            )

        prompt = self.detokenize(self._last_prompt_tokens)
        if not prompt:
            raise RuntimeError("Unable to reconstruct prompt for verification.")

        sampling_params = self._build_sampling_params(
            config=GenerationConfig(max_tokens=len(draft_tokens), temperature=0.0),
            max_tokens=len(draft_tokens),
            temperature=0.0,
            logprobs=self._logprobs,
        )
        outputs = self._llm.generate(prompt, sampling_params)
        completion = self._first_completion(outputs)
        if completion is None:
            raise RuntimeError("vLLM returned no output for verification.")

        token_ids = list(getattr(completion, "token_ids", []) or [])
        token_logprobs = self._extract_logprobs(completion)

        accepted_tokens: list[Token] = []
        accepted_count = 0
        for index, draft_token in enumerate(draft_tokens):
            if index >= len(token_ids):
                break
            predicted_id = token_ids[index]
            predicted_logprob = self._resolve_logprob(token_logprobs, index, predicted_id)
            predicted_text = self._decode_token(predicted_id)

            if predicted_id == draft_token.id:
                accepted_tokens.append(draft_token)
                accepted_count += 1
                continue

            accepted_tokens.append(
                Token(
                    id=predicted_id,
                    text=predicted_text,
                    logprob=predicted_logprob,
                )
            )
            break

        return accepted_tokens, accepted_count

    def verify_tokens(self, prompt: str, draft_tokens: list[Token]) -> tuple[list[Token], int]:
        """
        Backwards-compatible verification helper that accepts a prompt string.
        """
        if not prompt:
            raise ValueError("prompt must be non-empty")
        self._last_prompt_tokens = self.tokenize(prompt)
        return self.verify(draft_tokens)

    def get_kv_cache(self) -> KVCache:
        """
        Return KV cache state for the model.

        Raises:
            RuntimeError: vLLM KV cache access is not yet supported.
        """
        self._ensure_loaded()
        raise RuntimeError("vLLM KV cache access is not yet supported.")

    def set_kv_cache(self, cache: KVCache) -> None:
        """
        Restore KV cache state for the model.

        Raises:
            RuntimeError: vLLM KV cache access is not yet supported.
        """
        _ = cache
        self._ensure_loaded()
        raise RuntimeError("vLLM KV cache access is not yet supported.")

    def tokenize(self, text: str) -> list[int]:
        """
        Tokenize text into token IDs.
        """
        self._ensure_loaded()
        tokenizer = self._tokenizer
        if tokenizer is None:
            raise RuntimeError("vLLM tokenizer is unavailable.")
        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            return list(encode(text))
        if callable(tokenizer):
            encoded = tokenizer(text)
            if isinstance(encoded, dict) and "input_ids" in encoded:
                return list(encoded["input_ids"])
        raise RuntimeError("Unable to tokenize prompt with vLLM tokenizer.")

    def detokenize(self, tokens: list[int]) -> str:
        """
        Convert token IDs to text.
        """
        self._ensure_loaded()
        if not tokens:
            return ""
        tokenizer = self._tokenizer
        if tokenizer is None:
            raise RuntimeError("vLLM tokenizer is unavailable.")
        decode = getattr(tokenizer, "decode", None)
        if callable(decode):
            return decode([int(token) for token in tokens], skip_special_tokens=False)
        raise RuntimeError("Unable to detokenize tokens with vLLM tokenizer.")

    def _build_sampling_params(
        self,
        config: GenerationConfig,
        max_tokens: int,
        temperature: float,
        logprobs: int,
    ) -> Any:
        """Build SamplingParams with compatibility filtering."""
        max_tokens = max_tokens if max_tokens > 0 else 512
        top_k = config.top_k if config.top_k > 0 else -1
        top_p = config.top_p if config.top_p < 1.0 else 1.0
        min_p = config.min_p if config.min_p > 0.0 else 0.0

        params_data = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "min_p": min_p,
            "repetition_penalty": config.repetition_penalty,
            "presence_penalty": config.presence_penalty,
            "frequency_penalty": config.frequency_penalty,
            "stop": config.stop_sequences or None,
            "seed": config.seed,
            "logprobs": logprobs,
        }

        sampling_params_cls = self._sampling_params_cls
        if sampling_params_cls is None:
            raise RuntimeError("vLLM sampling parameters are unavailable; call load() first.")

        allowed = set(inspect.signature(sampling_params_cls).parameters)
        filtered = {key: value for key, value in params_data.items() if key in allowed}
        return sampling_params_cls(**filtered)

    def _first_completion(self, outputs: list[Any]) -> Any | None:
        """Return the first completion output from vLLM."""
        if not outputs:
            return None
        output = outputs[0]
        completions = getattr(output, "outputs", None) or []
        return completions[0] if completions else None

    def _yield_tokens(self, completion: Any) -> Iterator[Token]:
        """Yield Token objects from a completion."""
        token_ids = list(getattr(completion, "token_ids", []) or [])
        token_logprobs = self._extract_logprobs(completion)

        for index, token_id in enumerate(token_ids):
            logprob = self._resolve_logprob(token_logprobs, index, token_id)
            text = self._decode_token(token_id)
            yield Token(id=token_id, text=text, logprob=logprob)

    def _extract_logprobs(self, completion: Any) -> list[dict[int, float]]:
        """Normalize logprob payloads into a list of token->logprob mappings."""
        logprobs = getattr(completion, "logprobs", None)
        if not logprobs:
            return []

        normalized: list[dict[int, float]] = []
        for entry in logprobs:
            if entry is None:
                normalized.append({})
                continue
            if isinstance(entry, dict):
                normalized.append(self._normalize_logprob_entry(entry))
                continue
            if hasattr(entry, "logprob"):
                token_id = getattr(entry, "token_id", None)
                if token_id is None:
                    normalized.append({})
                else:
                    normalized.append({int(token_id): float(entry.logprob)})
                continue
            normalized.append({})
        return normalized

    def _normalize_logprob_entry(self, entry: dict[Any, Any]) -> dict[int, float]:
        """Normalize a single logprob entry mapping."""
        normalized: dict[int, float] = {}
        for token_id, value in entry.items():
            if value is None:
                continue
            if hasattr(value, "logprob"):
                normalized[int(token_id)] = float(value.logprob)
            else:
                normalized[int(token_id)] = float(value)
        return normalized

    def _resolve_logprob(
        self, logprobs: list[dict[int, float]], index: int, token_id: int
    ) -> float:
        """Resolve logprob value for a token at a given index."""
        if index >= len(logprobs):
            return 0.0
        return float(logprobs[index].get(int(token_id), 0.0))

    def _decode_token(self, token_id: int) -> str:
        """Decode a token id into text using the available tokenizer."""
        tokenizer = self._tokenizer
        if tokenizer is None:
            return f"<tok_{token_id}>"
        decode = getattr(tokenizer, "decode", None)
        if callable(decode):
            return decode([int(token_id)], skip_special_tokens=False)
        return f"<tok_{token_id}>"

    def _resolve_tokenizer(self) -> Any | None:
        """Resolve tokenizer from vLLM engine."""
        if self._llm is None:
            return None
        get_tokenizer = getattr(self._llm, "get_tokenizer", None)
        if callable(get_tokenizer):
            return get_tokenizer()
        return getattr(self._llm, "tokenizer", None)

    def _resolve_vocab_size(self) -> int:
        """Resolve vocabulary size from tokenizer or model config."""
        if self._tokenizer is not None:
            vocab_size = getattr(self._tokenizer, "vocab_size", None)
            if vocab_size is None:
                try:
                    vocab_size = len(self._tokenizer)
                except TypeError:
                    vocab_size = None
            if vocab_size is not None:
                return int(vocab_size)
        return 0

    def _resolve_context_length(self) -> int:
        """Resolve context length from model config if available."""
        if self._llm is None:
            return 0
        engine = getattr(self._llm, "llm_engine", None)
        model_config = getattr(engine, "model_config", None) if engine is not None else None
        max_model_len = getattr(model_config, "max_model_len", None)
        if max_model_len is not None:
            return int(max_model_len)
        return 0


__all__ = ["VLLMAdapter"]
