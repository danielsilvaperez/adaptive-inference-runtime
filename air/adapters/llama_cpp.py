"""
llama.cpp adapter for AIR.

This module provides an adapter for llama-cpp-python models, exposing the
AIR ModelAdapter interface for local inference on CPU/Metal/CUDA backends.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch

from air.adapters.base import ModelAdapter
from air.types import Token

if TYPE_CHECKING:
    from air.types import GenerationConfig, KVCache, Logits


class LlamaCppAdapter(ModelAdapter):
    """
    Adapter for llama-cpp-python inference engine.

    Example:
        >>> adapter = LlamaCppAdapter(model_id="llama-7b", model_path="model.gguf")
        >>> adapter.load()
        >>> config = GenerationConfig(max_tokens=64, temperature=0.7)
        >>> for token in adapter.generate("Hello", config):
        ...     print(token.text, end="")
    """

    def __init__(
        self,
        model_id: str,
        model_path: str | None = None,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        n_threads: int | None = None,
        n_batch: int = 512,
        use_mmap: bool = True,
        use_mlock: bool = False,
        seed: int | None = None,
        logits_all: bool = False,
    ) -> None:
        """
        Initialize the llama.cpp adapter.

        Args:
            model_id: Identifier for logging and routing (e.g., "llama-7b-q4").
            model_path: Path to the GGUF model file.
            n_ctx: Context length to allocate in tokens.
            n_gpu_layers: Number of layers to offload to GPU.
            n_threads: Number of CPU threads for inference (None for default).
            n_batch: Batch size for prompt evaluation.
            use_mmap: Enable memory-mapped model loading.
            use_mlock: Lock model in memory.
            seed: Random seed for reproducibility.
            logits_all: Keep logits for all tokens (required for get_logits/verify).
        """
        super().__init__(model_id)
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._n_threads = n_threads
        self._n_batch = n_batch
        self._use_mmap = use_mmap
        self._use_mlock = use_mlock
        self._seed = seed
        self._logits_all = logits_all

        self._llama: Any | None = None
        self._is_loaded = False
        self._vocab_size = 0
        self._context_length = 0
        self._last_prompt_tokens = None
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
        if not self._is_loaded or self._llama is None:
            raise RuntimeError(
                f"Model '{self._model_id}' is not loaded. "
                "Set model_path or call load() method first."
            )

    def load(self, model_path: str | None = None) -> None:
        """
        Load the llama.cpp model.

        Args:
            model_path: Optional path override for the GGUF model.

        Raises:
            ValueError: If no model path is provided.
            RuntimeError: If llama-cpp-python is not installed.
        """
        if self._is_loaded:
            return

        path = model_path or self._model_path
        if not path:
            raise ValueError("model_path must be provided to load llama.cpp model")

        try:
            from llama_cpp import Llama
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "llama-cpp-python is required for LlamaCppAdapter. "
                "Install with `pip install llama-cpp-python`."
            ) from exc

        kwargs: dict[str, Any] = {
            "model_path": path,
            "n_ctx": self._n_ctx,
            "n_gpu_layers": self._n_gpu_layers,
            "n_batch": self._n_batch,
            "use_mmap": self._use_mmap,
            "use_mlock": self._use_mlock,
        }
        if self._n_threads is not None:
            kwargs["n_threads"] = self._n_threads
        if self._seed is not None:
            kwargs["seed"] = self._seed
        if self._logits_all:
            kwargs["logits_all"] = True

        self._llama = Llama(**kwargs)
        self._is_loaded = True
        self._vocab_size = self._resolve_model_int("n_vocab", default=0)
        self._context_length = self._resolve_model_int("n_ctx", default=self._n_ctx)

    def load_model(self, path: str) -> None:
        """Compatibility wrapper for interface load_model."""
        self.load(model_path=path)

    def unload(self) -> None:
        """Unload the model to free resources."""
        if self._llama is not None:
            reset = getattr(self._llama, "reset", None)
            if callable(reset):
                reset()
            self._llama = None
        self._is_loaded = False
        self._vocab_size = 0
        self._context_length = 0

    def unload_model(self) -> None:
        """Compatibility wrapper for interface unload_model."""
        self.unload()

    def generate(self, prompt: str, config: GenerationConfig) -> Iterator[Token]:
        """
        Generate tokens from the llama.cpp model.

        Args:
            prompt: Input text prompt.
            config: Generation configuration.

        Yields:
            Generated tokens with logprobs.
        """
        self._ensure_loaded()
        if not prompt:
            raise ValueError("prompt must be non-empty")
        assert self._llama is not None

        self._last_prompt_tokens = self._tokenize(prompt)
        max_tokens = config.max_tokens if config.max_tokens > 0 else 512
        top_k = config.top_k if config.top_k > 0 else 0
        top_p = config.top_p if config.top_p < 1.0 else 1.0
        min_p = config.min_p if config.min_p > 0.0 else 0.0

        stream = self._llama(
            prompt,
            max_tokens=max_tokens,
            temperature=config.temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repeat_penalty=config.repetition_penalty,
            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty,
            stop=config.stop_sequences or None,
            seed=config.seed,
            stream=True,
            logprobs=1,
        )

        for chunk in stream:
            yield from self._tokens_from_chunk(chunk)

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
        assert self._llama is not None

        self._last_prompt_tokens = list(tokens)

        reset = getattr(self._llama, "reset", None)
        if callable(reset):
            reset()
        self._llama.eval(tokens)

        scores = self._get_scores()
        if scores is None:
            raise RuntimeError(
                "llama.cpp logits are unavailable. Initialize with logits_all=True."
            )
        last_scores = scores[-1]
        return torch.tensor(last_scores, dtype=torch.float32)

    def verify(self, draft_tokens: list[Token]) -> tuple[list[Token], int]:
        """
        Verify draft tokens for speculative decoding.

        Args:
            draft_tokens: Draft tokens to verify.

        Returns:
            Accepted tokens and count.
        """
        self._ensure_loaded()
        assert self._llama is not None

        if not draft_tokens:
            return [], 0

        if self._last_prompt_tokens is None:
            raise RuntimeError(
                "No prompt context available for verification. "
                "Call generate() or get_logits() first."
            )

        prompt_ids = self._last_prompt_tokens
        full_ids = prompt_ids + [token.id for token in draft_tokens]

        reset = getattr(self._llama, "reset", None)
        if callable(reset):
            reset()
        self._llama.eval(full_ids)

        scores = self._get_scores()
        if scores is None:
            raise RuntimeError(
                "llama.cpp logits are unavailable. Initialize with logits_all=True."
            )

        accepted_tokens: list[Token] = []
        accepted_count = 0
        prompt_len = len(prompt_ids)

        for i, draft_token in enumerate(draft_tokens):
            position = prompt_len + i - 1
            position_logits = torch.tensor(scores[position], dtype=torch.float32)
            predicted_token_id = torch.argmax(position_logits).item()

            if predicted_token_id == draft_token.id:
                accepted_tokens.append(draft_token)
                accepted_count += 1
                continue

            log_probs = torch.nn.functional.log_softmax(position_logits, dim=-1)
            corrected_logprob = log_probs[predicted_token_id].item()
            corrected_text = self._detokenize([predicted_token_id])
            accepted_tokens.append(
                Token(id=predicted_token_id, text=corrected_text, logprob=corrected_logprob)
            )
            break

        return accepted_tokens, accepted_count

    def verify_tokens(self, prompt: str, draft_tokens: list[Token]) -> tuple[list[Token], int]:
        """
        Backwards-compatible verification helper that accepts a prompt string.
        """
        if not prompt:
            raise ValueError("prompt must be non-empty")
        self._last_prompt_tokens = self._tokenize(prompt)
        return self.verify(draft_tokens)

    def get_kv_cache(self) -> KVCache:
        """
        Return KV cache state for the model.

        Raises:
            RuntimeError: llama.cpp KV cache access is not yet supported.
        """
        self._ensure_loaded()
        raise RuntimeError("llama.cpp KV cache access is not yet supported.")

    def set_kv_cache(self, cache: KVCache) -> None:
        """
        Restore KV cache state for the model.

        Raises:
            RuntimeError: llama.cpp KV cache access is not yet supported.
        """
        _ = cache
        self._ensure_loaded()
        raise RuntimeError("llama.cpp KV cache access is not yet supported.")

    def tokenize(self, text: str) -> list[int]:
        """
        Tokenize text into token IDs.
        """
        if not text:
            return []
        return self._tokenize(text)

    def detokenize(self, tokens: list[int]) -> str:
        """
        Convert token IDs to text.
        """
        if not tokens:
            return ""
        return self._detokenize(tokens)

    def _tokens_from_chunk(self, chunk: dict[str, Any]) -> list[Token]:
        """Convert a llama.cpp streaming chunk into Token objects."""
        choices = chunk.get("choices") or []
        if not choices:
            return []

        choice = choices[0] or {}
        logprobs = choice.get("logprobs") or {}
        token_texts = logprobs.get("tokens") or []
        token_logprobs = logprobs.get("token_logprobs") or []

        tokens: list[Token] = []
        if token_texts and len(token_texts) == len(token_logprobs):
            for token_text, token_logprob in zip(token_texts, token_logprobs):
                if not token_text:
                    continue
                token_id = self._token_id_from_text(token_text)
                tokens.append(
                    Token(
                        id=token_id,
                        text=token_text,
                        logprob=0.0 if token_logprob is None else float(token_logprob),
                    )
                )
            return tokens

        text = choice.get("text", "")
        if not text:
            return []
        token_id = self._token_id_from_text(text)
        logprob = token_logprobs[0] if token_logprobs else 0.0
        tokens.append(Token(id=token_id, text=text, logprob=float(logprob)))
        return tokens

    def _token_id_from_text(self, text: str) -> int:
        """Return a token id for a single token string."""
        token_ids = self._tokenize(text)
        return token_ids[0] if token_ids else -1

    def _tokenize(self, text: str) -> list[int]:
        """Tokenize text using llama.cpp."""
        self._ensure_loaded()
        assert self._llama is not None
        encoded = text.encode("utf-8")
        return list(self._llama.tokenize(encoded, add_bos=False))

    def _detokenize(self, token_ids: list[int]) -> str:
        """Detokenize token ids using llama.cpp."""
        self._ensure_loaded()
        assert self._llama is not None
        decoded = self._llama.detokenize(token_ids)
        return decoded.decode("utf-8", errors="ignore")

    def _resolve_model_int(self, attr: str, default: int) -> int:
        """Resolve model attribute or method returning an integer."""
        if self._llama is None:
            return default
        value = getattr(self._llama, attr, None)
        if callable(value):
            try:
                return int(value())
            except (TypeError, ValueError):
                return default
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _get_scores(self) -> list[list[float]] | None:
        """Return logits scores if exposed by llama.cpp."""
        if self._llama is None:
            return None
        scores = getattr(self._llama, "scores", None)
        if scores is None:
            scores = getattr(self._llama, "_scores", None)
        return scores


__all__ = ["LlamaCppAdapter"]
