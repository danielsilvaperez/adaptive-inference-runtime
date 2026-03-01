"""
HuggingFace Transformers adapter for AIR.

This module provides an adapter for using HuggingFace Transformers models
with the Adaptive Inference Runtime. It wraps the HF Transformers API to
provide the standard AIR ModelAdapter interface.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch

from air.adapters.base import ModelAdapter
from air.types import Token

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from air.types import GenerationConfig, KVCache, Logits


class HuggingFaceAdapter(ModelAdapter):
    """
    Adapter for HuggingFace Transformers models.

    This adapter wraps HuggingFace's transformers library to provide
    a unified interface compatible with AIR's routing and speculation
    systems.

    Example:
        >>> adapter = HuggingFaceAdapter(
        ...     model_id="meta-llama/Llama-2-7b-hf",
        ...     device="cuda:0"
        ... )
        >>> config = GenerationConfig(max_tokens=100, temperature=0.7)
        >>> for token in adapter.generate("Hello, world!", config):
        ...     print(token.text, end="")
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        torch_dtype: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> None:
        """
        Initialize the HuggingFace adapter.

        Args:
            model_id: HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b-hf").
            device: Device to load the model on ("cpu", "cuda", "cuda:0", etc.).
            torch_dtype: PyTorch dtype for model weights ("auto", "float16", "bfloat16").
            load_in_8bit: Whether to load model in 8-bit quantization.
            load_in_4bit: Whether to load model in 4-bit quantization.
        """
        super().__init__(model_id)
        self._device = device
        self._torch_dtype = torch_dtype
        self._load_in_8bit = load_in_8bit
        self._load_in_4bit = load_in_4bit

        # Lazy load model and tokenizer when first used
        self._model: AutoModelForCausalLM | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._is_loaded = False
        self._vocab_size = 0
        self._context_length = 0
        self._last_prompt_tokens: list[int] | None = None

    def _ensure_loaded(self) -> None:
        """Ensure model and tokenizer are loaded."""
        if not self._is_loaded or self._model is None or self._tokenizer is None:
            raise RuntimeError(
                f"Model '{self._model_id}' is not loaded. "
                "Set model_path or call load() method first."
            )

    def load(self, model_path: str | None = None) -> None:
        """
        Load the model and tokenizer.

        Args:
            model_path: Optional path to load model from. If None, uses model_id as HF identifier.
        """
        if self._is_loaded:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = model_path or self._model_id

        # Parse torch_dtype
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self._torch_dtype, "auto")

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model with appropriate settings
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self._device if self._device != "cpu" else None,
            load_in_8bit=self._load_in_8bit,
            load_in_4bit=self._load_in_4bit,
        )

        if self._device == "cpu":
            self._model = self._model.to(self._device)

        self._model.eval()
        self._is_loaded = True
        self._vocab_size = self._resolve_vocab_size()
        self._context_length = self._resolve_context_length()
        self._last_prompt_tokens = None

    def load_model(self, path: str) -> None:
        """Compatibility wrapper for interface load_model."""
        self.load(model_path=path)

    def unload(self) -> None:
        """Unload the model and tokenizer to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._is_loaded = False
        self._vocab_size = 0
        self._context_length = 0
        self._last_prompt_tokens = None
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_model(self) -> None:
        """Compatibility wrapper for interface unload_model."""
        self.unload()

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

    def generate(self, prompt: str, config: GenerationConfig) -> Iterator[Token]:
        """
        Generate tokens from the HuggingFace model.

        Args:
            prompt: Input text prompt.
            config: Generation configuration.

        Yields:
            Generated tokens with logprobs.
        """
        self._ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None

        # Tokenize input
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._model.device)
        self._last_prompt_tokens = input_ids[0].tolist()

        # Prepare generation kwargs
        max_tokens = config.max_tokens if config.max_tokens > 0 else 512
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": config.temperature > 0,
            "temperature": config.temperature if config.temperature > 0 else 1.0,
            "top_k": config.top_k if config.top_k > 0 else None,
            "top_p": config.top_p if config.top_p < 1.0 else None,
            "repetition_penalty": config.repetition_penalty,
            "return_dict_in_generate": True,
            "output_scores": True,
        }

        # Generate tokens
        with torch.no_grad():
            outputs = self._model.generate(input_ids, **gen_kwargs)

        # Extract generated token ids (excluding input)
        generated_ids = outputs.sequences[0][input_ids.shape[1] :]
        scores = outputs.scores

        # Convert to Token objects
        for i, token_id in enumerate(generated_ids):
            token_id_item = token_id.item()
            token_text = self._tokenizer.decode([token_id_item], skip_special_tokens=False)

            # Calculate logprob from scores
            if i < len(scores):
                token_logits = scores[i][0]  # [vocab_size]
                log_probs = torch.nn.functional.log_softmax(token_logits, dim=-1)
                token_logprob = log_probs[token_id_item].item()
            else:
                token_logprob = 0.0

            yield Token(id=token_id_item, text=token_text, logprob=token_logprob)

    def get_logits(self, tokens: list[int]) -> Logits:
        """
        Get logits for next token prediction.

        Args:
            tokens: Token IDs for the prompt context.

        Returns:
            Logits tensor for next token.
        """
        self._ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None
        if not tokens:
            raise ValueError("tokens must be non-empty")

        input_ids = torch.tensor([tokens], device=self._model.device)
        self._last_prompt_tokens = list(tokens)

        # Get logits
        with torch.no_grad():
            outputs = self._model(input_ids)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]

        # Return logits for the last token position
        return logits[0, -1, :]  # [vocab_size]

    def verify(self, draft_tokens: list[Token]) -> tuple[list[Token], int]:
        """
        Verify draft tokens for speculative decoding.

        Args:
            draft_tokens: Draft tokens to verify.

        Returns:
            Accepted tokens and count.
        """
        self._ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None

        if not draft_tokens:
            return [], 0

        if self._last_prompt_tokens is None:
            raise RuntimeError(
                "No prompt context available for verification. "
                "Call generate() or get_logits() first."
            )

        # Build sequence with draft tokens
        draft_ids = torch.tensor([t.id for t in draft_tokens], device=self._model.device)
        prompt_ids = torch.tensor(self._last_prompt_tokens, device=self._model.device)
        full_ids = torch.cat([prompt_ids, draft_ids])

        # Get model predictions for the entire sequence
        with torch.no_grad():
            outputs = self._model(full_ids.unsqueeze(0))
            logits = outputs.logits[0]  # [seq_len, vocab_size]

        # Verify each draft token
        accepted_tokens = []
        accepted_count = 0

        prompt_len = prompt_ids.shape[0]
        for i, draft_token in enumerate(draft_tokens):
            # Get logits that predict the current draft token position
            # Logits at position (prompt_len + i - 1) predict token at position (prompt_len + i)
            position_logits = logits[prompt_len + i - 1]
            predicted_token_id = torch.argmax(position_logits).item()

            if predicted_token_id == draft_token.id:
                # Token matches, accept it
                accepted_tokens.append(draft_token)
                accepted_count += 1
            else:
                # Token doesn't match, reject and generate correction
                log_probs = torch.nn.functional.log_softmax(position_logits, dim=-1)
                corrected_logprob = log_probs[predicted_token_id].item()
                corrected_text = self._tokenizer.decode(
                    [predicted_token_id], skip_special_tokens=False
                )
                corrected_token = Token(
                    id=predicted_token_id, text=corrected_text, logprob=corrected_logprob
                )
                accepted_tokens.append(corrected_token)
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
            RuntimeError: HuggingFace KV cache access is not yet supported.
        """
        self._ensure_loaded()
        raise RuntimeError("HuggingFace KV cache access is not yet supported.")

    def set_kv_cache(self, cache: KVCache) -> None:
        """
        Restore KV cache state for the model.

        Raises:
            RuntimeError: HuggingFace KV cache access is not yet supported.
        """
        _ = cache
        self._ensure_loaded()
        raise RuntimeError("HuggingFace KV cache access is not yet supported.")

    def tokenize(self, text: str) -> list[int]:
        """
        Tokenize text into token IDs.
        """
        self._ensure_loaded()
        assert self._tokenizer is not None
        encoded = self._tokenizer(text, return_tensors="pt")
        input_ids = encoded.get("input_ids")
        if input_ids is None:
            raise RuntimeError("Tokenizer did not return input_ids.")
        return input_ids[0].tolist()

    def detokenize(self, tokens: list[int]) -> str:
        """
        Convert token IDs to text.
        """
        self._ensure_loaded()
        assert self._tokenizer is not None
        if not tokens:
            return ""
        return self._tokenizer.decode(tokens, skip_special_tokens=False)

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
        if self._model is not None:
            vocab_size = getattr(self._model.config, "vocab_size", None)
            if vocab_size is not None:
                return int(vocab_size)
        return 0

    def _resolve_context_length(self) -> int:
        """Resolve context length from model config if available."""
        if self._model is None:
            return 0
        config = getattr(self._model, "config", None)
        for attr in ("max_position_embeddings", "max_seq_len", "n_positions"):
            value = getattr(config, attr, None)
            if value is not None:
                return int(value)
        return 0


__all__ = ["HuggingFaceAdapter"]
