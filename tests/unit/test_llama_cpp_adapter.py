"""
Unit tests for llama.cpp adapter implementation.

Tests the LlamaCppAdapter class with mocked llama.cpp models to avoid
requiring actual model files or llama-cpp-python installations.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
import torch

from air.adapters.llama_cpp import LlamaCppAdapter
from air.types import GenerationConfig, Token


class FakeLlama:
    """Minimal fake llama.cpp model for adapter tests."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.scores: list[list[float]] = []
        self.stream_chunks: list[dict[str, Any]] = []
        self.last_call: dict[str, Any] = {}
        self.last_eval_tokens: list[int] = []
        self.reset_called = False

    def __call__(self, prompt: str, **kwargs: Any):
        self.last_call = kwargs
        return iter(self.stream_chunks)

    def eval(self, tokens: list[int]) -> None:
        self.last_eval_tokens = tokens

    def reset(self) -> None:
        self.reset_called = True

    def tokenize(self, data: bytes, add_bos: bool = False) -> list[int]:
        _ = add_bos
        if data == b"hello":
            return [1]
        if data == b" world":
            return [2]
        return [3, 4]

    def detokenize(self, tokens: list[int]) -> bytes:
        return f"<{tokens[0]}>".encode()

    def n_vocab(self) -> int:
        return 42

    def n_ctx(self) -> int:
        return 128


def _install_fake_llama() -> types.SimpleNamespace:
    return types.SimpleNamespace(Llama=FakeLlama)


class TestLlamaCppAdapterInit:
    """Tests for LlamaCppAdapter initialization."""

    def test_initialization_default(self) -> None:
        adapter = LlamaCppAdapter(model_id="test-model")
        assert adapter.model_id == "test-model"
        assert adapter._model_path is None
        assert adapter._n_ctx == 2048
        assert adapter._n_gpu_layers == 0
        assert not adapter.is_loaded


class TestLlamaCppAdapterLoading:
    """Tests for model loading and unloading."""

    def test_load_model_default(self) -> None:
        adapter = LlamaCppAdapter(model_id="test-model", model_path="model.gguf")

        with pytest.raises(RuntimeError):
            list(adapter.generate("test", GenerationConfig(max_tokens=1)))

        with pytest.raises(RuntimeError):
            adapter.get_logits("test")

        with pytest.raises(RuntimeError):
            adapter.verify_tokens("test", [])

        fake_module = _install_fake_llama()
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
            adapter.load()

        assert adapter.is_loaded
        assert adapter._llama is not None
        assert adapter.vocab_size == 42
        assert adapter.context_length == 128
        assert adapter._llama.kwargs["model_path"] == "model.gguf"

    def test_load_model_with_path_override(self) -> None:
        adapter = LlamaCppAdapter(model_id="test-model", model_path="model.gguf")

        fake_module = _install_fake_llama()
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
            adapter.load(model_path="override.gguf")

        assert adapter._llama is not None
        assert adapter._llama.kwargs["model_path"] == "override.gguf"

    def test_unload_model(self) -> None:
        adapter = LlamaCppAdapter(model_id="test-model", model_path="model.gguf")
        fake_module = _install_fake_llama()
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
            adapter.load()
            assert adapter.is_loaded
            adapter.unload()

        assert not adapter.is_loaded
        assert adapter._llama is None


class TestLlamaCppAdapterGenerate:
    """Tests for token generation."""

    def test_generate_basic(self) -> None:
        adapter = LlamaCppAdapter(model_id="test-model", model_path="model.gguf")
        fake_module = _install_fake_llama()

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
            adapter.load()

        assert adapter._llama is not None
        adapter._llama.stream_chunks = [
            {
                "choices": [
                    {"text": "hello", "logprobs": {"tokens": ["hello"], "token_logprobs": [-0.1]}}
                ]
            },
            {
                "choices": [
                    {
                        "text": " world",
                        "logprobs": {"tokens": [" world"], "token_logprobs": [-0.2]},
                    }
                ]
            },
        ]

        config = GenerationConfig(max_tokens=2, temperature=0.7)
        tokens = list(adapter.generate("Hello", config))

        assert len(tokens) == 2
        assert all(isinstance(t, Token) for t in tokens)
        assert tokens[0].text == "hello"
        assert tokens[1].text == " world"


class TestLlamaCppAdapterLogits:
    """Tests for logits extraction and verification."""

    def test_get_logits(self) -> None:
        adapter = LlamaCppAdapter(model_id="test-model", model_path="model.gguf", logits_all=True)
        fake_module = _install_fake_llama()

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
            adapter.load()

        assert adapter._llama is not None
        adapter._llama.scores = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        logits = adapter.get_logits("hello")
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == torch.Size([3])
        assert torch.allclose(logits, torch.tensor([0.4, 0.5, 0.6]))

    def test_verify_tokens(self) -> None:
        adapter = LlamaCppAdapter(model_id="test-model", model_path="model.gguf", logits_all=True)
        fake_module = _install_fake_llama()

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
            adapter.load()

        assert adapter._llama is not None
        adapter._llama.scores = [
            [0.0, 0.0, 0.0],
            [0.1, 0.9, 0.0],  # Predict token id 1
            [0.2, 0.1, 0.7],  # Predict token id 2
        ]

        draft_tokens = [
            Token(id=1, text="hello", logprob=-0.1),
            Token(id=0, text="wrong", logprob=-2.0),
        ]

        accepted, accepted_count = adapter.verify_tokens("prompt", draft_tokens)
        assert accepted_count == 1
        assert len(accepted) == 2
        assert accepted[1].id == 2
        assert accepted[1].text == "<2>"
