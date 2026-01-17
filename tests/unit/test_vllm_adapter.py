"""
Unit tests for vLLM adapter implementation.

Tests the VLLMAdapter class with mocked vLLM components to avoid
requiring actual model downloads or vLLM installation.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
import torch

from air.adapters.vllm import VLLMAdapter
from air.types import GenerationConfig, Token


class FakeSamplingParams:
    """Capture sampling params in a version-compatible signature."""

    def __init__(
        self,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        repetition_penalty: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
        logprobs: int | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.stop = stop
        self.seed = seed
        self.logprobs = logprobs


class FakeCompletion:
    """Fake vLLM completion output."""

    def __init__(self, token_ids: list[int], logprobs: list[dict[int, float]]) -> None:
        self.token_ids = token_ids
        self.logprobs = logprobs
        self.text = "".join(f"<{token_id}>" for token_id in token_ids)


class FakeRequestOutput:
    """Fake vLLM request output."""

    def __init__(self, outputs: list[FakeCompletion]) -> None:
        self.outputs = outputs


class FakeTokenizer:
    """Minimal tokenizer for decoding."""

    def __init__(self, vocab_size: int = 5) -> None:
        self.vocab_size = vocab_size

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        _ = skip_special_tokens
        return f"<{token_ids[0]}>"

    def __len__(self) -> int:
        return self.vocab_size


class FakeLLM:
    """Minimal vLLM engine for adapter tests."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.outputs: list[FakeRequestOutput] = []
        self.last_prompt: str | None = None
        self.last_sampling_params: Any | None = None
        self._tokenizer = FakeTokenizer()
        self.llm_engine = types.SimpleNamespace(
            model_config=types.SimpleNamespace(max_model_len=256)
        )

    def generate(self, prompt: str, sampling_params: Any) -> list[FakeRequestOutput]:
        self.last_prompt = prompt
        self.last_sampling_params = sampling_params
        return self.outputs

    def get_tokenizer(self) -> FakeTokenizer:
        return self._tokenizer


def _install_fake_vllm() -> types.SimpleNamespace:
    return types.SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams)


class TestVLLMAdapterInit:
    """Tests for VLLMAdapter initialization."""

    def test_initialization_default(self) -> None:
        adapter = VLLMAdapter(model_id="test-model")
        assert adapter.model_id == "test-model"
        assert adapter._tensor_parallel_size == 1
        assert adapter._gpu_memory_utilization == 0.9
        assert adapter._max_model_len is None
        assert adapter._dtype == "auto"
        assert not adapter.is_loaded


class TestVLLMAdapterLoading:
    """Tests for model loading and unloading."""

    def test_load_model_default(self) -> None:
        adapter = VLLMAdapter(model_id="test-model")
        fake_module = _install_fake_vllm()

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setitem(sys.modules, "vllm", fake_module)
            adapter.load()

        assert adapter.is_loaded
        assert adapter._llm is not None
        assert adapter._llm.kwargs["model"] == "test-model"
        assert adapter.vocab_size == 5
        assert adapter.context_length == 256

    def test_load_model_with_path_override(self) -> None:
        adapter = VLLMAdapter(model_id="test-model")
        fake_module = _install_fake_vllm()

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setitem(sys.modules, "vllm", fake_module)
            adapter.load(model_path="/path/to/model")

        assert adapter._llm is not None
        assert adapter._llm.kwargs["model"] == "/path/to/model"

    def test_unload_model(self) -> None:
        adapter = VLLMAdapter(model_id="test-model")
        fake_module = _install_fake_vllm()

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setitem(sys.modules, "vllm", fake_module)
            adapter.load()
            adapter.unload()

        assert not adapter.is_loaded
        assert adapter._llm is None


class TestVLLMAdapterGenerate:
    """Tests for token generation."""

    def test_generate_basic(self) -> None:
        adapter = VLLMAdapter(model_id="test-model")
        fake_module = _install_fake_vllm()

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setitem(sys.modules, "vllm", fake_module)
            adapter.load()

        assert adapter._llm is not None
        completion = FakeCompletion(
            token_ids=[1, 2],
            logprobs=[{1: -0.1}, {2: -0.2}],
        )
        adapter._llm.outputs = [FakeRequestOutput(outputs=[completion])]

        config = GenerationConfig(max_tokens=2, temperature=0.7)
        tokens = list(adapter.generate("Hello", config))

        assert len(tokens) == 2
        assert tokens[0].text == "<1>"
        assert tokens[0].logprob == -0.1
        assert tokens[1].text == "<2>"
        assert tokens[1].logprob == -0.2


class TestVLLMAdapterLogits:
    """Tests for logits extraction and verification."""

    def test_get_logits(self) -> None:
        adapter = VLLMAdapter(model_id="test-model", logprobs=3)
        fake_module = _install_fake_vllm()

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setitem(sys.modules, "vllm", fake_module)
            adapter.load()

        assert adapter._llm is not None
        completion = FakeCompletion(
            token_ids=[1],
            logprobs=[{1: -0.1, 4: -0.3}],
        )
        adapter._llm.outputs = [FakeRequestOutput(outputs=[completion])]

        logits = adapter.get_logits("hello")
        assert logits.shape == torch.Size([5])
        assert torch.isclose(logits[1], torch.tensor(-0.1))
        assert torch.isclose(logits[4], torch.tensor(-0.3))
        assert torch.isneginf(logits[0])

    def test_verify_tokens(self) -> None:
        adapter = VLLMAdapter(model_id="test-model", logprobs=2)
        fake_module = _install_fake_vllm()

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setitem(sys.modules, "vllm", fake_module)
            adapter.load()

        assert adapter._llm is not None
        completion = FakeCompletion(
            token_ids=[1, 2],
            logprobs=[{1: -0.1}, {2: -0.2}],
        )
        adapter._llm.outputs = [FakeRequestOutput(outputs=[completion])]

        draft_tokens = [
            Token(id=1, text="<1>", logprob=-0.1),
            Token(id=3, text="<3>", logprob=-1.0),
        ]

        accepted, accepted_count = adapter.verify_tokens("prompt", draft_tokens)
        assert accepted_count == 1
        assert len(accepted) == 2
        assert accepted[1].id == 2
        assert accepted[1].logprob == -0.2
