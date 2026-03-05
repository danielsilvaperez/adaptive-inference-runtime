"""Unit tests for DraftModel draft generation."""

from __future__ import annotations

import pytest

from air.speculation.draft import DraftModel
from air.types import GenerationConfig, Token


class FakeAdapter:
    """Simple adapter stub for draft generation tests."""

    def __init__(self, tokens: list[Token], is_loaded: bool = True) -> None:
        self._tokens = tokens
        self._is_loaded = is_loaded
        self.model_id = "fake-model"
        self.last_config: GenerationConfig | None = None

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def generate(self, prompt: str, config: GenerationConfig):  # type: ignore[override]
        self.last_config = config
        max_tokens = config.max_tokens if config.max_tokens != -1 else len(self._tokens)
        yield from self._tokens[:max_tokens]


def test_generate_draft_respects_default_length() -> None:
    tokens = [
        Token(id=1, text="a", logprob=-0.1),
        Token(id=2, text="b", logprob=-0.2),
        Token(id=3, text="c", logprob=-0.3),
        Token(id=4, text="d", logprob=-0.4),
    ]
    adapter = FakeAdapter(tokens)
    model = DraftModel(adapter, default_draft_tokens=3, adaptive_draft=False)
    result = model.generate_draft("hello", GenerationConfig(max_tokens=10))

    assert len(result.tokens) == 3
    assert result.max_draft_tokens == 3
    assert adapter.last_config is not None
    assert adapter.last_config.max_tokens == 3


def test_generate_draft_clamps_to_config_max_tokens() -> None:
    tokens = [
        Token(id=1, text="a", logprob=-0.1),
        Token(id=2, text="b", logprob=-0.2),
        Token(id=3, text="c", logprob=-0.3),
    ]
    adapter = FakeAdapter(tokens)
    model = DraftModel(adapter, default_draft_tokens=5, adaptive_draft=False)
    config = GenerationConfig(max_tokens=2)
    result = model.generate_draft("hello", config)

    assert len(result.tokens) == 2
    assert result.max_draft_tokens == 2


def test_generate_draft_adaptive_stop() -> None:
    tokens = [
        Token(id=1, text="a", logprob=-5.0),
        Token(id=2, text="b", logprob=-5.0),
        Token(id=3, text="c", logprob=-5.0),
    ]
    adapter = FakeAdapter(tokens)
    model = DraftModel(
        adapter,
        default_draft_tokens=5,
        min_draft_tokens=2,
        logprob_stop_threshold=-1.0,
        logprob_window=2,
        adaptive_draft=True,
    )
    result = model.generate_draft("hello", GenerationConfig(max_tokens=10))

    assert len(result.tokens) == 2
    assert result.stopped_early is True


def test_generate_draft_requires_loaded_adapter() -> None:
    adapter = FakeAdapter([], is_loaded=False)
    model = DraftModel(adapter)

    with pytest.raises(RuntimeError, match="not loaded"):
        model.generate_draft("hello", GenerationConfig())


def test_generate_draft_requires_prompt() -> None:
    adapter = FakeAdapter([Token(id=1, text="a", logprob=-0.1)])
    model = DraftModel(adapter)

    with pytest.raises(ValueError, match="prompt must be non-empty"):
        model.generate_draft("", GenerationConfig())


def test_generate_draft_rejects_explicit_zero_max_draft_tokens() -> None:
    adapter = FakeAdapter([Token(id=1, text="a", logprob=-0.1)])
    model = DraftModel(adapter)

    with pytest.raises(ValueError, match="max_draft_tokens must be positive"):
        model.generate_draft("hello", GenerationConfig(), max_draft_tokens=0)
