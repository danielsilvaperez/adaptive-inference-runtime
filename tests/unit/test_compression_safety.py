"""Unit tests for compression safety guards and quality monitoring."""

from air.compression.safety import (
    CompressionQualityMonitor,
    CompressionSafetyGuard,
    CompressionSafetyManager,
    CompressionUseCase,
    QualityMonitorConfig,
)
from air.state import InferenceState
from air.types import CompressionConfig, Token


def _build_state_with_logprobs(logprobs: list[float]) -> InferenceState:
    state = InferenceState(model_id="test-model")
    for idx, logprob in enumerate(logprobs):
        state.add_token(Token(id=idx, text=str(idx), logprob=logprob))
    return state


def test_safety_guard_blocks_code_generation() -> None:
    state = InferenceState(model_id="test-model")
    state.set_metadata("use_case", CompressionUseCase.CODE_GENERATION.value)
    guard = CompressionSafetyGuard()

    assert guard.is_compression_allowed(state) is False


def test_safety_guard_allows_override() -> None:
    state = InferenceState(model_id="test-model")
    state.set_metadata("use_case", CompressionUseCase.CODE_GENERATION.value)
    state.set_metadata("force_compression", True)
    guard = CompressionSafetyGuard()

    assert guard.is_compression_allowed(state) is True


def test_safety_guard_honors_explicit_disable() -> None:
    state = InferenceState(model_id="test-model")
    state.set_metadata("use_case", CompressionUseCase.GENERAL.value)
    state.set_metadata("disable_compression", True)
    guard = CompressionSafetyGuard()

    assert guard.is_compression_allowed(state) is False


def test_quality_monitor_sets_baseline_without_changes() -> None:
    state = _build_state_with_logprobs([-1.0])
    config = CompressionConfig(target_ratio=0.5)
    monitor = CompressionQualityMonitor()

    decision = monitor.assess(state, config)

    assert decision.enabled is True
    assert decision.target_ratio == config.target_ratio
    assert state.metadata["compression_baseline_avg_logprob"] == -1.0


def test_quality_monitor_relaxes_compression_on_drop() -> None:
    state = _build_state_with_logprobs([-1.0, -4.0])
    config = CompressionConfig(target_ratio=0.5)
    monitor = CompressionQualityMonitor(
        QualityMonitorConfig(
            min_avg_logprob=-2.0,
            min_logprob=-6.0,
            max_avg_logprob_drop=0.5,
            target_ratio_step=0.2,
        )
    )

    decision = monitor.assess(state, config)

    assert decision.enabled is True
    assert decision.target_ratio == 0.7
    assert decision.reason == "quality_drop_relax_compression"
    assert "avg_logprob_below_min" in decision.violations


def test_quality_monitor_disables_when_fully_relaxed() -> None:
    state = _build_state_with_logprobs([-1.0, -6.0])
    config = CompressionConfig(target_ratio=0.95)
    monitor = CompressionQualityMonitor(
        QualityMonitorConfig(
            min_avg_logprob=-2.0,
            min_logprob=-5.0,
            max_avg_logprob_drop=0.5,
            target_ratio_step=0.1,
            disable_on_violation=True,
        )
    )

    decision = monitor.assess(state, config)

    assert decision.enabled is False
    assert decision.reason == "quality_drop_disable_compression"


def test_safety_manager_combines_guard_and_monitor() -> None:
    state = _build_state_with_logprobs([-1.0, -4.0])
    state.set_metadata("use_case", CompressionUseCase.RETRIEVAL_QA.value)
    config = CompressionConfig(target_ratio=0.5)
    manager = CompressionSafetyManager()

    decision = manager.evaluate(state, config)

    assert decision.enabled is False
    assert decision.reason == "use_case_guard"
