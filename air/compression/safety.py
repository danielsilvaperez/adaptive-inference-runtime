"""
Safety guards and quality monitoring for KV cache compression.

This module provides lightweight, configurable mechanisms for:
- Disabling compression for sensitive use cases (e.g., code generation, retrieval QA)
- Monitoring token-level quality signals and adjusting compression aggressiveness
- Falling back to no compression when quality degradation is detected
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

from air.state import InferenceState
from air.types import CompressionConfig


class CompressionUseCase(str, Enum):
    """Supported use-case labels for compression safety decisions."""

    GENERAL = "general"
    CODE_GENERATION = "code_generation"
    RETRIEVAL_QA = "retrieval_qa"


@dataclass(frozen=True)
class SafetyGuardConfig:
    """
    Configuration for compression safety guards.

    Attributes:
        disabled_use_cases: Use cases where compression should be disabled.
        metadata_use_case_key: Metadata key for use case labeling.
        metadata_disable_key: Metadata key for explicitly disabling compression.
        metadata_force_key: Metadata key for forcing compression.
    """

    disabled_use_cases: set[CompressionUseCase] = field(
        default_factory=lambda: {
            CompressionUseCase.CODE_GENERATION,
            CompressionUseCase.RETRIEVAL_QA,
        }
    )
    metadata_use_case_key: str = "use_case"
    metadata_disable_key: str = "disable_compression"
    metadata_force_key: str = "force_compression"

    def is_use_case_disabled(self, use_case: CompressionUseCase) -> bool:
        """Return True if compression is disabled for the given use case."""
        return use_case in self.disabled_use_cases


class CompressionSafetyGuard:
    """
    Evaluate whether compression should be enabled for a given inference state.

    The guard inspects the inference state's metadata for explicit disable/force
    flags and use-case labels.
    """

    def __init__(self, config: SafetyGuardConfig | None = None) -> None:
        self._config = config or SafetyGuardConfig()

    @property
    def config(self) -> SafetyGuardConfig:
        """Return the guard configuration."""
        return self._config

    def is_compression_allowed(self, state: InferenceState) -> bool:
        """
        Determine if compression is allowed for the given state.

        Priority:
        1. Explicit disable flag -> disallow
        2. Explicit force flag -> allow
        3. Use-case check -> allow if not disabled
        """
        metadata = state.metadata
        if metadata.get(self._config.metadata_disable_key, False):
            return False
        if metadata.get(self._config.metadata_force_key, False):
            return True

        use_case = metadata.get(self._config.metadata_use_case_key, CompressionUseCase.GENERAL)
        if isinstance(use_case, str):
            use_case = self._parse_use_case(use_case)

        if isinstance(use_case, CompressionUseCase):
            return not self._config.is_use_case_disabled(use_case)

        return True

    @staticmethod
    def _parse_use_case(value: str) -> CompressionUseCase:
        """Parse string metadata into a CompressionUseCase enum."""
        normalized = value.strip().lower()
        for use_case in CompressionUseCase:
            if use_case.value == normalized:
                return use_case
        return CompressionUseCase.GENERAL


@dataclass(frozen=True)
class QualityMonitorConfig:
    """
    Configuration for compression quality monitoring.

    Attributes:
        min_avg_logprob: Minimum acceptable average logprob.
        min_logprob: Minimum acceptable per-token logprob.
        max_avg_logprob_drop: Maximum acceptable drop from baseline avg logprob.
        target_ratio_step: Step size to relax compression when quality drops.
        disable_on_violation: Whether to disable compression on severe drops.
        metadata_baseline_key: Metadata key to store baseline avg logprob.
    """

    min_avg_logprob: float = -2.5
    min_logprob: float = -8.0
    max_avg_logprob_drop: float = 0.7
    target_ratio_step: float = 0.1
    disable_on_violation: bool = True
    metadata_baseline_key: str = "compression_baseline_avg_logprob"

    def relaxed_target_ratio(self, current_ratio: float) -> float:
        """Return a more conservative target ratio."""
        return min(1.0, current_ratio + self.target_ratio_step)


@dataclass(frozen=True)
class CompressionDecision:
    """Decision output from safety/quality evaluation."""

    enabled: bool
    target_ratio: float
    reason: str | None = None
    violations: tuple[str, ...] = ()


class CompressionQualityMonitor:
    """
    Monitor quality signals and adjust compression aggressiveness.

    Uses InferenceState token stats as a lightweight proxy for quality.
    """

    def __init__(self, config: QualityMonitorConfig | None = None) -> None:
        self._config = config or QualityMonitorConfig()

    @property
    def config(self) -> QualityMonitorConfig:
        """Return the monitor configuration."""
        return self._config

    def assess(self, state: InferenceState, config: CompressionConfig) -> CompressionDecision:
        """
        Assess quality signals and produce a compression decision.

        Returns the original config if not enough data is available.
        """
        if not config.enabled:
            return CompressionDecision(enabled=False, target_ratio=config.target_ratio)

        stats = state.stats
        if stats.total_tokens == 0:
            return CompressionDecision(enabled=config.enabled, target_ratio=config.target_ratio)

        baseline = self._get_or_set_baseline(state, stats.avg_logprob)
        windowed_min = self._compute_windowed_min(state)
        violations = self._evaluate_violations(stats.avg_logprob, windowed_min, baseline)

        if not violations:
            return CompressionDecision(enabled=config.enabled, target_ratio=config.target_ratio)

        relaxed_ratio = self._config.relaxed_target_ratio(config.target_ratio)
        if relaxed_ratio >= 1.0 and self._config.disable_on_violation:
            return CompressionDecision(
                enabled=False,
                target_ratio=config.target_ratio,
                reason="quality_drop_disable_compression",
                violations=violations,
            )

        return CompressionDecision(
            enabled=config.enabled,
            target_ratio=relaxed_ratio,
            reason="quality_drop_relax_compression",
            violations=violations,
        )

    def _get_or_set_baseline(self, state: InferenceState, avg_logprob: float) -> float:
        metadata = state.metadata
        key = self._config.metadata_baseline_key
        if key not in metadata:
            metadata[key] = avg_logprob
        return float(metadata[key])

    def _compute_windowed_min(self, state: InferenceState) -> float:
        """
        Compute minimum logprob from recent window instead of global minimum.

        Returns float('inf') when no recent tokens are available, which ensures
        no violation is triggered for the minimum logprob check.
        """
        recent = state.recent_logprobs
        if not recent:
            return float("inf")
        return min(recent)

    def _evaluate_violations(
        self, avg_logprob: float, min_logprob: float, baseline: float
    ) -> tuple[str, ...]:
        violations: list[str] = []
        if avg_logprob < self._config.min_avg_logprob:
            violations.append("avg_logprob_below_min")
        if min_logprob < self._config.min_logprob:
            violations.append("min_logprob_below_min")
        if (baseline - avg_logprob) > self._config.max_avg_logprob_drop:
            violations.append("avg_logprob_drop_exceeded")
        return tuple(violations)


class CompressionSafetyManager:
    """
    Combine safety guard and quality monitor into a single decision flow.
    """

    def __init__(
        self,
        guard: CompressionSafetyGuard | None = None,
        monitor: CompressionQualityMonitor | None = None,
    ) -> None:
        self._guard = guard or CompressionSafetyGuard()
        self._monitor = monitor or CompressionQualityMonitor()

    @property
    def guard(self) -> CompressionSafetyGuard:
        """Return the active safety guard."""
        return self._guard

    @property
    def monitor(self) -> CompressionQualityMonitor:
        """Return the active quality monitor."""
        return self._monitor

    def evaluate(self, state: InferenceState, config: CompressionConfig) -> CompressionDecision:
        """
        Evaluate whether compression should be applied and at what intensity.
        """
        if not self._guard.is_compression_allowed(state):
            return CompressionDecision(
                enabled=False,
                target_ratio=config.target_ratio,
                reason="use_case_guard",
            )
        return self._monitor.assess(state, config)


def allowed_use_cases(config: SafetyGuardConfig) -> Iterable[str]:
    """Return allowed use-case strings for external documentation tooling."""
    return [
        use_case.value
        for use_case in CompressionUseCase
        if use_case not in config.disabled_use_cases
    ]
