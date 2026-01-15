"""Unit tests for TopKDisagreementScorer."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from air.routing.confidence import TopKDisagreementScorer  # noqa: E402


class TestTopKDisagreementScorer:
    """Tests for top-k disagreement confidence scoring."""

    def test_peaked_distribution_scores_higher_than_uniform(self) -> None:
        """Peaked distributions should yield higher confidence than uniform ones."""
        scorer = TopKDisagreementScorer(k=5)

        peaked_logits = torch.tensor([[10.0, 2.0, 1.0, 0.5, 0.1]])
        uniform_logits = torch.ones((1, 100))

        peaked_score = scorer.score(peaked_logits)
        uniform_score = scorer.score(uniform_logits)

        assert 0.0 <= peaked_score <= 1.0
        assert 0.0 <= uniform_score <= 1.0
        assert peaked_score > uniform_score

    def test_score_accepts_1d_logits(self) -> None:
        """1D logits should be accepted and return a valid score."""
        scorer = TopKDisagreementScorer(k=3)
        logits = torch.tensor([1.0, 2.0, 3.0])

        score = scorer.score(logits)

        assert 0.0 <= score <= 1.0

    def test_score_on_gpu_device(self) -> None:
        """Score should run on available GPU devices without device mismatch."""
        device = None
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")

        if device is None:
            pytest.skip("No GPU device available for top-k disagreement test.")

        scorer = TopKDisagreementScorer(k=4)
        logits = torch.randn(2, 32, device=device)

        score = scorer.score(logits)

        assert 0.0 <= score <= 1.0
