"""
Unit tests for HuggingFace adapter implementation.

Tests the HuggingFaceAdapter class with mocked models to avoid
requiring actual model downloads.
"""

from unittest.mock import Mock, patch

import pytest
import torch

from air.adapters.huggingface import HuggingFaceAdapter
from air.types import GenerationConfig, Token

# Test constants
HIGH_LOGIT_VALUE = 10.0  # Value to set for logits to ensure argmax selection


class TestHuggingFaceAdapterInit:
    """Tests for HuggingFaceAdapter initialization."""

    def test_initialization_default(self):
        """Test adapter initialization with default parameters."""
        adapter = HuggingFaceAdapter(model_id="test-model")
        assert adapter.model_id == "test-model"
        assert adapter._device == "cpu"
        assert adapter._torch_dtype == "auto"
        assert not adapter._load_in_8bit
        assert not adapter._load_in_4bit
        assert not adapter.is_loaded

    def test_initialization_custom(self):
        """Test adapter initialization with custom parameters."""
        adapter = HuggingFaceAdapter(
            model_id="test-model",
            device="cuda:0",
            torch_dtype="float16",
            load_in_8bit=True,
        )
        assert adapter._device == "cuda:0"
        assert adapter._torch_dtype == "float16"
        assert adapter._load_in_8bit
        assert not adapter.is_loaded


class TestHuggingFaceAdapterLoading:
    """Tests for model loading and unloading."""

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_load_model_default(self, mock_tokenizer, mock_model):
        """Test loading model with default settings."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model_instance.device = "cpu"
        mock_model_instance.eval.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        # Test loading
        adapter = HuggingFaceAdapter(model_id="test-model")
        adapter.load()

        assert adapter.is_loaded
        assert adapter._model is not None
        assert adapter._tokenizer is not None
        mock_tokenizer.from_pretrained.assert_called_once_with("test-model")
        mock_model.from_pretrained.assert_called_once()

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_load_model_with_path(self, mock_tokenizer, mock_model):
        """Test loading model from custom path."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model_instance.device = "cpu"
        mock_model_instance.eval.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        # Test loading with custom path
        adapter = HuggingFaceAdapter(model_id="test-model")
        adapter.load(model_path="/path/to/model")

        mock_tokenizer.from_pretrained.assert_called_once_with("/path/to/model")

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_unload_model(self, mock_tokenizer, mock_model):
        """Test unloading model."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model_instance.device = "cpu"
        mock_model_instance.eval.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        # Load and unload
        adapter = HuggingFaceAdapter(model_id="test-model")
        adapter.load()
        assert adapter.is_loaded

        adapter.unload()
        assert not adapter.is_loaded
        assert adapter._model is None
        assert adapter._tokenizer is None

    def test_ensure_loaded_raises_when_not_loaded(self):
        """Test that methods raise error when model not loaded."""
        adapter = HuggingFaceAdapter(model_id="test-model")
        config = GenerationConfig()

        with pytest.raises(RuntimeError, match="not loaded"):
            list(adapter.generate("test", config))

        with pytest.raises(RuntimeError, match="not loaded"):
            adapter.get_logits([1])

        with pytest.raises(RuntimeError, match="not loaded"):
            adapter.verify([])


class TestHuggingFaceAdapterGenerate:
    """Tests for token generation."""

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_generate_basic(self, mock_tokenizer, mock_model):
        """Test basic token generation."""
        # Setup tokenizer mock
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer_instance.pad_token = "<eos>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Mock tokenization
        mock_input_ids = torch.tensor([[1, 2, 3]])
        mock_tokenizer_instance.return_value = {"input_ids": mock_input_ids}

        def mock_decode(token_ids, **kwargs):
            return f"token_{token_ids[0]}"

        mock_tokenizer_instance.decode = mock_decode

        # Setup model mock
        mock_model_instance = Mock()
        mock_model_instance.device = "cpu"
        mock_model_instance.eval.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance

        # Mock generation output
        generated_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])  # Include input + 3 new tokens
        scores = [
            torch.randn(1, 100),  # Logits for token 4
            torch.randn(1, 100),  # Logits for token 5
            torch.randn(1, 100),  # Logits for token 6
        ]
        mock_output = Mock()
        mock_output.sequences = generated_ids
        mock_output.scores = scores

        mock_model_instance.generate.return_value = mock_output
        mock_model.from_pretrained.return_value = mock_model_instance

        # Test generation
        adapter = HuggingFaceAdapter(model_id="test-model")
        adapter.load()

        config = GenerationConfig(max_tokens=3, temperature=1.0)
        tokens = list(adapter.generate("Hello", config))

        assert len(tokens) == 3  # 3 new tokens
        assert all(isinstance(t, Token) for t in tokens)
        assert all(
            hasattr(t, "id") and hasattr(t, "text") and hasattr(t, "logprob") for t in tokens
        )

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_generate_with_temperature_zero(self, mock_tokenizer, mock_model):
        """Test generation with temperature=0 (greedy)."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer_instance.pad_token = "<eos>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_input_ids = torch.tensor([[1, 2, 3]])
        mock_tokenizer_instance.return_value = {"input_ids": mock_input_ids}
        mock_tokenizer_instance.decode = lambda ids, **kw: f"token_{ids[0]}"

        mock_model_instance = Mock()
        mock_model_instance.device = "cpu"
        mock_model_instance.eval.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance

        generated_ids = torch.tensor([[1, 2, 3, 4]])
        mock_output = Mock()
        mock_output.sequences = generated_ids
        mock_output.scores = [torch.randn(1, 100)]

        mock_model_instance.generate.return_value = mock_output
        mock_model.from_pretrained.return_value = mock_model_instance

        # Test with temperature=0
        adapter = HuggingFaceAdapter(model_id="test-model")
        adapter.load()

        config = GenerationConfig(max_tokens=1, temperature=0.0)
        list(adapter.generate("Hello", config))

        # Verify generation was called with do_sample=False
        call_kwargs = mock_model_instance.generate.call_args[1]
        assert call_kwargs["do_sample"] is False


class TestHuggingFaceAdapterLogits:
    """Tests for logits extraction."""

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_get_logits(self, mock_tokenizer, mock_model):
        """Test logits extraction."""
        # Setup tokenizer mock
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer_instance.pad_token = "<eos>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Setup model mock
        mock_model_instance = Mock()
        mock_model_instance.device = "cpu"
        mock_model_instance.eval.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance

        # Mock forward pass
        vocab_size = 1000
        mock_logits = torch.randn(1, 3, vocab_size)  # [batch, seq_len, vocab]
        mock_output = Mock()
        mock_output.logits = mock_logits
        mock_model_instance.return_value = mock_output

        mock_model.from_pretrained.return_value = mock_model_instance

        # Test logits extraction
        adapter = HuggingFaceAdapter(model_id="test-model")
        adapter.load()

        logits = adapter.get_logits([1, 2, 3])

        assert logits.shape == (vocab_size,)  # Should return last position only
        assert torch.is_tensor(logits)


class TestHuggingFaceAdapterVerifyTokens:
    """Tests for speculative decoding token verification."""

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_verify_tokens_all_accepted(self, mock_tokenizer, mock_model):
        """Test verification when all draft tokens are accepted."""
        # Setup tokenizer mock
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer_instance.pad_token = "<eos>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_input_ids = torch.tensor([[1, 2, 3]])
        mock_tokenizer_instance.return_value = {"input_ids": mock_input_ids}
        mock_tokenizer_instance.decode = lambda ids, **kw: f"token_{ids[0]}"

        # Setup model mock
        mock_model_instance = Mock()
        mock_model_instance.device = "cpu"
        mock_model_instance.eval.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance

        # Create logits that predict the draft tokens
        # Draft tokens: 4, 5, 6
        draft_tokens = [
            Token(id=4, text="token_4", logprob=-0.1),
            Token(id=5, text="token_5", logprob=-0.2),
            Token(id=6, text="token_6", logprob=-0.3),
        ]

        vocab_size = 100
        # Create logits where argmax matches draft tokens
        mock_logits = torch.randn(6, vocab_size)  # 3 prompt + 3 draft
        mock_logits[2, 4] = HIGH_LOGIT_VALUE  # Position 2 predicts token 4
        mock_logits[3, 5] = HIGH_LOGIT_VALUE  # Position 3 predicts token 5
        mock_logits[4, 6] = HIGH_LOGIT_VALUE  # Position 4 predicts token 6

        mock_output = Mock()
        mock_output.logits = mock_logits.unsqueeze(0)
        mock_model_instance.return_value = mock_output

        mock_model.from_pretrained.return_value = mock_model_instance

        # Test verification
        adapter = HuggingFaceAdapter(model_id="test-model")
        adapter.load()

        adapter._last_prompt_tokens = mock_input_ids[0].tolist()
        accepted, count = adapter.verify(draft_tokens)

        assert count == 3  # All tokens accepted
        assert len(accepted) == 3
        assert all(isinstance(t, Token) for t in accepted)

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_verify_tokens_partial_acceptance(self, mock_tokenizer, mock_model):
        """Test verification when some draft tokens are rejected."""
        # Setup tokenizer mock
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer_instance.pad_token = "<eos>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_input_ids = torch.tensor([[1, 2, 3]])
        mock_tokenizer_instance.return_value = {"input_ids": mock_input_ids}
        mock_tokenizer_instance.decode = lambda ids, **kw: f"token_{ids[0]}"

        # Setup model mock
        mock_model_instance = Mock()
        mock_model_instance.device = "cpu"
        mock_model_instance.eval.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance

        # Draft tokens: 4, 5, 6
        draft_tokens = [
            Token(id=4, text="token_4", logprob=-0.1),
            Token(id=5, text="token_5", logprob=-0.2),
            Token(id=6, text="token_6", logprob=-0.3),
        ]

        vocab_size = 100
        # Create logits where first token matches but second doesn't
        mock_logits = torch.randn(6, vocab_size)
        mock_logits[2, 4] = HIGH_LOGIT_VALUE  # Position 2 predicts token 4 (match)
        mock_logits[3, 7] = HIGH_LOGIT_VALUE  # Position 3 predicts token 7 (no match, should be 5)

        mock_output = Mock()
        mock_output.logits = mock_logits.unsqueeze(0)
        mock_model_instance.return_value = mock_output

        mock_model.from_pretrained.return_value = mock_model_instance

        # Test verification
        adapter = HuggingFaceAdapter(model_id="test-model")
        adapter.load()

        adapter._last_prompt_tokens = mock_input_ids[0].tolist()
        accepted, count = adapter.verify(draft_tokens)

        assert count == 1  # Only first token accepted
        assert len(accepted) == 2  # First accepted + correction token
        assert accepted[0].id == 4  # First draft token accepted
        assert accepted[1].id == 7  # Correction token

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_verify_tokens_empty_draft(self, mock_tokenizer, mock_model):
        """Test verification with empty draft tokens."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer_instance.pad_token = "<eos>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model_instance.device = "cpu"
        mock_model_instance.eval.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        # Test with empty draft
        adapter = HuggingFaceAdapter(model_id="test-model")
        adapter.load()

        mock_input_ids = torch.tensor([[1, 2, 3]])
        adapter._last_prompt_tokens = mock_input_ids[0].tolist()
        accepted, count = adapter.verify([])

        assert count == 0
        assert len(accepted) == 0
