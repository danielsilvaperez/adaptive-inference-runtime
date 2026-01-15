"""
AIR Speculative Decoding Module

Implements speculative decoding for faster inference by using a smaller draft
model to generate candidate tokens that are then verified by the target model.

Key Components:
    - DraftModel: Interface for draft model implementations
    - SpeculativeDecoder: Main speculative decoding engine
    - TokenVerifier: Verifies draft tokens against target model
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from air.speculation.decoder import SpeculativeDecoder
    from air.speculation.draft import DraftModel
    from air.speculation.verifier import TokenVerifier

__all__ = [
    "SpeculativeDecoder",
    "DraftModel",
    "TokenVerifier",
]


def __getattr__(name: str):
    """Lazy import mechanism for speculation components."""
    if name == "SpeculativeDecoder":
        from air.speculation import decoder
        return decoder.SpeculativeDecoder
    elif name == "DraftModel":
        from air.speculation import draft
        return draft.DraftModel
    elif name == "TokenVerifier":
        from air.speculation import verifier
        return verifier.TokenVerifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
