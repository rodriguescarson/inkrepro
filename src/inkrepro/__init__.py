"""Measure how much an ink prediction depends on which checkpoint produced it."""

from .metrics import Agreement, agreement, align, to_unit, valid_mask

__all__ = ["Agreement", "agreement", "align", "to_unit", "valid_mask"]
__version__ = "0.1.0"
