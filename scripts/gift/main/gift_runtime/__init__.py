"""Isolated runtime overlay for the GIFT piecewise-ranking experiments."""

from .patch import apply_runtime_patches

__all__ = ["apply_runtime_patches"]
