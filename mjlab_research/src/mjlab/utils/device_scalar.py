"""Cached device scalars for writes through tensor indices."""

from __future__ import annotations

import functools

import torch


@functools.cache
def _cached(value: bool | int | float, dtype: torch.dtype, device: torch.device):
  return torch.tensor(value, dtype=dtype, device=device)


def device_scalar(value: bool | int | float, like: torch.Tensor) -> torch.Tensor:
  """``value`` as a 0-dim tensor on ``like``'s device and dtype.

  ``t[idx] = 0.0`` with tensor indices copies the Python scalar host->device
  from pageable memory, which synchronizes the stream; assigning a cached
  device scalar writes the same values without the copy.
  """
  return _cached(value, like.dtype, like.device)


@functools.cache
def device_tensor(values: tuple, device: torch.device | str) -> torch.Tensor:
  """A cached device tensor of constant ``values`` (never written to)."""
  return torch.tensor(values, device=device)
