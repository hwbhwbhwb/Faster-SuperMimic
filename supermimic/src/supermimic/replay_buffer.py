from dataclasses import dataclass

import torch
from tensordict import TensorDict


@dataclass(frozen=True)
class FieldSpec:
  dim: int = 0  # 0 = scalar (no trailing axis); >0 = vector with trailing dim
  dtype: torch.dtype = torch.float32

  def shape(self, num_envs: int, buffer_size: int) -> tuple[int, ...]:
    return (num_envs, buffer_size, self.dim) if self.dim else (num_envs, buffer_size)


class ReplayBuffer:
  """Per-env circular buffer over a TensorDict, with env-stratified window sampling."""

  def __init__(
    self,
    schema: dict[str, FieldSpec],
    buffer_size: int,
    num_envs: int,
    device: str = "cpu",
  ):
    if not {"terminated", "truncated"} <= schema.keys():
      raise ValueError("schema needs 'terminated' and 'truncated'")
    self._version = 0
    # Per window length: (version, csum, total, env weights) of the valid window
    # starts. The buffer only changes through insert() or by assigning _step /
    # num_transitions (a restore), and both bump the version.
    self._valid_cache: dict[int, tuple] = {}
    self.num_transitions = 0
    self.buffer_size = buffer_size
    self.num_envs = num_envs
    self.device = device
    self._step = 0
    self._storage = TensorDict(
      {
        name: torch.zeros(
          spec.shape(num_envs, buffer_size), dtype=spec.dtype, device=device
        )
        for name, spec in schema.items()
      },
      batch_size=[num_envs, buffer_size],
    )

  @property
  def _step(self) -> int:
    return self.__step

  @_step.setter
  def _step(self, value: int) -> None:
    self.__step = value
    self._version += 1

  @property
  def num_transitions(self) -> int:
    return self.__num_transitions

  @num_transitions.setter
  def num_transitions(self, value: int) -> None:
    self.__num_transitions = value
    self._version += 1

  def insert(self, td: TensorDict) -> None:
    if td.batch_size != torch.Size([self.num_envs]):
      raise ValueError(
        f"insert expects batch_size=[{self.num_envs}], got {list(td.batch_size)}"
      )
    if set(td.keys()) != set(self._storage.keys()):
      raise ValueError("schema mismatch")
    self._storage[:, self._step] = td
    self._step = (self._step + 1) % self.buffer_size
    self.num_transitions = min(self.buffer_size, self.num_transitions + 1)

  def sample_causal(
    self, sequence_length: int, batch_size: int
  ) -> tuple[TensorDict, torch.Tensor]:
    """Sample a window of ``L`` frames ENDING at a uniformly-drawn filled frame.

    The newest frame (index ``-1``) is the sampled "current" state — drawn like
    a 1-step ``sample``, so its distribution is independent of ``L``. Preceding
    slots hold the current frame's real in-episode predecessors; slots before
    the episode start (a stored ``terminated``/``truncated``) or before the
    oldest filled frame are clamped to the first in-episode frame, matching the
    real-sim history ring's reset padding.

    Returns ``(window, padded)``: the ``(B, L)`` window and a ``(B, L)`` bool
    mask that is ``True`` on the clamp-padded slots, so the caller can noise the
    oldest real frame once and share that realization across them.
    """
    L, B, E, T = sequence_length, batch_size, self.num_envs, self.buffer_size
    dev, n = self.device, self.num_transitions
    if L > T:
      raise ValueError(f"sequence_length {L} exceeds buffer_size {T}")
    if n < 1:
      raise RuntimeError("sample_causal on empty buffer")
    base = self._step if n == T else 0
    env_idx = torch.randint(E, (B,), device=dev)
    current_age = torch.randint(n, (B,), device=dev)
    ages = current_age[:, None] + (
      torch.arange(L, device=dev) - (L - 1)
    )  # (B, L), newest last
    ended = (self._storage["terminated"] | self._storage["truncated"])[
      env_idx[:, None], (base + ages.clamp(min=0)) % T
    ]
    # Reachable = contiguous in-episode suffix ending at the (always-valid) newest
    # frame; a stored done at an earlier slot ends the prior episode there.
    valid = (ages >= 0) & ~ended.bool()
    valid[:, -1] = True
    reachable = torch.cummin(valid.flip(-1).int(), dim=-1).values.flip(-1).bool()
    first_age = torch.where(reachable, ages, current_age[:, None]).amin(
      dim=-1, keepdim=True
    )
    slot = (base + torch.where(reachable, ages, first_age)) % T
    return self._storage[env_idx[:, None], slot], ~reachable

  def sample(self, sequence_length: int, batch_size: int) -> TensorDict:
    L, B, E, T = sequence_length, batch_size, self.num_envs, self.buffer_size
    dev, n = self.device, self.num_transitions
    if L > T:
      raise ValueError(f"sequence_length {L} exceeds buffer_size {T}")
    # Index by age (0 = oldest filled, n-1 = newest); slot = (base + age) % T.
    # In age-space windows are linear, so wrap and warm-up collapse into one predicate.
    base = self._step if n == T else 0
    cached = self._valid_cache.get(L)
    if cached is not None and cached[0] == self._version:
      _, csum, total, weights = cached
    else:
      ages = torch.arange(T, device=dev)
      if L == 1:
        # A one-frame window has no interior: every filled frame is a start.
        if n < 1:
          raise RuntimeError(f"no valid sequences L={L} n={n}")
        valid = (ages < n).expand(E, T)
      else:
        ended = (self._storage["terminated"] | self._storage["truncated"])[
          :, (base + ages) % T
        ].long()
        # Interior [a, a+L-2] is done-free iff c[a+L-1] == c[a] (c is cumsum with leading 0).
        c = torch.cat(
          [ended.new_zeros(E, 1), ended.cumsum(1), ended.new_zeros(E, L - 1)], dim=1
        )
        valid = (c[:, L - 1 : L - 1 + T] == c[:, :T]) & (ages + L - 1 < n)
      csum = valid.cumsum(dim=1, dtype=torch.long)
      total = csum[:, -1]
      if L != 1 and not (total > 0).any():
        raise RuntimeError(f"no valid sequences L={L} n={n}")
      weights = (total > 0).float()
      self._valid_cache[L] = (self._version, csum, total, weights)
    env_idx = torch.multinomial(weights, B, replacement=True)
    u = (torch.rand(B, device=dev) * total[env_idx].float()).long()
    age_start = torch.searchsorted(csum[env_idx], u[:, None], right=True).squeeze(-1)
    storage_idx = (base + age_start[:, None] + torch.arange(L, device=dev)) % T
    return self._storage[env_idx[:, None], storage_idx]
