# Changes

[← README](../README.md)

The agent worked in rounds: profile the current best tree with `bench.py profile` (torch.profiler, kernel-utilization
report), pick the hotspot, change it, time it with `bench.py time`. Tier N0 = values unchanged by construction; N1 =
floating-point-level differences (compiled/fused kernels). The full log, including the failed attempts, is
[`evidence/WORK_LOG.md`](../evidence/WORK_LOG.md).

## Baseline profile (unmodified tree)

10 profiled iterations: GPU busy 46%, 105k kernel launches, 1218 `cudaStreamSynchronize`. One iteration ≈ env step
34.5 ms (only 8.6 ms of GPU kernels: host/launch-bound) + 0.4 × one update of ~143 ms (world model 60 ms + policy
105 ms wall, both already CUDA-graph replays upstream). About 87 host syncs per iteration came from pageable
host-to-device copies of scalars, plus ~17 `.item()` calls.

## The nine commits

| Commit | Hotspot (from the profile) | Change | Tier | s/iteration |
|---|---|---|---|---|
| `edd13a6` Replay sampling | 12–20 ms of host work before the world-model graph: 10 × `ReplayBuffer.sample`, each recomputing a 4096 × 1000 done-gather/cumsum and a host sync; a `synchronize()` after every graph replay | Valid-window prefix sum cached per buffer version; no host sync after replay | N0 | 0.0815 (1.031×) |
| `30355d9` Critic fit | `GammaBetaBackward` (eager LayerNorm backward of the critic fit) at <5% of HBM bandwidth, 6 blocks, 40 calls per update | `torch.compile` of the critic minibatch loss → Inductor's fused LayerNorm backward | N1 | 0.0765 (1.098×) |
| `c7b817b` Env step | ~87 syncs/iteration; `set_joint_position_target` syncing every physics substep (5.2 ms/iteration), `torch.isin` in `motion_end` (1.6 ms) | Cached 0-dim device scalars for indexed writes (`mjlab/utils/device_scalar.py`), `searchsorted` membership on the sorted clip ends, episode stats kept on device | N0 | 0.0728 (1.153×) |
| `bc68c33` Combo kernels | Per BPTT step the compiled world-model step launches 96 kernels, ~75 of them tiny independent pointwise kernels | Inductor `combo_kernels=True` (horizontal fusion) for the compiled learner steps (`SUPERMIMIC_COMBO_KERNELS`, default 1) | N1 | 0.0682 (1.232×) |
| `6a7c23a` Collection | The simulator's `critic` and `goal` observation groups (~19% of the env step's host time) are computed every step and never read | `ObservationManager.set_skipped_groups`, set by the runner to all but `actor` and `full_state` | N0 | 0.0646 (1.300×) |
| `10131e6` Host/GPU trims | Misaligned TF32 GEMMs (dynamics input K = 130); per-access pose/velocity construction; normalizer flush after the physics wait; eager collection actor | Dynamics first-layer K padded 130 → 132 (4096×130×2048 forward 35.1 → 14.7 µs); component accessors; a pre-reset hook so host work overlaps the physics; compiled collection actor | N1 | 0.0637 (1.318×) |
| `17079f7` Env managers | `RewardManager.compute` 4.6 ms and `TerminationManager.compute` 1.1 ms per step (~10 small eager term functions) | `torch.compile` of both (`MJLAB_COMPILE_MANAGERS`, default on) | N1 | 0.0626 (1.341×) |
| `72e8b02` MotionCommand | `_update_metrics` 2.4 ms and `update_relative_body_poses` 1.4 ms per step | Both compiled (`MJLAB_COMPILE_COMMAND`, default on) | N1 | 0.0624 (1.347×) |
| `3b3b3c6` Normalizers | The pre-reset hook waited 22 ms (profiled) in `EmpiricalNormalization.update`: `torch.tensor(x.shape[0], device=cuda)` = pageable copy + sync, 4× per step | The same update op for op with a cached device count | N0 | 0.0619 (1.357×) |

Each switch can be set to `0` to disable its change. After the last commit: env-only iteration 22.7 ms (from 34.5),
update iteration 120 ms (from ~178). The profile after round 8 showed 69.6k kernels per 10 iterations (from 105k) and
the GPU busy 67% of the time (from 46%).

## What did not work

- **Compiled autograd for the 32-step BPTT backward** (to fuse the 1,427 `CUDAFunctor_add` gradient accumulations per
  update): still compiling the first backward after 17 minutes; abandoned and reverted.
- **Variants chosen for gate-3 margin**: restoring the critic-loss formulation of an earlier round, a value-identical
  subset of the changes, and the final tree without combo kernels. None was more reliable under gate 3 and all were
  slower (see [measurement.md](measurement.md#gate-3-noise-found-during-the-run)); the fastest tree with a full-check
  PASS was kept.

## Why not 1.5×

At ~0.062 s/iteration the budget is update graphs ≈ 0.4 × 105 ms = 42 ms (policy ~66 ms + world model ~39 ms of GPU
work per update, gaps < 5%) plus an env step of ~23 ms (8.7 ms GPU, of which MuJoCo Warp graphs 5.8 ms). 1.5× needs
56 ms, i.e. removing the whole env host overhead *and* ~15% of the update graphs.

- TF32 GEMMs (~46 ms per update, ~45% of update GPU time) run near the TF32 peak, with K already
  aligned; lower precision is outside the rules.
- LayerNorm + GELU fused kernels (~18 ms per update) move their bytes at HBM bandwidth; fusing them into GEMM
  epilogues needs full-row reductions across tiles.
- `CUDAFunctor_add` (autograd accumulation over 32 BPTT steps): compiled autograd did not finish compiling; the
  parameter layout must stay, so no flat-parameter trick.
- Per-optimizer-step reductions and fused-Adam launches (~0.5 ms/iteration bound) would need a custom Adam that
  matches fused Adam's state updates.
- MuJoCo Warp's per-tree-level kernels (~13 ms per 10 iterations) are inside the frozen simulator.
- The reset path draws random numbers in a fixed order; compiling or reordering it would change the random stream.
