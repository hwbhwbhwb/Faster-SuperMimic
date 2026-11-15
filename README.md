# Faster-SuperMimic

Training of the official [SuperMimic](https://github.com/HybridRobotics/supermimic) (with
[mjlab_research](https://github.com/HybridRobotics/mjlab_research)), **1.38× faster** per iteration on one GPU,
optimized by [ResearchBoost-Agent](https://github.com/DAGroup-PKU/ResearchBoost-Agent) (Claude Code + Claude Opus 5.5)
under a judge frozen before optimization started.

| Item | Result |
|---|---|
| Workload | `SuperMimic-GenCoord-G1` world-model + actor-critic BPTT training, 4096 MuJoCo Warp envs, the 40 LAFAN1 clips (G1 retargets), every other setting the code default |
| Speedup | **1.38×** (0.0852 → 0.0617 s/iteration) |
| Hardware | 1 GPU, timed runs pinned to 32 CPU cores |
| Measurement scope | Steady-state iteration time: 600 iterations after a 300-iteration warm-up, reference → candidate → reference |
| Validation | Statistical judge PASS (20 training endpoints within 3σ of 6 reference seeds); end-to-end 20k-iteration training, 2 seeds per arm, benchmark return 0.860 (original) vs 0.862 (optimized) |
| Peak memory | 15.11 → 15.18 GiB allocated |

The code is the upstream tree with nine commits on top; the parameter layout, hyperparameters, precision (TF32),
simulator and the order of random draws are unchanged. Training is not bitwise identical to the original (compiled
and fused kernels reorder floating-point sums, and MuJoCo Warp is not bitwise reproducible run to run even for the
original), which is why correctness is judged statistically and by an end-to-end run.

## What changed

| # | Commit | Change | s/iteration (cumulative) |
|---|---|---|---|
| — | `3db3066` (`upstream-snapshot`) | supermimic@abc124c + mjlab_research@dd51a9f, byte-identical | 0.0840 |
| 1 | Replay sampling | Cache the valid-window prefix sum per buffer version; no host sync after CUDA-graph replay | 0.0815 (1.03×) |
| 2 | Critic fit | `torch.compile` the critic minibatch loss (fused LayerNorm backward) | 0.0765 (1.10×) |
| 3 | Env step | Remove host syncs: device scalars for indexed writes, `searchsorted` for `isin`, no `.item()` in reset stats | 0.0728 (1.15×) |
| 4 | Combo kernels | Inductor horizontal fusion (`combo_kernels`) for the compiled learner steps | 0.0682 (1.23×) |
| 5 | Collection | Skip the simulator observation groups training never reads (`critic`, `goal`) | 0.0646 (1.30×) |
| 6 | Host/GPU trims | Dynamics first-layer K padded 130 → 132 (aligned TF32 GEMM), component accessors, pre-reset hook, compiled collection actor | 0.0637 (1.32×) |
| 7 | Env managers | Compile the reward and termination managers' per-step compute | 0.0626 (1.34×) |
| 8 | MotionCommand | Compile the per-step metrics and relative body poses | 0.0624 (1.35×) |
| 9 | Normalizers | Sync-free normalizer update | 0.0619 (1.36×) |

Per-step numbers are the agent's own `bench.py time` runs; the headline 1.38× is the harness's final full check.
Details, the ideas that did not work, and why the 1.5× target was out of reach: [docs/changes.md](docs/changes.md).

## Validation

- **Judge** ([docs/measurement.md](docs/measurement.md)): frozen before the agent started; its evaluator hash
  `6f78985e…` is the same in every candidate check and the final check, and this repository reproduces it.
  Gate 1 (RNG stream, first 8 iterations) and gate 3 (20 endpoints of an 800-iteration run against 6 reference seeds,
  3σ) pass; gate 2 (bitwise numerics) does not hold for the reference itself across processes, so every candidate is
  judged by gate 3. Largest endpoint deviation in the final check: 1.53σ.
- **End-to-end** ([docs/validation.md](docs/validation.md)): 20,000 training iterations, seeds 0 and 1 for each arm,
  every checkpoint evaluated **with the original code** on all 40 clips (paper Appendix F protocol: each clip once
  from its first frame, deterministic policy, no noise or randomization).

| Iterations | 4k | 8k | 12k | 16k | 20k |
|---|---|---|---|---|---|
| Original, mean normalized return (2 seeds) | 0.768 | 0.809 | 0.842 | 0.854 | **0.860** |
| Optimized, mean normalized return (2 seeds) | 0.753 | 0.817 | 0.840 | 0.857 | **0.862** |
| Original, failed clips (s0 / s1) | 10 / 12 | 4 / 2 | 0 / 3 | 2 / 0 | 0 / 0 |
| Optimized, failed clips (s0 / s1) | 7 / 5 | 3 / 5 | 3 / 1 | 1 / 0 | 1 / 0 |

## Reproduce

```bash
git clone https://github.com/hwbhwbhwb/Faster-SuperMimic && cd Faster-SuperMimic
git worktree add targets/reference upstream-snapshot           # the original code (commit 3db3066, checked by the judge)
git -C ../ResearchBoost-Agent archive 9c176c2 numerics | tar -x # the judge's numerics library (ResearchBoost-Agent@9c176c2)

export ASSETS=$HOME/faster-supermimic-assets                   # outside the repository: the judge copies the tree
bash prep/install_env.sh                                       # Python 3.13, mjlab_research's uv.lock (torch 2.14.0+cu130)
hf download lvhaidong/LAFAN1_Retargeting_Dataset --repo-type dataset --include 'g1/*' --local-dir $ASSETS/raw
PYTHONPATH=targets/reference/mjlab_research/src $ASSETS/env/bin/python prep/convert_motions.py \
  $ASSETS/raw/g1 $ASSETS/data/lafan1                           # 40 clips, 30 -> 50 fps, mjlab_research's converter
(cd $ASSETS/data/lafan1 && sha256sum *.npz > SHA256SUMS)
PYTHONPATH=. $ASSETS/env/bin/python prep/make_fixture.py targets/reference $ASSETS/data/lafan1 /tmp/fixture-scratch \
  $ASSETS/fixture/update_state.pt                              # reference state after 300 iterations (bitwise cases)

export TMPDIR=/tmp
# the harness's final check: reference -> candidate -> reference timing + correctness gates
touch task/final-evaluation                                    # the harness's marker (not part of the judge hash)
$ASSETS/env/bin/python bench/bench.py check --tree . --out runs/final --final
rm task/final-evaluation
# end-to-end comparison (one GPU per queue)
bash e2e/e2e_run.sh 0 ref:0 opt:1 & bash e2e/e2e_run.sh 1 opt:0 ref:1 & wait
```

Notes on the frozen judge, kept byte-identical so its hash can be checked:

- `task/formulation.json` and `bench/records/ref/update.r0.json` fingerprint the input clips and the bitwise fixture
  under the absolute paths of the machine the judge was frozen on (`/mnt/pfs/wenbo/rba-work/supermimic_official/assets`);
  point that path at your `$ASSETS` (a symlink is enough) for the input check. The clip fingerprints are sha256 of the
  converted `.npz` files; a different converter build can produce different bytes, which the check reports.
- `bench/bench.json` pins timed runs to cores `10-41` and keeps a GPU lock under `$TMPDIR`.
- The agent's commits keep their original hashes (`3db3066` → `3b3b3c6`), the ones named in `evidence/`.

## Layout

| Path | Content |
|---|---|
| `supermimic/`, `mjlab_research/` | The optimized code (history: `upstream-snapshot` + 9 commits) |
| `bench/`, `task/` | The frozen judge and task definition (byte-identical) |
| `e2e/` | End-to-end training/evaluation scripts and the results of the four runs |
| `evidence/` | Final check output and the agent's work log |
| `prep/` | Environment and data preparation |

## License

Apache License 2.0, as the upstream projects. The modifications are listed commit by commit above.
SuperMimic: Copyright 2026 The SuperMimic Authors. The LAFAN1 G1 retargets (CC BY-NC-ND 4.0) are not included.
