# End-to-end validation

[← README](../README.md)

The judge's gate 3 compares 800 iterations, and single runs of it are noisy
([measurement.md](measurement.md#gate-3-noise-found-during-the-run)). So both trees were also trained for 20,000
iterations and their checkpoints evaluated on the full clip set.

## Setup

- Arms: **original** = `upstream-snapshot` (`3db3066`), **optimized** = `3b3b3c6` (this repository), the frozen
  recipe of the judge (`bench/recipe.py`: 40 LAFAN1 clips, `--env.seed`, every other setting the code default), with
  the bench's offline W&B stand-in.
- Seeds 0 and 1 per arm, 20,000 iterations (`--runner.max-iterations 20001`), on two GPUs at once:
  GPU 0 ran original s0 then optimized s1, GPU 1 optimized s0 then original s1 (`e2e/e2e_run.sh`).
- The runner overwrites `world_model.pt` / `actor.pt` / `critic.pt` every 1,000 iterations; `e2e/e2e_train.py` keeps
  copies of the saves at 4k, 8k, 12k, 16k and 20k.
- Every checkpoint is evaluated **with the original code** (`e2e/e2e_eval.py`), so the optimized checkpoints also have
  to load into the unmodified models. The official repository ships no tracking evaluation (its `benchmark.py` times
  the simulator), so the script follows the paper's Appendix F protocol: one env per clip, started at the clip's first
  frame, deterministic actor (no action noise), no reset noise, no observation noise, no domain randomization (the
  startup DR events run so the sim context binds as in training, then every randomized model field is restored to its
  default and the encoder bias to 0). A clip fails if a non-time-out termination fires before the clip's end;
  normalized return = summed step reward / (max step reward × steps), max step reward = sum of the positive reward
  weights × step dt = 0.1.

## Results

Mean normalized return over the 40 clips:

| Iterations | 4k | 8k | 12k | 16k | 20k |
|---|---|---|---|---|---|
| Original s0 | 0.7655 | 0.8023 | 0.8404 | 0.8547 | 0.8583 |
| Original s1 | 0.7697 | 0.8154 | 0.8429 | 0.8542 | 0.8610 |
| Optimized s0 | 0.7333 | 0.8136 | 0.8389 | 0.8554 | 0.8624 |
| Optimized s1 | 0.7727 | 0.8199 | 0.8410 | 0.8584 | 0.8622 |
| **Original, mean** | 0.768 | 0.809 | 0.842 | 0.854 | **0.860** |
| **Optimized, mean** | 0.753 | 0.817 | 0.840 | 0.857 | **0.862** |

Failed clips (of 40):

| Iterations | 4k | 8k | 12k | 16k | 20k |
|---|---|---|---|---|---|
| Original s0 / s1 | 10 / 12 | 4 / 2 | 0 / 3 | 2 / 0 | 0 / 0 |
| Optimized s0 / s1 | 7 / 5 | 3 / 5 | 3 / 1 | 1 / 0 | 1 / 0 |

Tracking errors at 20k (mjlab `MotionCommand` metrics, step-weighted over all clips):

| | Original s0 | Original s1 | Optimized s0 | Optimized s1 |
|---|---|---|---|---|
| `error_body_pos` | 0.0330 | 0.0323 | 0.0308 | 0.0312 |
| `error_joint_pos` | 0.4657 | 0.4681 | 0.4676 | 0.4323 |
| `error_anchor_rot` | 0.0479 | 0.0489 | 0.0483 | 0.0435 |

The two arms learn at the same rate and end at the same tracking quality; the differences are within the
seed-to-seed spread of each arm.

Wall-clock training time of these runs (not a timing measurement: two runs shared the host's CPUs, without pinning):
original 1,729 s / 2,258 s, optimized 1,277 s / 1,785 s for 20k iterations including startup.

## Files

`e2e/results/<arm>_s<seed>/`: `benchmark_it<N>.json` (per-clip results and summary), `metrics.jsonl` (every training log
row the runner wrote, one per 10 iterations), `train_done.json`; `e2e/results/queue_gpu{0,1}.log`. Absolute machine
paths in these files were made repository-relative. The scripts in `e2e/` are the ones that ran, with their
machine paths replaced by repository-relative paths and `$ASSETS`.
