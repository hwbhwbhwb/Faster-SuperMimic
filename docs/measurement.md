# Measurement and judge

[← README](../README.md)

## Workload

The frozen recipe (`bench/recipe.py`): `supermimic/src/supermimic/scripts/train.py SuperMimic-GenCoord-G1
--motion-file <40 LAFAN1 npz> --env.seed <seed>`, every other setting the code's default (4096 envs, replay ratio
16 = 0.4 updates per env step, world model 10 batches of 2048 eight-step windows, actor-critic BPTT with batch 4096
and horizon 32, TF32 matmuls as mjlab's `configure_torch_backends()` sets).

- Data: the 40 LAFAN1 clips retargeted to the 29-DoF G1 (Hugging Face `lvhaidong/LAFAN1_Retargeting_Dataset`, `g1/`),
  converted with mjlab_research's own `csv_to_npz` MotionLoader (30 → 50 fps): 441,140 frames, 2.451 h. The code's
  default is a single clip behind a private W&B registry; the recipe uses the paper's LAFAN1 set instead.
- W&B: every run uses the offline stand-in `bench/wandb_stub` (the runner calls `wandb.Api()`), which changes nothing
  in training.
- One **iteration** is one `SuperMimicRunner.run()` loop step: one vectorized env step of 4096 MuJoCo Warp envs with
  the actor acting, written to the replay buffer, plus the updates the `Ratio` schedules for it (0.4 on average; one
  update = world-model update + actor-critic BPTT policy update + critic fit).

Software: Python 3.13, torch 2.14.0+cu130, mujoco 3.16.0, mujoco-warp 3.16.0, warp-lang 1.18.0, triton 3.8.0 (the
versions of mjlab_research's `uv.lock`; supermimic's own lock predates the published mjlab_research).

## Timing

`bench.py` runs the real entry point in a fresh process with a cold compile cache, pinned to CPUs 10–41: 300 warm-up
iterations, then two windows of 300 iterations. The final check measures reference → candidate → reference in fresh
processes and divides the mean reference window time by the candidate's.

| Run (final check) | s/iteration | s/update | Peak allocated |
|---|---|---|---|
| Reference A | 0.0858 | 0.2146 | 15.11 GiB |
| Candidate | **0.0617** | 0.1542 | 15.18 GiB |
| Reference B | 0.0846 | 0.2116 | 15.11 GiB |

Speedup 0.0852 / 0.0617 = **1.3821×** (reference spread 1.4%; frozen reference 0.084 s/iteration, spread 0.45% over
three runs; benefit bar 1.028×). Startup to the second iteration (env build, MuJoCo Warp kernel loads, torch.compile,
CUDA-graph capture) is about 290 s for both and is not in the window. Raw output: `evidence/final-check/`.

## Judge

The judge (`bench/`, `task/`, and ResearchBoost-Agent's `numerics/` library at 9c176c2) was written and validated
before the agent started and stayed frozen. Its joint hash (`bench.py evaluator_sha256()`) is
`6f78985e6467ca1cb370adf92a2b824476757728604c2850ed82dd1824b805ea` in every candidate check from the first
(2026-11-13 00:37) to the final one, and this repository reproduces it once `numerics/` is copied in. No judge file
was modified after the agent was launched (2026-11-12 22:30 CST); the harness's acceptance step (`goal/clawo/accept.py`)
compared the judge with the digests recorded at launch and passed.

Gates, in order (a check stops at the first failure):

0. **Structure**: `train.py` and everything in `SuperMimicRunner.run()` outside the loop body keep the reference AST.
1. **Benefit**: the timing above. The random stream at the start of each of the first 8 iterations must match the
   frozen reference (later iterations follow MuJoCo Warp's run-to-run resets and part even between two reference runs);
   no compilation inside a timing window; same command, math mode and library versions.
2. **Bitwise**: 5 updates from a frozen state (the reference after 300 iterations) in deterministic mode; exact items
   (RNG fingerprints, buffer counters, resolved config, motion digest, TF32 math mode, tensor names/shapes, finiteness)
   must hold. The reference is not bitwise identical to itself across processes (compiled imagination and world-model
   steps differ in the last bits), so no candidate can be N0 here and every candidate goes to gate 3.
3. **Statistical**: an 800-iteration trajectory from scratch on seed 0, 20 endpoints measured by the judge (per-model
   parameter-update RMS, drift and norm; per optimizer group the first Adam steps, which detect learning-rate changes;
   the latest `Loss/state`, `Policy/reward`, `Policy/grad_norm`, `Policy/value_function`). σ is the reference's own
   seed-to-seed standard deviation over 6 seeds, frozen in `bench/traj/frozen.json`; every |d| ≤ 3σ → PASS (N1).

Final check, gate 3 (d in units of σ): largest `actor_drift_rms` +1.53, `Policy_reward` −1.12, `Loss_state` +0.88,
all others below 0.9.

### Acceptance tests of the judge (before the agent started)

Trees built by `prep/make_accept.py` (one known change each) and two control runs:

| Planted change | Expected | Judge |
|---|---|---|
| None: unmodified reference, full check | no benefit | UNAVAILABLE (0.99×) |
| None: unmodified reference, quick check | not bitwise, exact items hold | PASS (not bitwise) |
| `TrainConfig.num_envs` 4096 → 4095 in `train.py` | ERROR | ERROR (gate 0) |
| An extra `torch.cuda.synchronize()` in `SuperMimicRunner.run()` before the loop | ERROR | ERROR (gate 0) |
| The world model skips its last minibatch per update | INCORRECT | INCORRECT (gate 3) |
| The actor objective without the critic bootstrap term | INCORRECT | INCORRECT (gate 3) |
| Learning rate × 1.02 (adapter variant, trajectory vs the frozen σ) | INCORRECT | INCORRECT (critic first step 200σ) |
| The discounted-return product taken in another order (`gamma**t * survival * reward` → `reward * survival * gamma**t`) | PASS | equivalent |
| Same-seed rerun of the reference | PASS | equivalent |

One adjustment was made while validating, before launch: the logger's running mean of the finished-episode return was
dropped from the gate-3 endpoints after the equivalent-reordering tree alone came out at 3.08σ on it (an
autocorrelated running mean whose 6-seed σ underestimates its spread).

### Gate-3 noise (found during the run)

Single 800-iteration runs move some endpoints by 2–3σ between reruns of the same tree. The agent measured it: every
candidate passed about half of its gate-3 runs, including a tree with only value-identical changes, while one rerun of
the unmodified reference passed at 1.45σ. The frozen seed-0 run sits about 2σ off the center of the seed
distribution on `Policy_grad_norm`, `Policy_reward` and `actor_update_rms`, and any rerun whose trajectory
decorrelates from it lands near that distribution's center. The judge was not changed. The final candidate (rounds
1–10, commit `3b3b3c6`) passed its own full check (1.360×, largest 2.90σ) and the harness's final check (1.382×,
largest 1.53σ). Because of this noise the end-to-end comparison in [validation.md](validation.md) was run as well.

## Fixture

`prep/make_fixture.py` rebuilds `$ASSETS/fixture/update_state.pt` (the reference's replay buffer, models, optimizer
states, iteration and RNG streams after 300 iterations of the recipe on seed 0) for gate 2.
