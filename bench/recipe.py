"""The frozen workload: official SuperMimic (HybridRobotics/supermimic@abc124c on HybridRobotics/mjlab_research@dd51a9f),
task SuperMimic-GenCoord-G1, on the 40 LAFAN1 clips of the paper's LAFAN1 experiment (Fig. 4a) instead of the
registry default (the single clip lafan_run1_subject5, behind a private W&B registry). Every other setting is the
code's default (TrainConfig: 4096 envs; SuperMimicRunnerCfg: replay ratio 16 = 0.4 updates per env step, world model
10 batches of 2048 x 8-step windows, actor-critic BPTT policy with horizon 32, ...). The bench adds --env.seed (the
default is a random seed per run) and runs W&B through an offline stand-in (bench/wandb_stub); motions are the LAFAN1
CSVs converted with mjlab_research's own csv_to_npz MotionLoader (30 -> 50 fps), saved locally.
"""
from pathlib import Path

TASK_ID = 'SuperMimic-GenCoord-G1'
ENTRY = 'supermimic/src/supermimic/scripts/train.py'
SRC_DIRS = ('supermimic/src', 'mjlab_research/src')   # put first on sys.path, in this order
UTD = 0.4                        # updates per env step: num_envs / num_batches / batch / horizon * replay_ratio


def argv(npz_dir, seed=0):
    return [TASK_ID, '--motion-file', f'{Path(npz_dir)}/*.npz', '--env.seed', str(seed)]
