"""End-to-end training run of one official SuperMimic tree with the frozen bench recipe (bench/recipe.py: task
SuperMimic-GenCoord-G1, the 40 LAFAN1 clips, --env.seed, every other setting the code default), stopped after
--iters iterations. The runner overwrites world_model.pt / actor.pt / critic.pt every save_interval iterations;
the saves at --snap-its are copied to <out>/snap/it<N>/. W&B goes through the bench's offline stub, which appends
every wandb.log row to <out>/metrics.jsonl.
  python e2e_train.py --tree <dir> --npz-dir <dir> --seed <s> --out <dir> [--iters 20000] [--snap-its 4000,8000]
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

STUB = str(Path(__file__).resolve().parents[1] / 'bench' / 'wandb_stub')   # the bench's offline W&B stand-in
TASK = 'SuperMimic-GenCoord-G1'
CKPT_FILES = ('world_model.pt', 'actor.pt', 'critic.pt')   # saved in this order in SuperMimicRunner.run()

p = argparse.ArgumentParser()
p.add_argument('--tree', required=True)
p.add_argument('--npz-dir', required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--out', required=True)
p.add_argument('--iters', type=int, default=20000)
p.add_argument('--snap-its', default='4000,8000,12000,16000,20000')
a = p.parse_args()

tree, out = Path(a.tree).resolve(), Path(a.out).resolve()
out.mkdir(parents=True, exist_ok=True)
for d in (STUB, tree / 'mjlab_research/src', tree / 'supermimic/src'):   # -> supermimic, mjlab_research, stub
    sys.path.insert(0, str(d))
os.environ['SM_BENCH_WANDB_LOG'] = str(out / 'metrics.jsonl')
os.chdir(out)                                    # run() writes logs/<task>/<timestamp>/ relative to cwd

import torch  # noqa: E402
from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg  # noqa: E402
from supermimic.scripts import train  # noqa: E402

save_every = SuperMimicRunnerCfg().save_interval
snap_its = {int(x) for x in a.snap_its.split(',') if x}
bad = sorted(i for i in snap_its if i % save_every or i > a.iters)
if bad:
    raise SystemExit(f'--snap-its {bad}: not a save point (every {save_every} iterations, up to {a.iters})')

_save = torch.save
n_saves = -1


def save(obj, f, *args, **kwargs):
    global n_saves
    _save(obj, f, *args, **kwargs)
    name = os.path.basename(f) if isinstance(f, (str, os.PathLike)) else None
    if name not in CKPT_FILES:
        return
    if name == CKPT_FILES[0]:
        n_saves += 1
    it = n_saves * save_every                    # the loop saves at step % save_every == 0, from step 0
    if it in snap_its:
        d = out / 'snap' / f'it{it}'
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, d / name)


torch.save = save

sys.argv = ['train.py', TASK, '--motion-file', f'{Path(a.npz_dir).resolve()}/*.npz', '--env.seed', str(a.seed),
            '--runner.max-iterations', str(a.iters + 1)]      # step a.iters is the last one, and a save point
t0 = time.time()
train.main()
missing = sorted(i for i in snap_its if not (out / 'snap' / f'it{i}' / 'actor.pt').exists())
(out / 'train_done.json').write_text(json.dumps({'argv': sys.argv, 'tree': str(tree), 'seed': a.seed,
                                                 'iters': a.iters, 'wall_s': round(time.time() - t0, 1),
                                                 'saves': n_saves + 1, 'missing_snaps': missing}))
if missing:
    raise SystemExit(f'snapshots missing for iterations {missing}')
print(f'[e2e] done {a.iters} iterations in {time.time() - t0:.0f} s, {n_saves + 1} saves')
