"""Acceptance trees of the official task: copies of the reference snapshot with one known change each.

    python make_accept.py <reference tree> <out dir>

  b_config      TrainConfig.num_envs 4096 -> 4095 in train.py               gate 0: ERROR
  b_setup       an extra statement in SuperMimicRunner.run before the loop   gate 0: ERROR
  c_wm_batches  the world model skips its last minibatch per update          INCORRECT (gate 3)
  c_no_boot     the actor objective without the critic bootstrap             INCORRECT (gate 3)
  d_reorder     the discounted-return product taken in another order         PASS (N1, gate 3)
The learning-rate control arm (x1.02) is the adapter variant `lr_x1.02`, run as a trajectory against the frozen sigma.
"""
import json
import shutil
import sys
from pathlib import Path

SKIP = shutil.ignore_patterns('.git', '__pycache__', 'logs')
P = 'supermimic/src/supermimic/'
EDITS = {
    'b_config': (P + 'scripts/train.py', '  num_envs: int = 4096', '  num_envs: int = 4095', 'ERROR'),
    'b_setup': (P + 'runners/supermimic_runner.py', '    self._seed_sim_history()\n',
                '    self._seed_sim_history()\n    torch.cuda.synchronize()\n', 'ERROR'),
    'c_wm_batches': (P + 'learners/wm_learner.py', '    for td in batches.unbind(0):',
                     '    for td in batches.unbind(0)[:-1]:', 'INCORRECT'),
    'c_no_boot': (P + 'learners/policy_learner.py', '    ret = ret + self._bootstrap(obs, survival, term_prob)',
                  '    ret = ret + 0.0 * self._bootstrap(obs, survival, term_prob)', 'INCORRECT'),
    'd_reorder': (P + 'learners/policy_learner.py', '      ret = ret + cfg.gamma**t * survival * reward',
                  '      ret = ret + reward * survival * cfg.gamma**t', 'PASS'),
}


def main():
    ref, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    expected = {}
    for name, (rel, old, new, verdict) in EDITS.items():
        dest = out / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(ref, dest, ignore=SKIP)
        f = dest / rel
        text = f.read_text()
        if text.count(old) != 1:
            raise SystemExit(f'{name}: {rel} has {text.count(old)} matches for the edit')
        f.write_text(text.replace(old, new))
        expected[name] = {'file': rel, 'expected': verdict}
    (out / 'expected.json').write_text(json.dumps(expected, indent=1) + '\n')
    print(json.dumps(expected))


if __name__ == '__main__':
    main()
