"""Evaluate one saved SuperMimic checkpoint on every LAFAN1 clip with the paper's benchmark protocol (Appendix F, as
the reproduction's scripts/eval_benchmark.py): one env per clip, started at the clip's first frame, deterministic
actor (no action noise), no reset noise (play config: no RSI pose/velocity ranges; joint_position_range set to 0),
no domain randomization (the startup DR events still run so SimContext binds its ranges as in training, then every
DR field is written back to the model default and the encoder bias to 0), no observation noise.
A clip fails if a non-time-out termination fires before the clip's end (motion_end); normalised return = summed
step reward / (max step reward * steps), max step reward = sum of the positive reward weights * step_dt.
  python e2e_eval.py --tree <reference tree> --npz-dir <dir> --ckpt-dir <snap/itN> --out <json>
"""
import argparse
import glob
import json
import sys
from pathlib import Path

STUB = str(Path(__file__).resolve().parents[1] / 'bench' / 'wandb_stub')   # the bench's offline W&B stand-in
TASK = 'SuperMimic-GenCoord-G1'

p = argparse.ArgumentParser()
p.add_argument('--tree', required=True)
p.add_argument('--npz-dir', required=True)
p.add_argument('--ckpt-dir', required=True)
p.add_argument('--out', required=True)
p.add_argument('--device', default='cuda:0')
a = p.parse_args()

tree = Path(a.tree).resolve()
for d in (STUB, tree / 'mjlab_research/src', tree / 'supermimic/src'):
    sys.path.insert(0, str(d))

import mjlab.tasks  # noqa: E402,F401
import torch  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402

import supermimic.tasks  # noqa: E402,F401
from supermimic.representations import extract_actor_goal  # noqa: E402
from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg, build_actor, build_world_model  # noqa: E402
from supermimic.tasks.task_spec import build_actor_obs_td  # noqa: E402

configure_torch_backends()
device = a.device
files = sorted(glob.glob(f'{Path(a.npz_dir).resolve()}/*.npz'))
if not files:
    raise SystemExit(f'no clips in {a.npz_dir}')

env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = len(files)
cmd_cfg = env_cfg.commands['motion']
cmd_cfg.motion_file = files
cmd_cfg.joint_position_range = (0.0, 0.0)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))
base = env.unwrapped

# Domain randomization off: startup events ran in the constructor; restore every model field they registered
# (base_com also rewrites the derived inertia fields) to the defaults, and zero the encoder bias.
events = set(base.cfg.events)
if events != {'foot_friction', 'base_com', 'encoder_bias'}:
    raise SystemExit(f'unexpected events in the play config: {sorted(events)}')
dr_fields = base.event_manager.domain_randomization_fields
for field in dr_fields:
    cur = getattr(base.sim.model, field)
    cur[:] = base.sim.get_default_field(field).expand_as(cur)
base.scene['robot'].data.encoder_bias[:] = 0.0

runner_cfg = SuperMimicRunnerCfg()
wm = build_world_model(runner_cfg, env, device)
ckpt = Path(a.ckpt_dir)
wm.load_bundle(torch.load(ckpt / 'world_model.pt', map_location=device))
wm.dynamics.eval()
representation = wm.representation
actor = build_actor(runner_cfg, wm, device)
actor.set_slot_mix({'motion': 1.0})
actor.load_state_dict(torch.load(ckpt / 'actor.pt', map_location=device))
actor.eval()
proprio = representation.actor_obs_mode == 'proprio'

# Env e plays clip e from its first frame.
cmd = base.command_manager.get_term('motion')
ends = cmd.motion.motion_ends
starts = torch.cat([ends.new_zeros(1), ends[:-1]])
n = base.num_envs
if len(ends) != n:
    raise SystemExit(f'{len(ends)} clips loaded for {n} envs')
cmd.cfg.sampling_mode = 'uniform'
cmd._uniform_sampling = lambda env_ids: starts[env_ids]
env.reset()
# reset() resamples, then runs command_manager.compute(dt=0), which advances every reset env by one frame.
if not torch.equal(cmd.time_steps.long(), starts + 1):
    raise SystemExit(f'reset did not place every env one frame past its clip start: {cmd.time_steps.tolist()}')
sim_ctx = wm.sim_context.read(base)

rm, tm = base.reward_manager, base.termination_manager
term_names = list(rm._term_names)
weights = [c.weight for c in rm._term_cfgs]
max_r = sum(w for w in weights if w > 0) * (base.step_dt if base.cfg.scale_rewards_by_dt else 1.0)
metric_names = sorted(cmd.metrics)
frames = (ends - starts).tolist()

active = torch.ones(n, dtype=torch.bool, device=device)
ret = torch.zeros(n, device=device)
steps = torch.zeros(n, dtype=torch.long, device=device)
term_sum = torch.zeros(n, len(term_names), device=device)
metric_sum = torch.zeros(n, len(metric_names), device=device)
results = {}
history = None
with torch.inference_mode():
    for _ in range(max(frames) + 10):
        obs = env.get_observations()
        rep = representation.represent(obs['full_state'])
        sensor = representation.compute_actor_sensor(rep, sim_ctx, sim_actor_obs=obs['actor'] if proprio else None)
        if actor.history_length == 1:
            history = sensor.unsqueeze(1)
        elif history is None:
            history = sensor.unsqueeze(1).repeat(1, actor.history_length, 1)
        else:
            history = torch.cat([history[:, 1:], sensor.unsqueeze(1)], dim=1)
        actions = actor(build_actor_obs_td({'motion': extract_actor_goal(obs['goal'])}, history))
        _, rew, dones, _ = env.step(actions)
        live = active.float()
        ret += rew * live
        steps += active.long()
        term_sum += rm._step_reward * live[:, None]
        metric_sum += torch.stack([cmd.metrics[k] for k in metric_names], dim=-1) * live[:, None]
        failed = tm.terminated & ~tm.time_outs
        done = dones.bool() & active
        for e in done.nonzero().squeeze(-1).tolist():
            st = int(steps[e])
            results[e] = {'name': Path(files[e]).stem, 'frames': frames[e], 'steps': st,
                          'norm_return': float(ret[e]) / (max_r * st), 'failed': bool(failed[e]),
                          'fail_step': st if bool(failed[e]) else -1,
                          'terms': {k: float(term_sum[e, i]) / st for i, k in enumerate(term_names)},
                          'metrics': {k: float(metric_sum[e, i]) / st for i, k in enumerate(metric_names)}}
        active &= ~done
        if not active.any():
            break
env.close()
if len(results) != n:
    raise SystemExit(f'{n - len(results)} clips did not finish within their length')

clips = [results[i] for i in range(n)]
tot = sum(r['steps'] for r in clips)
summary = {'clips': n, 'failed': sum(r['failed'] for r in clips),
           'mean_norm_return': sum(r['norm_return'] for r in clips) / n,
           'step_weighted_norm_return': sum(r['norm_return'] * r['steps'] for r in clips) / tot,
           'completed_frame_fraction': tot / sum(f - 1 for f in frames),   # a full clip runs frames - 1 steps
           'max_step_reward': max_r,
           'terms': {k: sum(r['terms'][k] * r['steps'] for r in clips) / tot for k in term_names},
           'metrics': {k: sum(r['metrics'][k] * r['steps'] for r in clips) / tot for k in metric_names},
           'dr_fields_restored': list(dr_fields)}
Path(a.out).write_text(json.dumps({'summary': summary, 'ckpt': str(ckpt.resolve()), 'clips': clips}, indent=1))
print(f"\n{n} clips | failed {summary['failed']} | mean norm return {summary['mean_norm_return']:.4f} "
      f"| step-weighted {summary['step_weighted_norm_return']:.4f}")
for k, v in summary['metrics'].items():
    print(f'  {k:28s} {v:.4f}')
for r in sorted(clips, key=lambda r: r['norm_return'])[:8]:
    print(f"  {r['name']:28s} ret {r['norm_return']:.3f} steps {r['steps']}/{r['frames']}"
          f"{' FAIL@' + str(r['fail_step']) if r['failed'] else ''}")
print('saved', a.out)
