"""numerics adapter for the official SuperMimic (supermimic + mjlab_research tree): builds the tree's own runner the
way train.py main() does on the frozen recipe, then runs either updates from a frozen state (bitwise cases) or the
tree's own per-iteration loop body (trajectories).

build(): the tree's TrainConfig is parsed from the frozen recipe arguments (task SuperMimic-GenCoord-G1, the 40
LAFAN1 clips, --env.seed = the case seed), the env is created and SuperMimicRunner(env, cfg, ...) constructed exactly
as in train.py main(); then the statements of SuperMimicRunner.run() before its `for step in ...` loop run as written
in the tree's runner module (gate 0 guarantees they equal the reference's), with the bench's offline W&B stand-in.

Bitwise cases (mode "update"): MuJoCo Warp is not deterministic from run to run, so the env step is not recorded.
The case restores a frozen state into the tree's objects (load_state: the replay buffer's storage and counters, the
world-model dynamics, representation, sim context, actor, critic, the three optimizer lists' states and every RNG
stream, saved from the reference after `preroll` iterations), then each step runs one update: the reference loop
body's update block (the body of `for _ in range(should_train(step)):`: replay sampling, world_model_learner.update,
policy_learner.update) in the tree's runner module, so it calls the tree's learners, models, buffer and wm env.
The official learners replay CUDA graphs after their first calls, so optimizer step() hooks do not fire on every
update; instead each step returns, per model, the parameter change of the update and the optimizer state after it
(Adam exp_avg / exp_avg_sq / step concatenated over parameters), plus the update's loss dict, as loss/ tensors.
Exact invariants: replay-buffer counters, the resolved runner configuration (all fields except run names / W&B
fields), a digest of the motion data at fixed frames, the fp32 math mode.

Trajectories (mode "loop", losses only): fresh from the seed, every step is one execution of the tree's own loop
body (`step` = the iteration index). Each step returns what the judge measures from the objects: per model the RMS of
the iteration's parameter change, the RMS distance from the initial parameters and the parameter norm; for each
optimizer group (wm, actor, critic) the RMS change of its first step() and the mean over its first 3 (the first
learner calls run eagerly, before graph capture: Adam's first step moves every element by about the learning rate,
a learning-rate detector with little seed dependence); and the latest values of the loop's loss_dict
(Loss/state, Policy/reward, Policy/grad_norm, Policy/value_function), repeated until the next update, and the
mean return of the logger's finished-episode buffer.
"""
import ast
import dataclasses
import hashlib
import importlib
import json
import math
import os
import sys
from pathlib import Path

import torch

from numerics.adapter import Adapter, Objects

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
import recipe  # noqa: E402

FREE_CONFIG_KEYS = {'run_name', 'wandb_project', 'wandb_run_path', 'headless'}
FIRST_STEPS = 3
TRAJ_LOSSES = ('Loss/state', 'Policy/reward', 'Policy/grad_norm', 'Policy/value_function')


def run_function(module):
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == 'SuperMimicRunner':
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == 'run':
                    return fn
    raise ValueError('no SuperMimicRunner.run')


def loop_parts(source):
    """(setup statements, loop body statements) of SuperMimicRunner.run in the given runner source."""
    fn = run_function(ast.parse(source))
    for idx, stmt in enumerate(fn.body):
        if isinstance(stmt, ast.For):
            return fn.body[:idx], stmt.body
    raise ValueError('SuperMimicRunner.run has no top-level for loop')


def update_block(source):
    """Body of the `for _ in range(should_train(step)):` loop inside run()'s loop body: one update."""
    _, body = loop_parts(source)
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.For) and 'should_train' in ast.unparse(node.iter):
            return node.body
    raise ValueError('no `for _ in range(should_train(step))` update loop in SuperMimicRunner.run')


def wrap_once(stmts):
    wrapped = ast.parse('for _bench_once in (0,):\n    pass')
    wrapped.body[0].body = stmts
    return ast.fix_missing_locations(wrapped)


def flat_config(obj, prefix=''):
    out = {}
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            out.update(flat_config(getattr(obj, f.name), f'{prefix}{f.name}.'))
    elif isinstance(obj, dict):
        for k in sorted(obj):
            out.update(flat_config(obj[k], f'{prefix}{k}.'))
    else:
        out[prefix.rstrip('.')] = repr(obj)
    return out


def flat(tensors):
    return torch.cat([t.detach().reshape(-1).float() for t in tensors]) if tensors else torch.zeros(0)


class SuperMimicAdapter(Adapter):
    name = 'supermimic-official'

    def __init__(self, repo, npz_dir, scratch, reference, variant=None):
        # absolute: build() changes the working directory before importing the tree
        self.repo, self.npz_dir, self.scratch = Path(repo).resolve(), Path(npz_dir).resolve(), Path(scratch).resolve()
        self.reference = Path(reference).resolve()
        self.variant = variant
        self.case, self.mode, self.preroll = {}, 'update', 0

    def configure(self, case):
        self.case = case
        self.mode = case.get('mode', 'update')
        if self.mode not in ('update', 'loop'):
            raise ValueError(f'unknown case mode {self.mode!r}')
        self.preroll = int(case.get('preroll', 0))

    # ------------------------------------------------------------------ build

    def build(self, seed, golden=False):
        if golden:
            raise NotImplementedError('no fp64 golden variant for this task')
        sys.path.insert(0, str(BENCH / 'wandb_stub'))
        for d in reversed(recipe.SRC_DIRS):
            sys.path.insert(0, str(self.repo / d))
        self.scratch.mkdir(parents=True, exist_ok=True)
        os.chdir(self.scratch)                         # run() writes logs/<task>/<timestamp>/ relative to cwd
        import mjlab
        import tyro
        train = importlib.import_module('supermimic.scripts.train')
        runner_mod = importlib.import_module('supermimic.runners.supermimic_runner')
        argv = recipe.argv(self.npz_dir, seed=seed)
        task = argv[0]
        args = tyro.cli(train.TrainConfig, args=argv[1:], default=train.TrainConfig.from_task(task),
                        config=mjlab.TYRO_FLAGS)
        if self.variant:
            self._apply_variant(args)
        # train.py main(), with its motion-file branch (the recipe always passes --motion-file)
        device = train.setup_device(args.device)
        args.env.commands['motion'].motion_file = os.path.expanduser(args.motion_file)
        args.registry_name = None
        args.env.commands['motion'].sampling_mode = (
            'external' if args.runner.sampling_mode == 'return' else args.runner.sampling_mode)
        env = train.create_env(args.env, args.num_envs, device)
        self.r = runner_mod.SuperMimicRunner(env=env, cfg=args.runner, task_id=task, registry_name=None,
                                             device=device)
        if torch.are_deterministic_algorithms_enabled():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        cfg_flat = {k: v for k, v in flat_config(args.runner).items() if k.split('.')[0] not in FREE_CONFIG_KEYS}
        cfg_flat['num_envs'] = repr(args.num_envs)
        self.config_digest = hashlib.sha256(json.dumps(cfg_flat, sort_keys=True).encode()).hexdigest()[:24]
        self.data_digest = self._data_digest()
        src = Path(runner_mod.__file__).read_text()
        setup, body = loop_parts(src)
        self.ns = dict(vars(runner_mod))
        self.ns['self'] = self.r
        exec(compile(ast.Module(body=setup, type_ignores=[]), f'{runner_mod.__file__}:run(setup)', 'exec'), self.ns)
        self.body = compile(wrap_once(body), f'{runner_mod.__file__}:run(loop body)', 'exec')
        ref_src = (self.reference / 'supermimic/src/supermimic/runners/supermimic_runner.py').read_text()
        self.update = compile(wrap_once(update_block(ref_src)), 'reference run(update block, run in the tree\'s module)',
                              'exec')
        self.iteration = 0
        for _ in range(self.preroll):
            self._iterate()
        self.prev = self._snapshot()
        self.init = {m: v.clone() for m, v in self.prev.items()}
        self.last = {}
        self.first_steps = {g: [] for g in self._optimizer_groups()}
        if self.mode == 'loop':
            self._watch_first_steps()

    def _apply_variant(self, args):
        if self.variant == 'lr_x1.02':
            args.runner.wm.lr *= 1.02
            args.runner.policy.lr *= 1.02
        else:
            raise ValueError(f'unknown variant {self.variant!r}')

    def _data_digest(self):
        m = self.r.motion
        g = torch.Generator().manual_seed(1234)
        frames = torch.randint(0, m.time_step_total, (4096,), generator=g).to(m.joint_pos.device)
        h = hashlib.sha256()
        for t in (m.root_pos_w[frames], m.root_quat_w[frames], m.joint_pos[frames], m.joint_vel[frames],
                  m.root_lin_vel_w[frames], m.root_ang_vel_w[frames], *m.body_state(frames), m.motion_ends):
            t = t.detach().float().contiguous().cpu()
            h.update(str(tuple(t.shape)).encode() + t.view(-1).view(torch.uint8).numpy().tobytes())
        return h.hexdigest()[:24]

    def _iterate(self):
        self.ns['step'] = self.iteration
        exec(self.body, self.ns)
        self.iteration += 1

    # ------------------------------------------------------------------ objects and state

    def _models(self):
        r = self.r
        return {'dynamics': r._dynamics, 'representation': r._representation, 'sim_context': r._sim_context,
                'actor': r.policy_learner.actor, 'critic': r.policy_learner.critic}

    def _optimizer_groups(self):
        pl = self.r.policy_learner
        return {'wm': list(self.r.world_model_learner.optimizers), 'actor': list(pl.actor_optimizers),
                'critic': list(pl.critic_optimizers)}

    def objects(self):
        # optimizers deliberately not registered: graph replays bypass step() hooks (see module docstring)
        return Objects(models={k: v for k, v in self._models().items() if isinstance(v, torch.nn.Module)},
                       sync={}, count_forwards=False)

    def save_state(self, path):
        r = self.r
        buf = r.replay_buffer
        state = {
            'buffer': {'storage': buf._storage.clone(), '_step': buf._step, 'num_transitions': buf.num_transitions},
            'models': {m: mod.state_dict() for m, mod in self._models().items() if hasattr(mod, 'state_dict')},
            'optimizers': {g: [o.state_dict() for o in opts] for g, opts in self._optimizer_groups().items()},
            'iteration': self.iteration,
            'rng': {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()},
        }
        torch.save(state, path)

    def load_state(self, path):
        state = torch.load(path, map_location=self.r.device, weights_only=False)
        buf = self.r.replay_buffer
        src = state['buffer']['storage']
        for key in src.keys(include_nested=True, leaves_only=True):
            cur, v = buf._storage.get(key), src.get(key)
            if cur.shape != v.shape or cur.dtype != v.dtype:
                raise RuntimeError(f'replay buffer field {key}: the tree has {(cur.dtype, tuple(cur.shape))}, the '
                                   f'frozen state {(v.dtype, tuple(v.shape))}; keep ReplayBuffer\'s storage layout')
            cur.copy_(v)
        buf._step, buf.num_transitions = state['buffer']['_step'], state['buffer']['num_transitions']
        for m, mod in self._models().items():
            if m in state['models']:
                mod.load_state_dict(state['models'][m])
        for g, opts in self._optimizer_groups().items():
            for o, sd in zip(opts, state['optimizers'][g], strict=True):
                o.load_state_dict(sd)
        self.iteration = state['iteration']
        torch.set_rng_state(state['rng']['cpu'].cpu())
        torch.cuda.set_rng_state(state['rng']['cuda'].cpu())
        self.prev = self._snapshot()

    # ------------------------------------------------------------------ steps

    def _snapshot(self):
        return {m: flat(list(mod.parameters())).clone() for m, mod in self._models().items()
                if isinstance(mod, torch.nn.Module) and any(True for _ in mod.parameters())}

    def _optimizer_state(self, opts):
        ts = []
        for o in opts:
            for group in o.param_groups:
                for p in group['params']:
                    for k in ('exp_avg', 'exp_avg_sq', 'step'):
                        v = o.state.get(p, {}).get(k)
                        if isinstance(v, torch.Tensor):
                            ts.append(v)
        return flat(ts)

    def _watch_first_steps(self, n=FIRST_STEPS):
        for g, opts in self._optimizer_groups().items():
            params = [p for o in opts for grp in o.param_groups for p in grp['params']]
            before = {}
            for o in opts:
                def pre(_o, _a, _k, g=g, params=params, before=before):
                    if len(self.first_steps[g]) < n and not torch.cuda.is_current_stream_capturing():
                        before['flat'] = flat(params).clone()

                def post(_o, _a, _k, g=g, params=params, before=before):
                    if 'flat' in before and len(self.first_steps[g]) < n:
                        cur = flat(params)
                        self.first_steps[g].append((cur - before.pop('flat')).double().pow(2).mean().sqrt())
                o.register_step_pre_hook(pre)
                o.register_step_post_hook(post)

    def batch(self, step, rank):
        return None

    def step(self, step, batch):
        ns = self.ns
        if self.mode == 'update':
            ns['step'] = self.iteration
            exec(self.update, ns)
            torch.cuda.synchronize()
            out = {}
            cur = self._snapshot()
            for m in cur:
                out[f'delta/{m}'] = cur[m] - self.prev[m]
            self.prev = cur
            for g, opts in self._optimizer_groups().items():
                out[f'optim/{g}'] = self._optimizer_state(opts)
            for k, v in (ns.get('loss_dict') or {}).items():
                out[f'metric/{k}'] = torch.as_tensor(v).detach().float().reshape(-1)
            return out
        self._iterate()
        cur = self._snapshot()
        out = {}
        for m in cur:
            out[f'{m}_update_rms'] = (cur[m] - self.prev[m]).double().pow(2).mean().sqrt()
            out[f'{m}_drift_rms'] = (cur[m] - self.init[m]).double().pow(2).mean().sqrt()
            out[f'{m}_param_norm'] = cur[m].double().norm()
        self.prev = cur
        for g, rms in self.first_steps.items():
            out[f'{g}_step1_rms'] = rms[0] if rms else math.nan
            out[f'{g}_first_steps_rms'] = torch.stack(rms).mean() if len(rms) == FIRST_STEPS else math.nan
        ld = ns.get('loss_dict') or {}
        for k in TRAJ_LOSSES:
            if k in ld:
                self.last[k] = float(ld[k])
            out[f'last/{k}'] = self.last.get(k, math.nan)
        rew = getattr(self.r.logger, 'rewbuffer', None)
        out['episode_return_mean'] = sum(rew) / len(rew) if rew else math.nan
        return out

    def phases(self):
        return ['update', 'iteration']

    def invariants(self):
        buf = self.r.replay_buffer
        return {'iteration': self.iteration, 'buffer_step': buf._step, 'buffer_transitions': buf.num_transitions,
                'config_digest': self.config_digest, 'data_digest': self.data_digest,
                'fp32_precision': f'matmul={torch.backends.cuda.matmul.fp32_precision} '
                                  f'cudnn={torch.backends.cudnn.fp32_precision}'}
