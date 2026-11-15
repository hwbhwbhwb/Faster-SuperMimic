"""Iteration timing of a tree's real official SuperMimic entrypoint (supermimic/src/supermimic/scripts/train.py run
as __main__), nothing patched in the training code.

  python <bench>/timing.py --tree <tree> --out <prefix> [options] -- <train.py arguments>

Run in a scratch working directory (the runner writes logs/<task>/<timestamp>/ relative to it). The tree's
supermimic/src and mjlab_research/src and the bench's offline W&B stand-in (bench/wandb_stub) are put first on
sys.path. One iteration of SuperMimicRunner.run() is one call of SuperMimicRunner.step() (one vectorised env step of
4096 environments) followed by the updates the Ratio schedules for it (0.4 per iteration on average: world-model
update + policy update). The hook wraps SuperMimicRunner.step at the class level and fires when a step() call
begins, so a window from the start of iteration a to the start of iteration b holds iterations a..b-1 whole (env
step, updates, logging). The first `warmup` iterations are skipped (env build, compiles, CUDA-graph captures), then
`repeats` windows of `window` iterations are timed; window edges are bracketed by torch.cuda.synchronize(). At every
iteration start the CPU and CUDA random generator states are fingerprinted (host-side, no sync): the bench compares
that sequence with the reference, so every timed run draws the same random numbers in the same order.

--profile torch: torch.profiler over `profile_iters` iterations after the warmup, with record_function ranges around
SuperMimicRunner.step, the world-model update, the policy update and env.step (installed only in this mode); writes a
summary and a Chrome trace. --profile nsys: cudaProfilerStart/Stop around the same span.

Writes <prefix>.r0.json.
"""
import argparse
import hashlib
import json
import os
import runpy
import signal
import sys
import time
from pathlib import Path

import torch

T_START = time.perf_counter()
BENCH = Path(__file__).resolve().parent
ANNOTATION_PREFIXES = ('## ', 'Call CompiledFxGraph', 'Optimizer.', 'CompiledFunction', 'ProfilerStep')
sys.path.insert(0, str(BENCH))
import recipe  # noqa: E402


def compile_count():
    utils = sys.modules.get('torch._dynamo.utils')
    if utils is None:
        return 0
    return utils.counters['stats']['unique_graphs'] + utils.counters['aot_autograd']['total']


def rng_fingerprint():
    h = hashlib.sha256(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_initialized():
        h.update(torch.cuda.get_rng_state().numpy().tobytes())
    return h.hexdigest()[:16]


def precision_flags():
    b = torch.backends
    return {'cuda.matmul.fp32_precision': b.cuda.matmul.fp32_precision, 'cudnn.fp32_precision': b.cudnn.fp32_precision,
            'cudnn.conv.fp32_precision': b.cudnn.conv.fp32_precision, 'cudnn.rnn.fp32_precision': b.cudnn.rnn.fp32_precision}


class StopTiming(BaseException):
    """BaseException, so `except Exception` blocks in the training code cannot swallow it."""


def kill_descendants():
    children = {}
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        try:
            with open(f'/proc/{d}/stat') as f:
                ppid = int(f.read().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(d))
    stack, found = [os.getpid()], []
    while stack:
        for c in children.get(stack.pop(), []):
            found.append(c)
            stack.append(c)
    for pid in found:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    return len(found)


class TorchProfile:
    def __init__(self, out):
        self.out, self.prof, self.undo = out, None, []

    def _wrap(self, obj, attr, label):
        fn = getattr(obj, attr, None)
        if fn is None:
            return

        def wrapped(*a, **k):
            with torch.autograd.profiler.record_function(f'## {label}'):
                return fn(*a, **k)
        setattr(obj, attr, wrapped)
        self.undo.append((obj, attr, fn))

    def start(self, runner):
        if runner is not None:
            self._wrap(getattr(runner, 'world_model_learner', None), 'update', 'wm_update')
            self._wrap(getattr(runner, 'policy_learner', None), 'update', 'policy_update')
            self._wrap(getattr(runner, 'env', None), 'step', 'env_step')
        from torch.profiler import ProfilerActivity, profile
        self.prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False,
                            with_stack=False, profile_memory=False)
        self.prof.__enter__()

    def stop(self, span_s, iterations):
        self.prof.__exit__(None, None, None)
        for obj, attr, fn in self.undo:
            setattr(obj, attr, fn)
        kernels, annotations, intervals, ops = {}, {}, [], {}
        for e in self.prof.events():
            dev = str(getattr(e, 'device_type', ''))
            annotation = getattr(e, 'is_user_annotation', False) or e.name.startswith(ANNOTATION_PREFIXES)
            if dev.endswith('CUDA') and not annotation:
                k = kernels.setdefault(e.name, [0.0, 0])
                k[0] += e.time_range.elapsed_us()
                k[1] += 1
                intervals.append((e.time_range.start, e.time_range.end))
            elif e.name.startswith('## ') and not dev.endswith('CUDA'):
                a = annotations.setdefault(e.name[3:], [0.0, 0, 0.0])
                a[0] += e.device_time_total
                a[1] += 1
                a[2] += e.cpu_time_total
            elif e.name.startswith('aten::') or e.name.startswith('cuda'):
                o = ops.setdefault(e.name, [0.0, 0])
                o[0] += e.self_cpu_time_total
                o[1] += 1
        intervals.sort()
        busy, cur = 0.0, None
        for s, t in intervals:
            if cur is None or s > cur[1]:
                if cur:
                    busy += cur[1] - cur[0]
                cur = [s, t]
            else:
                cur[1] = max(cur[1], t)
        if cur:
            busy += cur[1] - cur[0]
        total_kernel = sum(v[0] for v in kernels.values())
        summary = {
            'span_s': span_s, 'iterations': iterations,
            'gpu_busy_s': busy / 1e6, 'gpu_busy_fraction': (busy / 1e6) / span_s if span_s else None,
            'kernel_time_s': total_kernel / 1e6, 'kernel_launches': sum(v[1] for v in kernels.values()),
            'kernels': [{'name': n[:300], 'total_ms': round(v[0] / 1000, 3), 'calls': v[1],
                         'share_of_kernel_time': round(v[0] / total_kernel, 4) if total_kernel else None}
                        for n, v in sorted(kernels.items(), key=lambda kv: -kv[1][0])[:80]],
            'annotations': {n: {'cuda_ms': round(v[0] / 1000, 2), 'calls': v[1], 'cpu_ms': round(v[2] / 1000, 2)}
                            for n, v in sorted(annotations.items())},
            'cpu_ops_self_time': [{'name': n, 'self_cpu_ms': round(v[0] / 1000, 2), 'calls': v[1]}
                                  for n, v in sorted(ops.items(), key=lambda kv: -kv[1][0])[:40]],
        }
        Path(f'{self.out}.torchprof.json').write_text(json.dumps(summary, indent=1) + '\n')
        self.prof.export_chrome_trace(f'{self.out}.trace.json')


class IterTimer:
    def __init__(self, warmup, window, repeats, profile=None, profile_iters=10, out=None):
        self.warmup, self.window, self.repeats, self.profile = warmup, window, repeats, profile
        self.marks = {warmup + k * window: k for k in range(repeats + 1)}   # iteration index -> edge k
        self.last = max(self.marks)
        if profile:
            self.marks = {warmup: 0}
            self.last = warmup + profile_iters
        self.profile_iters = profile_iters
        self.it = 0                     # index of the iteration that begins now
        self.edges = {}
        self.first_iter_s = None
        self.starts = []
        self.rng = []
        self.torchprof = TorchProfile(out) if profile == 'torch' else None
        self.profile_t0 = None

    def begin(self, runner):
        """Called when SuperMimicRunner.step() starts iteration self.it."""
        i = self.it
        self.starts.append(time.perf_counter())
        self.rng.append(rng_fingerprint())
        if i == 1:
            torch.cuda.synchronize()
            self.first_iter_s = time.perf_counter() - T_START
        if self.profile:
            if i == self.warmup:
                torch.cuda.synchronize()
                self.profile_t0 = time.perf_counter()
                self.edges[0] = (self.profile_t0, time.time(), compile_count())
                if self.profile == 'nsys':
                    torch.cuda.cudart().cudaProfilerStart()
                else:
                    self.torchprof.start(runner)
            elif i == self.last:
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                self.edges[1] = (t1, time.time(), compile_count())
                if self.profile == 'nsys':
                    torch.cuda.cudart().cudaProfilerStop()
                else:
                    self.torchprof.stop(t1 - self.profile_t0, self.profile_iters)
                raise StopTiming()
        else:
            k = self.marks.get(i)
            if k is not None:
                torch.cuda.synchronize()
                self.edges[k] = (time.perf_counter(), time.time(), compile_count())
                if i == self.last:
                    raise StopTiming()
        if torch.cuda.is_initialized():
            torch.cuda.nvtx.range_push(f'iter {i}')
            torch.cuda.nvtx.range_pop()
        self.it += 1

    def report(self):
        windows = []
        n = 1 if self.profile else self.repeats
        size = self.profile_iters if self.profile else self.window
        for k in range(n):
            if k not in self.edges or k + 1 not in self.edges:
                continue
            a, b = self.edges[k], self.edges[k + 1]
            windows.append({'k': k, 'iterations': size, 'seconds': b[0] - a[0], 'wall_start': a[1],
                            'wall_end': b[1], 'compiles_in_window': b[2] - a[2]})
        return windows


def main():
    if '--' not in sys.argv:
        raise SystemExit('usage: timing.py --tree <tree> --out <prefix> [options] -- <train.py arguments>')
    split = sys.argv.index('--')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tree', required=True)
    p.add_argument('--out', required=True, help='output prefix; writes <prefix>.r0.json')
    p.add_argument('--warmup', type=int, default=300)
    p.add_argument('--window', type=int, default=300)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--profile', choices=('nsys', 'torch'))
    p.add_argument('--profile-iters', type=int, default=10)
    args = p.parse_args(sys.argv[1:split])
    train_argv = sys.argv[split + 1:]

    tree = Path(args.tree).resolve()
    sys.path.remove(str(BENCH))
    for d in reversed(recipe.SRC_DIRS):
        sys.path.insert(0, str(tree / d))
    sys.path.insert(0, str(BENCH / 'wandb_stub'))
    timer = IterTimer(args.warmup, args.window, args.repeats, args.profile, args.profile_iters, args.out)
    from supermimic.runners import supermimic_runner as runner_module
    original_step = runner_module.SuperMimicRunner.step

    def step_hook(self, *a, **k):
        timer.begin(self)
        return original_step(self, *a, **k)
    runner_module.SuperMimicRunner.step = step_hook

    entry = tree / recipe.ENTRY
    sys.argv = [str(entry), *train_argv]
    status, error = 'incomplete', None
    try:
        runpy.run_path(str(entry), run_name='__main__')
    except StopTiming:
        status = 'ok'
    except BaseException as exc:  # report, then exit nonzero
        import traceback
        traceback.print_exc()
        status, error = 'error', f'{type(exc).__name__}: {exc}'
    result = {
        'status': status, 'error': error, 'rank': 0,
        'warmup': args.warmup, 'window': args.window, 'repeats': args.repeats, 'profile': args.profile,
        'iterations_seen': timer.it, 'startup_to_second_iteration_s': timer.first_iter_s,
        'peak_memory_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
        'peak_memory_reserved_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
        'windows': timer.report() if status == 'ok' else [],
        'iteration_start_s': [round(t - T_START, 4) for t in timer.starts],
        'rng_fingerprints': timer.rng,
        'train_argv': train_argv,
        'math_flags_at_end': precision_flags(),
    }
    Path(f'{args.out}.r0.json').write_text(json.dumps(result, indent=1) + '\n')
    sys.stdout.flush()
    sys.stderr.flush()
    kill_descendants()
    os._exit(0 if status == 'ok' else 1)


if __name__ == '__main__':
    main()
