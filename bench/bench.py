"""Official SuperMimic (supermimic + mjlab_research tree; SuperMimic-GenCoord-G1, 40 LAFAN1 clips, one GPU) training
bench: one command per question. The gates and their rules are described in GOAL.md; this is the command summary.

Run from the workspace root with the environment's Python. GPU runs take the GPU lock (bench.json gpu_lock)
themselves, one at a time; do NOT wrap these commands in flock (that would deadlock).

  python bench/bench.py check   --tree <dir> --out <dir> [--config <file>] [--quick | --skip-timing | --n1]
        the verdict -> <out>/check.json
          (no flag)      gates 0, 1, 2, 3; gate 1 times the candidate once against the frozen reference timing
          --quick        gates 0 and 2 (exact items and records), no timing, no trajectory
          --skip-timing  gates 0, 2, 3
          --n1           gates 0 and 3 only (after a check whose gate 2 was not bitwise identical)
  python bench/bench.py time    --tree <dir> --out <dir> [--config <file>]
        one timing run of the tree (2 windows of 300 iterations after 300 warmup iterations), with its speed relative
        to the frozen reference timing -> <out>/time.json
  python bench/bench.py profile --tree <dir> --out <dir> [--config <file>] [--iters N]
        torch.profiler over N iterations (default 10) after the warmup: GPU busy fraction, kernel launches, top
        kernels, kernel categories, time inside env step / world-model update / policy update, and the kernel
        utilization report (skills/kernel-utilization-analysis) -> <out>/profile.json, profile.md, kernel_util.md,
        a Chrome trace

  gate 0  coverage: supermimic/src/supermimic/scripts/train.py, and SuperMimicRunner.run() outside the body of its
          `for step in range(...)` loop, equal the reference AST, else ERROR.
  gate 1  benefit: real entrypoint, cold compile caches, fingerprints, RNG stream equal to the frozen reference's on
          the iterations before the first update; speedup <= max(1.02, 1 + 2 x reference spread) -> UNAVAILABLE.
  gate 2  records: 5 updates from the frozen state (bench/sm_adapter.py), exact items else INCORRECT; all tensors
          bitwise -> PASS (N0).
  gate 3  statistical: 800 iterations from scratch on the first frozen seed, endpoints of bench/traj/spec.json within
          3 x the reference's seed sigma -> PASS (N1), else INCORRECT.

Every GPU run holds the GPU lock, waits until the visible GPU is empty and idle (INVALID after 10 minutes), runs
the code from a fresh copy of the tree in its own process group with a timeout, a fresh TMPDIR and fresh compile
caches, and kills what it started afterwards. Every command checks the frozen inputs (task/formulation.json) before
and after, and the source digest of the tree copy before and after. Candidate switches: a JSON file
{"environment": {"NAME": "value"}} given with --config; variables the bench manages (bench.json reserved_env) are
refused.
"""
import argparse
import ast
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent                       # workspace


def _expand(value):
    """bench.json and task/formulation.json name machine paths through ${ASSETS} (the task's read-only assets)."""
    if isinstance(value, str) and '$' in value:
        out = os.path.expandvars(value)
        if '$' in out:
            raise SystemExit(f'{value}: set ASSETS (and any other variable it names) before running the bench')
        return out
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


CFG = _expand(json.loads((BENCH / 'bench.json').read_text()))
sys.path.insert(0, str(BENCH))
import recipe  # noqa: E402

LOCK = Path(CFG['gpu_lock'])
PY = Path(CFG['python'])
TIERS = ['N0', 'N1', 'N2', 'N3', 'N4', 'N5']
SCRATCH = Path(CFG.get('scratch', '/tmp'))
TIMEOUT = CFG['timeouts_s']
RESERVED = set(CFG['reserved_env'])
SIGNATURES = [(re.compile(x['regex']), x['reason']) for x in CFG.get('invalid_run_signatures', [])]
COST = {'gpu_seconds': 0.0, 'gpu_runs': 0}
TREE_SKIP = {'.git', '__pycache__', '.venv'}
TOP_SKIP = {'logs', 'outputs', 'media', 'data'}   # run outputs and data next to the code, top level only
TAG_VAR = 'SM_BENCH_RUN'
FINAL_MARKER = ROOT / 'task' / 'final-evaluation'


def path(key):
    p = Path(_expand(key))
    return p if p.is_absolute() else (ROOT / p).resolve()


def log(msg):
    print(f'[bench {datetime.now():%H:%M:%S}] {msg}', flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def evaluator_sha256():
    """Joint hash of the frozen evaluator: numerics/, bench/ code and config, frozen records' manifests, task/."""
    files = sorted(p for p in (ROOT / 'numerics').rglob('*.py'))
    files += sorted(p for p in BENCH.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                    and p.suffix in ('.py', '.json') and 'ref' not in p.parts)
    files += sorted(p for p in (BENCH / 'records').rglob('*.json'))
    files += sorted(p for p in (ROOT / 'task').rglob('*') if p.is_file() and p != FINAL_MARKER)
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(ROOT)).encode() + b'\0' + sha256_file(p).encode() + b'\n')
    return h.hexdigest()


# ---------------------------------------------------------------- GPU runs

class GpuLock:
    def __init__(self, what, timeout=8 * 3600):
        self.what, self.timeout = what, timeout

    def __enter__(self):
        self.f = open(LOCK, 'a+')
        t0, said = time.time(), False
        while True:
            try:
                fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not said:
                    log(f'waiting for {LOCK} ({self.what}); if this command is wrapped in flock, unwrap it')
                    said = True
                if time.time() - t0 > self.timeout:
                    raise SystemExit(f'could not get {LOCK} within {self.timeout}s for {self.what}')
                time.sleep(5)
        self.t0 = time.time()
        log(f'GPU lock held: {self.what}')
        return self

    def __exit__(self, *_):
        COST['gpu_seconds'] += (time.time() - self.t0) * CFG['world_size']
        COST['gpu_runs'] += 1
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def cost():
    return {'gpu_hours': round(COST['gpu_seconds'] / 3600, 3), 'gpu_runs': COST['gpu_runs'],
            'method': f'GPU lock hold time x {CFG["world_size"]} card (startup, warmup and failed runs included)'}


def scratch_dir(prefix):
    free = shutil.disk_usage(SCRATCH).free / 2**30
    if free < CFG.get('min_scratch_free_gib', 20):
        raise SystemExit(f'{SCRATCH} has only {free:.1f} GiB free (< {CFG.get("min_scratch_free_gib", 20)}); '
                         f'clean up before measuring')
    return tempfile.TemporaryDirectory(prefix=prefix, dir=SCRATCH)


def gpu_state():
    q = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=60)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    keep = None if not visible else {v.strip() for v in visible.split(',') if v.strip()}
    out = []
    for line in q.stdout.strip().splitlines():
        idx, uid, mem, util = [x.strip() for x in line.split(',')]
        if keep is not None and idx not in keep and uid not in keep:
            continue
        out.append({'index': int(idx), 'uuid': uid, 'memory_mib': int(mem), 'util': int(util)})
    return out


def wait_idle(what, patience=600):
    t0 = time.time()
    while True:
        state = gpu_state()
        busy = [g for g in state if g['memory_mib'] > 1024 or g['util'] > 5]
        if not busy:
            return state, None
        if time.time() - t0 > patience:
            return state, f'GPUs busy for {patience}s before {what}: {busy}'
        time.sleep(15)


def _tagged(tag):
    me, pids = os.getuid(), []
    mark = f'{TAG_VAR}={tag}'.encode()
    for d in Path('/proc').iterdir():
        if not d.name.isdigit() or int(d.name) == os.getpid():
            continue
        try:
            if d.stat().st_uid == me and mark in (d / 'environ').read_bytes().split(b'\0'):
                pids.append(int(d.name))
        except OSError:
            continue
    return pids


def _kill_tagged(tag):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        pids = _tagged(tag)
        if not pids:
            return 0
        for p in pids:
            try:
                os.kill(p, sig)
            except ProcessLookupError:
                pass
        time.sleep(5)
    return len(_tagged(tag))


def base_env(tmp, cand_env=None, pythonpath=None):
    """Environment of every bench-launched process: the caller's environment minus reserved variables, then the
    candidate's switches, then what the bench manages (scratch, caches, offline, devices)."""
    env = {k: v for k, v in os.environ.items() if k not in RESERVED and not k.startswith('SM_BENCH')}
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        env['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
    env.update(cand_env or {})
    caches = Path(tmp) / 'caches'
    env.update({
        'TMPDIR': str(tmp), 'TMP': str(tmp), 'TEMP': str(tmp),
        'TORCHINDUCTOR_CACHE_DIR': str(caches / 'inductor'), 'TRITON_CACHE_DIR': str(caches / 'triton'),
        'CUDA_CACHE_PATH': str(caches / 'cuda'), 'XDG_CACHE_HOME': str(caches / 'xdg'),
        'WARP_CACHE_PATH': str(caches / 'warp'), 'MPLCONFIGDIR': str(caches / 'mpl'),
        'HF_HUB_OFFLINE': '1', 'WANDB_MODE': 'disabled', 'PYTHONNOUSERSITE': '1', 'MUJOCO_GL': 'egl',
        'PYTHONUNBUFFERED': '1',
        'PATH': f'{PY.parent}:{env.get("PATH", "/usr/local/bin:/usr/bin:/bin")}',
    })
    env.pop('PYTHONPATH', None)
    if pythonpath:
        env['PYTHONPATH'] = ':'.join(str(p) for p in pythonpath)
    for v in ('inductor', 'triton', 'cuda', 'xdg', 'warp', 'mpl'):
        (caches / v).mkdir(parents=True, exist_ok=True)
    return env


def run_gpu(cmd, cwd, env, log_file, what, timeout):
    """One GPU job under the lock: idle check, own process group, timeout, leftovers killed. Returns a dict with
    status (ok / TIMEOUT / INVALID), returncode, wall seconds and the GPU state before and after."""
    tag = uuid.uuid4().hex[:12]
    with GpuLock(what):
        before, problem = wait_idle(what)
        if problem:
            return {'status': 'INVALID', 'reason': problem, 'returncode': None, 'gpus_before': before}
        env = dict(env, **{TAG_VAR: tag})
        if CFG.get('cpu_affinity'):        # the same CPUs for every run alike
            cmd = ['taskset', '-c', CFG['cpu_affinity'], *cmd]
        t0 = time.time()
        status = None
        with open(log_file, 'w') as lf:
            proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                rc, status = None, 'TIMEOUT'
        wall = time.time() - t0
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        left = _kill_tagged(tag)
        after, _ = wait_idle(f'{what} (after)', patience=60)
    res = {'status': status or 'ok', 'returncode': rc, 'wall_s': round(wall, 1), 'gpus_before': before,
           'gpus_after': after}
    if status == 'TIMEOUT':
        res['reason'] = f'{what} exceeded {timeout}s; process group killed; see {Path(log_file).name}'
    if left:
        res['warning'] = f'{left} processes of the run survived SIGKILL'
    if any(g['memory_mib'] > 1024 for g in after or []):
        res['warning'] = (res.get('warning', '') + ' GPU memory still in use 60s after the run').strip()
    return res


def signatures(log_file):
    hits, text = [], Path(log_file).read_text(errors='replace') if Path(log_file).exists() else ''
    for pattern, reason in SIGNATURES:
        m = pattern.search(text)
        if m:
            line = text[text.rfind('\n', 0, m.start()) + 1:text.find('\n', m.end())]
            hits.append({'reason': reason, 'line': line.strip()[:300]})
    return hits


# ---------------------------------------------------------------- inputs and sources

def tree_digest(tree):
    h = hashlib.sha256()
    tree = Path(tree)
    for f in sorted(tree.rglob('*')):
        rel = f.relative_to(tree)
        if any(part in TREE_SKIP for part in rel.parts) or rel.parts[0] in TOP_SKIP or f.is_symlink() \
                or not f.is_file():
            continue
        h.update(str(rel).encode() + b'\0' + sha256_file(f).encode() + b'\n')
    return h.hexdigest()


def file_fingerprint(p, kind):
    if kind == 'sha256':
        return sha256_file(p)
    st = Path(p).stat()
    return f'{st.st_size}:{st.st_mtime_ns}'


def verify_inputs(when):
    """Every frozen input file of task/formulation.json (the LAFAN1 npz clips): sha256."""
    spec = json.loads((ROOT / 'task' / 'formulation.json').read_text())['inputs']
    t0, bad = time.time(), {}
    for rel, want in spec.items():
        p = path(rel)
        got = file_fingerprint(p, want['kind']) if p.exists() else None
        if got != want['value']:
            bad[rel] = {'want': want['value'], 'got': got}
    log(f'inputs {when}: {len(spec) - len(bad)}/{len(spec)} match the formulation ({time.time() - t0:.0f}s)')
    return {'when': when, 'ok': not bad, 'mismatches': bad, 'checked': len(spec), 'seconds': round(time.time() - t0, 1)}


def freeze_inputs(paths, big=400 << 20):
    """Formulator: fingerprints of every file under the given paths."""
    out = {}
    for root in paths:
        root = Path(root)
        files = [root] if root.is_file() else sorted(f for f in root.rglob('*') if f.is_file())
        for f in files:
            kind = 'stat' if f.stat().st_size > big else 'sha256'
            out[str(f)] = {'kind': kind, 'value': file_fingerprint(f, kind)}
    return out


def check_reference():
    ref = path(CFG['reference'])
    git = ['git', '-c', 'safe.directory=*', '-C', str(ref)]
    dirty = subprocess.run([*git, 'status', '--porcelain', '--untracked-files=no'], capture_output=True,
                           text=True).stdout.strip()
    head = subprocess.run([*git, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    if dirty or head != CFG['reference_commit']:
        raise SystemExit(f'reference tree is not the frozen upstream commit (head {head}, dirty: {bool(dirty)})')
    return ref


def materialize(src, dest):
    top = Path(src).resolve()

    def ignore(directory, names):
        at_top = Path(directory).resolve() == top
        return [n for n in names if n in TREE_SKIP or n.endswith('.pyc') or (at_top and n in TOP_SKIP)]
    shutil.copytree(src, dest, ignore=ignore, symlinks=False)
    for d in [Path(dest), *(p for p in Path(dest).rglob('*') if p.is_dir())]:
        d.chmod(d.stat().st_mode | 0o700)
    return Path(dest)


def candidate_environment(config_file):
    if not config_file:
        return {}
    data = json.loads(Path(config_file).read_text())
    extra = sorted(set(data) - {'environment'})
    if extra:
        raise SystemExit(f'{config_file}: only an "environment" mapping is accepted, found {extra}')
    env = {str(k): str(v) for k, v in (data.get('environment') or {}).items()}
    bad = sorted(k for k in env if k in RESERVED or k.startswith('SM_BENCH'))
    if bad:
        raise SystemExit(f'{config_file}: {bad} are managed by the bench and cannot be candidate switches')
    return env


def versions():
    out = {'python': sys.version.split()[0]}
    for pkg in ('torch', 'triton', 'warp-lang', 'mujoco', 'mujoco-warp', 'mjlab', 'numpy'):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


FINGERPRINT_KEYS = ('argv', 'math_mode', 'gpus', 'compile_cache', 'instrumentation', 'versions', 'timing_plan')


def fingerprint_diff(a, b):
    if a is None or b is None:
        return ['missing fingerprint']
    return [k for k in FINGERPRINT_KEYS if a.get(k) != b.get(k)]


# ---------------------------------------------------------------- gate 0: coverage

GUARDED_MODULES = ('supermimic/src/supermimic/scripts/train.py',)
LOOP_FILE = 'supermimic/src/supermimic/runners/supermimic_runner.py'


def setup_fingerprint(tree):
    """AST of the entry script, the configuration module, and Trainer.run with its `while` body blanked."""
    tree = Path(tree)
    parts = {rel: ast.dump(ast.parse((tree / rel).read_text()), include_attributes=False) for rel in GUARDED_MODULES}
    module = ast.parse((tree / LOOP_FILE).read_text())
    loops = 0
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == 'SuperMimicRunner':
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == 'run':
                    for stmt in fn.body:
                        if isinstance(stmt, ast.For):
                            stmt.body = [ast.Pass()]
                            loops += 1
                    parts['SuperMimicRunner.run'] = ast.dump(fn, include_attributes=False)
    return parts, loops


def setup_guard(ref, tree):
    want, _ = setup_fingerprint(ref)
    try:
        got, loops = setup_fingerprint(tree)
    except (OSError, SyntaxError) as exc:
        return [f'entry, config or training loop unreadable: {exc}']
    problems = []
    if loops != 1:
        problems.append(f'expected exactly one top-level `for` loop in SuperMimicRunner.run(), found {loops}')
    for name in (*GUARDED_MODULES, 'SuperMimicRunner.run'):
        if got.get(name) != want.get(name):
            problems.append(f'{name} differs from the reference' +
                            (' outside the body of its for loop' if name == 'SuperMimicRunner.run' else ''))
    return problems


def coverage(tree):
    """Gate 0 (CPU): setup guard. Returns (status, correctness) or None when it holds."""
    ref_src = check_reference()
    problems = setup_guard(ref_src, tree)
    if problems:
        return 'ERROR', {'passed': False, 'stage': 'setup_guard', 'problems': problems,
                         'note': 'the numerics adapter runs the reference setup (entry script, TrainConfig, the part '
                                 'of Trainer.run outside its while loop) and then the tree\'s loop body; move the change '
                                 'into code the trainer calls (Trainer.__init__, the trainers, models, env, buffer) '
                                 'or into the loop body'}
    return None


# ---------------------------------------------------------------- gate 2: bitwise records

def adapter_kwargs(tree, scratch, variant=None):
    kw = {'repo': str(tree), 'npz_dir': str(path(CFG['fixture']['npz_dir'])), 'scratch': str(scratch),
          'reference': str(path(CFG['reference']))}
    if variant:
        kw['variant'] = variant
    return kw


def record_cases(tree, rec, cases, logs_dir, cand_env, label, variant=None, losses_only=False):
    """One `numerics record` process per case. Returns None on success, else an error dict."""
    n = CFG['numerics']
    for case in cases:
        with scratch_dir('sm-r-') as tmp:
            cases_file = Path(tmp) / 'case.json'
            cases_file.write_text(json.dumps([_expand(case)]))
            env = base_env(tmp, cand_env, pythonpath=[ROOT])
            # deterministic mode also for compiled code: inductor skips on-device benchmarking (autotuning) whose
            # choice of kernel configuration changes the numerics from one process to the next
            env['TORCHINDUCTOR_DETERMINISTIC'] = '1'
            # and for TorchScript: mjlab's quaternion math is torch.jit.script; the profiling executor runs a function
            # unfused first and fused afterwards (different rounding), and re-profiles when shapes change
            env['PYTORCH_JIT'] = '0'
            cmd = [str(PY), '-m', 'numerics', 'record', '--adapter', n['adapter'], '--adapter-kwargs',
                   json.dumps(adapter_kwargs(tree, Path(tmp) / 'train', variant)), '--cases', str(cases_file),
                   *(['--losses-only'] if losses_only else ['--sketch-dim', str(n['sketch_dim'])]),
                   # no --dtype-trace: under its dispatch mode compiled regions fall back to eager
                   '--math-mode', CFG['math_mode'], '--deterministic', '--out', str(rec)]
            log_file = Path(logs_dir) / f'{label}_{case["name"]}.log'
            r = run_gpu(cmd, ROOT, env, log_file, f'{label}: case {case["name"]}',
                        TIMEOUT['trajectory' if losses_only else 'numerics'])
        if r['status'] != 'ok' or r['returncode']:
            return {'status': 'INVALID' if r['status'] == 'INVALID' else 'ERROR',
                    'reason': r.get('reason') or f'record exited {r["returncode"]}; see {log_file.name}'}
    return None


def missing_tensor_failures(report):
    """Judge errors that are really evidence of different work, turned into exact failures: a step whose tensor set
    is missing reference tensors and adds none of its own did not do the work that produced them."""
    out = []
    for err in report.get('errors') or []:
        if isinstance(err, dict) and err.get('missing') and not err.get('unexpected'):
            out.append({'where': err['where'], 'check': 'missing_tensors',
                        'note': 'the candidate produced none of these tensors (e.g. no gradient for parameters the '
                                'reference computes one for, or no step of an optimizer): it did not do that work',
                        'missing': err['missing'][:20]})
    return out


def numerics(tree, out, cand_env):
    n = CFG['numerics']
    rec = out / 'records'
    shutil.rmtree(rec, ignore_errors=True)
    cases = json.loads(path(n['cases']).read_text())
    failed = record_cases(tree, rec, cases, out, cand_env, 'record')
    if failed:
        return failed
    report = out / 'compare.json'
    env = base_env(out, pythonpath=[ROOT])
    r = subprocess.run([str(PY), '-m', 'numerics', 'compare', '--ref', str(path(n['ref'])), '--cand', str(rec),
                        '--thresholds', str(path(n['thresholds'])),
                        *[a for x in n.get('exclude', []) for a in ('--exclude', f'{x["regex"]}={x["reason"]}')],
                        '--out', str(report)], cwd=ROOT, env=env, capture_output=True, text=True)
    (out / 'compare.log').write_text(r.stdout + r.stderr)
    shutil.rmtree(out / 'caches', ignore_errors=True)
    if not report.exists():
        return {'status': 'ERROR', 'reason': f'compare exited {r.returncode}; see compare.log'}
    data = json.loads(report.read_text())
    if not n.get('keep_records'):
        shutil.rmtree(rec, ignore_errors=True)
    data['exact_failures'] = (data.get('exact_failures') or []) + missing_tensor_failures(data)
    summary = {k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in data.items()
               if k not in ('status', 'measured_tier')}
    bitwise = (data.get('checked') == data.get('bitwise_identical') and not data.get('dtype_changes')
               and not data.get('precision_mode_changes') and data.get('bitwise_trusted', False))
    return {'status': data.get('status'), 'measured_tier': data.get('measured_tier'), 'report': str(report),
            'errors': len(data.get('errors') or []), 'exact_failures': data.get('exact_failures') or [],
            'bitwise': bitwise, 'summary': summary}


# ---------------------------------------------------------------- gate 3: trajectory

def traj_case(spec, seed):
    return dict(spec['case'], name=f'traj_s{seed}', seed=seed)


def trajectories(tree, out, cand_env):
    """Gate 3: the candidate's trajectory on the first frozen reference seed (trajectory.seeds of them) against the
    bias bound sigma_k x the frozen seed sigma per endpoint."""
    tj = CFG['trajectory']
    spec = json.loads(path(tj['spec']).read_text())
    frozen_path = path(tj['frozen'])
    seeds = json.loads(frozen_path.read_text())['reference_seeds'][:tj['seeds']]
    rec = out / 'trajectory'
    shutil.rmtree(rec, ignore_errors=True)
    failed = record_cases(tree, rec, [traj_case(spec, s) for s in seeds], out, cand_env, 'traj', losses_only=True)
    if failed:
        return {'verdict': 'error', **failed}
    report_path = out / 'traj_report.json'
    env = base_env(out, pythonpath=[ROOT])
    r = subprocess.run([str(PY), '-m', 'numerics', 'traj-compare', '--frozen', str(frozen_path), '--cand', str(rec),
                        '--sigma-k', str(tj['sigma_k']), '--out', str(report_path)],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    shutil.rmtree(out / 'caches', ignore_errors=True)
    if not report_path.exists():
        return {'verdict': 'error', 'reason': f'traj-compare produced no report: {r.stderr[-1500:]}'}
    report = json.loads(report_path.read_text())
    report['report'] = str(report_path)
    return report


# ---------------------------------------------------------------- gate 1: timing

def time_tree(tree, out, label, cand_env=None, profile=None, profile_updates=4):
    t = CFG['timing']
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / f'timing_{label}'
    for f in out.glob(f'timing_{label}.r*.json'):
        f.unlink()
    with scratch_dir('sm-t-') as tmp:
        train_args = recipe.argv(path(CFG['fixture']['npz_dir']))
        hook_args = ['--warmup', str(t['warmup']), '--window', str(t['window']), '--repeats', str(t['repeats']),
                     *(['--profile', profile, '--profile-iters', str(profile_updates)] if profile else [])]
        argv = [str(PY), str(BENCH / 'timing.py'), '--tree', str(tree), '--out', str(prefix), *hook_args, '--',
                *train_args]
        stable_argv = ['python', 'timing.py', *hook_args, '--', *[x.replace(str(tmp), '<scratch>') for x in train_args]]
        env = base_env(tmp, cand_env)
        if profile == 'nsys':
            nsys = shutil.which('nsys') or CFG.get('nsys')
            if not nsys or not Path(nsys).exists():
                raise SystemExit('nsys not found')
            argv = [nsys, 'profile', '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
                    '--trace=cuda,nvtx,osrt,cublas', '--cuda-graph-trace=node', '--sample=none',
                    '--cuda-memory-usage=false', '--force-overwrite=true', '-o', str(out / f'nsys_{label}'), *argv]
        log_file = out / f'timing_{label}.log'
        what = f'timing {label}{f" ({profile})" if profile else ""}'
        work = Path(tmp) / 'work'
        work.mkdir()
        r = run_gpu(argv, work, env, log_file, what, TIMEOUT['profile' if profile else 'timing'])
    plan = [t['warmup'], t['window'], t['repeats']] if not profile else [t['warmup'], profile_updates, 1]
    fingerprint = {'argv': stable_argv, 'candidate_environment': cand_env or {},
                   'math_mode': CFG['math_mode'], 'gpus': [(g['index'], g['uuid']) for g in r.get('gpus_before') or []],
                   'compile_cache': 'cold (fresh per run)', 'instrumentation': profile or 'none',
                   'timing_plan': plan, 'versions': versions()}
    run = {'run_status': r['status'], 'wall_s': r.get('wall_s'), 'fingerprint': fingerprint, 'log': str(log_file),
           **({'warning': r['warning']} if r.get('warning') else {})}
    if r['status'] != 'ok':
        return {'status': r['status'], 'reason': r['reason'], **run}
    f = Path(f'{prefix}.r0.json')
    rank = json.loads(f.read_text()) if f.exists() else None
    if r['returncode'] or rank is None or rank['status'] != 'ok':
        err = rank.get('error') if rank else None
        return {'status': 'ERROR', 'reason': f'timing exited {r["returncode"]}; {err}; see {log_file.name}', **run}
    hits = signatures(log_file)
    if hits:
        return {'status': 'INVALID', 'reason': 'invalid-run signature in the log: ' + hits[0]['reason'],
                'signatures': hits, **run}
    return {**summarize_timing(rank, r['wall_s'], profile, profile_updates), **run, '_rank': rank}


def summarize_timing(rank, wall, profile=None, profile_updates=4):
    t = CFG['timing']
    repeats = 1 if profile else t['repeats']
    size = profile_updates if profile else t['window']
    problems = []
    ws = [w for w in rank['windows'] if w['k'] < repeats]
    if len(ws) != repeats:
        problems.append(f'{len(ws)} of {repeats} windows')
    for w in ws:
        if w.get('compiles_in_window'):
            problems.append(f'window {w["k"]}: torch.compile compiled {w["compiles_in_window"]} graphs inside the '
                            f'timing window (still compiling in steady state)')
    secs = [w['seconds'] for w in ws]
    total = sum(secs) if secs else None
    frozen = None
    rng_note = None
    ref_file = ROOT / 'task' / 'timing_ref.json'
    if not profile and ref_file.exists():
        frozen = json.loads(ref_file.read_text())
        # Only the iterations before the first update are compared exactly: from then on the number of random draws
        # per iteration follows the simulation (resets of terminated envs), which MuJoCo Warp does not reproduce from
        # run to run, so even two reference runs part after a few hundred iterations. Later divergence is a note.
        n_check = t['rng_check_iterations']
        want = frozen['rng_fingerprints']
        got = rank['rng_fingerprints']
        k = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
        rng_note = None if k is None else k
        if k is not None and k < n_check:
            problems.append(f'the random-number stream differs from the reference at iteration {k}, before the first '
                            f'update (random numbers drawn in a different amount or order change the work)')
    res = {
        'status': 'ok' if not problems and secs else 'ERROR', 'problems': problems,
        'rng_first_divergence_from_frozen_reference': rng_note,
        'window_s': [round(s, 3) for s in secs], 'total_s': round(total, 3) if total else None,
        'iterations_timed': size * len(secs),
        's_per_iteration': round(total / (size * len(secs)), 4) if secs else None,
        's_per_update': round(total / (size * len(secs)) / recipe.UTD, 4) if secs else None,
        'peak_memory_allocated_gib': round(rank['peak_memory_allocated_bytes'] / 2**30, 2),
        'peak_memory_reserved_gib': round(rank['peak_memory_reserved_bytes'] / 2**30, 2),
        'startup_to_second_iteration_s': round(rank['startup_to_second_iteration_s'] or 0, 1),
        'process_wall_s': round(wall, 1),
        'math_flags_at_end': rank.get('math_flags_at_end'),
    }
    if len(secs) == 2:
        res['window_ratio'] = round(secs[0] / secs[1], 4)
        if frozen:
            res['window_ratio_vs_frozen'] = round(res['window_ratio'] / frozen['window_ratio'] - 1, 4)
    return res


def strip(run):
    return {k: v for k, v in run.items() if k != '_rank'} if run else run


def benefit(tree, out, cand_env, final=False):
    t = CFG['timing']
    frozen = json.loads((ROOT / 'task' / 'timing_ref.json').read_text())
    if not final:
        cand = time_tree(tree, out, 'cand', cand_env=cand_env)
        runs = {'cand': strip(cand)}
        if cand['status'] != 'ok':
            status = 'INVALID' if cand['status'] in ('INVALID', 'TIMEOUT') else 'ERROR'
            return status, {'speedup': None, 'runs': runs, 'reason': cand.get('reason') or
                            '; '.join(cand.get('problems') or [])}
        diffs = [d for d in fingerprint_diff(frozen['fingerprint'], cand['fingerprint']) if d != 'gpus']
        if diffs:
            return 'ERROR', {'speedup': None, 'runs': runs, 'fingerprint_diffs': diffs,
                             'reason': f'not comparable with the frozen reference timing: fingerprint differs in {diffs}'}
        perf = {'mode': 'check (candidate once against the frozen reference timing)',
                'reference_s_per_iteration': frozen['s_per_iteration'], 'candidate_s_per_iteration': cand['s_per_iteration'],
                'speedup': round(frozen['total_s'] / cand['total_s'], 4),
                'reference_spread': frozen['spread'], 'runs': runs}
        notes = []
        dev = cand.get('window_ratio_vs_frozen')
        if dev is not None and abs(dev) > t['window_band_candidate']:
            notes.append(f'window ratio {cand["window_ratio"]} is {dev:+.1%} off the frozen reference ratio: one '
                         f'window may have been disturbed (note only; the final check re-measures)')
        perf['notes'] = notes
    else:
        ref_src = check_reference()
        with scratch_dir('sm-ref-') as tmp:
            ref_tree = materialize(ref_src, Path(tmp) / 'reference')
            ref_a = time_tree(ref_tree, out, 'ref_a')
            cand = time_tree(tree, out, 'cand', cand_env=cand_env) if ref_a['status'] == 'ok' else None
            ref_b = time_tree(ref_tree, out, 'ref_b') if cand and cand['status'] == 'ok' else None
        runs = {'ref_a': strip(ref_a), 'cand': strip(cand), 'ref_b': strip(ref_b)}
        bad = {k: v for k, v in runs.items() if v is None or v['status'] != 'ok'}
        if bad:
            status = 'INVALID' if any(v and v['status'] in ('INVALID', 'TIMEOUT') for v in bad.values()) else 'ERROR'
            return status, {'speedup': None, 'runs': runs}
        diffs = {k: [d for d in fingerprint_diff(ref_a.get('fingerprint'), v.get('fingerprint'))
                     if d != 'candidate_environment'] for k, v in runs.items() if k != 'ref_a'}
        badfp = {k: d for k, d in diffs.items() if d}
        if badfp:
            return 'ERROR', {'speedup': None, 'runs': runs, 'fingerprint_diffs': diffs,
                             'reason': f'timing runs not comparable: fingerprint differs from ref_a in {badfp}'}
        ra, rb, c = ref_a['total_s'], ref_b['total_s'], cand['total_s']
        ref = (ra + rb) / 2
        perf = {'mode': 'final (reference -> candidate -> reference)',
                'reference_s_per_iteration': round(ref / ref_a['iterations_timed'], 4),
                'candidate_s_per_iteration': cand['s_per_iteration'], 'speedup': round(ref / c, 4),
                'reference_spread': round(abs(ra - rb) / ref, 4),
                'frozen_reference_s_per_iteration': frozen['s_per_iteration'], 'runs': runs, 'fingerprint_diffs': diffs}
        problems = []
        for name, run in (('ref_a', ref_a), ('ref_b', ref_b)):
            off = run['total_s'] / frozen['total_s'] - 1
            if abs(off) > t['frozen_band']:
                problems.append(f'{name} took {run["total_s"]:.1f}s, {off:+.1%} off the frozen reference '
                                f'{frozen["total_s"]:.1f}s (> {t["frozen_band"]:.0%})')
        for name, run in (('ref_a', ref_a), ('cand', cand), ('ref_b', ref_b)):
            band = t['window_band'] if name != 'cand' else t['window_band_candidate']
            dev = run.get('window_ratio_vs_frozen')
            if dev is not None and abs(dev) > band:
                problems.append(f'{name}: window ratio {run["window_ratio"]} is {dev:+.1%} off the frozen reference '
                                f'ratio {frozen["window_ratio"]} (> {band:.0%}): one window was disturbed')
        perf['problems'] = problems
        if problems:
            return 'INVALID', perf
    bar = max(CFG['benefit']['min_speedup'], 1 + 2 * perf['reference_spread'])
    perf['benefit'] = {'speedup': perf['speedup'], 'speedup_bar': round(bar, 4),
                       'peak_memory_allocated_gib': cand['peak_memory_allocated_gib'],
                       'reference_peak_memory_allocated_gib': frozen['peak_memory_allocated_gib'],
                       'passed': perf['speedup'] > bar}
    return ('ok' if perf['benefit']['passed'] else 'UNAVAILABLE'), perf


# ---------------------------------------------------------------- verdict

def evaluate(tree, out, mode, cand_env=None):
    """Gates by mode (full / final / quick / skip_timing / n1). Returns (status, correctness, performance)."""
    cand_env = cand_env or {}
    failed = coverage(tree)
    if failed:
        return failed[0], failed[1], None
    if mode == 'n1':
        traj = trajectories(tree, out, cand_env)
        correctness = {'stage': 'statistical (n1: gates 0 and 3 only)', 'trajectory': traj}
        return traj_verdict(traj, correctness, None)
    performance = None
    if mode in ('full', 'final'):
        status, performance = benefit(tree, out, cand_env, final=mode == 'final')
        if status in ('ERROR', 'INVALID'):
            return status, {'passed': False, 'stage': 'benefit',
                            'note': (performance or {}).get('reason') or '; '.join((performance or {}).get('problems')
                                                                                 or []) or 'timing failed'}, performance
        if status == 'UNAVAILABLE':
            performance['measured_speedup'] = performance.pop('speedup')
            return 'UNAVAILABLE', {'passed': None, 'stage': 'benefit', 'verdict': 'no_benefit',
                                   'note': 'not faster than the reference: rejected without a precision evaluation'}, \
                performance
    num = numerics(tree, out, cand_env)
    tier = num.get('measured_tier')
    correctness = {'stage': 'bitwise', 'measured_tier': tier, 'numerics': num}
    if num.get('exact_failures'):
        return 'INCORRECT', dict(correctness, passed=False, verdict='structure',
                                 note='exact checks failed: the candidate does different work'), performance
    if num.get('status') == 'INVALID':
        return 'INVALID', dict(correctness, passed=False, note=num.get('reason') or 'record run invalid; measure again'), \
            performance
    if num.get('errors') or num.get('status') not in ('PASS', 'INCORRECT'):
        return 'ERROR', dict(correctness, passed=False, note='numerics could not compare the records'), performance
    if tier not in TIERS or TIERS.index(tier) > TIERS.index(CFG['agent_max_tier']):
        return 'INCORRECT', dict(correctness, passed=False, verdict='needs_signoff',
                                 note=f'measured tier {tier} exceeds {CFG["agent_max_tier"]}'), performance
    if num['bitwise']:
        return 'PASS', dict(correctness, passed=True, verdict='bitwise'), performance
    if mode == 'quick':
        return 'PASS', dict(correctness, passed=True, stage='bitwise (quick: gates 1 and 3 not run)',
                            verdict='not_bitwise', note='not bitwise identical: run check --n1 for the statistical '
                                                        'gate before relying on this candidate'), performance
    traj = trajectories(tree, out, cand_env)
    correctness.update(stage='statistical', trajectory=traj)
    return traj_verdict(traj, correctness, performance)


def traj_verdict(traj, correctness, performance):
    if traj['verdict'] == 'equivalent':
        return 'PASS', dict(correctness, passed=True, verdict='equivalent'), performance
    if traj['verdict'] == 'error':
        return ('INVALID' if traj.get('status') == 'INVALID' else 'ERROR'), \
            dict(correctness, passed=False, verdict='error'), performance
    return 'INCORRECT', dict(correctness, passed=False, verdict=traj['verdict'],
                             note=f'an endpoint is beyond {CFG["trajectory"]["sigma_k"]} x its reference seed sigma'), \
        performance


def guarded(out, fn, tree_src=None):
    """Inputs checked before (nothing measured on a mismatch) and after; the tree is copied first and its digest
    checked again at the end (the code must not write into its own tree)."""
    out.mkdir(parents=True, exist_ok=True)
    start = verify_inputs('start')
    if not start['ok']:
        return {'status': 'ERROR', 'stage': 'inputs', 'inputs': [start],
                'note': 'a frozen input differs from task/formulation.json; nothing was measured'}
    if tree_src is None:
        res = fn(None)
        digest = None
    else:
        with scratch_dir('sm-src-') as tmp:
            tree = materialize(tree_src, Path(tmp) / 'tree')
            digest = tree_digest(tree)
            try:
                res = fn(tree)
            except SystemExit as exc:
                res = {'status': 'ERROR', 'note': f'the evaluation stopped: {exc}'}
            res['source_digest'] = digest
            if tree_digest(tree) != digest:
                res['status'] = 'ERROR'
                res['note'] = 'the tree changed while it was being measured (the code wrote into its own tree)'
    end = verify_inputs('end')
    res['inputs'] = [start, end]
    if not end['ok']:
        res['status'] = 'ERROR'
        res['note'] = 'a frozen input changed during the run'
    return res


def finish(out, name, res):
    res['cost'] = cost()
    res['evaluator_sha256'] = evaluator_sha256()
    res['finished'] = datetime.now().isoformat(timespec='seconds')
    (out / name).write_text(json.dumps(res, indent=1, default=str) + '\n')
    log(f'wrote {out / name}: status {res.get("status")}')
    summary = {'status': res.get('status')}
    perf = res.get('performance')
    if isinstance(perf, dict):
        summary['speedup'] = perf.get('speedup', perf.get('measured_speedup'))
    for k in ('speedup_vs_frozen_reference', 's_per_iteration'):
        if k in res:
            summary[k] = res[k]
    print(json.dumps(summary))
    return 0 if res.get('status') in ('PASS', 'ok') else 1


def cmd_check(a):
    out = Path(a.out).resolve()
    cand_env = candidate_environment(a.config)
    if a.final and not FINAL_MARKER.exists():
        raise SystemExit('check --final is the harness\'s final evaluation (goal/clawo/judge.sh); use check')
    flags = [x for x in ('quick', 'skip_timing', 'n1', 'final') if getattr(a, x)]
    if len(flags) > 1:
        raise SystemExit(f'choose one of --quick, --skip-timing, --n1: got {flags}')
    mode = flags[0] if flags else 'full'

    def run(tree):
        status, correctness, performance = evaluate(tree, out, mode, cand_env=cand_env)
        return {'status': status, 'mode': mode, 'final': mode == 'final', 'n1': mode == 'n1',
                'quick': mode == 'quick', 'skip_timing': mode == 'skip_timing',
                'tree': str(Path(a.tree).resolve()), 'config': a.config, 'environment': cand_env,
                'correctness': correctness, 'performance': performance}
    return finish(out, 'check.json', guarded(out, run, Path(a.tree).resolve()))


def cmd_time(a):
    out = Path(a.out).resolve()
    cand_env = candidate_environment(a.config)

    def run(tree):
        res = strip(time_tree(tree, out, 'tree', cand_env=cand_env))
        ref_file = ROOT / 'task' / 'timing_ref.json'
        if ref_file.exists():
            frozen = json.loads(ref_file.read_text())
            if res['status'] == 'ok':
                res['speedup_vs_frozen_reference'] = round(frozen['total_s'] / res['total_s'], 4)
            res['frozen_reference_s_per_iteration'] = frozen['s_per_iteration']
        return dict(res, tree=str(Path(a.tree).resolve()), config=a.config, environment=cand_env)
    return finish(out, 'time.json', guarded(out, run, Path(a.tree).resolve()))


CATEGORIES = [
    ('warp (simulation)', r'_warp_|mujoco|wp_|warp'),
    ('gemm', r'gemm|cutlass|sm90_xmma|sm80_xmma|sm100|cublas|ampere_|hopper_|s16816|nvjet'),
    ('optimizer', r'adam|multi_tensor|foreach'),
    ('triton', r'^triton_'),
    ('memcpy/cast', r'copy|memcpy|cast|convert|_to_copy'),
    ('elementwise/reduce', r'elementwise|reduce|vectorized|unrolled|index|cat|softmax|mul|add|scatter|gather'),
]


def categorize(kernels):
    cats = {}
    for k in kernels:
        n = k['name'].lower()
        cat = next((c for c, rx in CATEGORIES if re.search(rx, n)), 'other')
        cats[cat] = cats.get(cat, 0.0) + k['total_ms']
    return {k: round(v, 1) for k, v in sorted(cats.items(), key=lambda kv: -kv[1])}


def kernel_util_report(trace, out):
    script = BENCH / 'kernel_util.py'
    r = subprocess.run([str(PY), str(script), str(trace), '--out', str(out / 'kernel_util.md'),
                        '--json', str(out / 'kernel_util.json')], capture_output=True, text=True, timeout=1800)
    return {'returncode': r.returncode, 'report': str(out / 'kernel_util.md'),
            **({'stderr': r.stderr[-1500:]} if r.returncode else {})}


def cmd_profile(a):
    out = Path(a.out).resolve()
    cand_env = candidate_environment(a.config)
    mode = 'nsys' if a.nsys else 'torch'

    def run(tree):
        res = time_tree(tree, out, 'profile', cand_env=cand_env, profile=mode, profile_updates=a.iters)
        rank = res.pop('_rank', None)
        if res['status'] != 'ok':
            return dict(res, tree=str(Path(a.tree).resolve()), config=a.config, environment=cand_env)
        res['note'] = (f'{a.iters} iterations (~{a.iters * recipe.UTD:g} updates) after a warmup of '
                       f'{CFG["timing"]["warmup"]} updates; instrumented: the time is not comparable with `time`')
        ends = rank['iteration_start_s']
        first = CFG['timing']['warmup']
        res['iteration_s_host'] = [round(ends[i + 1] - ends[i], 3) for i in range(first, min(len(ends) - 1, first + a.iters))]
        if mode == 'torch':
            summ = json.loads((out / 'timing_profile.torchprof.json').read_text())
            summ['categories_ms'] = categorize(summ['kernels'])
            res['torch_profile'] = summ
            res['trace'] = str(out / 'timing_profile.trace.json')
            res['kernel_util'] = kernel_util_report(out / 'timing_profile.trace.json', out)
            md = ['# profile', '', f'tree {a.tree}, config {a.config}', '', res['note'], '',
                  f'window {summ["span_s"]:.2f} s for {summ["iterations"]} iterations; GPU busy '
                  f'{summ["gpu_busy_s"]:.2f} s ({summ["gpu_busy_fraction"]:.0%}); kernel time {summ["kernel_time_s"]:.2f} s; '
                  f'{summ["kernel_launches"]} kernel launches', '',
                  'host time per iteration (s): ' + ', '.join(map(str, res['iteration_s_host'])), '',
                  f'kernel utilization report: {res["kernel_util"]["report"]}', '',
                  '## kernel time by category (ms)', '', '| category | ms |', '|---|---|']
            md += [f'| {k} | {v} |' for k, v in summ['categories_ms'].items()]
            md += ['', '## time inside the main calls (ranges: env_step, wm_update, policy_update)', '',
                   '| call | GPU ms | calls | CPU ms |', '|---|---|---|---|']
            md += [f'| {k} | {v["cuda_ms"]} | {v["calls"]} | {v["cpu_ms"]} |' for k, v in summ['annotations'].items()]
            md += ['', '## top kernels', '', '| ms | calls | share | kernel |', '|---|---|---|---|']
            md += [f'| {k["total_ms"]} | {k["calls"]} | {k["share_of_kernel_time"]} | `{k["name"][:140]}` |'
                   for k in summ['kernels'][:40]]
            md += ['', '## top CPU ops (self time)', '', '| ms | calls | op |', '|---|---|---|']
            md += [f'| {o["self_cpu_ms"]} | {o["calls"]} | {o["name"]} |' for o in summ['cpu_ops_self_time'][:25]]
            (out / 'profile.md').write_text('\n'.join(md) + '\n')
        else:
            res['trace'] = str(out / 'nsys_profile.nsys-rep')
        return dict(res, tree=str(Path(a.tree).resolve()), config=a.config, environment=cand_env)
    return finish(out, 'profile.json', guarded(out, run, Path(a.tree).resolve()))


def cmd_baseline(a):
    """Formulator: the reference timed `runs` times (noise floor, frozen timing and RNG stream)."""
    out = Path(a.out).resolve()

    def run(_):
        ref_src = check_reference()
        with scratch_dir('sm-ref-') as tmp:
            tree = materialize(ref_src, Path(tmp) / 'reference')
            runs = {f'ref_{i}': time_tree(tree, out, f'ref_{i}') for i in range(a.runs)}
        ok = [v for v in runs.values() if v['status'] == 'ok']
        tot = [v['total_s'] for v in ok]
        res = {'status': 'ok' if len(ok) == len(runs) else 'ERROR', 'runs': {k: strip(v) for k, v in runs.items()},
               'total_s': tot, 'spread': round((max(tot) - min(tot)) / statistics.mean(tot), 4) if tot else None}
        if ok:
            r = ok[0]['_rank']
            res['rng_fingerprints'] = r['rng_fingerprints']
            res['rng_identical_across_runs'] = all(v['_rank']['rng_fingerprints'] == r['rng_fingerprints'] for v in ok)
        return res
    return finish(out, 'baseline.json', guarded(out, run))


def cmd_freeze_timing(a):
    """Formulator: task/timing_ref.json from a `baseline` output (its first `--use` runs)."""
    b = json.loads(Path(a.baseline).read_text())
    runs = [v for v in b['runs'].values() if v['status'] == 'ok']
    n_check = CFG['timing']['rng_check_iterations']
    prefixes = {tuple(r_['rng_fingerprints'][:n_check]) for r_ in
                (json.loads(Path(v['log']).with_suffix('.r0.json').read_text()) for v in runs)}
    if len(runs) < 2 or len(prefixes) != 1:
        raise SystemExit(f'need at least two ok reference runs whose RNG streams agree on the first {n_check} iterations')
    tot = [r['total_s'] for r in runs]
    mean = statistics.mean(tot)
    ratios = [r['window_ratio'] for r in runs]
    ref = {'total_s': round(mean, 3), 's_per_iteration': round(statistics.mean(r['s_per_iteration'] for r in runs), 4),
           'spread': round((max(tot) - min(tot)) / mean, 4), 'window_ratio': round(statistics.mean(ratios), 4),
           'runs_total_s': tot, 'peak_memory_allocated_gib': max(r['peak_memory_allocated_gib'] for r in runs),
           'fingerprint': runs[0]['fingerprint'], 'rng_fingerprints': b['rng_fingerprints'],
           'source': str(Path(a.baseline).resolve()), 'frozen': datetime.now().isoformat(timespec='seconds')}
    (ROOT / 'task' / 'timing_ref.json').write_text(json.dumps(ref, indent=1) + '\n')
    print(json.dumps({k: ref[k] for k in ('total_s', 's_per_iteration', 'spread', 'window_ratio')}))
    return 0


# ---------------------------------------------------------------- formulator commands

def cmd_record(a):
    """Formulator: record cases (default bench cases) of a tree into a directory (reference records, variants)."""
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = json.loads(Path(a.cases).read_text()) if a.cases else json.loads(path(CFG['numerics']['cases']).read_text())
    tree = Path(a.tree).resolve()
    if a.traj_seeds:
        spec = json.loads(path(CFG['trajectory']['spec']).read_text())
        cases = [traj_case(spec, int(s)) for s in a.traj_seeds.split(',')]
    failed = record_cases(tree, out, cases, out, candidate_environment(a.config), 'formulate', variant=a.variant,
                          losses_only=bool(a.traj_seeds))
    print(json.dumps({'status': 'ok' if not failed else failed, 'out': str(out), 'cost': cost()}))
    return 1 if failed else 0


def cmd_freeze_inputs(a):
    inputs = freeze_inputs(a.paths)
    f = ROOT / 'task' / 'formulation.json'
    data = json.loads(f.read_text()) if f.exists() else {}
    data['inputs'] = inputs
    f.write_text(json.dumps(data, indent=1) + '\n')
    print(json.dumps({'files': len(inputs), 'sha256': sum(v['kind'] == 'sha256' for v in inputs.values())}))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    k = sub.add_parser('check')
    k.add_argument('--tree', required=True)
    k.add_argument('--out', required=True)
    k.add_argument('--config')
    k.add_argument('--quick', action='store_true')
    k.add_argument('--skip-timing', action='store_true')
    k.add_argument('--n1', action='store_true')
    k.add_argument('--final', action='store_true', help=argparse.SUPPRESS)
    t = sub.add_parser('time')
    t.add_argument('--tree', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--config')
    f = sub.add_parser('profile')
    f.add_argument('--tree', required=True)
    f.add_argument('--out', required=True)
    f.add_argument('--config')
    f.add_argument('--nsys', action='store_true')
    f.add_argument('--iters', type=int, default=10)
    b = sub.add_parser('baseline', help=argparse.SUPPRESS)
    b.add_argument('--out', required=True)
    b.add_argument('--runs', type=int, default=3)
    ft = sub.add_parser('freeze-timing', help=argparse.SUPPRESS)
    ft.add_argument('--baseline', required=True)
    r = sub.add_parser('record', help=argparse.SUPPRESS)
    r.add_argument('--tree', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--cases')
    r.add_argument('--variant')
    r.add_argument('--config')
    r.add_argument('--traj-seeds')
    fi = sub.add_parser('freeze-inputs', help=argparse.SUPPRESS)
    fi.add_argument('paths', nargs='+')
    a = p.parse_args()
    return {'freeze-inputs': cmd_freeze_inputs, 'check': cmd_check, 'time': cmd_time, 'profile': cmd_profile,
            'baseline': cmd_baseline, 'freeze-timing': cmd_freeze_timing, 'record': cmd_record}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
