"""Offline stand-in for the `wandb` package in bench runs (the GPU host cannot reach api.wandb.ai, and the official
runner calls wandb.Api() for checkpoint-artifact pruning, which would block or fail on the network).

Covers exactly what supermimic and mjlab_research use: init / log / finish, run.{id,name,entity,project,
log_artifact,use_artifact,link_artifact}, Artifact(name, type).add_file, Api().{artifacts,artifact,run}.
Every wandb.log row is appended as one JSON line to $SM_BENCH_WANDB_LOG when that is set (the bench reads the
training metrics from it). Artifacts are not uploaded anywhere; Api queries return nothing or raise, as an
unreachable server would, and the training code already treats those as non-fatal.
"""
import json
import math
import os
import uuid

__version__ = '0.0.0-bench-stub'
run = None


class _Run:
    def __init__(self, project=None, name=None, config=None):
        self.id = uuid.uuid4().hex[:8]
        self.name = name or f'bench-{self.id}'
        self.project = project or 'bench'
        self.entity = 'bench'
        self.config = dict(config or {})

    def log_artifact(self, artifact, *a, **k):
        return artifact

    def use_artifact(self, artifact, *a, **k):
        return artifact

    def link_artifact(self, *a, **k):
        return None

    def finish(self, *a, **k):
        finish()


class Artifact:
    def __init__(self, name, type=None, *a, **k):
        self.name, self.type, self.files = name, type, []
        self.version, self.aliases = 'v0', []

    def add_file(self, path, *a, **k):
        self.files.append(path)

    def delete(self, *a, **k):
        return None


class Api:
    def __init__(self, *a, **k):
        pass

    def artifacts(self, *a, **k):
        return []

    def artifact(self, name, *a, **k):
        raise RuntimeError(f'wandb bench stub: no registry access ({name})')

    def run(self, path, *a, **k):
        raise RuntimeError(f'wandb bench stub: no run access ({path})')


def _plain(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f if math.isfinite(f) else str(f)


def init(project=None, name=None, config=None, *a, **k):
    global run
    run = _Run(project, name, config)
    return run


def log(row, step=None, *a, **k):
    path = os.environ.get('SM_BENCH_WANDB_LOG')
    if path:
        with open(path, 'a') as f:
            f.write(json.dumps({'_step': step, **{key: _plain(v) for key, v in row.items()}}) + '\n')


def finish(*a, **k):
    global run
    run = None
