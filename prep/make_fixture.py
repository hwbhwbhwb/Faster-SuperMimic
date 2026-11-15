"""Freeze the state the bitwise cases start from: the official reference after 300 iterations of the recipe (seed 0):
replay buffer storage and counters, world model, actor, critic, optimizer states, the iteration and RNG streams.

    PYTHONPATH=<ws> python make_fixture.py <reference tree> <npz dir> <scratch> <out .pt>
"""
import sys

from bench.sm_adapter import SuperMimicAdapter

ref, npz, scratch, out = sys.argv[1:5]
a = SuperMimicAdapter(repo=ref, npz_dir=npz, scratch=scratch, reference=ref)
a.configure({'mode': 'loop', 'preroll': 300})
a.build(0)
buf = a.r.replay_buffer
print(f'iteration {a.iteration} buffer step {buf._step} transitions {buf.num_transitions}', flush=True)
print('loss_dict keys:', sorted((a.ns.get('loss_dict') or {}).keys()), flush=True)
print('models with parameters:', sorted(a._snapshot()), flush=True)
a.save_state(out)
print('saved', out)
