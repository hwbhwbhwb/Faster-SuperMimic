"""LAFAN1 G1 CSVs -> the official motion npz format, with mjlab_research's own converter
(mjlab.scripts.csv_to_npz.MotionLoader: 30 fps input, 50 fps output, its interpolation and velocities), saved locally
instead of uploaded to the W&B registry (the only change from csv_to_npz.main).

    PYTHONPATH=<snapshot>/mjlab_research/src python convert_motions.py <csv dir> <out dir> [device]
"""
import sys
from pathlib import Path

import numpy as np
from mjlab.scripts.csv_to_npz import MotionLoader

src, out = Path(sys.argv[1]), Path(sys.argv[2])
device = sys.argv[3] if len(sys.argv) > 3 else 'cuda:0'
out.mkdir(parents=True, exist_ok=True)
files = sorted(src.glob('*.csv'))
assert len(files) == 40, f'expected 40 LAFAN1 CSVs, found {len(files)}'
total = 0
for f in files:
    m = MotionLoader(motion_file=str(f), input_fps=30.0, output_fps=50.0, line_range=None, device=device)
    arrays = m.arrays()
    np.savez(out / f'{f.stem}.npz', **arrays)
    total += arrays['joint_pos'].shape[0]
print(f'{len(files)} clips, {total} frames, {total / 50 / 3600:.3f} h at 50 Hz -> {out}')
