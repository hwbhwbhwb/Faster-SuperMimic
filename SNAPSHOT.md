# Official SuperMimic snapshot

- supermimic: https://github.com/HybridRobotics/supermimic @ abc124c1bd5976341034668af3d12c087f2f1f55
- mjlab_research: https://github.com/HybridRobotics/mjlab_research @ dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd
  (the official README says kevinzakka/mjlab_research, which is not public; HybridRobotics/mjlab_research is the
  published copy: "mjlab with the motion-tracking extensions SuperMimic uses")
- supermimic/uv.lock pins torch 2.10 / mujoco 3.10 and predates the published mjlab_research (which requires
  mujoco~=3.16, torch>=2.14), so the environment follows mjlab_research's own uv.lock instead; see the task's prep.
