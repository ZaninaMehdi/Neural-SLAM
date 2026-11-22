# Phantom dynamic objects: Gibson val results

Pretrained ANS models, Gibson val (`val_mt`), 14 scenes x 71 episodes =
994 episodes, 1000 steps each. Objects: speed 0.1 m/step, radius 0.25 m,
contact distance 0.35 m, `--dyn_block 1` (agent's move undone on contact),
`--dyn_fov_only 0` (all objects given to planner and global policy),
seed 1. See `docs/DYNAMIC_OBJECTS.md`.

Slurm jobs 28343207-28343210 (2026-11-21, 1 H100 each, 75-84 min).

## Collisions (mean per episode)

| Mode | Objects | Contact events | Contact steps | Blocked steps | Object yields | Collision-free eps |
|---|---|---|---|---|---|---|
| known  | 1 | 5.039  | 90.118  | 90.118  | 126.275 | 16.7% |
| random | 1 | 5.325  | 97.492  | 97.492  | 138.307 | 14.9% |
| known  | 3 | 13.966 | 253.288 | 253.288 | 533.061 | 4.3%  |
| random | 3 | 14.619 | 268.613 | 268.613 | 619.979 | 3.2%  |

## Exploration after 1000 steps

| Run | Area (m²) | vs paper | vs reproduction | Ratio | vs paper | vs reproduction |
|---|---|---|---|---|---|---|
| Paper (ANS)               | 32.701 | – | – | 0.948 | – | – |
| Reproduction, no objects  | 31.574 | -1.127 | – | 0.9448 | -0.0032 | – |
| known, 1 object           | 31.301 | -1.400 | -0.274 | 0.9372 | -0.0108 | -0.0076 |
| random, 1 object          | 31.037 | -1.664 | -0.537 | 0.9305 | -0.0175 | -0.0143 |
| known, 3 objects          | 29.337 | -3.364 | -2.237 | 0.8845 | -0.0635 | -0.0603 |
| random, 3 objects         | 29.462 | -3.239 | -2.112 | 0.8858 | -0.0622 | -0.0591 |

Per-run files: `train.log` (summary at the end), `dyn_collisions.txt`
(one row per episode), `explored_area.txt` / `explored_ratio.txt` (one
array per episode, a value every 25 steps), `slurm.out`.
