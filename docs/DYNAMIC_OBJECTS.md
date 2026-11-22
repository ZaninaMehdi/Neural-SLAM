# Phantom dynamic objects (oracle-perception baseline)

This experiment adds moving obstacles to evaluation of the **pretrained**
Active Neural SLAM models, without retraining anything. It measures how
the existing planner and global policy cope with moving obstacles when
they are told exactly where the obstacles are.

Code: `env/habitat/utils/dynamic_objects.py` (objects and drawing);
hooks in `env/habitat/exploration_env.py` (step, collision, planner) and
`main.py` (global policy input, logging). Flags are in `arguments.py`.
With `--dyn_mode none` (the default) everything behaves exactly like the
original code.

## How it works

* **Objects are not rendered.** They never appear in the RGB image, so the
  Neural SLAM module never sees them. Only the *planning* side
  (global policy + path planner) is tested; perception is assumed perfect.
* **Movement.** Each object is a disk of radius `--dyn_radius` (0.25 m)
  that walks along the scene's walkable floor (the navmesh), using
  shortest paths, so it goes around walls. It moves `--dyn_speed` metres
  per environment step (agent forward step = 0.25 m). Positions are
  absolute world coordinates in metres.
* **dyn_obs matrix.** Every step a fresh binary matrix is drawn:
  1.0 inside each object's disk, 0 elsewhere (`rasterize()`). It is
  combined (element-wise max) with *copies* of the obstacle map that go to
  * the path planner (every step; it also adds a 1-cell safety margin), and
  * the global policy input channels (every 25 steps, when it picks a
    new long-term goal).

  The agent's own map (`local_map` / `full_map`) is never changed, so
  objects do not leave permanent traces in it.
* **Placement on the map.** An object's offset from the agent's **true**
  pose is applied to the agent's **estimated** pose, i.e. like an ideal
  on-board sensor: the object lines up with the agent's (possibly
  drifting) map.
* **Collision.** After each step, contact = distance between agent centre
  and object centre < `dyn_radius + dyn_agent_radius` (0.25 + 0.1 =
  0.35 m). Only end-of-step positions are checked (no speed involved), so
  a grazing pass between two checks can be missed; head-on hits are always
  caught at the default speeds.
* **Objects never walk into the agent.** If an object's next position
  would touch the agent, it waits that step (counted as a *yield*). Every
  collision is therefore caused by the agent moving.
* **Blocking** (`--dyn_block 1`). On contact the agent's move is undone:
  it is put back to its previous pose and the observation is re-rendered.
  Without blocking the collision is only counted and the agent passes
  through.

## Modes (`--dyn_mode`)

| Mode | Trajectory | Repeatable? |
|---|---|---|
| `none` | no objects (original ANS) | – |
| `known` | Each object loops forever along a closed route: start → `dyn_num_waypoints-1` random waypoints → back to start. Waypoints are not stops; the object moves continuously. | Yes. Generated once per (scene, episode) from a fixed seed and cached as JSON in `--dyn_traj_dir`, so every run and every method sees identical motion. |
| `random` | Each object walks the shortest path to a random goal; on arrival it picks a new random goal. Never repeats. | Only with the same `--seed`, number of processes (`-n`) and episode order. |

In both modes the agent only knows where objects are **now**, never their
future path. Starts are at least `--dyn_min_start_dist` from the agent and
on the agent's floor.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--dyn_mode` | `none` | `none` / `known` / `random` (see above) |
| `--num_dyn_obj` | 3 | objects per episode (fewer if no reachable start is found) |
| `--dyn_speed` | 0.1 | object speed, metres per env step |
| `--dyn_radius` | 0.25 | object radius in metres (map disk and collision) |
| `--dyn_agent_radius` | 0.1 | agent radius in metres, used only for the contact distance |
| `--dyn_block` | 0 | 0 = count collisions only; 1 = count and undo the agent's move |
| `--dyn_fov_only` | 0 | 1 = show only objects within 3.2 m and the camera's field of view (walls are ignored, no occlusion); 0 = show all objects (full oracle) |
| `--dyn_num_waypoints` | 4 | points per loop in `known` mode, start included |
| `--dyn_min_start_dist` | 1.5 | minimum start distance (m) from the agent |
| `--dyn_traj_dir` | `~/scratch/ans_runs/dyn_trajectories` | cache for `known` trajectories (`<scene>/ep<id>_n<n>_wp<k>_d<dist>.json`) |

The cache file name contains `num_dyn_obj`, `dyn_num_waypoints` and
`dyn_min_start_dist`, so changing those creates new trajectories;
`dyn_speed` and `dyn_radius` reuse the same routes.

## Running

Use the evaluation job script; extra arguments after the split are passed
to `main.py`. Outputs go to `~/scratch/ans_runs`.

```bash
# paper protocol: 14 scenes x 71 episodes = 994 episodes, 1000 steps
sbatch --time=02:00:00 apptainer/eval.sbatch dyn_known_block 71 val_mt \
    --dyn_mode known --dyn_block 1

sbatch --time=02:00:00 apptainer/eval.sbatch dyn_random_count 71 val_mt \
    --dyn_mode random --dyn_block 0

# quick pilot (14 x 2 episodes, a few minutes)
sbatch --time=00:15:00 apptainer/eval.sbatch dyn_pilot 2 val_mt --dyn_mode known
```

A full 994-episode run takes about 1 h 10 min – 1 h 30 min on one H100
(the original reproduction took 1 h 08 min). Add `--print_images 1` to
save visualisation frames (orange disks = objects); this is slow and
writes many files, so use it only for a few episodes.

## Outputs and how to read them

For `--exp_name <exp>`:

| File | Content |
|---|---|
| `~/scratch/ans_runs/dump/<exp>/dyn_collisions.txt` | one row per episode: `process episode events contact_steps blocked_steps object_yields` |
| `~/scratch/ans_runs/dump/<exp>/explored_area.txt` | per episode, explored area (m²) every 25 steps; last value = final |
| `~/scratch/ans_runs/dump/<exp>/explored_ratio.txt` | same, as fraction of explorable area |
| `~/scratch/ans_runs/models/<exp>/train.log` | at the end: mean exploration curves and the collision summary |
| `~/scratch/ans_runs/dump/<exp>/episodes/...` | frames, only with `--print_images 1` |

Collision summary at the end of `train.log` (averages per episode):

| Line | Meaning |
|---|---|
| `contact events/eps` | number of separate collisions (no contact → contact) |
| `contact steps/eps` | steps spent in contact; `contact steps / events` = how long the agent stays stuck per collision |
| `blocked steps/eps` | steps whose move was undone; equals contact steps with `--dyn_block 1`, 0 otherwise |
| `object yields/eps` | steps an object waited to avoid walking into the agent |
| `collision-free eps` | % of episodes with zero collisions |

`Final Exp Area` / `Final Exp Ratio` are the paper's metrics; the last
number is the score after 1000 steps (paper: 32.701 m² / 0.948; our
reproduction with `--dyn_mode none`: 31.574 m² / 0.945).

### Comparing runs

Run the same 994 episodes with only the dyn settings changed:

* **Safety:** contact events per episode and % collision-free episodes.
* **Cost of the obstacles:** final explored area / ratio versus the
  `none` baseline. With `--dyn_block 1`, the drop shows how much
  collisions slow exploration down.
* `known` runs see identical object motion, so differences between
  settings come from the agent, not from the objects.

## Limitations

* Contact is checked only at the end of each step (see Collision above).
* `--dyn_fov_only 1` does not check for walls between agent and object.
* Collisions recorded by the original code's collision map during blocked
  moves stay in that map for the episode.
* `dyn_obs` is binary (1.0 / 0); there is no uncertainty or detection
  noise.
* There is no dedicated planning metric (path length, steps with no path,
  closest distance to objects); only coverage and collisions are logged.
