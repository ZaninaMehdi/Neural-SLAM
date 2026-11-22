"""Phantom dynamic obstacles for oracle-perception experiments.

Objects are NOT rendered: they never appear in RGB/depth, so the SLAM module
never perceives them. They exist only as
  * positions on the scene navmesh, moved every environment step, and
  * a `dyn_obs` matrix that is overlaid (element-wise max) on *copies* of the
    obstacle map given to the planner and the global policy.
The persistent SLAM map (local_map / full_map) is never modified.

Modes (--dyn_mode):
  none    feature off (default, original ANS behaviour)
  known   per-episode waypoint loops generated from a fixed seed derived from
          (scene, episode id) and cached as JSON, so every run / method sees
          exactly the same motion
  random  random-waypoint walk, new goal on arrival, seeded from
          (--seed, process rank, episode count)

Coordinates: positions are kept in habitat-sim world coordinates
(x, y=height, z). For the map overlay each object is expressed relative to
the agent's TRUE pose and re-placed at the same offset from the agent's
ESTIMATED pose, i.e. as an ideal on-board sensor would report it, so it lines
up with the agent's own (drifting) map.
"""
import ctypes
import json
import os
import zlib

import numpy as np
import habitat_sim

import env.habitat.utils.pose as pu

_libc = ctypes.CDLL(None)


def _seed_navmesh_rng(seed):
    # habitat-sim's PathFinder.get_random_navigable_point() draws from C rand();
    # PathFinder.seed() is not exposed to Python in this version.
    _libc.srand(ctypes.c_uint(int(seed) % (2 ** 32)))


def sim_xy(position):
    """habitat world position -> planar (x, y), same convention as
    Exploration_Env.get_sim_location()."""
    return -position[2], -position[0]


def dist2d(a, b):
    """Planar distance between two habitat world positions (ignores height)."""
    return float(np.hypot(a[0] - b[0], a[2] - b[2]))


class _Object:
    def __init__(self, polyline, loop):
        self.polyline = [np.asarray(p, dtype=np.float64) for p in polyline]
        self.loop = loop
        self.seg = 0        # index of current segment start
        self.t = 0.0        # metres travelled along current segment
        self.pos = self.polyline[0].copy()

    def at_end(self):
        return (not self.loop) and self.seg >= len(self.polyline) - 1

    def propose(self, dist):
        """Position after travelling `dist` metres; does not commit."""
        seg, t = self.seg, self.t + dist
        n = len(self.polyline)
        for _ in range(10 * n + 10):
            if seg >= n - 1:
                if not self.loop:
                    return self.polyline[-1].copy(), n - 1, 0.0
                seg = 0
            a, b = self.polyline[seg], self.polyline[seg + 1]
            seg_len = dist2d(a, b)
            if t <= seg_len:
                frac = 0.0 if seg_len == 0 else t / seg_len
                return a + frac * (b - a), seg, t
            t -= seg_len
            seg += 1
        return self.pos.copy(), self.seg, self.t

    def commit(self, proposal):
        self.pos, self.seg, self.t = proposal


class DynamicObjects:
    def __init__(self, args, rank):
        self.args = args
        self.rank = rank
        self.mode = args.dyn_mode
        self.enabled = self.mode != "none"
        self.n = args.num_dyn_obj
        self.speed = args.dyn_speed
        self.radius = args.dyn_radius
        self.contact_dist = args.dyn_radius + args.dyn_agent_radius
        self.ep_count = 0
        self.objects = []
        self.prev_contact = False
        self.pf = None

    # ------------------------------------------------------------------ setup
    def _sample_point(self, from_pos, agent_pos, min_agent_dist, tries=200):
        """Random navigable point on the same floor, reachable from
        `from_pos`, at least `min_agent_dist` from the agent.
        Returns (point, path_points) or (None, None)."""
        for _ in range(tries):
            p = np.asarray(self.pf.get_random_navigable_point(), np.float64)
            if abs(p[1] - from_pos[1]) > 0.5:
                continue
            if dist2d(p, agent_pos) < min_agent_dist:
                continue
            path = habitat_sim.ShortestPath()
            path.requested_start = np.asarray(from_pos, np.float32)
            path.requested_end = p.astype(np.float32)
            if not self.pf.find_path(path) or \
                    not np.isfinite(path.geodesic_distance) or \
                    len(path.points) < 2:
                continue
            return p, [np.asarray(q, np.float64) for q in path.points]
        return None, None

    def _gen_known(self, agent_pos):
        objs = []
        for _ in range(self.n):
            start, _ = self._sample_point(agent_pos, agent_pos,
                                          self.args.dyn_min_start_dist)
            if start is None:
                continue
            poly, cur = [start], start
            for _ in range(self.args.dyn_num_waypoints - 1):
                wp, pts = self._sample_point(cur, agent_pos, 0.0)
                if wp is None:
                    break
                poly += pts[1:]
                cur = wp
            # close the loop back to the start
            path = habitat_sim.ShortestPath()
            path.requested_start = np.asarray(cur, np.float32)
            path.requested_end = np.asarray(start, np.float32)
            if len(poly) > 1 and self.pf.find_path(path) and \
                    np.isfinite(path.geodesic_distance):
                poly += [np.asarray(q, np.float64) for q in path.points[1:]]
            if len(poly) > 1:
                objs.append([p.tolist() for p in poly])
        return objs

    def reset(self, pathfinder, agent_pos, scene_name, episode_id):
        self.ep_count += 1
        self.objects = []
        self.prev_contact = False
        if not self.enabled:
            return
        self.pf = pathfinder
        agent_pos = np.asarray(agent_pos, np.float64)
        scene = os.path.basename(scene_name).split(".")[0]

        if self.mode == "known":
            seed = zlib.crc32("{}/{}".format(scene, episode_id).encode())
            fname = os.path.join(
                os.path.expanduser(self.args.dyn_traj_dir), scene,
                "ep{}_n{}_wp{}_d{}.json".format(
                    episode_id, self.n, self.args.dyn_num_waypoints,
                    self.args.dyn_min_start_dist))
            polylines = None
            if os.path.exists(fname):
                with open(fname) as f:
                    polylines = json.load(f)["objects"]
            else:
                _seed_navmesh_rng(seed)
                polylines = self._gen_known(agent_pos)
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                tmp = fname + ".tmp{}".format(os.getpid())
                with open(tmp, "w") as f:
                    json.dump({"scene": scene, "episode_id": str(episode_id),
                               "seed": seed, "objects": polylines}, f)
                os.replace(tmp, fname)
            self.objects = [_Object(p, loop=True) for p in polylines]
        else:  # random
            seed = (self.args.seed * 1000003 + self.rank * 10007 +
                    self.ep_count) % (2 ** 31)
            _seed_navmesh_rng(seed)
            for _ in range(self.n):
                start, _ = self._sample_point(agent_pos, agent_pos,
                                              self.args.dyn_min_start_dist)
                if start is None:
                    continue
                goal, pts = self._sample_point(start, agent_pos, 0.0)
                if goal is None:
                    continue
                self.objects.append(_Object(pts, loop=False))

    # ------------------------------------------------------------------- step
    def step_objects(self, agent_pos):
        """Advance every object by `speed` metres. An object whose next
        position would touch the agent waits this step instead (so objects
        never walk into the agent). Returns the number of such waits."""
        if not self.enabled:
            return 0
        yields = 0
        for obj in self.objects:
            if obj.at_end():  # random mode: pick a new goal from here
                goal, pts = self._sample_point(obj.pos, agent_pos, 0.0)
                if goal is not None:
                    obj.polyline, obj.seg, obj.t = pts, 0, 0.0
                    obj.pos = pts[0].copy()
            prop = obj.propose(self.speed)
            if dist2d(prop[0], agent_pos) < self.contact_dist:
                yields += 1
                continue
            obj.commit(prop)
        return yields

    def in_contact(self, agent_pos):
        return any(dist2d(o.pos, agent_pos) < self.contact_dist
                   for o in self.objects)

    def relative_positions(self, agent_sim_pose):
        """Object offsets (dx forward, dy left) in metres relative to the
        agent's true pose (x, y, o[rad]) from get_sim_location()."""
        rel = []
        if not self.enabled:
            return rel
        max_range = self.args.vision_range * self.args.map_resolution / 100.
        half_fov = np.deg2rad(self.args.hfov) / 2.
        for o in self.objects:
            ox, oy = sim_xy(o.pos)
            dx, dy, _ = pu.get_rel_pose_change((ox, oy, 0.), agent_sim_pose)
            if self.args.dyn_fov_only:
                if np.hypot(dx, dy) > max_range or \
                        abs(np.arctan2(dy, dx)) > half_fov:
                    continue
            rel.append((float(dx), float(dy)))
        return rel


def rasterize(rel, est_pose, shape, origin_rc, map_resolution, radius_m,
              scale=1.0):
    """Draw objects as filled disks (value 1) into a `shape` matrix.

    rel:       [(dx, dy), ...] offsets from DynamicObjects.relative_positions
    est_pose:  agent's estimated pose (x, y, o[deg]) in full-map metres
    origin_rc: (row, col) of the matrix's top-left cell in full-map cells
    scale:     cells of the full map per cell of this matrix (e.g. the
               global-policy downscaling factor)
    """
    m = np.zeros(shape, dtype=np.float32)
    if not rel:
        return m
    rad = radius_m * 100. / map_resolution / scale
    rows = np.arange(shape[0])[:, None]
    cols = np.arange(shape[1])[None, :]
    for dx, dy in rel:
        x, y, _ = pu.get_new_pose(tuple(est_pose), (dx, dy, 0.))
        r = (y * 100. / map_resolution - origin_rc[0]) / scale
        c = (x * 100. / map_resolution - origin_rc[1]) / scale
        m[(rows - r) ** 2 + (cols - c) ** 2 <= rad ** 2] = 1.
    return m
