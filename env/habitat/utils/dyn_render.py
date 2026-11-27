"""Rendered dynamic objects (-withRendering).

Each dynamic object from dynamic_objects.py is also added to the habitat-sim
scene as a box, moved every step with set_translation(), so it appears in RGB
and depth like any other surface. Movement, trajectories and collision checks
stay in dynamic_objects.py: this habitat-sim (9575dcd, built without Bullet)
only has KINEMATIC objects, and the agent walks through them (the navmesh
ignores objects).

Verified by apptainer/physics_render_test.py (job 28803834): objects render in
RGB and depth, move, are occluded by walls and leave no trace when removed.

Pitfalls of this habitat-sim version handled here:
  * habitat-api never passes enable_physics to habitat-sim, so
    enable_physics_in_habitat_api() patches HabitatSim.create_sim_config.
  * habitat-sim always reads ./data/default.phys_scene_config.json (relative
    to the working directory, not configurable from Python), and that file
    must contain "friction coefficient" and "restitution coefficient", or
    Simulator() fails with "IndexError: map::at".
    write_physics_assets() writes it plus the box mesh under ./data/
    (gitignored).
"""
import json
import os

import magnum as mn

PHYS_CONFIG = "data/default.phys_scene_config.json"
OBJ_DIR = "data/objects"
OBJ_NAME = "dyn_box"
COLOR = (1.0, 0.0, 0.0)  # red


def _write_atomic(path, text):
    # Several env processes start at once and write identical content
    tmp = "{}.tmp{}".format(path, os.getpid())
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def write_physics_assets(width, height):
    """Box mesh (width x height x width metres) + physics configs under
    ./data/. Must run before the simulator is created."""
    os.makedirs(OBJ_DIR, exist_ok=True)
    x, y, z = width / 2., height / 2., width / 2.
    v = [(-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
         (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)]
    faces = [(1, 2, 3), (1, 3, 4), (5, 8, 7), (5, 7, 6), (1, 5, 6), (1, 6, 2),
             (4, 3, 7), (4, 7, 8), (1, 4, 8), (1, 8, 5), (2, 6, 7), (2, 7, 3)]
    lines = ["mtllib {}.mtl".format(OBJ_NAME), "usemtl box"]
    lines += ["v %f %f %f" % p for p in v]
    for t in faces:  # both windings: visible regardless of face culling
        lines.append("f %d %d %d" % t)
        lines.append("f %d %d %d" % (t[0], t[2], t[1]))
    _write_atomic(os.path.join(OBJ_DIR, OBJ_NAME + ".obj"),
                  "\n".join(lines) + "\n")
    _write_atomic(os.path.join(OBJ_DIR, OBJ_NAME + ".mtl"),
                  "newmtl box\nKa %g %g %g\nKd %g %g %g\nKs 0 0 0\nd 1\n"
                  "illum 1\n" % (COLOR + COLOR))
    _write_atomic(os.path.join(OBJ_DIR, OBJ_NAME + ".phys_properties.json"),
                  json.dumps({"render mesh": OBJ_NAME + ".obj",
                              "collision mesh": OBJ_NAME + ".obj",
                              "mass": 1.0}, indent=2))
    _write_atomic(PHYS_CONFIG, json.dumps({
        "physics simulator": "none", "timestep": 0.01,
        "gravity": [0, -9.8, 0],
        "friction coefficient": 0.4, "restitution coefficient": 0.1,
        "rigid object paths": ["objects/" + OBJ_NAME]}, indent=2))


def enable_physics_in_habitat_api():
    """Make habitat-api create habitat-sim with enable_physics=True (it
    covers simulators re-created on scene changes too)."""
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
    if getattr(HabitatSim, "_dyn_physics_patched", False):
        return
    original = HabitatSim.create_sim_config

    def create_sim_config(self, sensor_suite):
        cfg = original(self, sensor_suite)
        cfg.sim_cfg.enable_physics = True
        return cfg

    HabitatSim.create_sim_config = create_sim_config
    HabitatSim._dyn_physics_patched = True


class RenderedObjects:
    """One box in the scene per dynamic object."""

    def __init__(self, height):
        self.height = height

    def reset(self, sim, n):
        """Remove every object left in the scene (the simulator is reused
        when consecutive episodes share a scene) and add n boxes."""
        for oid in list(sim.get_existing_object_ids()):
            sim.remove_object(oid)
        self.ids = [sim.add_object(0) for _ in range(n)]
        if any(i is None or i < 0 for i in self.ids):
            raise RuntimeError("add_object failed: is {} present?".format(
                PHYS_CONFIG))

    def hide(self, sim):
        """Move every box 100 m below the scene (out of every view)."""
        for oid in self.ids:
            sim.set_translation(mn.Vector3(0., -100., 0.), oid)

    def sync(self, sim, positions):
        """positions: habitat world (x, y=floor height, z) per object. The
        box mesh is centred on its bounding box, so lift it by height/2."""
        for oid, p in zip(self.ids, positions):
            sim.set_translation(mn.Vector3(float(p[0]),
                                           float(p[1]) + self.height / 2.,
                                           float(p[2])), oid)
