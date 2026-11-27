# Checks whether the pinned habitat-sim (9575dcd, built WITHOUT Bullet) can show a moving
# object in the RGB and depth images.
#
# It writes a box mesh plus the physics config files to <out_dir>/work, turns on
# enable_physics, adds the box 1.5 m in front of the agent, moves it, and compares frames.
# Without Bullet, objects are KINEMATIC: they can be placed with set_translation but don't
# fall or collide.
#
#   apptainer/ans.sh python apptainer/physics_render_test.py <scene.glb> <out_dir>
#
# Checks (each prints PASS/FAIL):
#   1 add_object returns a valid id
#   2 object visible: center depth drops to about 1.25 m, and red pixels appear in RGB
#   3 moving works: the box moves left to right across the image and doesn't leave a ghost
#   4 (info only) agent collision: does move_forward stop at the box, or pass through it?
import os, sys, json
import numpy as np
import magnum as mn
import habitat_sim

scene = os.path.abspath(sys.argv[1])
out_dir = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else "physics_test_out")
work = os.path.join(out_dir, "work")
os.makedirs(os.path.join(work, "data", "objects"), exist_ok=True)

# Box roughly the size of a person: 0.5 x 1.6 x 0.5 m, red material.
SX, SY, SZ = 0.5, 1.6, 0.5
DIST = 1.5  # metres in front of the agent


def write_box(path_obj, path_mtl):
    x, y, z = SX / 2, SY / 2, SZ / 2
    v = [(-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
         (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)]
    faces = [(1, 2, 3), (1, 3, 4), (5, 8, 7), (5, 7, 6), (1, 5, 6), (1, 6, 2),
             (4, 3, 7), (4, 7, 8), (1, 4, 8), (1, 8, 5), (2, 6, 7), (2, 7, 3)]
    with open(path_mtl, "w") as f:
        f.write("newmtl red\nKa 1 0 0\nKd 1 0 0\nKs 0 0 0\nd 1\nillum 1\n")
    with open(path_obj, "w") as f:
        f.write("mtllib box.mtl\nusemtl red\n")
        for p in v:
            f.write("v %f %f %f\n" % p)
        for t in faces:  # write both windings so the box is visible regardless of culling
            f.write("f %d %d %d\n" % t)
            f.write("f %d %d %d\n" % (t[0], t[2], t[1]))


write_box(os.path.join(work, "data/objects/box.obj"), os.path.join(work, "data/objects/box.mtl"))
with open(os.path.join(work, "data/objects/box.phys_properties.json"), "w") as f:
    json.dump({"render mesh": "box.obj", "collision mesh": "box.obj", "mass": 1.0}, f, indent=2)
with open(os.path.join(work, "data/default.phys_scene_config.json"), "w") as f:
    # friction/restitution are required here: at this commit loadScene() reads them with
    # getDouble() -> std::map::at, which throws "IndexError: map::at" if they are missing.
    json.dump({"physics simulator": "none", "timestep": 0.01, "gravity": [0, -9.8, 0],
               "friction coefficient": 0.4, "restitution coefficient": 0.1,
               "rigid object paths": ["objects/box"]}, f, indent=2)

# habitat-sim at this commit reads the hard-coded relative path data/default.phys_scene_config.json
os.chdir(work)

sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene.id = scene
sim_cfg.gpu_device_id = 0
sim_cfg.enable_physics = True
sensors = []
for uuid, st in [("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)]:
    s = habitat_sim.SensorSpec(); s.uuid = uuid; s.sensor_type = st; s.resolution = [256, 256]
    s.position = [0.0, 0.88, 0.0]; sensors.append(s)
agent_cfg = habitat_sim.agent.AgentConfiguration(); agent_cfg.sensor_specifications = sensors
sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
sim.seed(0); np.random.seed(0)
agent = sim.get_agent(0)

results = {}


def obs():
    o = sim.get_sensor_observations()
    return o["rgb"][..., :3].copy(), o["depth"].copy()


def center_depth(d):
    return float(np.median(d[118:138, 118:138]))


def red_mask(rgb):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    return (r > 120) & (r > g + 60) & (r > b + 60)


def save(name, rgb, d):
    try:
        import cv2
        cv2.imwrite(os.path.join(out_dir, name + "_rgb.png"), rgb[..., ::-1])
        cv2.imwrite(os.path.join(out_dir, name + "_depth.png"),
                    (np.clip(d / 5.0, 0, 1) * 255).astype(np.uint8))
    except Exception as e:
        np.save(os.path.join(out_dir, name + "_rgb.npy"), rgb); print("cv2 save failed:", e)


def check(key, ok, msg):
    results[key] = bool(ok)
    print("[%s] %s: %s" % ("PASS" if ok else "FAIL", key, msg))


print("physics object library size:", sim.get_physics_object_library_size())

# Find a pose facing -Z (identity rotation) with DIST+1 m of free, navigable floor ahead.
# Use a fresh AgentState: set_state() also restores each sensor to the WORLD pose stored in
# state.sensor_states, so reusing agent.get_state() would leave the camera at the start pose.
def put_agent(p):
    s = habitat_sim.AgentState()
    s.position = np.array(p); s.rotation = np.quaternion(1, 0, 0, 0)
    agent.set_state(s)


pose = None
for _ in range(200):
    p = np.array(sim.pathfinder.get_random_navigable_point())
    ahead = p + np.array([0, 0, -DIST])
    if not sim.pathfinder.is_navigable(ahead):
        continue
    put_agent(p)
    rgb0, d0 = obs()
    if center_depth(d0) > DIST + 1.0 and red_mask(rgb0).mean() < 0.01:
        pose = p; break
if pose is None:
    print("FAIL: no free pose found in this scene; try another scene"); sys.exit(2)
cam = agent.get_state().sensor_states["rgb"].position
print("camera at", np.round(cam, 2).tolist(), "(should be agent + [0, 0.88, 0])")
print("agent at", pose.tolist(), "| baseline center depth %.2f m" % center_depth(d0))
save("0_baseline", rgb0, d0)

# 1. add object
oid = sim.add_object(0)
check("add_object", oid is not None and oid >= 0, "id=%s" % oid)
if not results["add_object"]:
    sim.close(); sys.exit(1)

# The box mesh is re-centred on its bbox centre, so put the centre SY/2 above the floor.
floor_y = pose[1]


def place(x_off, z_off=-DIST):
    sim.set_translation(mn.Vector3(float(pose[0] + x_off), float(floor_y + SY / 2),
                                   float(pose[2] + z_off)), oid)


# 2. visible?
place(0.0)
rgb1, d1 = obs()
save("1_object", rgb1, d1)
cd = center_depth(d1); red = red_mask(rgb1).mean()
expected = DIST - SZ / 2
check("depth_sees_object", abs(cd - expected) < 0.15,
      "center depth %.2f m (expected ~%.2f, baseline %.2f)" % (cd, expected, center_depth(d0)))
check("rgb_sees_object", red > 0.02, "red pixel fraction %.3f" % red)

# 3. move across the view; the red blob's centre column should increase left to right.
cols = []
for i, x in enumerate(np.linspace(-0.8, 0.8, 5)):
    place(float(x))
    rgb, d = obs(); save("2_move_%d" % i, rgb, d)
    m = red_mask(rgb)
    cols.append(float(np.nonzero(m)[1].mean()) if m.any() else float("nan"))
print("red blob centre column per step:", ["%.0f" % c for c in cols])
mono = all(np.isfinite(cols)) and all(b > a for a, b in zip(cols, cols[1:]))
check("object_moves", mono, "blob moves left to right" if mono else "blob did not track the motion")

# Move it 50 m away: the view should match the baseline again (no ghost).
sim.set_translation(mn.Vector3(float(pose[0]), float(floor_y - 50.0), float(pose[2])), oid)
rgb2, d2 = obs(); save("3_removed", rgb2, d2)
check("no_ghost", abs(center_depth(d2) - center_depth(d0)) < 0.05 and red_mask(rgb2).mean() < 0.005,
      "center depth back to %.2f m" % center_depth(d2))

# 4. agent collision (info only): put the box back and walk forward 10 x 0.25 m.
put_agent(pose)
place(0.0)
for _ in range(10):
    sim.step("move_forward")
z_travel = float(pose[2] - agent.get_state().position[2])
box_near = DIST - SZ / 2
passed = z_travel > box_near
print("[INFO] agent_collision: agent moved %.2f m forward; box front face at %.2f m -> %s"
      % (z_travel, box_near, "PASSED THROUGH the box (no collision)" if passed else "stopped before the box"))
results["agent_passes_through"] = passed

sim.close()
with open(os.path.join(out_dir, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
core = ["add_object", "depth_sees_object", "rgb_sees_object", "object_moves", "no_ghost"]
print("frames + results.json in", out_dir)
print("OVERALL:", "RENDERED DYNAMIC OBJECTS WORK" if all(results.get(k) for k in core) else "NOT WORKING (see FAILs)")
