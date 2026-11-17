# Renders one RGB + depth frame with habitat-sim (headless EGL) to verify GPU rendering works.
import sys, numpy as np, habitat_sim
scene = sys.argv[1]
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene.id = scene
sim_cfg.gpu_device_id = 0
sensors = []
for uuid, st in [("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)]:
    s = habitat_sim.SensorSpec(); s.uuid = uuid; s.sensor_type = st; s.resolution = [256, 256]
    s.position = [0.0, 0.88, 0.0]; sensors.append(s)
agent_cfg = habitat_sim.agent.AgentConfiguration(); agent_cfg.sensor_specifications = sensors
sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
obs = sim.get_sensor_observations()
print("rgb", obs["rgb"].shape, "mean", float(obs["rgb"][..., :3].mean()),
      "| depth", obs["depth"].shape, "max", float(obs["depth"].max()))
assert obs["rgb"][..., :3].mean() > 0, "black frame: rendering failed"
sim.close(); print("RENDER OK")
