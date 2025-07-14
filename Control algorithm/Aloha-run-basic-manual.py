import mujoco
from mujoco import viewer
import time

# === Load the model ===
xml_path = r"D:\PhD\0PhD-Implementation\0ALOHA-ALL\mobile_aloha_sim-master\aloha_mujoco\aloha\meshes_mujoco\aloha_v1.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# === Launch the passive viewer ===
with viewer.launch_passive(model, data) as v:
    print("MuJoCo viewer launched. Use GUI to manually control the robot.")
    print("Press Esc or close the window to exit.")

    while v.is_running():
        mujoco.mj_step(model, data)
        v.sync()
        time.sleep(model.opt.timestep)
