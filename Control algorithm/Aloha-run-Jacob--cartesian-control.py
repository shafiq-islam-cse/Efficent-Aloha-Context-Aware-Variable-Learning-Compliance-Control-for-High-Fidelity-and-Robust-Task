import numpy as np
import torch
import mujoco
import mujoco.viewer
import time
from collections import deque
import matplotlib.pyplot as plt
from aloha_env import AlohaEnv  # Your environment wrapper

# -------------------------------
# Jacobian-based Inverse Kinematics
# -------------------------------
def cartesian_to_joint_delta(env, end_effector_site, desired_dx):
    """
    Convert desired Cartesian displacement delta_x to joint displacements delta_q
    using Jacobian pseudo-inverse.
    """
    jacp = np.zeros((3, env.model.nv))
    jacr = np.zeros((3, env.model.nv))
    site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, end_effector_site)
    mujoco.mj_jacSite(env.model, env.data, jacp, jacr, site_id)
    J = jacp  # Linear velocity Jacobian
    # Select columns for controlled joints only
    J_joint = J[:, env.ctrl_indices]
    # Compute pseudo-inverse of Jacobian to get joint delta
    dq = np.linalg.pinv(J_joint) @ desired_dx
    return dq

# -------------------------------
# Admittance Control in Cartesian Space
# -------------------------------
class CartesianAdmittanceController:
    def __init__(self, M=None, D=None, K=None):
        # Inertia, Damping, Stiffness matrices (3x3 diagonal)
        self.M = M if M is not None else np.diag([0.1, 0.1, 0.1])
        self.D = D if D is not None else np.diag([5.0, 5.0, 5.0])
        self.K = K if K is not None else np.diag([50.0, 50.0, 50.0])
        self.cart_vel = np.zeros(3)  # Initialize Cartesian velocity
        self.cart_pos = np.zeros(3)  # Initialize Cartesian position estimate (for stiffness)

    def step(self, force_error, dt):
        """
        Calculate next Cartesian position delta using admittance dynamics:
        M a + D v + K x = F
        """
        # Compute acceleration: a = M^-1(F - D v - K x)
        acc = np.linalg.inv(self.M) @ (force_error - self.D @ self.cart_vel - self.K @ self.cart_pos)
        # Update velocity and position
        self.cart_vel += acc * dt
        self.cart_pos += self.cart_vel * dt
        return self.cart_pos.copy()  # Return desired Cartesian position

# -------------------------------
# Main simulation and control loop
# -------------------------------
def get_ctrl_indices(model):
    """
    Returns list of actuator indices for controlled joints.
    """
    indices = [model.actuator(f"{side}_joint{i}").id for side in ("fl", "fr") for i in range(1, 9)]
    return indices

def main():
    # Paths and setup
    xml_path = r"D:\PhD\0PhD-Implementation\0ALOHA-ALL\mobile_aloha_sim-master\aloha_mujoco\aloha\meshes_mujoco\aloha_v1.xml"
    env = AlohaEnv(xml_path)
    ctrl_indices = get_ctrl_indices(env.model)
    env.ctrl_indices = ctrl_indices  # Attach indices for easy access

    end_effector_site = "fl_gripper_site"  # Define your end-effector site
    cartesian_goal_delta = np.array([0.0, 0.0, 0.01])  # Desired Cartesian incremental motion (can be updated)

    # Control limits for actuators (min, max) per joint
    ctrl_ranges = np.array([
        [-3.14158,  3.14158], [0, 3.14158], [0, 3.14158], [-2, 1.67], [-1.5708, 1.5708],
        [-3.14158, 3.14158], [0, 0.0475], [0, 0.0475], [-3.14158, 3.14158], [0, 3.14158],
        [0, 3.14158], [-2, 1.67], [-1.5708, 1.5708], [-3.14158, 3.14158], [0, 0.0475], [0, 0.0475]
    ])

    # Initialize buffers and parameters
    seq_len = 20
    seq_buffer = deque(maxlen=seq_len)
    obs = env.reset()
    while len(seq_buffer) < seq_len:
        state = obs[ctrl_indices + [i + len(obs)//2 for i in ctrl_indices]]
        seq_buffer.append(state)

    # Initialize compliance threshold
    threshold = np.zeros(len(ctrl_indices))
    threshold_added = False

    # Initialize admittance controller instance
    admittance_ctrl = CartesianAdmittanceController()

    # Initialize variables for logging
    rewards, joint_positions, latencies, compliance_factors = [], [], [], []
    step = 0

    # For joint compliance calculation
    prev_qvel = np.zeros_like(env.data.qvel[ctrl_indices])
    prev_manual_ctrl = env.data.ctrl[ctrl_indices].copy()

    dt = 0.01  # timestep duration

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        print("[INFO] Starting control loop. Close viewer window to stop.")
        while viewer.is_running():
            start = time.time()

            # Current manual joint control input
            manual_ctrl = env.data.ctrl[ctrl_indices].copy()
            # Joint velocities and acceleration
            qvel = env.data.qvel[ctrl_indices].copy()
            accel = qvel - prev_qvel
            # Change in manual input (for boosting compliance)
            delta_manual = np.abs(manual_ctrl - prev_manual_ctrl)

            # Simulate external Cartesian force error (for admittance input)
            # Example: proportional control towards goal delta
            cartesian_force_error = 100.0 * (cartesian_goal_delta - admittance_ctrl.cart_pos)

            # Admittance control step: get desired Cartesian position delta
            desired_cartesian_pos = admittance_ctrl.step(cartesian_force_error, dt)
            desired_cartesian_delta = desired_cartesian_pos - admittance_ctrl.cart_pos  # Should be near zero since integrated

            # Convert Cartesian delta to joint delta using Jacobian
            try:
                delta_q_from_cartesian = cartesian_to_joint_delta(env, end_effector_site, desired_cartesian_delta)
            except Exception as e:
                print(f"[WARNING] Cartesian to Joint Mapping Failed: {e}")
                delta_q_from_cartesian = np.zeros_like(manual_ctrl)

            # Compute compliance factor based on joint velocities and manual input
            v_norm = np.linalg.norm(qvel)
            a_norm = np.linalg.norm(accel)
            m_norm = np.linalg.norm(delta_manual)

            raw_compliance = 1 / (1 + v_norm + 0.5 * a_norm)
            manual_boost = np.tanh(m_norm)
            compliance_factor = np.clip(raw_compliance + 0.05 * manual_boost, 0.01, 0.2)
            compliance_factors.append(compliance_factor)

            # Initialize threshold after 10 steps
            if step == 10 and not threshold_added:
                threshold = env.data.qpos[ctrl_indices].copy() - manual_ctrl
                threshold_added = True
                print(f"[INFO] Threshold computed at step {step}: {threshold}")

            if threshold_added:
                joint_compliance_delta = compliance_factor * 0.001 * threshold
            else:
                joint_compliance_delta = np.zeros_like(manual_ctrl)

            # Combine manual control + compliance + admittance (joint space)
            assisted_ctrl = manual_ctrl + joint_compliance_delta + delta_q_from_cartesian

            # Clip commands within joint limits
            assisted_ctrl_clipped = np.clip(assisted_ctrl, ctrl_ranges[:, 0], ctrl_ranges[:, 1])

            # Apply control commands to environment
            full_ctrl = env.data.ctrl.copy()
            for idx, c_idx in enumerate(ctrl_indices):
                full_ctrl[c_idx] = assisted_ctrl_clipped[idx]
            env.data.ctrl[:] = full_ctrl

            # Step simulation environment
            next_obs, reward, done, _ = env.step(env.data.ctrl)

            # Update state buffer for next iteration
            next_state = next_obs[ctrl_indices + [i + len(next_obs)//2 for i in ctrl_indices]]
            seq_buffer.append(next_state)

            # Log data
            rewards.append(reward)
            joint_positions.append(assisted_ctrl_clipped.copy())
            latencies.append(time.time() - start)

            # Prepare for next step
            prev_qvel = qvel
            prev_manual_ctrl = manual_ctrl
            viewer.sync()
            step += 1
            time.sleep(dt)

    # Convert logged data to numpy arrays for plotting
    joint_positions = np.array(joint_positions)
    rewards = np.array(rewards)
    latencies = np.array(latencies)
    compliance_factors = np.array(compliance_factors)

    # Print summary stats
    print(f"\n[SESSION CLOSED] Total Steps: {step}")
    print(f"Cumulative Reward: {rewards.sum():.4f}")
    print(f"Average Reward: {rewards.mean():.4f}")
    print(f"Action Smoothness Δqpos: {np.mean(np.abs(np.diff(joint_positions, axis=0))):.6f}")
    print(f"Joint Variance: {np.var(joint_positions, axis=0)}")
    print(f"Latency per Step: {np.mean(latencies):.6f}s")

    # Plotting
    fig, axs = plt.subplots(6, 1, figsize=(14, 30), constrained_layout=True)

    axs[0].plot(rewards, label="Reward", color='green')
    axs[0].legend()
    axs[0].set_title("Reward")

    axs[1].plot(latencies, label="Latency (s)", color='red')
    axs[1].legend()
    axs[1].set_title("Latency")

    axs[2].plot(np.abs(np.diff(joint_positions, axis=0)).mean(axis=1), label="Δqpos (Action Smoothness)", color='purple')
    axs[2].legend()
    axs[2].set_title("Action Smoothness")

    axs[3].bar(np.arange(len(joint_positions[0])), np.var(joint_positions, axis=0), label="Joint Variance", color='brown')
    axs[3].legend()
    axs[3].set_title("Joint Variance")

    im = axs[4].imshow(joint_positions.T, aspect='auto', cmap='viridis', interpolation='none')
    fig.colorbar(im, ax=axs[4], orientation='vertical')
    axs[4].set_title("Heatmap of Joint Positions Over Time")
    axs[4].set_xlabel("Timestep")
    axs[4].set_ylabel("Joint Index")

    axs[5].plot(compliance_factors, label="Compliance Factor", color='navy')
    axs[5].set_title("Compliance Factor Over Time")
    axs[5].set_xlabel("Step")
    axs[5].set_ylabel("Compliance Value")
    axs[5].legend()
    axs[5].grid(True)

    plt.show()

if __name__ == "__main__":
    main()
