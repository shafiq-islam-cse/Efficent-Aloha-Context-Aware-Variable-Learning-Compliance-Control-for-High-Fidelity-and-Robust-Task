import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import mujoco
import mujoco.viewer
import time
from collections import deque
from aloha_env import AlohaEnv
import matplotlib.pyplot as plt


# ====== Positional Encoding ======
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


# ====== Causal Mask ======
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return mask


# ====== Graph Attention Layer ======
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        Wh = self.W(h)  # (batch_size*num_chunks, chunk_size, out_features)
        batch_size, N, _ = Wh.size()
        Wh_i = Wh.unsqueeze(2).repeat(1, 1, N, 1)
        Wh_j = Wh.unsqueeze(1).repeat(1, N, 1, 1)
        e = self.leakyrelu(self.a(torch.cat([Wh_i, Wh_j], dim=-1))).squeeze(-1)  # (batch_size*num_chunks, N, N)
        attention = torch.softmax(e, dim=-1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)


# ====== BiACT Policy with Graph Attention ======
class GraphAttentionBiACTPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=128, nhead=8, num_layers=3, chunk_size=4):
        super().__init__()
        self.chunk_size = chunk_size
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.gat = GraphAttentionLayer(d_model, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)

        remainder = seq_len % self.chunk_size
        if remainder != 0:
            pad_len = self.chunk_size - remainder
            padding = torch.zeros(batch_size, pad_len, x.size(2), device=x.device)
            x = torch.cat([x, padding], dim=1)
            seq_len += pad_len

        chunks = x.view(batch_size, -1, self.chunk_size, x.size(2))
        b, num_chunks, csize, d = chunks.size()
        chunks_reshaped = chunks.view(b * num_chunks, csize, d)
        gat_out = self.gat(chunks_reshaped)
        chunk_embeds = gat_out.mean(dim=1).view(b, num_chunks, d)

        causal_mask = generate_square_subsequent_mask(num_chunks).to(x.device)
        decoded = self.transformer_decoder(tgt=chunk_embeds, memory=chunk_embeds, tgt_mask=causal_mask)
        pooled = decoded.mean(dim=1)
        return self.fc_out(pooled)


# ====== Meta-RL Agent with MAML ======
class MetaRLAgentWithMAML:
    def __init__(self, input_dim, output_dim, seq_len=20, inner_lr=1e-2, outer_lr=1e-3):
        self.policy = GraphAttentionBiACTPolicy(input_dim, output_dim)
        self.meta_optimizer = torch.optim.Adam(self.policy.parameters(), lr=outer_lr)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.batch_size = 8
        self.seq_len = seq_len
        self.inner_lr = inner_lr
        self.losses = []

    def act(self, state_seq):
        self.policy.eval()
        with torch.no_grad():
            # state_seq shape might be (seq_len, feature_dim) -> add batch dim:
            if len(state_seq.shape) == 2:
                x = torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0)
            elif len(state_seq.shape) == 3:
                x = torch.tensor(state_seq, dtype=torch.float32)
            else:
                raise ValueError(f"Unexpected state_seq shape: {state_seq.shape}")

            # Debug print
            # print(f"[DEBUG] act() input shape: {x.shape}")  # (1, seq_len, input_dim)

            output = self.policy(x).squeeze(0)
            return output.cpu().numpy()

    def remember(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))

    def compute_loss(self, model, batch):
        s, a, r, ns, d = zip(*batch)
        s = torch.tensor(np.array(s), dtype=torch.float32)
        a = torch.tensor(np.array(a), dtype=torch.float32)
        r = torch.tensor(np.array(r), dtype=torch.float32).unsqueeze(1)
        ns = torch.tensor(np.array(ns), dtype=torch.float32)
        d = torch.tensor(np.array(d), dtype=torch.float32).unsqueeze(1)

        q_pred = model(s)
        q_next = model(ns).detach().max(dim=1, keepdim=True)[0]
        target = r + self.gamma * q_next * (1 - d)
        return F.mse_loss(q_pred, target)

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        support_batch = batch[:self.batch_size // 2]
        query_batch = batch[self.batch_size // 2:]

        inner_model = GraphAttentionBiACTPolicy(
            input_dim=self.policy.input_proj.in_features,
            output_dim=self.policy.fc_out.out_features
        )
        inner_model.load_state_dict(self.policy.state_dict())

        inner_loss = self.compute_loss(inner_model, support_batch)
        grads = torch.autograd.grad(inner_loss, inner_model.parameters(), create_graph=True)
        updated_params = {name: param - self.inner_lr * grad
                          for (name, param), grad in zip(inner_model.named_parameters(), grads)}

        for name, param in inner_model.named_parameters():
            param.data = updated_params[name].data.clone()

        query_loss = self.compute_loss(inner_model, query_batch)

        self.meta_optimizer.zero_grad()
        query_loss.backward()
        self.meta_optimizer.step()
        self.losses.append(query_loss.item())


# ============================
# Jacobian-based IK Function
# ============================
def cartesian_to_joint_delta(env, end_effector_site, desired_dx):
    jacp = np.zeros((3, env.model.nv))
    jacr = np.zeros((3, env.model.nv))
    site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, end_effector_site)
    mujoco.mj_jacSite(env.model, env.data, jacp, jacr, site_id)
    J = jacp
    # Use only ctrl_indices columns of J
    J_joint = J[:, env.ctrl_indices]
    dq = np.linalg.pinv(J_joint) @ desired_dx
    return dq


# ============================
# Controller Helper
# ============================
def get_ctrl_indices(model):
    indices = [model.actuator(f"{side}_joint{i}").id for side in ("fl", "fr") for i in range(1, 9)]
    return indices


REWARD_THRESHOLD = 0.2  # Adjust based on your environment's reward scale
MANUAL_IDLE_TIMEOUT = 1.0  # seconds
reward_window = deque(maxlen=20)
last_manual_input_time = time.time()


def sigmoid(x, center=25, scale=5):
    """Sigmoid ramp: smooth transition from 0 to 1"""
    return 1 / (1 + np.exp(-(x - center) / scale))


# ============================
# Main Simulation
# ============================
def cosine_similarity(a, b, eps=1e-8):
    a_norm = np.linalg.norm(a) + eps
    b_norm = np.linalg.norm(b) + eps
    return np.dot(a, b) / (a_norm * b_norm)


def main():
    xml_path = r"D:\PhD\0PhD-Implementation\0ALOHA-ALL\mobile_aloha_sim-master\aloha_mujoco\aloha\meshes_mujoco\aloha_v1.xml"
    env = AlohaEnv(xml_path)
    ctrl_indices = get_ctrl_indices(env.model)
    env.ctrl_indices = ctrl_indices
    import numpy as np
    end_effector_site = "fl_gripper_site"
    cartesian_goal_delta = np.array([0.0, 0.0, 0.01])
    input_dim = len(ctrl_indices) * 2
    output_dim = len(ctrl_indices)
    seq_len = 20
    agent = MetaRLAgentWithMAML(input_dim=input_dim, output_dim=output_dim, seq_len=seq_len)

    ctrl_ranges = np.array([
        [-3.14158, 3.14158],  # fl_joint1
        [0, 3.14158],  # fl_joint2
        [0, 3.14158],  # fl_joint3
        [-2, 1.67],  # fl_joint4
        [-1.5708, 1.5708],  # fl_joint5
        [-3.14158, 3.14158],  # fl_joint6
        [0, 0.0475],  # fl_joint7
        [0, 0.0475],  # fl_joint8

        [-3.14158, 3.14158],  # fr_joint1
        [0, 3.14158],  # fr_joint2
        [0, 3.14158],  # fr_joint3
        [-2, 1.67],  # fr_joint4
        [-1.5708, 1.5708],  # fr_joint5
        [-3.14158, 3.14158],  # fr_joint6
        [0, 0.0475],  # fr_joint7
        [0, 0.0475],  # fr_joint8
    ])

    rewards, joint_positions, latencies, compliance_factors, actions_history = [], [], [], [], []
    alignment_scores = []
    assist_weights = []

    obs = env.reset()
    mujoco.mj_forward(env.model, env.data)

    print("[INFO] Launching viewer. Waiting for manual input to begin...")

    with (mujoco.viewer.launch_passive(env.model, env.data) as viewer):
        viewer.user_warning = "Move any joint manually to start..."

        # Wait for manual input
        while True:
            manual_ctrl = env.data.ctrl[ctrl_indices].copy()
            if np.linalg.norm(manual_ctrl) > 1e-3:
                print("[INFO] Manual control detected. Starting main loop...")
                break
            viewer.sync()
            time.sleep(0.01)

        # Initialize sequence buffer
        seq_buffer = deque(maxlen=seq_len)
        while len(seq_buffer) < seq_len:
            state = obs[ctrl_indices + [i + len(obs) // 2 for i in ctrl_indices]]
            seq_buffer.append(state)

        prev_qvel = np.zeros(len(ctrl_indices))
        prev_manual_ctrl = env.data.ctrl[ctrl_indices].copy()
        step = 0
        threshold_added = False
        threshold = None
        assist_weight = 0.0
        last_manual_move_step = 0
        manual_active = False

        def sigmoid_ramp(x, start=100, slope=0.05):
            return 1 / (1 + np.exp(-slope * (x - start)))

        while viewer.is_running():
            start = time.time()

            manual_ctrl = env.data.ctrl[ctrl_indices].copy()
            qvel = env.data.qvel[ctrl_indices].copy()
            accel = qvel - prev_qvel
            delta_manual = np.abs(manual_ctrl - prev_manual_ctrl)

            # Manual activity detection
            if np.linalg.norm(manual_ctrl) > 1e-3:
                manual_active = True
                last_manual_move_step = step
            elif step - last_manual_move_step > 100:
                manual_active = False

            try:
                delta_q_from_cartesian = cartesian_to_joint_delta(env, end_effector_site, cartesian_goal_delta)
            except Exception as e:
                print("[WARNING] Cartesian to Joint Mapping Failed:", e)
                delta_q_from_cartesian = np.zeros_like(manual_ctrl)

            state_seq = np.array(seq_buffer)
            # Normalize sequence input
            state_seq = (state_seq - np.mean(state_seq)) / (np.std(state_seq) + 1e-8)

            agent_action = agent.act(state_seq)
            actions_history.append(agent_action.copy())

            # Smoothed adaptive threshold
            if step >= 10:
                current_threshold = manual_ctrl - agent_action
                if threshold is None:
                    threshold = current_threshold
                else:
                    alpha = 0.1
                    threshold = (1 - alpha) * threshold + alpha * current_threshold
                threshold_added = True

            # Compliance factor calculation
            v_norm = np.linalg.norm(qvel)
            a_norm = np.linalg.norm(accel)
            m_norm = np.linalg.norm(delta_manual)

            raw_compliance = np.exp(-0.5 * v_norm - 0.25 * a_norm)
            manual_boost = 0.2 * np.tanh(2 * m_norm)
            compliance_factor = np.clip(raw_compliance + manual_boost, 0.01, 1.0)
            compliance_factors.append(compliance_factor)

            if threshold_added:
                joint_level_delta = compliance_factor * 0.01 * threshold
            else:
                joint_level_delta = np.zeros_like(manual_ctrl)

            def assistive_component(manual_ctrl, agent_action):
                return agent_action if np.dot(manual_ctrl, agent_action) >= 0 else np.zeros_like(agent_action)

            assistive_agent_action = assistive_component(manual_ctrl, agent_action)

            def project_along_manual(manual, vector):
                manual_norm = np.linalg.norm(manual)
                if manual_norm < 1e-6:
                    return np.zeros_like(vector)
                direction = manual / manual_norm
                projection = np.dot(vector, direction) * direction
                return projection if np.dot(projection, manual) >= 0 else np.zeros_like(vector)

            aligned_joint_delta = project_along_manual(manual_ctrl, joint_level_delta)
            aligned_cartesian_delta = project_along_manual(manual_ctrl, delta_q_from_cartesian)
            aligned_assistive_action = project_along_manual(manual_ctrl, assistive_agent_action)

            # Alignment score with cosine similarity
            alignment_score = cosine_similarity(manual_ctrl, agent_action)
            alignment_scores.append(alignment_score)

            # Clamp negative alignment to zero for assist weight
            alignment_score = max(0, alignment_score)
            alignment_score_clamped = alignment_score
            assist_weight = alignment_score_clamped * sigmoid_ramp(step)
            assist_weights.append(assist_weight)

            # Compose final control signal with assist weight
            if manual_active and step >= 100:
                adaptive_variable_learning_compliance_control = (
                        manual_ctrl
                        # + assist_weight +
                        + 0.001 * aligned_joint_delta
                        + 0.001 * aligned_cartesian_delta
                        + 0.0001 * aligned_assistive_action)
            else:
                adaptive_variable_learning_compliance_control = manual_ctrl

            adaptive_variable_learning_compliance_control_clipped = np.clip(
                adaptive_variable_learning_compliance_control,
                ctrl_ranges[:, 0],
                ctrl_ranges[:, 1]
            )

            full_ctrl = env.data.ctrl.copy()
            for idx, c_idx in enumerate(ctrl_indices):
                full_ctrl[c_idx] = adaptive_variable_learning_compliance_control_clipped[idx]
            env.data.ctrl[:] = full_ctrl

            next_obs, reward, done, _ = env.step(env.data.ctrl)
            next_state = next_obs[ctrl_indices + [i + len(next_obs) // 2 for i in ctrl_indices]]
            seq_buffer.append(next_state)

            # Add a small reward bonus proportional to alignment score
            reward += 0.5 * alignment_score_clamped

            agent.remember(state_seq, adaptive_variable_learning_compliance_control_clipped, reward,
                           list(seq_buffer), done)

            if step >= 20 and step % 20 == 0:
                print(f"[INFO] Agent retraining at step {step}")
                agent.train()

            rewards.append(reward)
            joint_positions.append(adaptive_variable_learning_compliance_control_clipped.copy())
            latencies.append(time.time() - start)

            print(
                f"[STEP {step}] "
                f"Reward: {reward:.4f} | AssistWeight: {assist_weight:.3f} | "
                f"Compliance: {compliance_factor:.5f} | "
                f"V_Norm: {v_norm:.4f} | A_Norm: {a_norm:.4f} | M_Norm: {m_norm:.4f} | "
                f"Control Norm: {np.linalg.norm(adaptive_variable_learning_compliance_control_clipped):.4f} | "
                f"Agent Action Norm: {np.linalg.norm(agent_action):.4f} | "
                f"Alignment Score: {alignment_score:.4f}\n"
                f"Clipped Control: {adaptive_variable_learning_compliance_control_clipped}\n"
                f"Agent Action: {agent_action}"
            )

            prev_qvel = qvel
            prev_manual_ctrl = manual_ctrl
            viewer.sync()
            step += 1
            time.sleep(0.01)

    window_size = 10
    losses = agent.losses


    if len(losses) >= window_size:
        smooth_loss = np.convolve(losses, np.ones(window_size) / window_size, mode='valid') * 10
    else:
        smooth_loss = losses

    fig_titles = [
        "Training Dynamics",
        "Joint Behavior Analysis",
        "Policy Control Characteristics"
    ]
    import matplotlib.pyplot as plt
    # Figure 5: Training Dynamics
    fig1, axs1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle(fig_titles[0], fontsize=16)

    axs1[0, 0].plot(smooth_loss, label="Agent Smoothed Loss (x10)", color='blue')
    #axs1[0, 0].set_title("Smoothed Loss", fontsize=14)
    axs1[0, 0].set_xlabel("Step", fontsize=16)
    axs1[0, 0].set_ylabel("Smoothed Loss (x10)", fontsize=16)
    axs1[0, 0].legend(loc='upper right',fontsize=16)
    axs1[0, 0].tick_params(axis='both', labelsize=16)

    axs1[0, 1].plot(losses, label="Agent Raw Loss per step", color='orange')
    #axs1[0, 1].set_title("Raw Loss", fontsize=14)
    axs1[0, 1].set_xlabel("Step", fontsize=16)
    axs1[0, 1].set_ylabel("Raw Loss", fontsize=16)
    axs1[0, 1].legend(loc='upper right',fontsize=16)
    axs1[0, 1].tick_params(axis='both', labelsize=16)

    axs1[1, 0].plot(rewards, label="Reward per step", color='green')
    # axs1[1, 0].set_title("Reward per Step", fontsize=14)
    axs1[1, 0].set_xlabel("Step", fontsize=16)
    axs1[1, 0].set_ylabel("Reward", fontsize=16)
    axs1[1, 0].legend(loc='upper right',fontsize=16)
    axs1[1, 0].tick_params(axis='both', labelsize=16)

    axs1[1, 1].plot(latencies, label="Latency per step", color='green')
    #axs1[1, 1].set_title("Step Latency", fontsize=14)
    axs1[1, 1].set_xlabel("Step", fontsize=16)
    axs1[1, 1].set_ylabel("Latency (s)", fontsize=16)
    axs1[1, 1].legend(loc='upper right',fontsize=16)
    axs1[1, 1].tick_params(axis='both', labelsize=16)
    fig1.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.draw()
    plt.savefig("5.pdf", format='pdf', dpi=1000)
    plt.show()

    # Figure 6: Joint Behavior Analysis
    fig2, axs2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle(fig_titles[1], fontsize=16)

    if len(actions_history) > 1:
        axs2[0, 0].plot(np.abs(np.diff(actions_history, axis=0)).mean(axis=1), label="Action Smoothness over time", color='blue')
    else:
        axs2[0, 0].plot([], label="Action Smoothness", color='purple')
    #axs2[0, 0].set_title("Action Smoothness", fontsize=14)
    axs2[0, 0].set_xlabel("Step", fontsize=16)
    axs2[0, 0].set_ylabel("Change in joint position Δqpos", fontsize=16)
    axs2[0, 0].legend(loc='upper right',fontsize=16)
    axs2[0, 0].tick_params(axis='both', labelsize=16)

    axs2[0, 1].bar(
        np.arange(len(joint_positions[0])),
        np.var(joint_positions, axis=0),
        color='brown',
        label='Joint Stability (Variance)'
    )
    # Remove the title
    # axs2[0, 1].set_title("Joint Stability (Variance)", fontsize=14)

    axs2[0, 1].set_xlabel("Joint Index", fontsize=16)
    axs2[0, 1].set_ylabel("Variance", fontsize=16)
    axs2[0, 1].legend(loc='upper right',fontsize=16)
    axs2[0, 1].tick_params(axis='both', labelsize=16)


    im = axs2[1, 0].imshow(np.array(joint_positions).T, aspect='auto', cmap='viridis', interpolation='none')
    fig2.colorbar(im, ax=axs2[1, 0])

    # Remove title and use legend instead
    # axs2[1, 0].set_title("Heatmap of Joint Positions", fontsize=14)
    axs2[1, 0].set_xlabel("Timestep", fontsize=16)
    axs2[1, 0].set_ylabel("Joint Index", fontsize=16)

    # Add legend entry for the colorbar (as a workaround)
    axs2[1, 0].plot([], [], label="Heatmap Joint Positions", color='black')
    axs2[1, 0].legend(loc='upper right',fontsize=16)
    axs2[1, 0].tick_params(axis='both', labelsize=16)

    axs2[1, 1].bar(
        np.arange(len(joint_positions[0])),
        np.max(joint_positions, axis=0) - np.min(joint_positions, axis=0),
        color='darkcyan',
        label='Joint Movement Range'
    )
    # Remove title
    # axs2[1, 1].set_title("Joint Movement Range", fontsize=14)

    axs2[1, 1].set_xlabel("Joint Index", fontsize=16)
    axs2[1, 1].set_ylabel("Movement Range", fontsize=16)
    axs2[1, 1].legend(loc='upper right',fontsize=16)
    axs2[1, 1].tick_params(axis='both', labelsize=16)
    fig2.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.draw()
    plt.savefig("6.pdf", format='pdf', dpi=1000)
    plt.show()



    # Figure 7: Policy Control Characteristics
    fig3, axs3 = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle(fig_titles[2], fontsize=16)

    axs3[0, 0].plot(compliance_factors, label="Compliance Factor used per step", color='blue')
    #axs3[0, 0].set_title("Compliance Factor Over Time", fontsize=14)
    axs3[0, 0].set_xlabel("Step", fontsize=16)
    axs3[0, 0].set_ylabel("Compliance Factor", fontsize=16)
    axs3[0, 0].legend(loc='upper right',fontsize=16)
    axs3[0, 0].tick_params(axis='both', labelsize=16)

    axs3[0, 1].plot(np.linalg.norm(joint_positions, axis=1), label="Joint Velocity Norm over step", color='green')
    #axs3[0, 1].set_title("Joint Velocity Norm", fontsize=14)
    axs3[0, 1].set_xlabel("Step", fontsize=16)
    axs3[0, 1].set_ylabel("Velocity Norm ||q̇||", fontsize=16)
    axs3[0, 1].legend(loc='upper right',fontsize=16)
    axs3[0, 1].tick_params(axis='both', labelsize=16)

    axs3[1, 0].plot(np.linalg.norm(np.diff(joint_positions, axis=0), axis=1), label="Δ Change in Joint Velocity Norm", color='blue')
    #axs3[1, 0].set_title("Change in Joint Velocity Norm", fontsize=14)
    axs3[1, 0].set_xlabel("Step", fontsize=16)
    axs3[1, 0].set_ylabel("Δ||q̇|| Velocity Norm ", fontsize=16)
    axs3[1, 0].legend(loc='upper right',fontsize=16)
    axs3[1, 0].tick_params(axis='both', labelsize=16)

    axs3[1, 1].plot(np.gradient(compliance_factors), label="ΔChange in  Compliance Factor", color='green')
    #axs3[1, 1].set_title("Change in Compliance Factor", fontsize=14)
    axs3[1, 1].set_xlabel("Step", fontsize=16)
    axs3[1, 1].set_ylabel("Δ Compliance", fontsize=16)
    axs3[1, 1].legend(loc='upper right',fontsize=16)
    axs3[1, 1].tick_params(axis='both', labelsize=16)
    fig3.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.draw()
    plt.savefig("7.pdf", format='pdf', dpi=1000)
    plt.show()



    # Figure 8: Alignment Score Over Time
    fig4b = plt.figure(figsize=(14, 7))
    plt.title("Agent Assisted Alignment Score Over Time", fontsize=27)

    alignment_scores_arr = np.array(alignment_scores)
    plt.plot(alignment_scores_arr,
             label="Alignment Score",
             color='blue',
             linestyle='-',
             linewidth=2.5,
             marker='*',
             markersize=4)

    plt.xlabel("Step", fontsize=27)
    plt.ylabel("Alignment Score (Dot Product)", fontsize=27)
    plt.legend(loc='upper right',fontsize=27)
    plt.xticks(fontsize=27)
    plt.yticks(fontsize=27)
    plt.grid(True, linestyle=':', alpha=0.7)  # Optional: adds a subtle grid
    plt.tight_layout()
    plt.draw()
    plt.savefig("8.pdf", format='pdf', dpi=1000)
    plt.show()
    '''
    # Figure 9: Agent Actions Over Time
    from itertools import cycle

    # Define cycles for line styles and markers
    line_styles = cycle(['-', '--', '-.', ':'])
    markers = cycle(['', 'o', 's', 'D', 'x', '^', '*', 'v'])

    # Figure Agent Actions Over Time
    fig4a = plt.figure(figsize=(14, 7))
    plt.title("Agent Assisted Actions Over Time Step", fontsize=22)
    actions_arr = np.array(actions_history)

    for joint_idx in range(actions_arr.shape[1]):
        style = next(line_styles)
        marker = next(markers)
        plt.plot(actions_arr[:, joint_idx],
                 label=f"Joint {joint_idx + 1}",
                 linestyle=style,
                 marker=marker,
                 linewidth=1,
                 markersize=1)

    plt.xlabel("Step", fontsize=22)
    plt.ylabel("Agent Action Value", fontsize=22)
    plt.legend(loc='upper right', fontsize=18, ncol=4)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.draw()
    plt.savefig("9old.pdf", format='pdf', dpi=1000)
    plt.show()
    '''
    # Figure 9: Agent Actions Over Time - Two Subplots (Left Arm and Right Arm)
    from itertools import cycle
    import matplotlib.pyplot as plt
    import numpy as np

    # Define cycles for line styles and markers
    line_styles = cycle(['-', '--', '-.', ':'])
    markers = cycle(['', 'o', 's', 'D', 'x', '^', '*', 'v'])

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))  # wider for legend space
    fig.suptitle("Agent Assisted Actions Over Time Step", fontsize=20)

    # Convert actions_history to numpy array
    actions_arr = np.array(actions_history)

    # Left arm subplot (first 8 joints)
    ax1.set_title("Left Arm Joints", fontsize=20)
    for joint_idx in range(8):  # First 8 joints for left arm
        style = next(line_styles)
        marker = next(markers)
        ax1.plot(actions_arr[:, joint_idx],
                 label=f"Joint {joint_idx + 1}",
                 linestyle=style,
                 marker=marker,
                 linewidth=1,
                 markersize=1)

    ax1.set_ylabel("Agent Action Value", fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=20)

    # Legend outside (right side)
    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=16, ncol=1, frameon=True)

    # Reset cycles for right arm
    line_styles = cycle(['-', '--', '-.', ':'])
    markers = cycle(['', 'o', 's', 'D', 'x', '^', '*', 'v'])

    # Right arm subplot (next 8 joints)
    ax2.set_title("Right Arm Joints", fontsize=20)
    for joint_idx in range(8, 16):  # Next 8 joints for right arm
        style = next(line_styles)
        marker = next(markers)
        ax2.plot(actions_arr[:, joint_idx],
                 label=f"Joint {joint_idx + 1}",
                 linestyle=style,
                 marker=marker,
                 linewidth=1,
                 markersize=1)

    ax2.set_xlabel("Step", fontsize=20)
    ax2.set_ylabel("Agent Action Value", fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)

    # Legend outside (right side)
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=16, ncol=1, frameon=True)

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])  # leave space on right and top
    plt.subplots_adjust(top=0.9)
    plt.savefig("9.pdf", format='pdf', dpi=1000, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
