# === Necessary Imports ===
import mujoco
from mujoco import viewer
import time
import time as pytime  # avoid conflict with mujoco.time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from contextlib import nullcontext
import os
import itertools

# === Part 1: Run MuJoCo Manual Demonstration and Save to CSV ===
xml_path = r"D:\PhD\0PhD-Implementation\0ALOHA-ALL\mobile_aloha_sim-master\aloha_mujoco\aloha\meshes_mujoco\aloha_v1.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

ctrl_ranges = np.array([
    [-3.14158, 3.14158], [0, 3.14158], [0, 3.14158], [-2, 1.67], [-1.5708, 1.5708], [-3.14158, 3.14158], [0, 0.0475], [0, 0.0475],
    [-3.14158, 3.14158], [0, 3.14158], [0, 3.14158], [-2, 1.67], [-1.5708, 1.5708], [-3.14158, 3.14158], [0, 0.0475], [0, 0.0475]
])

demonstration_data = []
save_path = "NewData.csv"

with viewer.launch_passive(model, data) as v:
    print("MuJoCo viewer launched. Use GUI to manually control the robot.")
    print("Press Esc or close the window to exit.")
    while v.is_running():
        mujoco.mj_step(model, data)
        v.sync()
        pytime.sleep(model.opt.timestep)

        ctrl_clipped = np.clip(data.ctrl[:len(ctrl_ranges)], ctrl_ranges[:, 0], ctrl_ranges[:, 1])
        timestep_data = {
            'time': float(data.time),
            'qpos': data.qpos.tolist(),
            'qvel': data.qvel.tolist(),
            'ctrl': ctrl_clipped.tolist()
        }
        qpos_clean = data.qpos.tolist()
        qvel_clean = data.qvel.tolist()
        ctrl_clean = ctrl_clipped.tolist()

        print(f"[Time: {data.time:.4f}]")
        print(f"  qpos : {qpos_clean}")
        print(f"  qvel : {qvel_clean}")
        print(f"  ctrl : {ctrl_clean}")
        demonstration_data.append(timestep_data)

with open(save_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['time'] +
                    [f'qpos_{i}' for i in range(len(data.qpos))] +
                    [f'qvel_{i}' for i in range(len(data.qvel))] +
                    [f'ctrl_{i}' for i in range(ctrl_ranges.shape[0])])
    for step in demonstration_data:
        row = [step['time']] + step['qpos'] + step['qvel'] + step['ctrl']
        writer.writerow(row)

# === Load CSV and Prepare Dataset ===
data = pd.read_csv(save_path2)
qpos_cols = [c for c in data.columns if c.startswith('qpos_')]
qvel_cols = [c for c in data.columns if c.startswith('qvel_')]
ctrl_cols = [c for c in data.columns if c.startswith('ctrl_')]

X = data[qpos_cols + qvel_cols].values.astype(np.float32)
y = data[ctrl_cols].values.astype(np.float32)

class DemoDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs)
        self.targets = torch.tensor(targets)
    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx): return self.inputs[idx], self.targets[idx]

dataloader = DataLoader(DemoDataset(X, y), batch_size=32, shuffle=True)

input_dim = X.shape[1]
output_dim = y.shape[1]

# === Define Models ===
class FeedForwardPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim))
    def forward(self, x): return self.net(x)

class BiACT(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = FeedForwardPolicy(input_dim, output_dim)
    def forward(self, x): return self.net(x)

class LightweightGATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.attention = nn.Parameter(torch.empty(size=(1, out_features)))
        nn.init.xavier_uniform_(self.attention.data, gain=1.414)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, h, adj):
        Wh = self.linear(h)
        attn_scores = torch.matmul(Wh, self.attention.t())
        e = attn_scores + attn_scores.T
        e = self.leakyrelu(e)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = torch.softmax(attention, dim=1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, Wh)
        return self.layer_norm(h_prime)

class MetaRL_LightGAT_BiACT(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.gat = LightweightGATLayer(input_dim, 48, dropout=0.3)
        self.biact = BiACT(48, output_dim)
    def forward(self, x, adj):
        x = self.gat(x, adj)
        return self.biact(x)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, output_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

class DDPGActor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.8)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.8)
        self.fc3 = nn.Linear(64, output_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return torch.tanh(self.fc3(x))

class SACActor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout1 = nn.Dropout(0.8)
        self.fc2 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(0.8)
        self.mean = nn.Linear(64, output_dim)
        self.log_std = nn.Linear(64, output_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        std = torch.exp(torch.clamp(log_std, -20, 2))
        return mean, std

class PPOActor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.dropout1 = nn.Dropout(0.8)
        self.fc2 = nn.Linear(32, 32)
        self.dropout2 = nn.Dropout(0.8)
        self.mean = nn.Linear(32, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.dropout1(x)
        x = torch.sigmoid(self.fc2(x))
        x = self.dropout2(x)
        mean = self.mean(x)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

class GRAIL(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.gat1 = LightweightGATLayer(input_dim, 64)
        self.gat2 = LightweightGATLayer(64, 64)
        self.fc = nn.Linear(64, output_dim)
    def forward(self, x, adj):
        x = self.gat1(x, adj)
        x = F.relu(x)
        x = self.gat2(x, adj)
        return self.fc(x)

class A2C(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        action_logits = self.actor(x)
        state_value = self.critic(x)
        return action_logits, state_value

# === Metrics ===
def classification_accuracy(y_true, y_pred):
    y_true, y_pred = y_true.cpu().numpy(), y_pred.cpu().numpy()
    y_pred_classes = np.round(y_pred)
    y_true_classes = np.round(y_true)
    return (y_true_classes == y_pred_classes).mean()

def compute_metrics(y_true, y_pred):
    y_true, y_pred = y_true.cpu().numpy(), y_pred.cpu().numpy()
    mse = ((y_true - y_pred) ** 2).mean()
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, r2

def create_adj_matrix(bs):
    return torch.eye(bs).to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == 'cuda':
    autocast_context = torch.amp.autocast(device_type='cuda')
    scaler = GradScaler()
else:
    autocast_context = nullcontext()
    scaler = None

# Only keep the specified models
models = {
    "MetaRL (LightGAT+BiACT)": MetaRL_LightGAT_BiACT(input_dim, output_dim),
    "DQN": DQN(input_dim, output_dim),
    "DDPGActor": DDPGActor(input_dim, output_dim),
    "SACActor": SACActor(input_dim, output_dim),
    "PPOActor": PPOActor(input_dim, output_dim),
    "GRAIL": GRAIL(input_dim, output_dim),
    "A2C": A2C(input_dim, output_dim)
}

def train_supervised(model, dataloader, epochs=30):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)# All highest performance if i use lr=1e-3, and it got low if lr=1e-8
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    criterion = nn.MSELoss()

    losses, r2s, rmses, times, accs = [], [], [], [], []

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        all_preds, all_targets = [], []

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            with autocast_context:
                if isinstance(model, MetaRL_LightGAT_BiACT):
                    adj = create_adj_matrix(batch_x.size(0))
                    outputs = model(batch_x, adj)
                elif isinstance(model, GRAIL):
                    adj = create_adj_matrix(batch_x.size(0))
                    outputs = model(batch_x, adj)
                elif isinstance(model, A2C):
                    action_logits, _ = model(batch_x)
                    outputs = action_logits
                elif isinstance(model, (SACActor, PPOActor)):
                    mean, _ = model(batch_x)
                    outputs = mean
                else:
                    outputs = model(batch_x)

                loss = criterion(outputs, batch_y)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            all_preds.append(outputs.detach().cpu())
            all_targets.append(batch_y.cpu())

        scheduler.step()
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(dataloader.dataset)
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)

        with torch.no_grad():
            mse, rmse, r2 = compute_metrics(targets, preds)
            acc = classification_accuracy(targets, preds)

        losses.append(avg_loss)
        r2s.append(r2)
        rmses.append(rmse)
        times.append(epoch_time)
        accs.append(acc)

        print(f"Epoch {epoch+1}/{epochs} Accuracy: {acc:.4f} - Loss: {avg_loss:.6f} - R2: {r2:.4f} - RMSE: {rmse:.4f} - Time: {epoch_time:.2f}s")

    return losses, r2s, rmses, times, accs

# === Train All Models ===
train_histories = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    losses, r2s, rmses, times, accs = train_supervised(model, dataloader, epochs=30)
    train_histories[name] = {
        "acc": accs,
        "loss": losses,
        "r2": r2s,
        "rmse": rmses,
        "time": times
    }

# === Plotting with Average in Legend, Save as Figure 6.eps to Figure 10.eps ===
os.makedirs("plots", exist_ok=True)

line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'd', 'P', '*', 'X', 'H', 'v']
colors = plt.cm.tab20.colors  # Up to 20 distinct colors

metrics_to_plot = ["acc", "loss", "r2", "rmse", "time"]
metric_titles = {
    "acc": "Accuracy",
    "loss": "Training Loss",
    "r2": "R² Score",
    "rmse": "Root Mean Squared Error",
    "time": "Epoch Time (s)"
}
figure_filenames = ["11.pdf", "12.pdf", "r2.pdf", "13.pdf", "14.pdf"]

for metric, fig_name in zip(metrics_to_plot, figure_filenames):
    plt.figure(figsize=(7, 5.5))
    style_cycle = itertools.cycle(zip(line_styles, markers, colors))
    for name, hist in train_histories.items():
        if metric in hist:
            style, marker, color = next(style_cycle)
            values = hist[metric]
            plt.plot(values, label=f"{name} (avg={np.mean(values):.4f})",
                     linestyle=style, marker=marker, color=color, alpha=0.9)
    plt.xlabel("Epoch")
    plt.ylabel(metric_titles[metric])
    plt.grid(True)
    plt.legend(fontsize=12, loc="upper right")
    plt.tight_layout()

    for line in plt.gca().get_lines():
        line.set_alpha(1.0)
    plt.draw()
    # Save and Show
    plt.savefig(f"plots/{fig_name}", format='pdf', dpi=1000)
    plt.show()
    plt.close()
