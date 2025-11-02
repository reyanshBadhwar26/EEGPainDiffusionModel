import os
import numpy as np
import mne
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ========================== PART 1: Preprocessing =============================

def preprocess_all_eeg(root_dir="raw_data", save_path="preprocessed_eeg.npy"):
    all_data = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".vhdr"):
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                try:
                    raw = mne.io.read_raw_brainvision(file_path, preload=True)
                    raw.resample(250)
                    raw.filter(0.5, 100, fir_design='firwin')
                    raw.notch_filter(50)

                    ica = mne.preprocessing.ICA(n_components=20, random_state=42, max_iter='auto')
                    ica.fit(raw.copy().filter(l_freq=1.0, h_freq=None))
                    raw_clean = ica.apply(raw.copy())

                    epochs = mne.make_fixed_length_epochs(raw_clean, duration=3.0, preload=True)
                    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)

                    # Z-score normalization
                    for i in range(data.shape[0]):
                        for ch in range(data.shape[1]):
                            data[i, ch] = (data[i, ch] - np.mean(data[i, ch])) / (np.std(data[i, ch]) + 1e-8)

                    all_data.append(data)

                except Exception as e:
                    print(f"Skipping {file_path} due to error: {e}")

    if len(all_data) == 0:
        raise RuntimeError("No valid EEG files found.")

    combined_data = np.concatenate(all_data, axis=0)
    print(f"Total epochs collected: {combined_data.shape}")
    np.save(save_path, combined_data)
    return combined_data

# ========================= PART 2: DDPM Model =================================

# class SinusoidalPosEmb(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, x):
#         device = x.device
#         half_dim = self.dim // 2
#         emb = torch.exp(torch.arange(half_dim, device=device) * -(np.log(10000) / (half_dim - 1)))
#         emb = x[:, None] * emb[None, :]
#         return torch.cat((emb.sin(), emb.cos()), dim=-1)

# class ResidualBlockFixed(nn.Module):
#     def __init__(self, in_ch, out_ch, t_dim):
#         super().__init__()
#         self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
#         self.time_fc = nn.Linear(t_dim, out_ch)
#         self.bn1 = nn.BatchNorm1d(out_ch)
#         self.bn2 = nn.BatchNorm1d(out_ch)
#         self.res_conv = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

#     def forward(self, x, t):
#         h = self.conv1(x)
#         h = self.bn1(h)
#         h += self.time_fc(t).unsqueeze(-1)
#         h = F.relu(h)
#         h = self.conv2(h)
#         h = self.bn2(h)
#         return F.relu(h + self.res_conv(x))


# class EEGDiffusionModel(nn.Module):
#     def __init__(self, ch=32, t_dim=128):
#         super().__init__()
#         self.time_emb = SinusoidalPosEmb(t_dim)
#         self.time_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.ReLU())
#         self.conv_in = nn.Conv1d(ch, 64, 3, padding=1)
#         self.res1 = ResidualBlockFixed(64, 128, t_dim)
#         self.res2 = ResidualBlockFixed(128, 256, t_dim)
#         self.res3 = ResidualBlockFixed(256, 128, t_dim)
#         self.res4 = ResidualBlockFixed(128, 64, t_dim)
#         self.conv_out = nn.Conv1d(64, ch, 3, padding=1)

#     def forward(self, x, t):
#         t = self.time_mlp(self.time_emb(t))
#         x = self.conv_in(x)      # -> 64
#         x = self.res1(x, t)      # -> 128
#         x = self.res2(x, t)      # -> 256
#         x = self.res3(x, t)      # -> 128
#         x = self.res4(x, t)      # -> 64
#         return self.conv_out(x)  # -> ch

# ========================= UNet Components ====================================
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_fc = nn.Linear(t_dim, out_channels)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x, t):
        t_emb = self.time_fc(t).unsqueeze(-1)
        x = self.conv1(x) + t_emb
        x = F.relu(self.norm1(x))
        x = self.conv2(x)
        x = F.relu(self.norm2(x))
        x_down = self.pool(x)
        return x_down, x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, t_dim):
        super().__init__()
        self.upconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv1d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_fc = nn.Linear(t_dim, out_channels)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)

    def forward(self, x, skip, t):
        t_emb = self.time_fc(t).unsqueeze(-1)
        x = self.upconv(x)

        # Fix time dimension mismatch
        if x.shape[-1] != skip.shape[-1]:
            diff = skip.shape[-1] - x.shape[-1]
            if diff > 0:
                x = F.pad(x, (0, diff))
            elif diff < 0:
                skip = F.pad(skip, (0, -diff))

        x = torch.cat((x, skip), dim=1)
        x = self.conv1(x) + t_emb
        x = F.relu(self.norm1(x))
        x = self.conv2(x)
        x = F.relu(self.norm2(x))
        return x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

# ========================== UNet Model ========================================

class EEGUNet(nn.Module):
    def __init__(self, ch=66, t_dim=128):
        super().__init__()
        self.time_emb = SinusoidalPosEmb(t_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim),
            nn.ReLU()
        )

        self.in_conv = nn.Conv1d(ch, 64, kernel_size=3, padding=1)

        self.down1 = DownBlock(64, 128, t_dim)
        self.down2 = DownBlock(128, 256, t_dim)

        self.mid_conv1 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.mid_conv2 = nn.Conv1d(256, 256, kernel_size=3, padding=1)

        self.up1 = UpBlock(256, 128, skip_channels=256, t_dim=t_dim)
        self.up2 = UpBlock(128, 64, skip_channels=128, t_dim=t_dim)

        self.out_conv = nn.Conv1d(64, ch, kernel_size=3, padding=1)

    def forward(self, x, t):
        t = self.time_mlp(self.time_emb(t))

        x = self.in_conv(x)

        x1, skip1 = self.down1(x, t)
        x2, skip2 = self.down2(x1, t)

        x_mid = F.relu(self.mid_conv1(x2))
        x_mid = F.relu(self.mid_conv2(x_mid))

        x = self.up1(x_mid, skip2, t)
        x = self.up2(x, skip1, t)

        return self.out_conv(x)


# ========================= PART 3: Diffusion Core ==============================

def linear_beta_schedule(timesteps):
    return torch.linspace(1e-4, 0.02, timesteps)

def get_noise_schedule(T):
    beta = linear_beta_schedule(T)
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_bar

def forward_diffusion(x0, t, alpha_bar):
    noise = torch.randn_like(x0)
    sqrt_ab = alpha_bar[t][:, None, None]
    sqrt_one_minus_ab = (1 - alpha_bar[t])[:, None, None]
    return sqrt_ab * x0 + sqrt_one_minus_ab * noise, noise

# ========================= PART 4: Training Loop ===============================

def train_ddpm(data, epochs=30, T=200, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGUNet(ch=data.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    beta, alpha, alpha_bar = get_noise_schedule(T)
    alpha_bar = alpha_bar.to(device)

    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)  # (N, C, T)
    if data_tensor.ndim == 3:
        data_tensor = data_tensor.permute(0, 1, 2)  # [B, C, T]

    for epoch in range(epochs):
        model.train()
        losses = []
        perm = torch.randperm(len(data_tensor))
        for i in tqdm(range(0, len(data_tensor), batch_size), desc=f"Epoch {epoch+1}"):
            idx = perm[i:i + batch_size]
            x0 = data_tensor[idx]
            t = torch.randint(0, T, (x0.shape[0],), device=device).long()
            xt, noise = forward_diffusion(x0, t, alpha_bar)
            pred_noise = model(xt, t.float())
            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1} - Avg Loss: {np.mean(losses):.4f}")
    
    torch.save(model.state_dict(), "eeg_model/ddpm_model.pth")
    return model, alpha_bar

# ========================= PART 5: Sampling ====================================

def sample_ddpm(model, T=200, shape=(1, 32, 750)):
    device = next(model.parameters()).device
    beta, alpha, alpha_bar = get_noise_schedule(T)
    alpha = alpha.to(device)
    alpha_bar = alpha_bar.to(device)

    x = torch.randn(shape).to(device)
    for t in reversed(range(T)):
        t_tensor = torch.full((shape[0],), t, device=device)
        z = torch.randn_like(x) if t > 0 else 0
        pred_noise = model(x, t_tensor.float())
        alpha_t = alpha[t]
        alpha_bar_t = alpha_bar[t]
        x = (1 / torch.sqrt(alpha_t)) * (x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * pred_noise) + torch.sqrt(beta[t]) * z
    return x.detach().cpu().numpy()


# STEP 1: Preprocess EEG from all subjects
#data = preprocess_all_eeg("raw_data")

#data = np.load("eeg_model/preprocessed_eeg.npy")
# STEP 2: Train DDPM model
#model, alpha_bar = train_ddpm(data, epochs=50)

model = EEGUNet(ch=66)
model.load_state_dict(torch.load("ddpm_model.pth", map_location=torch.device("cpu")))

# STEP 3: Sample synthetic EEG
synthetic = sample_ddpm(model, shape=(10, 66, 750))  # generate 10 samples
np.save("synthetic_eeg.npy", synthetic)
