import os
import numpy as np
import mne
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ========================== MINIMAL PREPROCESSING =============================

def preprocess_all_eeg_minimal(root_dir="raw_data", save_path="preprocessed_eeg.npy"):
    """
    Minimal EEG preprocessing for diffusion model training:
    - Resample to 250 Hz
    - Bandpass filter 0.5–100 Hz
    - Epoch into fixed-length windows (3s)
    - Z-score normalization per channel
    """
    all_data = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".vhdr"):
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                try:
                    # Load raw EEG
                    raw = mne.io.read_raw_brainvision(file_path, preload=True)

                    # Resample to 250 Hz
                    raw.resample(250)

                    # Bandpass filter
                    raw.filter(0.5, 100, fir_design='firwin')

                    # Epoch: fixed-length 3s
                    epochs = mne.make_fixed_length_epochs(raw, duration=3.0, preload=True)
                    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)

                    # Z-score normalization per channel
                    mean = data.mean(axis=2, keepdims=True)
                    std = data.std(axis=2, keepdims=True) + 1e-8
                    data = (data - mean) / std

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


# ========================= UNet (Deep) ========================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freq = torch.exp(
            torch.arange(half, device=device) * -(torch.log(torch.tensor(10000.0, device=device)) / (half - 1))
        )
        emb = t[:, None] * freq[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TimeMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t):
        return self.net(t)


def _gn(c):
    return nn.GroupNorm(num_groups=min(32, c), num_channels=c)


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = _gn(out_ch)
        self.norm2 = _gn(out_ch)
        self.act = nn.SiLU()
        self.time_fc = nn.Linear(t_dim, out_ch)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = h + self.time_fc(t_emb).unsqueeze(-1)
        h = self.act(h)
        h = self.drop(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return self.act(h + self.skip(x))


class Downsample1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.convT = nn.ConvTranspose1d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.convT(x)


class EEGUNetDeep(nn.Module):
    def __init__(
        self,
        ch: int = 66,
        base_channels: int = 64,
        channel_mults=(1, 2, 4, 8, 8),
        num_res_blocks: int = 2,
        t_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_conv = nn.Conv1d(ch, base_channels, kernel_size=3, padding=1)

        self.time_emb = SinusoidalPosEmb(t_dim)
        self.time_mlp = TimeMLP(t_dim, t_dim)

        in_ch = base_channels
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_channels = []

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            num_blocks = num_res_blocks + (0 if i == 0 else 1)
            for j in range(num_blocks):
                blocks.append(ResBlock1D(in_ch, out_ch, t_dim, dropout))
                in_ch = out_ch
                self.skip_channels.append(out_ch)
            self.down_blocks.append(blocks)
            if i != len(channel_mults) - 1:
                self.downsamples.append(Downsample1D(in_ch))
            else:
                self.downsamples.append(nn.Identity())

        mid_ch = in_ch
        self.mid = nn.ModuleList([
            ResBlock1D(mid_ch, mid_ch, t_dim, dropout),
            ResBlock1D(mid_ch, mid_ch, t_dim, dropout),
        ])

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.skip_channels = list(reversed(self.skip_channels))

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks + (0 if i == len(channel_mults)-1 else 1)):
                skip_ch = self.skip_channels.pop(0)
                blocks.append(ResBlock1D(in_ch + skip_ch, out_ch, t_dim, dropout))
                in_ch = out_ch
            self.up_blocks.append(blocks)
            if i != 0:
                self.upsamples.append(Upsample1D(in_ch))
            else:
                self.upsamples.append(nn.Identity())

        self.out_norm = _gn(base_channels)
        self.out_act  = nn.SiLU()
        self.out_conv = nn.Conv1d(base_channels, ch, kernel_size=3, padding=1)

    def forward(self, x, t):
        if t.dtype != torch.float32:
            t = t.float()
        t_emb = self.time_mlp(self.time_emb(t))

        skips = []
        x = self.in_conv(x)
        for blocks, down in zip(self.down_blocks, self.downsamples):
            for blk in blocks:
                x = blk(x, t_emb)
                skips.append(x)
            x = down(x)

        for blk in self.mid:
            x = blk(x, t_emb)

        for blocks, up in zip(self.up_blocks, self.upsamples):
            for blk in blocks:
                skip = skips.pop()
                if x.shape[-1] != skip.shape[-1]:
                    diff = skip.shape[-1] - x.shape[-1]
                    if diff > 0:
                        x = F.pad(x, (0, diff))
                    elif diff < 0:
                        skip = F.pad(skip, (0, -diff))
                x = torch.cat([x, skip], dim=1)
                x = blk(x, t_emb)
            x = up(x)

        # ---------- Output ----------
        x = self.out_norm(x)
        x = self.out_act(x)
        x = self.out_conv(x)
        return x  # fixed: removed early return inside loop


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
    model = EEGUNetDeep(ch=data.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    beta, alpha, alpha_bar = get_noise_schedule(T)
    alpha_bar = alpha_bar.to(device)

    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    if data_tensor.ndim == 3:
        data_tensor = data_tensor.permute(0, 1, 2)

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
            if pred_noise.size(2) != noise.size(2):
                pred_noise = pred_noise[:, :, :noise.size(2)]
            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1} - Avg Loss: {np.mean(losses):.4f}")
    
    torch.save(model.state_dict(), "eeg_model/ddpm_model.pth")
    return model, alpha_bar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ========================= PART 5: Sampling ====================================
def sample_ddpm_gpu(model, T, batch_shape):
    """
    Generate one batch of samples entirely on GPU.
    """
    beta, alpha, alpha_bar = get_noise_schedule(T)
    beta = beta.to(device)
    alpha = alpha.to(device)
    alpha_bar = alpha_bar.to(device)

    x = torch.randn(batch_shape, device=device)  # Start from pure noise

    for t in reversed(range(T)):
        t_tensor = torch.full((batch_shape[0],), t, device=device)
        z = torch.randn_like(x) if t > 0 else 0

        # Forward pass through the model
        pred_noise = model(x, t_tensor.float())

        # Fix mismatched time dimension
        if pred_noise.size(2) > x.size(2):
            pred_noise = pred_noise[:, :, :x.size(2)]
        elif pred_noise.size(2) < x.size(2):
            x = x[:, :, :pred_noise.size(2)]

        alpha_t = alpha[t]
        alpha_bar_t = alpha_bar[t]

        # DDPM update rule
        x = (1 / torch.sqrt(alpha_t)) * (
            x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * pred_noise
        ) + torch.sqrt(beta[t]) * z

    return x  # stays on GPU


def generate_synthetic_eeg(model, total_samples=200, batch_size=10, T=200, save_path="synthetic_eeg_minimal.npy"):
    all_samples = []
    num_batches = (total_samples + batch_size - 1) // batch_size

    for _ in tqdm(range(num_batches), desc="Sampling EEG"):
        current_batch_size = min(batch_size, total_samples - len(all_samples))
        batch_shape = (current_batch_size, 66, 750)

        batch_samples = sample_ddpm_gpu(model, T=T, batch_shape=batch_shape)
        all_samples.append(batch_samples.detach().cpu().numpy())
        del batch_samples
        torch.cuda.empty_cache()

    # Concatenate all batches and save as proper .npy
    synthetic_array = np.concatenate(all_samples, axis=0)
    np.save(save_path, synthetic_array)
    print(f"Synthetic EEG saved: {save_path}, shape={synthetic_array.shape}")
    return save_path

# STEP 1: Preprocess EEG from all subjects
#data = preprocess_all_eeg_minimal("raw_data")
#np.save("preprocessed_eeg_minimal.npy", data)

# STEP 2: Train DDPM model
#model, alpha_bar = train_ddpm(data, epochs=250)

model = EEGUNetDeep(ch=66).to(device)  # <-- move model to GPU
state_dict = torch.load("eeg_model/ddpm_model.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Disable gradient tracking for sampling
torch.set_grad_enabled(False)

# STEP 3: Sample synthetic EEG
synthetic = generate_synthetic_eeg(
    model,
    total_samples=2000,
    batch_size=10,
    T=200,
    save_path="eeg_model/synthetic_eeg_minimal_2000_new.npy"
)
