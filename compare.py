import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import seaborn as sns

def load_data(real_path="eeg_model/preprocessed_eeg.npy", synthetic_path="eeg_model/synthetic_eeg.npy"):
    real = np.load(real_path)
    synthetic = np.load(synthetic_path)
    print(f"Real data shape: {real.shape}")
    print(f"Synthetic data shape: {synthetic.shape}")
    return real, synthetic

def plot_sample_channels(real, synthetic, sample_idx=0, channels=[0,1,2]):
    time = np.arange(real.shape[2])  # number of time points
    plt.figure(figsize=(15, 6))
    for i, ch in enumerate(channels):
        plt.subplot(len(channels), 2, 2*i+1)
        plt.plot(time, real[sample_idx, ch], label="Real")
        plt.title(f"Real EEG Sample {sample_idx} Channel {ch}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        
        plt.subplot(len(channels), 2, 2*i+2)
        plt.plot(time, synthetic[sample_idx, ch], label="Synthetic", color='orange')
        plt.title(f"Synthetic EEG Sample {sample_idx} Channel {ch}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig("eeg_model/sample_channels_comparison.png")

def print_basic_stats(data, name="Data"):
    print(f"--- {name} ---")
    print(f"Mean over all data: {np.mean(data):.4f}")
    print(f"Std dev over all data: {np.std(data):.4f}")
    print(f"Min: {np.min(data):.4f}, Max: {np.max(data):.4f}")

    with open (f"eeg_model/{name.lower().replace(' ', '_')}_stats.txt", "w") as f:
        f.write(f"Mean: {np.mean(data):.4f}\n")
        f.write(f"Std dev: {np.std(data):.4f}\n")
        f.write(f"Min: {np.min(data):.4f}, Max: {np.max(data):.4f}\n")

def plot_psd(data, fs=250, channel=0, nperseg=256, title="PSD"):
    # data shape: (samples, channels, times)
    f, Pxx = welch(data[:, channel, :].reshape(-1, data.shape[2]), fs=fs, nperseg=nperseg, axis=-1)
    mean_Pxx = np.mean(Pxx, axis=0)
    plt.figure()
    plt.semilogy(f, mean_Pxx)
    plt.title(f"{title} - Channel {channel}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.savefig("eeg_model/{}_channel_{}.png".format(title.lower().replace(" ", "_"), channel))

def plot_correlation_matrix(data, name="Data"): 
    # Flatten epochs and time to get (channels, samples*time)
    flattened = data.transpose(1,0,2).reshape(data.shape[1], -1)
    corr = np.corrcoef(flattened)
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, cmap='coolwarm', center=0, square=True)
    plt.title(f"Channel Correlation Matrix - {name}")
    plt.xlabel("Channels")
    plt.ylabel("Channels")
    plt.savefig("eeg_model/{}_correlation_matrix.png".format(name.lower().replace(" ", "_")))

def main():
    real, synthetic = load_data()

    # Basic stats
    print_basic_stats(real, "Real EEG")
    print_basic_stats(synthetic, "Synthetic EEG")

    # Plot some example channels for first sample
    plot_sample_channels(real, synthetic, sample_idx=0, channels=[0,1,2])

    # Plot PSD for one channel
    plot_psd(real, title="Real EEG PSD", channel=0)
    plot_psd(synthetic, title="Synthetic EEG PSD", channel=0)

    # Plot correlation matrices
    plot_correlation_matrix(real, name="Real EEG")
    plot_correlation_matrix(synthetic, name="Synthetic EEG")

if __name__ == "__main__":
    main()
