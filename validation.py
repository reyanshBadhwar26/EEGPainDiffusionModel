# analysis_full.py
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch
from scipy.stats import wasserstein_distance, skew, kurtosis
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import permanova, DistanceMatrix
import matplotlib.pyplot as plt

# ───────── Config knobs ───────── #
RFF_FEATURES = 2048
BATCH_SIZE   = 8192
RNG          = np.random.default_rng(0)

REAL_PATH  = "eeg_model/preprocessed_eeg.npy"
SYNTH_PATH = "eeg_model/synthetic_eeg_minimal_2000.npy"

# ──────── Improved safe loader ──────── #
def safe_load(path, reference_shape=None, dtype=np.float32, mmap_mode='r'):
    """
    Robust loader that:
      - tries np.load (for real .npy or .npz)
      - if that fails, tries np.memmap with reference_shape (for raw memmap dumps)
      - if reference_shape provided but memmap fails, tries to infer sample count from file size
      - otherwise falls back to np.fromfile
    Returns a numpy array (or memmap) of dtype dtype.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    # 1) Try standard numpy loader (handles .npy and .npz)
    try:
        # allow_pickle True only to be permissive for legacy files — but most
        # of your data should be plain numeric arrays.
        arr = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
        # If it's an npz archive, pick first array
        if isinstance(arr, np.lib.npyio.NpzFile):
            keys = list(arr.keys())
            if len(keys) == 0:
                raise ValueError("Empty .npz archive")
            arr = arr[keys[0]]
        # convert memmap to np.ndarray if mmap_mode == 'r' but keep dtype
        if isinstance(arr, np.memmap) and mmap_mode != 'r':
            arr = np.asarray(arr)
        return np.asarray(arr, dtype=dtype)
    except Exception as e_np:
        # fallthrough to other strategies
        pass

    filesize = os.path.getsize(path)
    itemsize = np.dtype(dtype).itemsize

    # 2) If a reference shape is provided, try opening as memmap of that shape
    if reference_shape is not None:
        # try memmap with the provided shape
        try:
            mm = np.memmap(path, dtype=dtype, mode=mmap_mode, shape=tuple(reference_shape))
            return mm  # memmap is fine; caller can use it like an array
        except Exception:
            # maybe the file is raw binary but with different leading dimension (num_samples)
            try:
                # reference_shape expected like (n_ref, channels, times) or (channels, times)
                # derive channels x times from reference_shape[1:]
                channels_time = int(np.prod(reference_shape[1:]))
                if channels_time <= 0:
                    raise ValueError("Invalid reference channels/time product")
                inferred_samples = filesize // (channels_time * itemsize)
                if inferred_samples > 0 and inferred_samples * channels_time * itemsize == filesize:
                    shape = (inferred_samples, ) + tuple(reference_shape[1:])
                    arr = np.fromfile(path, dtype=dtype).reshape(shape)
                    return arr
            except Exception:
                pass

    # 3) If filesize matches a sensible 3D shape using a guess (common case),
    #    attempt to infer shape if reference_shape absent.
    #    We won't guess channels/times automatically here — user should provide reference_shape.
    try:
        # fallback: read raw flat array
        arr = np.fromfile(path, dtype=dtype)
        return arr
    except Exception as e_final:
        raise RuntimeError(f"Unable to load file {path}: {e_final}") from e_final


# ──────── Load real first, then synthetic (using real's shape if needed) ──────── #
real_eeg = safe_load(REAL_PATH)  # expect this to be a true .npy saved with header
print("Real EEG shape:", getattr(real_eeg, "shape", None), "dtype:", real_eeg.dtype)

# Now load synthetic using real_eeg shape as reference (so memmap/raw dumps load correctly)
synthetic_eeg = safe_load(SYNTH_PATH, reference_shape=real_eeg.shape, dtype=np.float32)
print("Synthetic EEG shape:", getattr(synthetic_eeg, "shape", None), "dtype:", synthetic_eeg.dtype)

# If either returned a 1D flat array, attempt to reshape to match reference
if synthetic_eeg.ndim == 1:
    try:
        synthetic_eeg = synthetic_eeg.reshape(real_eeg.shape)
        print("Reshaped synthetic to:", synthetic_eeg.shape)
    except Exception:
        # keep as-is; later validation will catch shape mismatches
        pass


real_eeg = safe_load(REAL_PATH)
synthetic_eeg = safe_load(SYNTH_PATH)

# ──────── Validation ──────── #
def validate_data(real, synth):
    checks = {
        "channels_time_match": real.shape[1:] == synth.shape[1:],
        "no_nans": not (np.isnan(real).any() or np.isnan(synth).any()),
        "same_dtype": real.dtype == synth.dtype,
    }
    summary = {
        "real_mean": float(np.mean(real)),
        "synth_mean": float(np.mean(synth)),
        "real_std": float(np.std(real)),
        "synth_std": float(np.std(synth)),
        "real_min": float(real.min()),
        "synth_min": float(synth.min()),
        "real_max": float(real.max()),
        "synth_max": float(synth.max()),
    }
    return checks, summary

# ──────── MMD helpers ──────── #
def _median_gamma(X, max_n=4096):
    n = X.shape[0]
    if n > max_n:
        idx = RNG.choice(n, size=max_n, replace=False)
        X = X[idx]
    d2 = np.sum(X**2, axis=1, keepdims=True) + np.sum(X**2, axis=1) - 2*np.dot(X, X.T)
    d2 = d2[np.triu_indices_from(d2, k=1)]
    d2 = d2[d2 > 0]
    if d2.size == 0:
        return 1.0
    med = np.median(d2)
    return 1.0 / med if np.isfinite(med) and med > 0 else 1.0

def _mmd_rbf_rff(X, Y, gamma=None, n_features=RFF_FEATURES, batch=BATCH_SIZE):
    if gamma is None:
        gX = X if X.shape[0] <= 8192 else X[RNG.choice(X.shape[0], 8192, replace=False)]
        gamma = _median_gamma(gX)

    d = X.shape[1]
    W = RNG.normal(0.0, np.sqrt(2*gamma), size=(d, n_features)).astype(np.float32)
    b = RNG.uniform(0, 2*np.pi, size=(n_features,)).astype(np.float32)

    def mean_phi(A):
        m = A.shape[0]
        acc = np.zeros(n_features, dtype=np.float64)
        for i in range(0, m, batch):
            chunk = A[i:i+batch]
            Z = np.dot(chunk, W) + b
            np.cos(Z, out=Z)
            acc += Z.sum(axis=0)
        return acc / m

    muX = mean_phi(X)
    muY = mean_phi(Y)
    return float(np.mean((muX - muY)**2)), float(gamma)

def compute_mmd_rbf(X, Y, gamma=None):
    m, g_used = _mmd_rbf_rff(X, Y, gamma=gamma)
    print(f"[MMD] RFF approx (N={X.shape[0]}/{Y.shape[0]}, feats={RFF_FEATURES}), gamma={g_used:.3e}")
    return m

# ──────── Metrics ──────── #
def ks_distance(real, synth):
    r = real.reshape(-1)
    s = synth.reshape(-1)
    ks_stat, p_value = stats.ks_2samp(r, s)
    return {"ks_stat": float(ks_stat), "p_value": float(p_value)}

def wasserstein(real, synth):
    r = real.reshape(-1)
    s = synth.reshape(-1)
    return float(wasserstein_distance(r, s))

def moment_stats(real, synth):
    r = real.reshape(-1)
    s = synth.reshape(-1)
    return {
        "mean_diff": float(abs(np.mean(r) - np.mean(s))),
        "std_diff": float(abs(np.std(r) - np.std(s))),
        "skew_diff": float(abs(skew(r) - skew(s))),
        "kurtosis_diff": float(abs(kurtosis(r) - kurtosis(s))),
    }

def spectral_diffusion(real, synth, fs=256):
    def _trial_spectral_entropy(x):
        f, Pxx = welch(x, fs=fs, nperseg=fs*2)
        Pxx = Pxx / np.sum(Pxx)
        return -np.sum(Pxx * np.log(Pxx + 1e-12))
    real_scores = [_trial_spectral_entropy(trial.mean(axis=0)) for trial in real]
    synth_scores = [_trial_spectral_entropy(trial.mean(axis=0)) for trial in synth]
    return {
        "real_mean": float(np.mean(real_scores)),
        "synth_mean": float(np.mean(synth_scores)),
        "diff": float(abs(np.mean(real_scores) - np.mean(synth_scores)))
    }

def permanova_test(real, synth, n_perms=199):
    n = min(len(real), len(synth))
    real = real[:n].reshape(n, -1)
    synth = synth[:n].reshape(n, -1)
    data = np.vstack([real, synth])
    labels = ["real"] * n + ["synth"] * n
    dist = pdist(data, metric="euclidean")
    dm = DistanceMatrix(squareform(dist), ids=[f"{i}" for i in range(len(labels))])
    return permanova(dm, grouping=labels, permutations=n_perms)

# ──────── Run Analysis ──────── #
checks, summary = validate_data(real_eeg, synthetic_eeg)

real_flat = real_eeg.reshape(real_eeg.shape[0], -1)
synth_flat = synthetic_eeg.reshape(synthetic_eeg.shape[0], -1)

ks_result   = ks_distance(real_eeg, synthetic_eeg)
wass_dist   = wasserstein(real_eeg, synthetic_eeg)
moments     = moment_stats(real_eeg, synthetic_eeg)
spec_diff   = spectral_diffusion(real_eeg, synthetic_eeg)
mmd_score   = compute_mmd_rbf(real_flat, synth_flat)
permanova_res = permanova_test(real_eeg, synthetic_eeg)

print("\n=== RESULTS (FULL DATA) ===")
print("Validation:", checks)
print("Summary:", summary)
print("KS Test:", ks_result)
print("Wasserstein:", wass_dist)
print("Moments:", moments)
print("Spectral Diffusion:", spec_diff)
print("MMD^2 (RBF):", mmd_score)
print("PERMANOVA:", permanova_res)