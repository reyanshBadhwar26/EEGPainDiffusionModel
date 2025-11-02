import os

path = "synthetic_eeg_minimal_200.npy"

print("File size:", os.path.getsize(path), "bytes")

with open(path, "rb") as f:
    header = f.read(16)
print("First 16 bytes:", header)