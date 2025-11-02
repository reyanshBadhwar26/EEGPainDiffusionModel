#!/bin/bash
#SBATCH --job-name=my_eeg_job
#SBATCH --output=output_UNET_deep_minsynth.txt
#SBATCH --error=error_UNET_deep_minsynth.txt
#SBATCH --partition=gpu-v100
#SBATCH --nodes=1            # <- Choose one idle node (e.g., fg1, fg8, fg10–13)
#SBATCH --gres=gpu:2              # Request 1 GPU
#SBATCH --cpus-per-task=4         # Number of CPU cores (adjust as needed)
#SBATCH --mem=128G                 # RAM needed
#SBATCH --time=07:00:00           # Job time limit

# Load required modules
module load python/3.10 cuda/12.1.1

# Activate virtual environment (if applicable)
source ~/myenv/bin/activate

# Run your Python script
python eeg_model/eeg_UNET_deep_minimal.py
