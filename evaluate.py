import torch
from models import UNetGenerator1D
from data.dataset_loader import PreprocessedPPGDataset
from utils import extract_hr_from_ppg, compute_mae
from collections import defaultdict
import numpy as np
import os

DATA_ROOT = r"./preprocessed_data"
print("Path exists:", os.path.exists(DATA_ROOT))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "ppganet_generator.pth"

# Load model
G = UNetGenerator1D().to(DEVICE)
G.load_state_dict(torch.load(MODEL_PATH))
G.eval()

# Load full dataset
subject_dirs = [f"S{i}" for i in range(1, 16)]
dataset = PreprocessedPPGDataset(DATA_ROOT, subject_dirs)

# Group by activity name
results = defaultdict(list)

with torch.no_grad():
    for sample in dataset:
        ppg = sample['ppg'].to(DEVICE)  # [1, 512]
        hr_true = sample['hr'].item()
        activity = sample['activity_name']

        denoised = G(ppg.unsqueeze(0))  # [1, 1, 512]
        hr_pred = extract_hr_from_ppg(denoised.squeeze(0)).item()
        error = abs(hr_pred - hr_true)
        results[activity].append(error)

# Print MAE per activity
for activity, errors in results.items():
    errors = np.array(errors)
    mae = errors.mean()
    rmse = np.sqrt((errors ** 2).mean())
    print(f"{activity:10s} | N={len(errors):4d} | MAE={mae:.2f} bpm | RMSE={rmse:.2f} bpm")
