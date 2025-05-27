import os
import torch
from models import UNetGenerator1D
from data.dataset_loader import PreprocessedPPGDataset
from utils import extract_hr_from_ppg
import numpy as np

# Configuration
DATA_ROOT = "./preprocessed_data"
LOG_DIR   = "logs"
LOG_FILE  = os.path.join(LOG_DIR, "evaluate_all.log")

os.makedirs(LOG_DIR, exist_ok=True)
print("Data root exists:", os.path.exists(DATA_ROOT))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTIVITY_LIST = [
    "Sitting", "Stairs", "Soccer", "Cycling",
    "Driving", "Lunch", "Walking", "Working"
]

# Prepare dataset
subject_dirs = [f"S{i}" for i in range(1, 16)]
dataset = PreprocessedPPGDataset(DATA_ROOT, subject_dirs)

# Open log file
with open(LOG_FILE, "w") as log:
    log.write("Batch evaluation started\n")
    log.write("="*40 + "\n")

    for activity in ACTIVITY_LIST:
        model_path = (
            f"checkpoints/{activity.lower()}/"
            f"ppganet_generator_{activity.lower()}.pth"
        )
        if not os.path.exists(model_path):
            line = f"[SKIP] Model not found for {activity}: {model_path}\n"
            print(line.strip())
            log.write(line)
            continue

        header = f"Evaluating activity: {activity}"
        print("\n" + header)
        log.write("\n" + header + "\n")

        # Load generator
        G = UNetGenerator1D().to(DEVICE)
        G.load_state_dict(torch.load(model_path, map_location=DEVICE))
        G.eval()

        # Collect errors
        errors = []
        with torch.no_grad():
            for sample in dataset:
                if sample["activity_name"] != activity:
                    continue
                ppg     = sample["ppg"].to(DEVICE)
                hr_true = sample["hr"].item()
                denoised = G(ppg.unsqueeze(0))
                hr_pred = extract_hr_from_ppg(denoised.squeeze(0)).item()
                errors.append(abs(hr_pred - hr_true))

        if not errors:
            line = f"[WARN] No samples for {activity}\n"
            print(line.strip())
            log.write(line)
            continue

        errors = np.array(errors)
        mae  = errors.mean()
        rmse = np.sqrt(np.mean(errors**2))
        result_line = (
            f"{activity:10s} | Samples: {len(errors):4d} | "
            f"MAE = {mae:.2f} bpm | RMSE = {rmse:.2f} bpm\n"
        )
        print(result_line.strip())
        log.write(result_line)

    log.write("="*40 + "\n")
    log.write("Batch evaluation completed\n")

print(f"\nResults have been logged to {LOG_FILE}")