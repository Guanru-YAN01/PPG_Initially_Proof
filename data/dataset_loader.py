import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

class PreprocessedPPGDataset(Dataset):
    def __init__(self, data_root, subject_ids, activity_filter=None):
        self.samples = []
        for subject in subject_ids:
            subj_path = os.path.join(data_root, subject)
            if not os.path.isdir(subj_path):
                continue
            for category in ["minimal", "rhythmic", "irregular"]:
                category_path = os.path.join(subj_path, category)
                if not os.path.isdir(category_path):
                    continue
                for fname in os.listdir(category_path):
                    if not fname.endswith(".pkl"):
                        continue
                    activity_name = os.path.splitext(fname)[0]  # e.g., "Sitting"
                    if activity_filter and activity_name.lower() != activity_filter.lower():
                        continue  
                    file_path = os.path.join(category_path, fname)
                    with open(file_path, "rb") as f:
                        data = pickle.load(f, encoding="latin1")
                    for sample in data:
                        ppg = sample["bvp_preprocessed"]
                        hr = sample["label"]
                        activity_type = 0 if activity_name.lower() == "sitting" else 1
                        self.samples.append({
                            "ppg": torch.tensor(ppg, dtype=torch.float32).unsqueeze(0),  # [1, 512]
                            "hr": torch.tensor(hr, dtype=torch.float32),
                            "activity_name": activity_name,
                            "activity_type": activity_type
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def get_dataloaders(data_root, batch_size=32, activity_filter=None):
    subject_dirs = sorted([d for d in os.listdir(data_root) if d.startswith("S")])
    train_ids = subject_dirs[:12]  # S1 to S12
    val_ids = subject_dirs[12:]    # S13 to S15

    train_set = PreprocessedPPGDataset(data_root, train_ids, activity_filter)
    val_set = PreprocessedPPGDataset(data_root, val_ids, activity_filter)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader