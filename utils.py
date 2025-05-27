import torch
import torch.nn.functional as F
import numpy as np
from scipy.signal import find_peaks
import os


def extract_hr_from_ppg(ppg_signal, fs=64):
    """
    Estimate heart rate (bpm) from denoised PPG signal.
    Input: ppg_signal (Tensor [1, 512])
    Output: estimated HR (Tensor [1])
    """
    signal_np = ppg_signal.detach().squeeze().cpu().numpy()

    peaks, _ = find_peaks(signal_np, distance=fs*0.4)  # At least 0.4s apart

    if len(peaks) < 2:
        return torch.tensor([0.0])  # fallback

    rr_intervals = np.diff(peaks) / fs  # in seconds
    hr = 60.0 / np.mean(rr_intervals)
    return torch.tensor([hr], dtype=torch.float32)
import numpy as np
import torch
from scipy.signal import find_peaks

def extract_hr_from_ppg(ppg_signal, fs=64):

    sig_np = ppg_signal.detach().cpu().numpy()  # shape:[B,1,L] or [1,L]
    # [B,1,L] -> [B,L]
    if sig_np.ndim == 3:
        sig_np = sig_np.squeeze(1)
    # [1, L]
    if sig_np.ndim == 1:
        sig_np = sig_np[np.newaxis, :]
    # find peak
    hrs = []
    for one in sig_np:  
        peaks, _ = find_peaks(one, distance=fs * 0.4)  # 0.4s between two peaks
        duration_sec = one.shape[0] / fs              # windows duration
        hr = len(peaks) * (60.0 / duration_sec)       # HR = num of peaks * (60 / win duration)
        hrs.append(hr)

    return torch.tensor(hrs, dtype=torch.float32, device=ppg_signal.device)



def compute_mae(preds, targets):
    preds = torch.tensor(preds)
    targets = torch.tensor(targets)
    return F.l1_loss(preds, targets).item()

class EarlyStopping:
    def __init__(self, patience=5, mode='min', save_path=None):
        self.patience = patience
        self.mode = mode  # 'min' for loss/MAE, 'max' for accuracy
        self.save_path = save_path
        self.best_score = float('inf') if mode == 'min' else -float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, score, model_dict):
        improved = (score < self.best_score) if self.mode == 'min' else (score > self.best_score)

        if improved:
            self.best_score = score
            self.counter = 0
            self.save_models(model_dict)
            return True  # Saved new best
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # No improvement

    def save_models(self, model_dict):
        if self.save_path is None:
            return
        os.makedirs(self.save_path, exist_ok=True)
        for name, model in model_dict.items():
            torch.save(model.state_dict(), os.path.join(self.save_path, f"{name}.pth"))
