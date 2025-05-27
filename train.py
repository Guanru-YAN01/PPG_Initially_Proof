import torch
import torch.nn as nn
import torch.optim as optim
import os
from models import UNetGenerator1D, CNNDiscriminator1D
from data.dataset_loader import get_dataloaders
from utils import extract_hr_from_ppg, compute_mae, EarlyStopping

# Settings
DATA_ROOT = r"./preprocessed_data"
print("Path exists:", os.path.exists(DATA_ROOT))

BATCH_SIZE = 32
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAMBDA_ADV = 0.001
PATIENCE = 5

# Define activities
ACTIVITY_MAP = {
    "Sitting": "minimal",
    "Stairs": "rhythmic",
    "Soccer": "irregular",
    "Cycling": "rhythmic",
    "Driving": "irregular",
    "Lunch": "irregular",
    "Walking": "rhythmic",
    "Working": "minimal"
}

for activity_name in ACTIVITY_MAP:
    print(f"\n--- Training model for activity: {activity_name} ---")

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{activity_name.lower()}_train_log.txt")
    with open(log_file, "w") as f:
        f.write(f"Training log for activity: {activity_name}\n")

    # Load activity-specific data
    train_loader, val_loader = get_dataloaders(
        DATA_ROOT, batch_size=BATCH_SIZE, activity_filter=activity_name
    )

    # Initialize models
    G = UNetGenerator1D().to(DEVICE)
    D = CNNDiscriminator1D().to(DEVICE)

    # Optimizers and losses
    optimizer_G = optim.Adam(G.parameters(), lr=1e-4)
    optimizer_D = optim.Adam(D.parameters(), lr=1e-4)
    criterion_adv = nn.BCELoss()
    criterion_hr = nn.L1Loss()

    # Early stopping setup
    save_dir = os.path.join("checkpoints", activity_name.lower())
    early_stopper = EarlyStopping(patience=PATIENCE, mode='min', save_path=save_dir)

    for epoch in range(EPOCHS):
        G.train()
        D.train()
        total_loss_G, total_loss_D = 0, 0

        for batch in train_loader:
            ppg = batch['ppg'].to(DEVICE)
            # print(ppg.shape)
            hr = batch['hr'].to(DEVICE)
            activity_type = batch['activity_type'].to(DEVICE)

            # ==== Train Discriminator ====
            pred_clean = D(ppg[activity_type == 0])  # clean samples only
            pred_fake = D(G(ppg.detach()))

            real_labels = torch.ones_like(pred_clean)
            fake_labels = torch.zeros_like(pred_fake)

            loss_D_real = criterion_adv(pred_clean, real_labels)
            loss_D_fake = criterion_adv(pred_fake, fake_labels)
            loss_D = loss_D_real + loss_D_fake

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # ==== Train Generator ====
            denoised = G(ppg)
            pred_hr = extract_hr_from_ppg(denoised)
            loss_hr = criterion_hr(pred_hr, hr)

            pred_fake_for_G = D(denoised)
            loss_adv = criterion_adv(pred_fake_for_G, torch.ones_like(pred_fake_for_G))
            loss_G = loss_hr + LAMBDA_ADV * loss_adv

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            total_loss_G += loss_G.item()
            total_loss_D += loss_D.item()

        # ==== Validation ====
        G.eval()
        val_mae_total = 0
        with torch.no_grad():
            for batch in val_loader:
                ppg = batch['ppg'].to(DEVICE)
                hr = batch['hr'].to(DEVICE)
                denoised = G(ppg)
                pred_hr = extract_hr_from_ppg(denoised)
                val_mae_total += compute_mae(pred_hr, hr)

        val_mae_avg = val_mae_total / len(val_loader)
        log_line = f"Epoch {epoch+1}/{EPOCHS} | Loss_G: {total_loss_G:.4f} | Loss_D: {total_loss_D:.4f} | Val MAE: {val_mae_avg:.4f}"
        print(log_line)
        with open(log_file, "a") as f:
            f.write(log_line + "\n")


        saved = early_stopper(val_mae_avg, {
            f"ppganet_generator_{activity_name.lower()}": G,
            f"ppganet_discriminator_{activity_name.lower()}": D
        })

        if saved:
            msg = f"Saved best model for {activity_name}"
            print(msg)
            with open(log_file, "a") as f:
                f.write(msg + "\n")
        if early_stopper.early_stop:
            msg = f"Early stopping for {activity_name} at epoch {epoch+1}"
            print(msg)
            with open(log_file, "a") as f:
                f.write(msg + "\n")
            break
