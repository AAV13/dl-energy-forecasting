import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def weighted_mse_loss(pred, target, weights):
    """
    pred, target: (batch, n_targets, horizon)
    weights: list of floats, one per target
    """
    loss = 0
    for i, w in enumerate(weights):
        loss += w * nn.functional.mse_loss(pred[:, i, :], target[:, i, :])
    return loss

def train_model(model, train_loader, val_loader, config, weights, device):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(config["epochs"]):
        # ── train ──
        model.train()
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = weighted_mse_loss(pred, y, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)

        # ── validate ──
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += weighted_mse_loss(pred, y, weights).item()

        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)

        print(f"Epoch {epoch+1:03d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        # ── early stopping ──
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return train_losses, val_losses