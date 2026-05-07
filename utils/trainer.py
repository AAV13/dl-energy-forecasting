import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from models.loss import grq_loss

def get_prediction(model, x):
    """
    Extracts just the prediction tensor, discarding attention weights 
    during the standard training loop.
    """
    out = model(x)
    if isinstance(out, tuple):
        return out[0]
    return out

def train_model(model, train_loader, val_loader, config, weights, device):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 1e-4)
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(config.get("epochs", 30)):
        model.train()
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # pred shape: (Batch, Horizon, Targets, Quantiles) -> (B, 24, 3, 3)
            pred = get_prediction(model, x)
            
            # Calculate the new Gated Residual Quantile Loss
            # y shape: (Batch, Targets, Horizon) -> (B, 3, 24)
            loss = grq_loss(pred, y, target_weights=weights)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = get_prediction(model, x)
                val_loss += grq_loss(pred, y, target_weights=weights).item()

        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)

        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.get("patience", 10):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    return train_losses, val_losses