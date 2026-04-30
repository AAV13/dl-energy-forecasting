import yaml, pickle, argparse, torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils.seed import set_seed
from utils.trainer import train_model
from utils.metrics import compute_all_metrics
from models.lstm import BiLSTMForecaster

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def make_windows(df, feature_cols, target_cols, lookback, horizon):
    data = df[feature_cols].values
    targets = df[target_cols].values
    X, Y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i+lookback])
        Y.append(targets[i+lookback:i+lookback+horizon].T)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def main(config_path):
    config = load_config(config_path)
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(config["data"]["processed_path"], "rb") as f:
        data = pickle.load(f)

    train_df = data["train"]
    val_df   = data["val"]
    test_df  = data["test"]
    scaler   = data["scaler"]
    feat_cols   = data["feature_cols"]
    target_cols = data["target_cols"]

    L, H = config["lookback"], config["horizon"]
    X_train, Y_train = make_windows(train_df, feat_cols, target_cols, L, H)
    X_val,   Y_val   = make_windows(val_df,   feat_cols, target_cols, L, H)
    X_test,  Y_test  = make_windows(test_df,  feat_cols, target_cols, L, H)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train), torch.tensor(Y_train)),
        batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(
        torch.tensor(X_val), torch.tensor(Y_val)),
        batch_size=config["batch_size"])
    test_loader = DataLoader(TensorDataset(
        torch.tensor(X_test), torch.tensor(Y_test)),
        batch_size=config["batch_size"])

    model = BiLSTMForecaster(
        input_size=len(feat_cols),
        hidden_size=128,
        dropout=config["dropout"],
        horizon=H,
        n_targets=len(target_cols)
    ).to(device)

    weights = [
        config["loss_weights"]["solar"],
        config["loss_weights"]["wind_onshore"],
        config["loss_weights"]["wind_offshore"],
        config["loss_weights"]["price"]
    ]

    train_losses, val_losses = train_model(
        model, train_loader, val_loader, config, weights, device)

    # ── evaluate on test set ──
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.append(model(x.to(device)).cpu().numpy())
            actuals.append(y.numpy())

    preds   = np.concatenate(preds)
    actuals = np.concatenate(actuals)

    results = compute_all_metrics(actuals, preds, target_cols)
    print("\n── Test Results ──")
    for target, metrics in results.items():
        print(f"{target}: {metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)