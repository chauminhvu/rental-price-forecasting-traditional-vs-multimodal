import optuna
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from define_model import MLPModel


def train_xgboost(X_train, y_train, params):
    xgb_model = XGBRegressor(**params)
    xgb_model.fit(X_train, y_train)
    return xgb_model

def objective(trial):
    # Define search space for hyperparameters
    mlp_params = {
        "input_dim": X_train.shape[1],
        "hidden_size": trial.suggest_int("hidden_size", 32, 256),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
    }

    xgb_params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
    }

    # Train MLP model
    model = MLPModel(**mlp_params)
    trainer = pl.Trainer(gpus=1, max_epochs=10)
    trainer.fit(model, DataLoader(TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), 
                                                torch.tensor(y_train, dtype=torch.float32))))

    # Train XGBoost model
    xgb_model = train_xgboost(X_train, y_train, xgb_params)

    # Evaluate XGBoost model
    y_pred_xgb = xgb_model.predict(X_test)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)

    # Evaluate MLP model
    model.eval()
    with torch.no_grad():
        y_pred_mlp = model(torch.tensor(X_test_scaled, dtype=torch.float32)).cpu().numpy()
    mse_mlp = mean_squared_error(y_test, y_pred_mlp)

    return (mse_xgb + mse_mlp) / 2  # Return average MSE

def main():
    # Load data and split into features and target
    # Assuming you have data loaded into X and y
    clean_data = gd.read_csv('data/processed/rental_processed.csv')
    print(clean_data.info)
    print(clean_data.columns)
    exit()
    # Split data into train and test sets
    global X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features for MLP model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Optimize hyperparameters
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    # Print best parameters
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == "__main__":
    main()
