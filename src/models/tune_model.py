import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch.nn as nn
import optuna
import pickle
import lightning as L
from define_model import RentalDataModule, MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score


EPOCHS = 20
PATH_DATASETS = "data/processed/rental_processed.csv"
BATCHSIZE = 4096
NTRIALS = 100


def mlp_objective(trial):
    datamodule = RentalDataModule(data_dir=PATH_DATASETS, batch_size=BATCHSIZE,
                                  fraction=0.2, split_ratio=0.2, num_workers=16)
    datamodule.prepare_data()
    datamodule.setup("fit")
    datamodule.setup("validate")
    datamodule.setup("test")

    # Suggest activations funtions in ANN's core
    activation = trial.suggest_categorical('activation', ['nn.SELU()', 'nn.SiLU()', 'nn.ReLU()'])
    # Suggest number of layers
    hidden_layers = trial.suggest_int("hidden_layers", 2, 4)
    # Suggest dropout
    dropout = trial.suggest_discrete_uniform('droptout', 0.0, 0.5, 0.1)
    # Number of hidden neurons in each layer
    output_neurons = [trial.suggest_int(f"n_l{i}", 30, 240, step=10) for i in range(hidden_layers)]
    # Book keeping parameters
    mlp_params = dict(learning_rate=1e-2,
                      layer_sizes=[30] + output_neurons + [1],
                      hidden_activation=activation,
                      loss_func=nn.HuberLoss(),
                      dropout=dropout,
                      )

    mlp_model = MLPRegressor(**mlp_params)
    # Initialize PyTorch Lightning trainer
    trainer = L.Trainer(accelerator='auto',
                        max_epochs=EPOCHS,
                        gradient_clip_val=4.,
                        enable_checkpointing=False,
                        )

    # Train the model
    trainer.fit(mlp_model, datamodule)
    mlp_model.eval()
    x, y = datamodule.test_dataset[:]
    y_hat_test = mlp_model(x).detach().cpu().numpy()
    # Check for NaN values in predictions
    if np.isnan(y_hat_test).any():
        # Handle NaN values gracefully
        return float('-inf')  # Return negative infinity as a special value
        
    r2_test = r2_score(y_hat_test, y.detach().cpu().numpy())
    return r2_test


def xgb_objective(trial):
    datamodule = RentalDataModule(data_dir=PATH_DATASETS, batch_size=BATCHSIZE,
                                  fraction=0.2, split_ratio=0.2, num_workers=16)
    datamodule.prepare_data()
    datamodule.setup("fit")
    datamodule.setup("validate")
    datamodule.setup("test")

    # Suggest for XGboost
    max_depth = trial.suggest_int('max_depth', 3, 20)
    n_estimators =trial.suggest_int('n_estimators', 50, 1000, 50)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 1e-1)
    xgb_params = dict(
        device='gpu',
        tree_method='gpu_hist',
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        eval_set=[(datamodule.X_val_scaled, datamodule.y_val_scaled)],
        eval_metric="mphe",
    )

    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(datamodule.X_train_scaled, datamodule.y_train_scaled,)
    xgb_pred_test = xgb_model.predict(datamodule.X_test_scaled)
    xgb_r2_test = r2_score(xgb_pred_test, datamodule.y_test_scaled)
    return xgb_r2_test


def main():
    # # Optimize MLPRegressor hyperparameters
    # mlp_study = optuna.create_study(direction='maximize')
    # mlp_study.optimize(mlp_objective, n_trials=NTRIALS)

    # print("Best MLPRegressor parameters:", mlp_study.best_params)
    # print("Best MLPRegressor R2 score:", mlp_study.best_value)
    # # Save best MLPRegressor parameters to a file using pickle
    # with open("models/best_mlp_params.pkl", "wb") as f:
    #     pickle.dump(mlp_study.best_params, f)
    
    # Optimize XGBRegressor hyperparameters
    xgb_study = optuna.create_study(direction='maximize')
    xgb_study.optimize(xgb_objective, n_trials=NTRIALS)
    
    # Save best XGBRegressor parameters to a file using pickle
    with open("models/best_xgb_params.pkl", "wb") as f:
        pickle.dump(xgb_study.best_params, f)

    print("Best XGBRegressor parameters:", xgb_study.best_params)
    print("Best XGBRegressor R2 score:", xgb_study.best_value)


if __name__ == "__main__":
    main()
