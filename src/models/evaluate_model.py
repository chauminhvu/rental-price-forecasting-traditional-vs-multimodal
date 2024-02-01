import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
# from xgboost import XGBRegressor
from define_model import RentalDataModule, MLPRegressor, load_mlp_params
from src.data.data_processing import plot_predictions
np.random.seed(2024)
PATH_DATASETS = "data/processed/rental_processed.csv"
MODEL_PATH = "models"

# Load test data
datamodule = RentalDataModule(data_dir=PATH_DATASETS, split_ratio=0.2, num_workers=16)
datamodule.prepare_data()
datamodule.setup("test")

mlp_params = load_mlp_params(f"{MODEL_PATH}/best_mlp_params.pkl",
                             datamodule.X_test.shape[1],
                             datamodule.y_test.shape[1],
                             )
# Load MLPRegressor model
mlp_model = MLPRegressor(**mlp_params)
mlp_model.load_state_dict(torch.load(f"{MODEL_PATH}/mlp_model_80.ckpt"))

# Inference for MLPRegressor
mlp_model.eval()
x_mlp_test, y_mlp_test = datamodule.test_dataset[:]
mlp_pred_test = mlp_model(x_mlp_test).detach().cpu().numpy()
mlp_pred_unscaled_test = datamodule.y_scaler.inverse_transform(mlp_pred_test)

# x_mlp_test_unscaled = datamodule.x_scaler.inverse_transform(datamodule.X_test_scaled)
# mlp_pred_df = pd.DataFrame([x_mlp_test_unscaled, mlp_pred_unscaled_test])
# exit()
# Load XGBRegressor model
with open(f"{MODEL_PATH}/xgb_model_90.pkl", "rb") as f:
    xgb_model = pickle.load(f)

# Inference for XGBRegressor
xgb_pred_test = xgb_model.predict(datamodule.X_test_scaled)
xgb_pred_test_unscaled = datamodule.y_scaler.inverse_transform(xgb_pred_test.reshape(-1,1))

# Calculate R2 scores
r2_mlp = r2_score(mlp_pred_unscaled_test, datamodule.y_test)
r2_xgb = r2_score(xgb_pred_test_unscaled, datamodule.y_test)
print(f"R2 MLP: {r2_mlp}")
print(f"R2 XGBoost: {r2_xgb}")
xlabel = r"Living space $[m^2]$"
ylabel = "total rent [euros/month]"
title = r"Performence of MLP vs XGboost on $1\%$ data"


plot_predictions(datamodule, mlp_pred_unscaled_test, xgb_pred_test_unscaled,
                col=7, subset_percentage=0.05, xlabel=xlabel, ylabel=ylabel, title=title)
plt.savefig(f'figures/r2_MLP_XGBoost.png', dpi=300)