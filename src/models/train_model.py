import torch
import pickle
import torch.nn as nn
import lightning as L 
from define_model import (RentalDataModule, MLPRegressor,
                          RMSLELoss, LogCoshLoss, load_mlp_params)
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

PATH_DATASETS = "data/processed/rental_processed.csv"
MODEL_PATH = "models"
BATCHSIZE = 4096
EPOCHS = 300

# process data
datamodule = RentalDataModule(data_dir=PATH_DATASETS, batch_size=BATCHSIZE,
                              split_ratio=0.2, num_workers=16)
datamodule.prepare_data()
datamodule.setup("validate")
datamodule.setup("test")
datamodule.setup("fit")

mlp_params = load_mlp_params(f"{MODEL_PATH}/best_mlp_params.pkl",
                             datamodule.X_test.shape[1],
                             datamodule.y_test.shape[1],
                             )

# Load MLPRegressor model
mlp_model = MLPRegressor(**mlp_params)

# Initialize PyTorch Lightning trainer
trainer = L.Trainer(accelerator='auto',
                    max_epochs=EPOCHS,
                    gradient_clip_val=4.,
                    enable_checkpointing=False,
                    # profiler="simple"
                    )

# Train the model
trainer.fit(mlp_model, datamodule)
trainer.test(mlp_model, datamodule)

mlp_model.eval()
x, y = datamodule.test_dataset[:]
y_hat_test = mlp_model(x).detach().cpu().numpy()
mlp_r2_test = r2_score(y_hat_test, y.detach().cpu().numpy())
print(f"R2 score: {mlp_r2_test}")
torch.save(mlp_model.state_dict(), f"models/mlp_model_{int(100*mlp_r2_test)}.ckpt")

# Train XGBoost
# Load best XGBRegressor parameters from file
with open("models/best_xgb_params.pkl", "rb") as f:
    best_xgb_params = pickle.load(f)

xgb_params = dict(
    device='gpu',
    tree_method='gpu_hist',
    # enable_categorical=True,
    learning_rate=best_xgb_params['learning_rate'],
    max_depth=best_xgb_params['max_depth'],
    n_estimators=best_xgb_params['n_estimators'],
    eval_set=[(datamodule.X_val_scaled, datamodule.y_val_scaled)],
    eval_metric="rmsle",
    # eval_metric="mphe",
    verbosity=1
)
xgb_model = XGBRegressor(**xgb_params)
xgb_model.fit(datamodule.X_train_scaled, datamodule.y_train_scaled,)
xgb_pred_test = xgb_model.predict(datamodule.X_test_scaled)
xgb_r2_test = r2_score(xgb_pred_test, datamodule.y_test_scaled)
print(f"R2 score - XGBRegressor: {xgb_r2_test}")
# Save the trained XGBoost model using pickle
with open(f"models/xgb_model_{int(100*xgb_r2_test)}.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

