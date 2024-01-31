import torch
import torch.nn as nn
import lightning as L 
from define_model import RentalDataModule, MLPRegressor
from sklearn.metrics import r2_score

PATH_DATASETS = "data/processed/rental_processed.csv"
BATCHSIZE = 8000
# process data

datamodule = RentalDataModule(data_dir=PATH_DATASETS, batch_size=BATCHSIZE, split_ratio=0.2)
datamodule.prepare_data()
datamodule.setup("validate")
# print(datamodule.X_train.shape)
hyperparameters = dict(layer_sizes=[16, 128, 128, 32, 1], lr=1e-2, hidden_activation=nn.Mish(),
                       dropout=0.1, loss_func=nn.HuberLoss())
mlp_model = MLPRegressor(**hyperparameters)

# Initialize PyTorch Lightning trainer
trainer = L.Trainer(accelerator='auto',
                    max_epochs=100,
                    fast_dev_run=False
                    )

# Train the model
trainer.fit(mlp_model, datamodule)
trainer.test(mlp_model, datamodule)

mlp_model.eval()
x, y = datamodule.test_dataset[:]
y_hat_test = mlp_model(x).detach().cpu().numpy()
r2_test = r2_score(y_hat_test, y.detach().cpu().numpy())
print(f"R2 score: {r2_test}")
