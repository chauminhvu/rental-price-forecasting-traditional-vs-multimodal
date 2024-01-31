import pandas as pd
import argparse
from numpy import float32
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from src.data.data_processing import label_encoding
torch.set_float32_matmul_precision('high')

if torch.cuda.is_available():
    device = torch.device("cuda")
    # torch.set_default_dtype(torch.float32)
else:
    device = torch.device("cpu")
    # torch.set_default_dtype(torch.float32)



def load_data(file_path, verbose=False):
    """
    Load training data

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: training data
    """
    clean_data_df = pd.read_csv(file_path)

    # Convert object-type columns to category
    object_cols = clean_data_df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        clean_data_df[col] = clean_data_df[col].astype("category")

    prediction_df = clean_data_df.drop(["baseRent", "picturecount"], axis=1)

    # print data info
    if verbose:
        print(prediction_df.info())
        print(prediction_df.head(5))

    return prediction_df


# def train_xgboost(X_train, y_train):
#     xgb_model = XGBRegressor(tree_method='gpu_hist')  # Use GPU for training XGBoost
#     xgb_model.fit(X_train, y_train)
#     return xgb_model


class RentalDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=2048, split_ratio=0.1, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.num_workers = num_workers

    def prepare_data(self):
        self.data = load_data(self.data_dir, verbose=False)
        # Encode categorical columns
        categorical_cols = self.data.select_dtypes(include=["category"]).columns
        self.encoders = label_encoding(self.data, categorical_cols)
        # get in/output
        outputs = self.data['totalRent'].values.astype(float32).reshape(-1, 1)
        inputs = self.data.drop('totalRent', axis=1).values.astype(float32)
        # slit data
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(inputs, outputs,
                                                                    test_size=self.split_ratio,
                                                                    random_state=2024)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_temp, y_temp,
                                                                              test_size=self.split_ratio,
                                                                               random_state=2024)
        # fit scaler to data 
        self.x_scaler = RobustScaler().fit(self.X_train)
        self.y_scaler = RobustScaler().fit(self.y_train)

    def setup(self, stage: str):

        if stage == "fit":
            # transform data
            self.X_train_scaled = self.x_scaler.transform(self.X_train)
            self.y_train_scaled = self.y_scaler.transform(self.y_train)
            # convert to tensor
            self.train_inputs = torch.tensor(self.X_train_scaled)
            self.train_targets = torch.tensor(self.y_train_scaled)
            self.train_dataset = TensorDataset(self.train_inputs, self.train_targets)

        if stage == "validate":
            # transform data
            self.X_val_scaled = self.x_scaler.transform(self.X_val)
            self.y_val_scaled = self.y_scaler.transform(self.y_val)
            # convert to tensor
            self.val_inputs = torch.tensor(self.X_val_scaled)
            self.val_targets = torch.tensor(self.y_val_scaled)
            self.val_dataset = TensorDataset(self.val_inputs, self.val_targets)
            # print("="*30)
            # print(self.val_dataset.shape)

        if stage == "test":
            # transform data
            self.X_test_scaled = self.x_scaler.transform(self.X_test)
            self.y_test_scaled = self.y_scaler.transform(self.y_test)
            # convert to tensor
            self.test_inputs = torch.tensor(self.X_test_scaled)
            self.test_targets = torch.tensor(self.y_test_scaled)
            self.test_dataset = TensorDataset(self.test_inputs, self.test_targets)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            # generator=torch.Generator(device),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # generator=torch.Generator(device),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # generator=torch.Generator(device),
        )


class MLPModel(L.LightningModule):
    """
    Multi-Layer Perceptron class.

    Attributes:
        layers (nn.ModuleList): List of layers to be used in the model.

    Args:
        layer_sizes (list[int]): List of integers representing the sizes of each layer in the MLP.
        hidden_activation (str): String representation of the activation function to be used in hidden layers.

    Raises:
        Exception: If the given activation function is not supported.
    """
    def __init__(self, layer_sizes, hidden_activation, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Core hidden layers
        for k in range(len(layer_sizes)-2):
            self.layers.append(nn.Linear(layer_sizes[k], layer_sizes[k+1], bias=True))
            self.layers.append(nn.Dropout(dropout))
            self.layers.append(hidden_activation)
        # Final layer
        self.layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1], bias=True))

    def forward(self, x):
        # x = x.float()
        for layer in self.layers:
            x = layer(x)
        return x


class MLPRegressor(L.LightningModule):
    def __init__(self, layer_sizes, hidden_activation, dropout, loss_func, lr):
        super().__init__()
        self.save_hyperparameters(ignore=['loss_func'])
        self.layer_sizes = layer_sizes
        self.hidden_activation = hidden_activation
        self.lr = lr
        self.model = MLPModel(self.layer_sizes, self.hidden_activation, dropout)
        self.loss_func = loss_func

        # Initialize lists to store training and validation losses
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.training_epoch_avg = []
        self.validation_epoch_avg = []
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        # Get batch data
        x_batch, y_batch = batch
        predict = self.model(x_batch)
        loss = self.loss_func(predict, y_batch)
        # Store
        # self.training_step_outputs.append(loss)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        predict = self.model(x_batch)
        loss = self.loss_func(predict, y_batch)
        # Store
        # self.validation_step_outputs.append(stored_loss)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        predict = self.model(x_batch)
        loss = self.loss_func(predict, y_batch)
        # Store
        # self.validation_step_outputs.append(stored_loss)
        self.log("test_loss", loss)

    # def on_train_epoch_end(self):
    #     epoch_average = torch.stack(self.training_step_outputs).mean()
    #     self.training_epoch_avg.append(epoch_average)
    #     self.log("training_epoch_average", epoch_average)
    #     self.training_step_outputs.clear()

    # def on_validation_epoch_end(self):
    #     epoch_average = torch.stack(self.validation_step_outputs).mean()
    #     self.validation_epoch_avg.append(epoch_average)
    #     self.log("validation_epoch_average", epoch_average)
    #     self.validation_step_outputs.clear()
