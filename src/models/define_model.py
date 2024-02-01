import pandas as pd
import numpy as np
import pickle
import warnings
from numpy import float32
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler, MinMaxScaler
from src.data.data_processing import group_minor_categories
torch.set_float32_matmul_precision('high')
warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def load_data(file_path, verbose=False):
    """
    Load training data

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: training data
    """

    clean_data_df = pd.read_csv(file_path)
    prediction_df = clean_data_df.drop(["baseRent", "picturecount"], axis=1)

    # Convert object-type columns to category
    cat_cols = prediction_df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        # convert to category type
        prediction_df[col] = prediction_df[col].astype("category")

    # Group minior category items cumsum < 20% into "Other"
    prediction_df = group_minor_categories(prediction_df, cat_cols, threshold=0.2)

    # Encode categorical columns
    encoder = OneHotEncoder(sparse=False)
    categorical_cols_encoded = encoder.fit_transform(prediction_df[cat_cols])
    # Concatenate the encoded categorical columns with the original DataFrame
    encoded_df = pd.DataFrame(categorical_cols_encoded,
                              columns=encoder.get_feature_names_out(cat_cols))
    final_df = pd.concat([prediction_df.drop(cat_cols, axis=1), encoded_df], axis=1)
    # print data info
    if verbose:
        print(final_df.info())
        print(final_df.head(5))
    return final_df


def load_mlp_params(file_path, input_size, output_size):
    # Load best MLPRegressor parameters from file
    with open(file_path, "rb") as f:
        best_mlp_params = pickle.load(f)

    # Extract the sizes of hidden layers
    hidden_layer_sizes = [best_mlp_params[f'n_l{i}'] for i in range(
        best_mlp_params['hidden_layers'])]

    mlp_params = {
        'learning_rate': 1e-3,
        'layer_sizes': [input_size] + hidden_layer_sizes + [output_size],
        'hidden_activation': best_mlp_params['activation'],
        'loss_func': nn.HuberLoss(),
        'dropout': best_mlp_params['droptout'],
    }
    return mlp_params


class RentalDataModule(L.LightningDataModule):
    def __init__(self, data_dir, fraction=1., batch_size=2048, split_ratio=0.1, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.fraction = fraction
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.num_workers = num_workers

    def prepare_data(self):
        self.data = load_data(self.data_dir, verbose=False)
        self.data = self.data.sample(frac=self.fraction, random_state=2024)
        # get in/output
        print(self.data.drop('totalRent', axis=1).columns)
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
        # self.x_scaler = MinMaxScaler((0.1, 1)).fit(self.X_train)
        # self.y_scaler = MinMaxScaler((0.1, 1)).fit(self.y_train)
        

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
    def __init__(self, layer_sizes, hidden_activation, dropout, loss_func, learning_rate):
        super().__init__()
        self.save_hyperparameters(ignore=['loss_func', 'hidden_activation'])
        self.layer_sizes = layer_sizes
        self.hidden_activation = eval(f"{hidden_activation}")
        self.lr = learning_rate
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


class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, label):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(label + 1)))


class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        diff = y_true - y_pred
        return torch.mean(torch.log(torch.cosh(diff + 1e-12)))

