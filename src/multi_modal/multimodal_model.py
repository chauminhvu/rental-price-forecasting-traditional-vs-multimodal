from autogluon.multimodal import MultiModalPredictor
import cudf as gd
from src.data.data_processing import group_minor_categories
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error
torch.set_float32_matmul_precision('medium')


def root_mean_squared_error(y_pred, y_true):
    mse = mean_squared_error(y_pred, y_true)
    return np.sqrt(mse)

TRAINING = False
TRAINING_TIME = 60*60  # seconds
PATH_DATASETS = "data/processed/rental_text_processed.csv"
MODEL_PATH = "multi_models"

mix_text_df = gd.read_csv(PATH_DATASETS)
mix_text_df = mix_text_df.drop(["baseRent", "picturecount"], axis=1)
mix_text_df = mix_text_df.dropna(axis=0)
# conver to log-scale
log_cols = ['serviceCharge', 'totalRent']
for log in log_cols:
    mix_text_df.loc[:, log] = np.log1p(mix_text_df[log])

cat_cols = ['condition', 'heatingType', 'typeOfFlat']
mix_text_df[cat_cols] = mix_text_df[cat_cols].astype("category")
# mix_text_df = group_minor_categories(mix_text_df, cat_cols, threshold=0.2)

# Get small part to test first
small_sample_df = mix_text_df.sample(frac=0.3, random_state=2024).to_pandas()
small_sample_df.drop(['description', 'facilities'], axis=1)
# Split df
train_df = small_sample_df.sample(frac=0.8, random_state=2024)
test_df = small_sample_df.drop(train_df.index)

print(f"Numer of training sample: {train_df.shape[0]}")

if TRAINING:
    time_limit = 1 * 60  # 3 minutes
    multimodal = MultiModalPredictor(label='totalRent', path=MODEL_PATH)
    fit_params={"optimization.gradient_clip_val": 5,
            "optimization.lr_mult": 2}
    multimodal.fit(train_df, time_limit=time_limit, hyperparameters=fit_params)
else:
    multimodal = MultiModalPredictor.load(f"{MODEL_PATH}")


# Get predictions
# y_test = np.expm1(test_df['totalRent'])
y_test = test_df['totalRent']
# y_pred_test = np.expm1(multimodal.predict(test_df))
y_pred_test = multimodal.predict(test_df)


# Calculate R2 scores
r2_multi = r2_score(y_pred_test, y_test)
rmse_multi = root_mean_squared_error(y_pred_test, y_test)
print(f"R2: {r2_multi}")
print(f"RMSE: {rmse_multi}")
xlabel = r"Living space $[m^2]$"
ylabel = "total rent [euros/month]"
title = r"Performence of MLP vs XGboost on $1\%$ data"

print('Predictions:')
print('------------')
print(y_pred_test[:5])
print()
print('True Value:')
print('------------')
print(y_test[:5])
