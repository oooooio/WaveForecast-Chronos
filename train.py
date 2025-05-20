from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import pandas as pd
import os
import yaml
from pathlib import Path


# 加载 YAML 配置
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
# 验证配置项
required_keys = ["data", "training"]
for key in required_keys:
    if key not in config:
        raise ValueError(f"Missing required config key: {key}")

filepaths = config["data"]["filepaths"]
prediction_lengths = config["data"]["prediction_lengths"]
TRAINSIZE = config["training"]["train_size"]
VALSIZE = config["training"]["val_size"]
TESTSIZE = config["training"]["test_size"]

context_length = config["context-length"]


def load_train_data(path, prediction_length):
    df = pd.read_csv(path)
    windows = []
    item_id = 0
    windows_size = context_length + prediction_length
    i = 0
    while i <= len(df) - windows_size:
        window = df.iloc[i : i + windows_size].copy()
        lost = window["SWH"].isnull().mean()
        if lost > 0.05:
            i += 1
            continue
        if window["SWH"].iloc[-prediction_length:].isna().any():
            i += 1
            continue
        item_id += 1
        window["item_id"] = item_id
        windows.append(window)
        i += prediction_length

    result = pd.concat(windows)
    train_size = int(item_id * TRAINSIZE)
    val_size = int(item_id * VALSIZE)
    test_size = int(item_id * TESTSIZE)

    train_data = result[result["item_id"] <= train_size]
    val_data = result[
        (result["item_id"] > train_size) & (result["item_id"] <= train_size + val_size)
    ]
    test_data = result[
        (result["item_id"] > train_size + val_size)
        & (result["item_id"] <= train_size + val_size + test_size)
    ]
    train_data = TimeSeriesDataFrame.from_data_frame(train_data)
    val_data = TimeSeriesDataFrame.from_data_frame(val_data)
    test_data = TimeSeriesDataFrame.from_data_frame(test_data)
    return train_data, val_data, test_data


for filepath in filepaths:
    for prediction_length in prediction_lengths:
        filename = Path(filepath).stem
        workdir = f"models/{filename}/{prediction_length}/"
        if not os.path.exists(workdir):
            os.makedirs(workdir)
        train_data, val_data, test_data = load_train_data(filepath, prediction_length)
        TimeSeriesPredictor(
            prediction_length=prediction_length,
            target="SWH",
            freq="h",
            eval_metric="RMSE",
            path=workdir,
        ).fit(
            # All Model Use default hyperparameters
            train_data=train_data,
            tuning_data=val_data,
            presets="best_quality",
            verbosity=2,
            hyperparameters={
                "DeepAR": {},
                "AutoETS": {},
                "DirectTabular": {},
                "RecursiveTabular": {},
                "DynamicOptimizedTheta": {},
                "TemporalFusionTransformer": {},
                "NPTS": {},
                "SeasonalNaive": {},
                "PatchTST": {},
                "TiDE": {},
                "Chronos": [
                    {
                        "model_path": "bolt_base",
                        "ag_args": {"name_suffix": "ZeroShot"},
                    },
                    {
                        "model_path": "bolt_small",
                        "fine_tune": True,
                        "ag_args": {"name_suffix": "FineTuned"},
                    },
                ],
            },
        )
