from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import pandas as pd
import yaml
import os

with open("conf/config.yaml", "r") as f:
    config = yaml.safe_load(f)
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
    # 是数据长度
    train_size = int(item_id * TRAINSIZE)
    val_size = int(item_id * VALSIZE)
    test_size = int(item_id * TESTSIZE)

    # 合并所有窗口
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


stations = [
    "41008",
    "44007",
    "41010",
    "42003",
    "51003",
]


for station in stations:
    if not os.path.exists(f"/20zhaiyilin/predict_wave/scores/{station}"):
        os.mkdir(f"/20zhaiyilin/predict_wave/scores/{station}")
    for predict_length in prediction_lengths:
        _, _, test_data = load_train_data(
            f"/20zhaiyilin/predict_wave/data/self/{station}.csv", predict_length
        )

        TimeSeriesPredictor.load(
            f"/20zhaiyilin/predict_wave/models/{station}/{predict_length}"
        ).leaderboard(
            test_data,
            extra_metrics=[
                "MAE",
                "MASE",
                "RMSLE",
                "SMAPE",
            ],
        ).to_csv(
            f"/20zhaiyilin/predict_wave/scores/{station}/{predict_length}.csv"
        )
