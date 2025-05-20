from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from autogluon.tabular import TabularPredictor
from joblib import Parallel, delayed
import pandas as pd
import joblib
import numpy as np
import os

TRAINSIZE = 0.7
VALSIZE = 0.15
TESTSIZE = 0.15


def load_data(path, prediction_length):
    data = TimeSeriesDataFrame.from_path(path).fill_missing_values()
    length = len(data)

    train_size = int(length * TRAINSIZE)
    val_size = int(length * VALSIZE)
    test_size = int(length * TESTSIZE)

    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size : train_size + val_size]
    test_data = data.iloc[train_size + val_size : train_size + val_size + test_size]

    predict_series = [
        pd.concat([val_data[-2048:], test_data[: i + 1 - prediction_length]])
        for i in range(prediction_length, val_size)
    ]

    return train_data, val_data, test_data, predict_series


models_list = [
    "SeasonalNaive",
    "RecursiveTabular",
    "DirectTabular",
    "NPTS",
    "DynamicOptimizedTheta",
    "AutoETS",
    "ChronosZeroShot[bolt_base]",
    "ChronosFineTuned[bolt_small]",
    "TemporalFusionTransformer",
    "DeepAR",
    "PatchTST",
    "TiDE",
]


def process_data(filename, prediction_length, models_list):
    train_data, val_data, test_data, predict_series = load_data(
        filename, prediction_length
    )
    length = len(predict_series)
    workdir = filename.split(".")[0].split("/")[-1] + "/" + str(prediction_length) + "/"
    modeldir = "models/" + workdir
    predictdir = "predict/" + workdir
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    if not os.path.exists(predictdir):
        os.makedirs(predictdir)
    predictor = TimeSeriesPredictor.load(modeldir)
    print(f"Processing {filename} with prediction length {prediction_length}")
    results = [] 

    for model in models_list:
        predictions = []
        for i in range(0, length, prediction_length):
            predictions.append(predictor.predict(predict_series[i], model=model))

        df = pd.concat(predictions)
        df = test_data.join(df, how="outer").dropna()
    
        df.to_csv("predict" + workdir + model + ".csv")
        results.append((model, df))
    print(f"Finished processing {filename} with prediction length {prediction_length}")
    return results


def save_results(workdir, results):
    for model, df in results:
        df.to_csv(workdir + model + ".csv")


filenames = [
    "data/ndbc/41008.csv",
    "data/ndbc/44007.csv",
]
prediction_lengths = [1, 3, 6, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120]


tasks = []
for filename in filenames:
    for prediction_length in prediction_lengths:
        tasks.append((filename, prediction_length, models_list))


results = Parallel(n_jobs=-1)(
    delayed(process_data)(filename, prediction_length, models_list)
    for filename, prediction_length, models_list in tasks
)
