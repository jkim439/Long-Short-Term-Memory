# Title: LSTM
# Author: Junghwan Kim
# Date: April 21, 2021

import pandas as pd

dataset = pd.read_csv("data/ny.csv", index_col="id").iloc[:, 1:]
dataset.columns = ["distance to target", "period", "trip_time_in_secs", "trip_distance", "longitude", "latitude", "is_pickup"]
dataset.index.name = "id"
dataset = dataset.dropna(axis=0)
temp_column = dataset.pop("distance to target")
dataset.insert(6, "distance to target", temp_column)
dataset.to_csv("data/ny_preprocessed.csv")
print(dataset)
