# Title: LSTM
# Author: Junghwan Kim
# Date: April 21, 2021

import pandas as pd

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from matplotlib import pyplot
from keras.callbacks import EarlyStopping

variables = ["id", "period,trip_time_in_secs", "trip_distance", "longitude", "latitude", "is_pickup",
             "distance to target"]
data = pd.read_csv("data/ny_preprocessed.csv", variables, delimiter=",")

print("\n\ndataframe----------------------------------------------------------\n")
print("NULL:\n")

print(data.isnull().sum())

print("\n\ndata----------------------------------------------------------\n")
print(data)
print(data.shape)


def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)


def FindMinMax(data):
    Mindata = np.min(data, 0)
    Maxdata = np.max(data, 0)
    return Mindata, Maxdata


def MinMaxReturn(val, Min, Max):
    return val * (Max - Min + 1e-7) + Min


Mintemp, Maxtemp = FindMinMax(data)
data_minmax = MinMaxScaler(data)

train = data_minmax.loc[:107488]
test = data_minmax.loc[107489:]
test = test.reset_index(drop=True)

print("\n\ntrain----------------------------------------------------------\n")
print(train)
print(train.shape)
print("\n\ntest----------------------------------------------------------\n")
print(test)
print(test.shape)

train_shifted = train.copy()
test_shifted = test.copy()

for s in range(1, 6):
    train_shifted["shift_{}".format(s)] = train_shifted["distance to target"].shift(s)
    test_shifted["shift_{}".format(s)] = test_shifted["distance to target"].shift(s)

print("\n\ntrain_shifted----------------------------------------------------------\n")
print(train_shifted)
print(train_shifted.shape)

print("\n\ntest_shifted----------------------------------------------------------\n")
print(test_shifted)
print(test_shifted.shape)

train_shifted_drop = train_shifted.dropna()
train_shifted_drop = train_shifted_drop.reset_index(drop=True)

test_shifted_drop = test_shifted.dropna()
test_shifted_drop = test_shifted_drop.reset_index(drop=True)

print("\n\ntrain_shifted_drop----------------------------------------------------------\n")
print(train_shifted_drop)
print(train_shifted_drop.shape)

print("\n\ntest_shifted_drop----------------------------------------------------------\n")
print(test_shifted_drop)
print(test_shifted_drop.shape)

X_train = train_shifted_drop.dropna().drop("shift_5", axis=1)
y_train = train_shifted_drop.dropna()[["shift_5"]]

X_test = test_shifted_drop.dropna().drop("shift_5", axis=1)
y_test = test_shifted_drop.dropna()[["shift_5"]]

print("\n\nX_train----------------------------------------------------------\n")
print(X_train)
print(X_train.shape)
train_id_count = list(X_train.groupby(["id"]).count()["period"])

print("\n\ny_train----------------------------------------------------------\n")
print(y_train)
print(y_train.shape)

print("\n\nX_test----------------------------------------------------------\n")
print(X_test)
print(X_test.shape)
test_id_count = list(X_test.groupby(["id"]).count()["period"])

print("\n\ny_test----------------------------------------------------------\n")
print(y_test)
print(y_test.shape)

print("\n\nX (Concat X_train, X_test)----------------------------------------------------------\n")
X = pd.concat([X_train, X_test])
X = X.reset_index(drop=True)
print(X)
df = pd.DataFrame(X)
df.to_csv("data/ny_X.csv", index=False, header=False)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

X_train_t = X_train.reshape(X_train.shape[0], 12, 1)
X_test_t = X_test.reshape(X_test.shape[0], 12, 1)

early_stop = EarlyStopping(monitor="loss", patience=1, verbose=1)
model = Sequential()
model.add(LSTM(50, input_shape=(12, 1)))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
model.summary()
history = model.fit(X_train_t, y_train, validation_data=(X_test_t, y_test), epochs=50, batch_size=32, verbose=1,
                    shuffle=False, callbacks=[early_stop])
pyplot.plot(history.history["loss"], label="train")
pyplot.plot(history.history["val_loss"], label="test")
pyplot.legend()
pyplot.show()
model.save("my_model")
score = model.evaluate(X_test_t, y_test, batch_size=32)
print("\nModel Evaluate Score:", score)
