# Title: LSTM
# Author: Junghwan Kim
# Date: April 21, 2021

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow import keras
from numpy import array
import pandas as pd
import time

min = pd.Series({"id": 1.0, "period": 0.0, "trip_time_in_secs": 0.0, "trip_distance": 0.0, "longitude": -110.53633,
                 "latitude": -0.56333297, "is_pickup": 0.0, "distance to target": 0.0})
max = pd.Series({"id": 143.0, "period": 6.0, "trip_time_in_secs": 10620.0, "trip_distance": 97.8, "longitude": 0.0,
                 "latitude": 74.006752, "is_pickup": 1.0, "distance to target": 38975.0})
id_count = [3071, 2318, 3310, 3066, 1286, 168, 2410, 752, 2212, 776, 2318, 2418, 1620, 1576, 2266, 1020, 2232, 2568,
            1996, 3362, 1802, 2340, 2452, 3352, 2360, 2460, 766, 2130, 2080, 2224, 3076, 2474, 3053, 3278, 2222, 1510,
            3384, 1610, 1392, 2126, 2862, 2710, 1998, 2322, 2334, 1370, 2124, 1962, 1020, 1946, 2137, 2892, 2364, 2720,
            3184, 2994, 3410, 2478, 2272, 2314, 2374, 3344, 2394, 1238, 2710, 2294, 1358, 1460, 3130, 2312, 2962, 186,
            926, 2438, 1580, 3058, 2672, 2428, 1500, 2980, 2350, 2950, 2168, 886, 2550, 2654, 2078, 572, 2152, 1808,
            2670, 2336, 546, 902, 1358, 2236, 2652, 1434, 2, 3078, 2864, 66, 348, 1778, 1994, 1902, 3088, 642, 1840,
            2014, 1992, 1804, 2932, 1536, 1164, 2978, 728, 2726, 2514, 2492, 2256, 2390, 252, 2714, 1178, 2550, 2570,
            1506, 2402, 2922, 2728, 2622, 3066, 2380, 1474, 2774, 2334, 1058, 882, 948, 3060, 2544, 1002, 0]
id_line = [3071, 5389, 8699, 11765, 13051, 13219, 15629, 16381, 18593, 19369, 21687, 24105, 25725, 27301, 29567, 30587,
           32819, 35387, 37383, 40745, 42547, 44887, 47339, 50691, 53051, 55511, 56277, 58407, 60487, 62711, 65787,
           68261, 71314, 74592, 76814, 78324, 81708, 83318, 84710, 86836, 89698, 92408, 94406, 96728, 99062, 100432,
           102556, 104518, 105538, 107484, 109621, 112513, 114877, 117597, 120781, 123775, 127185, 129663, 131935,
           134249, 136623, 139967, 142361, 143599, 146309, 148603, 149961, 151421, 154551, 156863, 159825, 160011,
           160937, 163375, 164955, 168013, 170685, 173113, 174613, 177593, 179943, 182893, 185061, 185947, 188497,
           191151, 193229, 193801, 195953, 197761, 200431, 202767, 203313, 204215, 205573, 207809, 210461, 211895,
           211897, 214975, 217839, 217905, 218253, 220031, 222025, 223927, 227015, 227657, 229497, 231511, 233503,
           235307, 238239, 239775, 240939, 243917, 244645, 247371, 249885, 252377, 254633, 257023, 257275, 259989,
           261167, 263717, 266287, 267793, 270195, 273117, 275845, 278467, 281533, 283913, 285387, 288161, 290495,
           291553, 292435, 293383, 296443, 298987, 299989]
variables = ["id", "period", "trip_time_in_secs", "trip_distance", "longitude", "latitude", "is_pickup",
             "distance to target", "shift_1", "shift_2", "shift_3", "shift_4"]
columns = ["id", "timestamp", "predicted_distance_to_target"]
df = pd.read_csv("data/ny_X.csv", header=None, names=variables)
model = keras.models.load_model("/Users/jkim/Classes/DTM/Project/Python/LSTM/my_model")

def MinMaxReturn(val, Min, Max):
    result = float(val * (Max - Min + 1e-7) + Min)
    return "{:.4f}".format(result)


output = pd.DataFrame([[0, 0, 0]], columns=columns)

id = 1
line = 0
start = time.time()
for id in range(1, 144):
    timestamp = 0
    while line < id_line[id - 1]:
        x_input = array(df.loc[line:line].values.tolist()).reshape((1, 12, 1))
        x_output = model.predict(x_input, verbose=0)
        x_output_value = MinMaxReturn(x_output, min[7], max[7])
        output_line = pd.DataFrame([[id, timestamp, x_output_value]], columns=columns)
        #print("Taxi ID:", id, "\t\tCSV line:", line, "\t\tPredicted distance to target:", x_output_value)
        output = pd.concat([output, output_line])
        line += 1
        timestamp += 1

output = output.iloc[1:]
output = output.reset_index(drop=True)
output.to_csv("results/ny_output.csv", columns=columns, index=False, header=True)
print(output)
print("Elapsed time:", time.time() - start)
