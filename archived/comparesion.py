# Reading csv using csv
import numpy as np
import pandas as pd
import csv
import math
import time
import itertools as it
time_start = time.time()
# for specific time period
with open("C:\\Users\\simpl\\PycharmProjects\\BW1\\BY_walking_10_1500-05.csv") as raw:
    raw_data = csv.DictReader(raw)
    acc_collection = pd.DataFrame()
    nrow = 0
    # for specific time period
    starttime = '2018-12-10 15:01:35.010'
    endtime = '2018-12-10 15:01:39.010'

    for row in raw_data:
        if (row['time'] > starttime) & (row['x'] != 'NaN' ):
            tem = pd.DataFrame(data={'time': row['time'], 'x': row['x'], 'y': row['y'], 'z': row['z']}, index=[nrow])
            tem['time'] = pd.to_datetime(tem['time'])
            tem['x'] = pd.to_numeric(tem['x'])
            tem['y'] = pd.to_numeric(tem['y'])
            tem['z'] = pd.to_numeric(tem['z'])
            x_2 = math.pow(tem['x'], 2)
            y_2 = math.pow(tem['y'], 2)
            z_2 = math.pow(tem['z'], 2)
            sum_xyz_2 = x_2 + y_2 + z_2
            tem['x^y'] = math.sqrt(x_2 + y_2)
            tem['x^z'] = math.sqrt(x_2 + z_2)  # due to the g in z therefore only the
            tem['y^z'] = math.sqrt(y_2 + z_2)  # data with z is calculated by - 1 ?
            if sum_xyz_2 >= 1:
                tem['x^y^z'] = math.sqrt(sum_xyz_2 - 1)
            else:
                tem['x^y^z'] = 0
            acc_collection = acc_collection.append(tem)
        if row['time'] > endtime:  # define which data is wanted
            break
        nrow += 1

acc_collection.to_csv('BY_det_walking_10_1500-05.csv')
# plotting the raw data
import matplotlib.pyplot as plt
plt.plot(acc_collection['x'], label="x")
plt.plot(acc_collection['y'], label="y")
plt.plot(acc_collection['z'], label="z")  # significant lower values, around -1. may indicating normalization is needed
plt.legend(bbox_to_anchor=(0.85, 0.99), loc=2, borderaxespad=0.)
plt.show()

plt.plot(acc_collection['x^y^z'], label="x^y^z")
plt.legend(bbox_to_anchor=(0.80, 0.99), loc=2, borderaxespad=0.)
plt.show()
time_end = time.time()
print 'time cost:', (time_end - time_start), 's'

