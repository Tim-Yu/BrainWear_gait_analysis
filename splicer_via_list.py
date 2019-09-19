# Reading csv using csv
import numpy as np
import pandas as pd
import csv
import math
import time
import itertools as it
import matplotlib.pyplot as plt
import os
from datetime import timedelta


def intervals_splicer(walking_interval, file_dir, save_csv=1, save_fig=1):  # need to optimising the speed
    with open(file_dir) as raw:
        # reading the interval from a list
        raw_data = csv.DictReader(raw)
        for interval in walking_interval:
            time_start = time.time()  # programme timer
            start_time = interval._short_repr
            end_time = (interval + timedelta(seconds=29))._short_repr
            # storing the file with detailed time range in the name to the spliced dir
            file_name = ('.\\spliced\\' + file_dir.split("\\")[-1].split(".")[0] + start_time.replace(":", ".") + 'to' +
                         end_time.replace(":", ".") + '.csv')
            acc_collection = pd.DataFrame()
            if not os.path.isdir('spliced'):
                os.makedirs('spliced')
            if not os.path.isfile(file_name):
                n_row = 0
                # for specific time period
                for row in raw_data:  # 0 is Y 1 is X 2 is Z 4 is datetime
                    if (row.values()[4] >= start_time) & (row.values()[0] != 'NaN'):
                        tem = pd.DataFrame(data={'time': row.values()[4], 'x': row.values()[1], 'y': row.values()[0],
                                                 'z': row.values()[2]}, index=[n_row])
                        tem['time'] = pd.to_datetime(tem['time'])
                        tem['x'] = pd.to_numeric(tem['x'])
                        tem['y'] = pd.to_numeric(tem['y'])
                        tem['z'] = pd.to_numeric(tem['z'])
                        x_2 = math.pow(tem['x'], 2)
                        y_2 = math.pow(tem['y'], 2)
                        z_2 = math.pow(tem['z'], 2)
                        sum_xyz_2 = x_2 + y_2 + z_2
                        # tem['x^y'] = math.sqrt(x_2 + y_2)
                        # tem['x^z'] = math.sqrt(x_2 + z_2)  # due to the g so Euclidian Norm Minus One is not applyful
                        # tem['y^z'] = math.sqrt(y_2 + z_2)  # for the X^Y X^Z or so panel
                        tem['x^y^z'] = math.sqrt(sum_xyz_2)
                        acc_collection = acc_collection.append(tem)
                    if row.values()[4] >= end_time:  # define which data is wanted
                        break
                    n_row += 1
                if save_csv == 1:
                    acc_collection.to_csv(file_name)
            else:  # load the preexist file
                acc_collection = pd.read_csv(file_name, index_col=0)
            #  for plotting
            if not os.path.isdir('fig'):
                os.makedirs('fig')
            if not os.path.isdir('fig\\' + file_dir.split("\\")[-1].split(".")[0]):
                os.makedirs('fig\\' + file_dir.split("\\")[-1].split(".")[0])
            # determine if fig exist and if fig is needed
            if (not os.path.isfile(
                    '.\\fig\\' + file_dir.split("\\")[-1].split(".")[0] + '\\' + 'xyz' + start_time.replace(":", ".") +
                    ' to ' + end_time.replace(":", ".") + '.png') & save_fig == 1):
                plt.figure(1)
                plt.plot(acc_collection['x'], label="x")
                plt.plot(acc_collection['y'], label="y")
                plt.plot(acc_collection['z'],
                         label="z")  # significant lower values, around -1. may indicating normalization is needed
                plt.legend(bbox_to_anchor=(0.85, 0.99), loc=2, borderaxespad=0.)
                plt.savefig(
                    '.\\fig\\' + file_dir.split("\\")[-1].split(".")[0] + '\\' + 'xyz' + start_time.replace(":", ".") +
                    ' to ' + end_time.replace(":", ".") + '.png')
                plt.clf()
                plt.figure(2)
                plt.plot(acc_collection['x^y^z'], label="x^y^z")
                plt.legend(bbox_to_anchor=(0.80, 0.99), loc=2, borderaxespad=0.)
                plt.savefig(
                    '.\\fig\\' + file_dir.split("\\")[-1].split(".")[0] + '\\' + 'av' + start_time.replace(":", ".") +
                    ' to ' + end_time.replace(":", ".") + '.png')
                plt.clf()
            time_end = time.time()
            print 'time cost:', (time_end - time_start), 's'
    return acc_collection



