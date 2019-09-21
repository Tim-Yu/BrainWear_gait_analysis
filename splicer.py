# Reading csv using csv
import pandas as pd
import csv
import math
import time
import matplotlib.pyplot as plt
import os
from datetime import timedelta
# quick note: the input data is using AX3 GUI processed raw data,


def splicer(start_time, end_time, file_dir, save=1):
    # storing the file with detailed time range in the name to the spliced dir
    if not os.path.isdir('spliced'):
        os.makedirs('spliced')
    if not os.path.isdir('./spliced/' + file_dir.split("/")[-1].split(".")[0]):
        os.makedirs('./spliced/' + file_dir.split("/")[-1].split(".")[0])
    file_name = ('./spliced/' + file_dir.split("/")[-1].split(".")[0] + '/' + file_dir.split("/")[-1].split(".")[0] +
                 start_time.replace(":", ".") + 'to' + end_time.replace(":", ".") + '.csv')
    if not os.path.isfile(file_name):  # in case the setting is already run before. search for preexist file
        with open(file_dir) as raw:
            raw_data = csv.DictReader(raw)
            acc_collection = pd.DataFrame()
            n_row = 0
            # for specific time period
            for row in raw_data:
                if (row['time'] > start_time) and (row['x'] != 'NaN'):
                    tem = pd.DataFrame(data={'time': row['time'], 'x': row['x'], 'y': row['y'], 'z': row['z']},
                                       index=[n_row])
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
                if row['time'] > end_time:  # define which data is wanted
                    break
                n_row += 1
        if save == 1:
            acc_collection.to_csv(file_name)
    else:  # load the preexist file
        acc_collection = pd.read_csv(file_name, index_col=0)
    return acc_collection


def intervals_splicer(walking_intervals, file_dir, save_csv=1, save_fig=1, save_xyzfug=0):
    with open(file_dir) as raw:
        # reading the interval from a list
        if not os.path.isdir('5s_spliced'):
            os.makedirs('5s_spliced')
        if not os.path.isdir('./5s_spliced/' + file_dir.split("/")[-1].split(".")[0]):
            os.makedirs('./5s_spliced/' + file_dir.split("/")[-1].split(".")[0])
        raw_data = csv.DictReader(raw)
        for intervals in walking_intervals:
            time_starts = time.time()  # programme timer
            start_time = intervals._short_repr
            end_time = (intervals + timedelta(seconds=30))._short_repr
            # storing the file with detailed time range in the name to the spliced dir
            file_name = ('./5s_spliced/' + file_dir.split("/")[-1].split(".")[0] + '/' +
                         file_dir.split("/")[-1].split(".")[0] + start_time.replace(":", ".") + 'to' +
                         end_time.replace(":", ".") + '.csv')
            acc_collection = pd.DataFrame()
            if not os.path.isfile(file_name):
                n_row = 0
                # for specific time period
                for row in raw_data:  # 0 is Y 1 is X 2 is Z 4 is datetime, from the omi is 0 is datetime x is 1 y is 2
                    if (row.values()[2] >= start_time) and (row.values()[1] != 'NaN'):
                        tem = pd.DataFrame(data={'time': row.values()[2], 'x': row.values()[0], 'y': row.values()[1],
                                                 'z': row.values()[3]}, index=[n_row])
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
                    if row.values()[2] >= end_time:  # define which data is wanted
                        break
                    n_row += 1
                if save_csv == 1:
                    acc_collection.to_csv(file_name)
            else:  # load the preexist file
                acc_collection = pd.read_csv(file_name, index_col=0)
            #  for plotting
            if not os.path.isdir('5s_fig'):
                os.makedirs('5s_fig')
            if not os.path.isdir('5s_fig/' + file_dir.split("/")[-1].split(".")[0]):
                os.makedirs('5s_fig/' + file_dir.split("/")[-1].split(".")[0])
            # determine if fig exist and if fig is needed
            if (not os.path.isfile(
                    './5s_fig/' + file_dir.split("/")[-1].split(".")[0] + '/' + 'av' + start_time.replace(":", ".") +
                    ' to ' + end_time.replace(":", ".") + '.png') and save_fig == 1):
                if save_xyzfug == 1:
                    plt.figure(1)
                    plt.plot(acc_collection['x'], label="x")
                    plt.plot(acc_collection['y'], label="y")
                    plt.plot(acc_collection['z'],
                             label="z")  # significant lower values, around -1. may indicating normalization is needed
                    plt.legend(bbox_to_anchor=(0.85, 0.99), loc=2, borderaxespad=0.)
                    plt.savefig(
                        './5s_fig/' + file_dir.split("/")[-1].split(".")[0] + '/' + 'xyz' + start_time.replace(":", ".") +
                        ' to ' + end_time.replace(":", ".") + '.png')
                plt.clf()
                plt.figure(2)
                plt.plot(acc_collection['x^y^z'], label="x^y^z")
                plt.legend(bbox_to_anchor=(0.80, 0.99), loc=2, borderaxespad=0.)
                plt.savefig(
                    './5s_fig/' + file_dir.split("/")[-1].split(".")[0] + '/' + 'av' + start_time.replace(":", ".") +
                    ' to ' + end_time.replace(":", ".") + '.png')
                plt.clf()
            time_ends = time.time()
            print 'time cost:', (time_ends - time_starts), 's'


# function from module accUtils edited to be functional with out installation
def loadTimeSeriesCSV(tsFile):
    """Load time series csv.gz file and append date/time column to it

    The time associated with each reading can be inferred from the very first
    row, which describes the sample rate, start and end times of the data.

    For example header
    "acceleration (mg) - 2014-05-07 13:29:50 - 2014-05-13 09:50:25 - sampleRate = 5 seconds, imputed"
    indicates that the first measurement time is at 2014-05-07 13:29:50, the second
    at 2014-05-07 13:29:55, the third at 2014-05-07 13:30:00 ... and the last at
    2014-05-13 09:50:25.

    :param str tsFile: Output filename for .csv.gz file

    :return: Pandas dataframe of epoch data
    :rtype: pandas.DataFrame

    :Example:
    >>> import accUtils
    >>> import pandas as pd
    >>> df = accUtils.loadTimeSeriesCSV("sample-timeSeries.csv.gz")
    <returns pd.DataFrame>
    """
    DAYS = ['mon', 'tue', 'wed', 'thur', 'fri', 'sat', 'sun']
    TIME_SERIES_COL = 'time'
    # get header
    header = pd.read_csv(tsFile, nrows=1, header=0, compression='gzip')
    headerInfo = header.columns[0]
    if header.columns[0] == TIME_SERIES_COL:
        headerInfo = header.columns[1]
        header.columns = [TIME_SERIES_COL, 'acc'] + header.columns[2:].tolist()
    else:
        header.columns = ['acc'] + header.columns[1:].tolist()
        # read start time, endtime, and sample rate
        startDate = headerInfo.split(" - ")[1]
        endDate = headerInfo.split(" - ")[2]
        sampleRate = headerInfo.split("sampleRate = ")[1].split(" ")[0]

    # read data
    tsData = pd.read_csv(tsFile, skiprows=1, header=None, names=header.columns,
                         compression='gzip')
    if header.columns[0] != TIME_SERIES_COL:
        tsData.index = pd.date_range(start=startDate, end=endDate,
                                     freq=str(sampleRate) + 's')
    return tsData


current_dir = os.path.dirname(__file__)
os.chdir(current_dir)
# try to get the time interval from the time series file
df = loadTimeSeriesCSV("./example_data/volunteer2-timeSeries.csv.gz")
walking_interval = df[df['walking'] == 1].index  # ._short_repr will be the str
sleeping_interval = df[df['sleep'] == 1].index
sedentary_interval = df[df['sedentary'] == 1].index
moderate_interval = df[df['moderate'] == 1].index
imputed_interval = df[df['imputed'] == 1].index
# defining the path for the raw data
dir_of_file = "./example_data/volunteer2.csv"

# faster way to do is using the intervals_splicer, it includes the plotting functions

intervals_splicer(walking_interval, dir_of_file)

# individual splicer
'''
# define the parameter used for the function. For specific time period, defining the time period as below
start_datetime = '2018-10-18 13:05:30.010'
end_datetime = '2018-10-18 13:05:38.010'


time_start = time.time()
spliced = splicer(start_datetime, end_datetime, dir_of_file)

# plotting the individual result

if not os.path.isdir('fig'):
    os.makedirs('fig')
if not os.path.isdir('fig/' + dir_of_file.split("/")[-1].split(".")[0]):
    os.makedirs('fig/' + dir_of_file.split("/")[-1].split(".")[0])
plt.figure(1)
plt.plot(spliced['x'], label="x")
plt.plot(spliced['y'], label="y")
plt.plot(spliced['z'], label="z")  # significant lower values, around -1. may indicating normalization is needed
plt.legend(bbox_to_anchor=(0.85, 0.99), loc=2, borderaxespad=0.)
plt.savefig('./fig/' + dir_of_file.split("/")[-1].split(".")[0] + '/' + 'xyz' + start_datetime.replace(":", ".") +
            ' to ' + end_datetime.replace(":", ".") + '.png')
plt.clf()
plt.figure(2)
plt.plot(spliced['x^y^z'], label="x^y^z")
plt.legend(bbox_to_anchor=(0.80, 0.99), loc=2, borderaxespad=0.)
plt.savefig('./fig/' + dir_of_file.split("/")[-1].split(".")[0] + '/' + 'av' + start_datetime.replace(":", ".") +
            ' to ' + end_datetime.replace(":", ".") + '.png')
plt.clf()
time_end = time.time()
print 'time cost:', (time_end - time_start), 's'
'''

# try to split a period of time to walking intervals via the classification
'''for interval in walking_interval:
    time_start = time.time()
    interval_end = interval + timedelta(seconds=30)
    spliced = splicer(interval._short_repr, interval_end._short_repr, dir_of_file)
    if not os.path.isdir('fig'):
        os.makedirs('fig')
    if not os.path.isdir('fig/' + dir_of_file.split("/")[-1].split(".")[0]):
        os.makedirs('fig/' + dir_of_file.split("/")[-1].split(".")[0])
    plt.figure(1)
    plt.plot(spliced['x'], label="x")
    plt.plot(spliced['y'], label="y")
    plt.plot(spliced['z'], label="z")  # significant lower values, around -1. may indicating normalization is needed
    plt.legend(bbox_to_anchor=(0.85, 0.99), loc=2, borderaxespad=0.)
    plt.savefig(
        './fig/' + dir_of_file.split("/")[-1].split(".")[0] + '/' + 'xyz' + interval._short_repr.replace(":", ".") +
        ' to ' + interval_end._short_repr.replace(":", ".") + '.png')
    plt.clf()
    plt.figure(2)
    plt.plot(spliced['x^y^z'], label="x^y^z")
    plt.legend(bbox_to_anchor=(0.80, 0.99), loc=2, borderaxespad=0.)
    plt.savefig(
        './fig/' + dir_of_file.split("/")[-1].split(".")[0] + '/' + 'av' + interval._short_repr.replace(":", ".") +
        ' to ' + interval_end._short_repr.replace(":", ".") + '.png')
    plt.clf()
    time_end = time.time()
    print 'time cost:', (time_end - time_start), 's'
'''

# characteristic of the period

