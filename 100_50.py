from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
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

df = loadTimeSeriesCSV(r'C:\Users\simpl\Documents\VM_Ubuntu\50\45676_50hz_CWA-DATA-timeSeries.csv.gz')
walking_interval2 = df[df['walking'] == 1].index  # ._short_repr will be the str
sleeping_interval2 = df[df['sleep'] == 1].index
sedentary_interval2 = df[df['sedentary'] == 1].index
moderate_interval2 = df[df['moderate'] == 1].index
imputed_interval2 = df[df['imputed'] == 1].index


plt.figure()
barWidth = 0.375
names = ['walking', 'sleeping', 'sedentary', 'moderate', 'imputed']

r1 = range(5)
r2 = map(lambda x: (x + 0.5 * barWidth), r1)
r1 = map(lambda x: (x - 0.5 * barWidth), r1)

# Create red Bars walking
plt.bar(r1, [len(walking_interval), len(sleeping_interval), len(sedentary_interval), len(moderate_interval), len(imputed_interval)], color='#CD6155', edgecolor='white', width=barWidth,
        label="100Hz")
# Create yellow Bars moderate
plt.bar(r2, [len(walking_interval2), len(sleeping_interval2), len(sedentary_interval2), len(moderate_interval2), len(imputed_interval2)], color='#F5B041', edgecolor='white', width=barWidth, label="50Hz")
# Custom axis
plt.xticks(range(5), names)
plt.xlabel("Activities")
plt.ylabel("Epochs")
plt.legend(loc="NorthOutside", fontsize=20)
# Show graphic
plt.show()

