from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import pandas as pd
from datetime import timedelta
# generating animation
raw_data = pd.read_csv(r'C:\Users\simpl\Data_folder\5s_out\BYB-CWA-DATAEpoch\BYB-CWA-DATAEpoch2018-12-07 19.03.24to2018-12-07 19.03.29.csv')
raw_data.drop(raw_data.columns[raw_data.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
fig, ax = plt.subplots()
x = np.arange(0, 500, 1)
line, = ax.plot(x, raw_data['x^y^z'][x])
def animate(i):
    line.set_ydata(raw_data['x^y^z'][x + i])
    return line,
def init():
    line.set_ydata(raw_data['x^y^z'][x])
    return line,
ani = animation.FuncAnimation(fig=fig,
                              func=animate,
                              frames=499,
                              init_func=init,
                              interval=20,
                              blit=False)
plt.show()



def update_points(num):
    point_ani.set_data(x[num], raw_data['x'][num])
    return point_ani,
def update_points2(num):
    point_ani.set_data(x[num], raw_data['y'][num])
    return point_ani,
def update_points3(num):
    point_ani.set_data(x[num], raw_data['z'][num])
    return point_ani,

fig = plt.figure(tight_layout=True)
plt.plot(x, raw_data['x'])
plt.plot(x, raw_data['y'])
plt.plot(x, raw_data['z'])
point_ani, = plt.plot(x[0], raw_data['x'][0], "ro")
point_ani2, = plt.plot(x[0], raw_data['y'][0], "ro")
point_ani3, = plt.plot(x[0], raw_data['z'][0], "ro")
plt.grid(ls="--")
ani = animation.FuncAnimation(fig, update_points, np.arange(0, 500), interval=1, blit=True)
ani2 = animation.FuncAnimation(fig, update_points2, np.arange(0, 500), interval=1, blit=True)
ani3 = animation.FuncAnimation(fig, update_points3, np.arange(0, 500), interval=1, blit=True)
plt.xlabel("ms")
plt.ylabel("g(9.8m/s^2)")
ani.save('walking2.gif', writer='imagemagick', fps=1000)

plt.show()


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# plotting the raw accelerometer data
x = raw_data['x']
y = raw_data['y']
z = raw_data['z']
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x, y, z)
plt.show()

# plotting the moving pattern

vx, vy, vz = [], [], []

for row in np.arange(len(raw_data)):
    vx.append(0.01 * 9.8 * sum(raw_data['x'][:row]))
    vy.append(0.01 * 9.8 * sum(raw_data['y'][:row]))
    vz.append(0.01 * 9.8 * sum(raw_data['z'][:row]))

sx, sy, sz = [], [], []
sx.append(0)
sy.append(0)
sz.append(0)

for row in np.arange(len(raw_data)):
    sx.append(sx[row] + vx[row] * 0.01 + 0.5 * x[row] * 0.01 * 0.01)
    sy.append(sy[row] + vy[row] * 0.01 + 0.5 * y[row] * 0.01 * 0.01)
    sz.append(sz[row] + vz[row] * 0.01 + 0.5 * z[row] * 0.01 * 0.01)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(sx, sy, sz)
plt.show()


# making figures showing the rate of walking and sleeping ect.

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


# try to get the time interval from the time series file
df = loadTimeSeriesCSV(r'C:\Users\simpl\Documents\VM_Ubuntu\data\BW_015_20190220-timeSeries.csv.gz')
walking_interval = df[df['walking'] == 1].index  # ._short_repr will be the str
sleeping_interval = df[df['sleep'] == 1].index
sedentary_interval = df[df['sedentary'] == 1].index
moderate_interval = df[df['moderate'] == 1].index
imputed_interval = df[df['imputed'] == 1].index

date_range = df.index[-1] - df.index[1]

if df.index[1].time() > df.index[-1].time():
    day_total = date_range.days + 1  # plus the first day and the last day
else:
    day_total = date_range.days  # only missing one day
walking_sum = []
sleeping_sum = []
sedentary_sum = []
moderate_sum = []
imputed_sum = []
epoch_time = 5
for day in range(day_total):
    current_date = df.index[1] + timedelta(days=day)
    walking_sum.append(epoch_time * sum(map(lambda current: (current.date() == current_date.date()), walking_interval)))
    sleeping_sum.append(epoch_time * sum(map(lambda current: (current.date() == current_date.date()), sleeping_interval)))
    sedentary_sum.append(epoch_time * sum(map(lambda current: (current.date() == current_date.date()), sedentary_interval)))
    moderate_sum.append(epoch_time * sum(map(lambda current: (current.date() == current_date.date()), moderate_interval)))
    imputed_sum.append(epoch_time * sum(map(lambda current: (current.date() == current_date.date()), imputed_interval)))
# taking log
'''walking_sum = []
sleeping_sum = []
sedentary_sum = []
moderate_sum = []
imputed_sum = []
for day in range(day_total):
    current_date = df.index[1] + timedelta(days=day)
    walking_sum.append(math.log(sum(map(lambda current: (current.date() == current_date.date()), walking_interval)) + 1))
    sleeping_sum.append(math.log(sum(map(lambda current: (current.date() == current_date.date()), sleeping_interval)) + 1))
    sedentary_sum.append(math.log(sum(map(lambda current: (current.date() == current_date.date()), sedentary_interval)) + 1))
    moderate_sum.append(math.log(sum(map(lambda current: (current.date() == current_date.date()), moderate_interval)) + 1))
    imputed_sum.append(math.log(sum(map(lambda current: (current.date() == current_date.date()), imputed_interval)) + 1))
'''
# plot
barWidth = 0.85
names = map(lambda x: ('Day ' + str(int(x))), (range(day_total) + np.ones(day_total)))
# set the start date
start = 1
day_total = day_total - start
plt.figure()
# Create green Bars imputed
plt.bar(range(day_total), imputed_sum[start:], color='#52BE80', edgecolor='white', width=barWidth, label="Imputed")
# Create red Bars walking
plt.bar(range(day_total), walking_sum[start:], bottom=imputed_sum[start:], color='#CD6155', edgecolor='white', width=barWidth,
        label="Walking")
# Create blue Bars sleeping
plt.bar(range(day_total), sleeping_sum[start:], bottom=map(lambda i, j: (i + j), imputed_sum[start:], walking_sum[start:]), color='#5499C7',
        edgecolor='white', width=barWidth, label="Sleeping")
# Create gray Bars sedentary
plt.bar(range(day_total), sedentary_sum[start:], bottom=map(lambda i, j, h: (i + j + h), imputed_sum[start:], walking_sum[start:], sleeping_sum[start:])
        , color='#99A3A4', edgecolor='white', width=barWidth, label="Sit/Stand")
# Create yellow Bars moderate
plt.bar(range(day_total), moderate_sum[start:], bottom=map(lambda i, j, h, k: (i + j + h + k), imputed_sum[start:], walking_sum[start:],
                                                   sleeping_sum[start:], sedentary_sum[start:]),
        color='#F5B041', edgecolor='white', width=barWidth, label="Light working")
# Custom axis
plt.xticks(range(day_total), names)
plt.xlabel("Days")
plt.ylabel("Seconds")
plt.legend(loc="upper right", fontsize=20)
# Show graphic
plt.show()


# showing the light working and walking

plt.figure()
barWidth = 0.375
names = map(lambda x: ('Day ' + str(int(x))), (range(day_total) + np.ones(day_total)))

r1 = range(day_total)
r2 = map(lambda x: (x + 0.5 * barWidth), r1)
r1 = map(lambda x: (x - 0.5 * barWidth), r1)

# Create red Bars walking
plt.bar(r1, walking_sum[start:], color='#CD6155', edgecolor='white', width=barWidth,
        label="Walking")
# Create yellow Bars moderate
plt.bar(r2, moderate_sum[start:], color='#F5B041', edgecolor='white', width=barWidth, label="Light working")
# Custom axis
plt.xticks(range(day_total), names)
plt.xlabel("Days")
plt.ylabel("Seconds")
plt.legend(loc="NorthOutside", fontsize=20)
# Show graphic
plt.show()


# calculating the percentage reduced

