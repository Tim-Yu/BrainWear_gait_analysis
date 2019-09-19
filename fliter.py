import pandas as pd
import os
import numpy as np
import pywt
# import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from scipy.signal import find_peaks_cwt
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
files_path = r'C:\Users\simpl\PycharmProjects\BW1\spliced'
gait_data = []
labels = []
file_names = []

os.chdir(files_path)
label = 0
for folder_name in os.listdir(files_path):
    second_folder = os.path.join(os.path.abspath(files_path), folder_name)
    for file_name_pre in os.listdir(second_folder):
        gait_file = None
        file_name = os.path.join(second_folder, file_name_pre)
        if file_name.startswith(second_folder):
            # label = 0
            gait_file = pd.read_csv(file_name)
            gait_file = gait_file['x^y^z']
        if (gait_file is not None) and sum(abs(np.fft.fft(gait_file).real) > 100) > 3:
            labels.append(label)
            file_names.append(file_name)
            gait_data.append(gait_file)
    label += 1


# Using cwt to generating features
pca = PCA(n_components=1)
n = 0
features = np.empty((0, 149))
scale = np.arange(1, 150)
for ind in range(len(gait_data)):
    n += 1
    print(str(n) + '/' + str(len(gait_data)))  # progress indicator
    coef, freq = pywt.cwt(gait_data[ind][:2000], scale, 'gaus5')
    features = np.vstack([features, pca.fit_transform(coef).flatten()])

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20)

clf = SVC(gamma='scale')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print (accuracy)
# save the model for future use
joblib.dump(clf, "filter_model.m")
# test on how to differentiate good quality data and bad quality data

# stationary_test = pd.read_csv(r'C:\Users\simpl\PycharmProjects\BW1\spliced_back\KLC_20181030_CWA-DATAEpoch2018-10-17 17.51.35to2018-10-17 17.52.04.csv')
# Changing_test = pd.read_csv(r'C:\Users\simpl\PycharmProjects\BW1\spliced_back\KLC_20181030_CWA-DATAEpoch2018-10-17 19.05.05to2018-10-17 19.05.34.csv')



# os.path.splitext(os.listdir(r'C:\Users\simpl\PycharmProjects\BW1\selected\changing')[1])[0].split('av')[1].split('to')[0]


# getting the needed file names for filtering the useful feed in data
selected_file_names = []
selected_file_data = []
labels = []
for file_selected in os.listdir(r'C:\Users\simpl\PycharmProjects\BW1\selected\BY_5s'):
    test ='C://Users//simpl//PycharmProjects//BW1//5s_spliced//BYB-CWA-DATAEpoch' + '//' + 'BYB-CWA-DATAEpoch' + os.path.splitext(file_selected)[0].split('av')[1].split(' to ')[0] + 'to' + os.path.splitext(file_selected)[0].split('av')[1].split(' to ')[1] + '.csv'
    tem_pd = pd.read_csv(test)
    tem_pd = tem_pd['x^y^z']
    selected_file_names.append(test)
    selected_file_data.append(tem_pd)
    labels.append(0)

for file_selected in os.listdir(r'C:\Users\simpl\PycharmProjects\BW1\selected\BY_5s_changing'):
    test ='C://Users//simpl//PycharmProjects//BW1//5s_spliced//BYB-CWA-DATAEpoch' + '//' + 'BYB-CWA-DATAEpoch' + os.path.splitext(file_selected)[0].split('av')[1].split(' to ')[0] + 'to' + os.path.splitext(file_selected)[0].split('av')[1].split(' to ')[1] + '.csv'
    tem_pd = pd.read_csv(test)
    tem_pd = tem_pd['x^y^z']
    selected_file_names.append(test)
    selected_file_data.append(tem_pd)
    labels.append(1)

pca = PCA(n_components=1)
n = 0
features = np.empty((0, 149))
scale = np.arange(1, 150)
for ind in range(len(selected_file_data)):
    n += 1
    print(str(n) + '/' + str(len(selected_file_data)))  # progress indicator
    coef, freq = pywt.cwt(selected_file_data[ind][:2800], scale, 'morl')
    features = np.vstack([features, pca.fit_transform(coef).flatten()])

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20)

clf = SVC(gamma='scale')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print (accuracy)

# try the new feed in data

selected_file_names = []
selected_file_data = []
labels = []
for file_selected in os.listdir(r'C:\Users\simpl\PycharmProjects\BW1\selected\KLC_stationary'):
    test ='C://Users//simpl//PycharmProjects//BW1//spliced//KLC_20181030_CWA-DATAEpoch' + '//' + 'KLC_20181030_CWA-DATAEpoch' + os.path.splitext(file_selected)[0].split('av')[1].split(' to ')[0] + 'to' + os.path.splitext(file_selected)[0].split('av')[1].split(' to ')[1] + '.csv'
    tem_pd = pd.read_csv(test)
    tem_pd = tem_pd['x^y^z']
    selected_file_names.append(test)
    selected_file_data.append(tem_pd)
    labels.append(0)

for file_selected in os.listdir(r'C:\Users\simpl\PycharmProjects\BW1\selected\BY_stationary'):
    test ='C://Users//simpl//PycharmProjects//BW1//spliced//BYB-CWA-DATAEpoch' + '//' + 'BYB-CWA-DATAEpoch' + os.path.splitext(file_selected)[0].split('av')[1].split(' to ')[0] + 'to' + os.path.splitext(file_selected)[0].split('av')[1].split(' to ')[1] + '.csv'
    tem_pd = pd.read_csv(test)
    tem_pd = tem_pd['x^y^z']
    selected_file_names.append(test)
    selected_file_data.append(tem_pd)
    labels.append(1)

pca = PCA(n_components=2)
n = 0
features = np.empty((0, 198))
scale = np.arange(1, 100)
# coef_cluster = np.empty((0, 199))
for ind in range(len(selected_file_data)):
    n += 1
    print(str(n) + '/' + str(len(selected_file_data)))  # progress indicator
    coef, freq = pywt.cwt(selected_file_data[ind][:2800], scale, 'morl')
    features = np.vstack([features, pca.fit_transform(coef).flatten()])
    # coef_cluster = np.vstack([coef_cluster, pca.transform(coef)])

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20)

clf = SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print (accuracy)


# try with DWT the performance is same

n = 0
coef_cluster = np.empty((0, 204))
for ind in range(len(selected_file_data)):
    n += 1
    print(str(n) + '/' + str(len(selected_file_data)))  # progress indicator
    cA, cD = pywt.dwt(selected_file_data[ind][:400], 'db5')
    coef_cluster = np.vstack([coef_cluster, cA])

X_train, X_test, y_train, y_test = train_test_split(coef_cluster, labels, test_size=0.20)

clf = SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print (accuracy)
# try with fft poor performance

n = 0
coef_cluster = np.empty((0, 2000))
for ind in range(len(gait_data)):
    n += 1
    print(str(n) + '/' + str(len(gait_data)))  # progress indicator
    coef = np.fft.fft(gait_data[ind]['x^y^z'][:2000]).real[:2000]
    coef_cluster = np.vstack([coef_cluster, coef])

X_train, X_test, y_train, y_test = train_test_split(coef_cluster, labels, test_size=0.20)

clf = SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print (accuracy)

# find peak in every 1 s
'''
peakind = find_peaks_cwt(oranginal, np.arange(1, 100))
oranginal[peakind]
peakind, _ = find_peaks(gait_data[1]['x^y^z'], distance=70)
'''
# the filter for getting the stationary walking data

# !!!!! need to average the input data to same amount in case of bias
selected_label = []
selected_file_data = []
selected_file_names = []
for n in range(len(gait_data)):
    peak_ind, _ = find_peaks(gait_data[n]['x^y^z'], distance=55)
    low_ind, _ = find_peaks(-gait_data[n]['x^y^z'], distance=55)
    range_hl = gait_data[n]['x^y^z'][peak_ind].mean() - gait_data[n]['x^y^z'][low_ind].mean()
    diff = gait_data[n]['x^y^z'][peak_ind].max() - gait_data[n]['x^y^z'][peak_ind].min()
    if diff <= 0.4 and range_hl > 0.5:
        selected_file_data.append(gait_data[n]['x^y^z'])
        selected_label.append(labels[n])
        selected_file_names.append(file_names[n])

pca = PCA(n_components=1)
n = 0
sele_features = np.empty((0, 149))
scale = np.arange(1, 150)
for ind in range(len(selected_file_data)):
    n += 1
    print(str(n) + '/' + str(len(selected_file_data)))  # progress indicator
    coef, freq = pywt.cwt(selected_file_data[ind][:500], scale, 'gaus5')
    sele_features = np.vstack([sele_features, pca.fit_transform(coef).flatten()])
X_train, X_test, y_train, y_test = train_test_split(sele_features, selected_label, test_size=0.20)
clf = SVC(gamma='scale')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print (accuracy)

plt.figure(1)
plt.plot(selected_file_data[4000], label="x^y^z")
plt.legend(bbox_to_anchor=(0.80, 0.99), loc=2, borderaxespad=0.)

# visualise the selected data
for i in selected_file_data:
    plt.figure(1)
    plt.plot(i, label="x^y^z")
    plt.legend(bbox_to_anchor=(0.80, 0.99), loc=2, borderaxespad=0.)
    plt.savefig('./fig_t/' + str(n) + '.png')
    n += 1
    plt.clf()

# present the miss matched data

matches = pd.concat([y_pred, y_test], axis=1, ignore_index=True)
matched = pd.DataFrame(matches[map(lambda x, y: (x == y), matches[0], matches[1])])
miss_match = pd.DataFrame(matches[map(lambda x, y: (x != y), matches[0], matches[1])])
# notice 0 is byb; 1 is jw; 2 is klc; 3 is mw
# showing the attribution of miss matches
B2B = sum(matched[1] == 0)
B2J = sum(miss_match[0][miss_match[1] == 0] == 1)
B2K = sum(miss_match[0][miss_match[1] == 0] == 2)
B2M = sum(miss_match[0][miss_match[1] == 0] == 3)

J2B = sum(miss_match[0][miss_match[1] == 1] == 0)
J2J = sum(matched[1] == 1)
J2K = sum(miss_match[0][miss_match[1] == 1] == 2)
J2M = sum(miss_match[0][miss_match[1] == 1] == 3)

K2B = sum(miss_match[0][miss_match[1] == 2] == 0)
K2J = sum(miss_match[0][miss_match[1] == 2] == 1)
K2K = sum(matched[1] == 2)
K2M = sum(miss_match[0][miss_match[1] == 2] == 3)

M2B = sum(miss_match[0][miss_match[1] == 3] == 0)
M2J = sum(miss_match[0][miss_match[1] == 3] == 1)
M2K = sum(miss_match[0][miss_match[1] == 3] == 2)
M2M = sum(matched[1] == 3)
print ('B2J', B2J)

# try to see the performance of using JW

selected_label_new = map(lambda x: int(x == 1), selected_label)

