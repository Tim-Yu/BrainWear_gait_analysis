import pandas as pd
import os
import numpy as np
import pywt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.externals import joblib
'''
Folder_Path = 'C:/Users/simpl/PycharmProjects/BW1/test'
SaveFile_Path = 'C:/Users/simpl/PycharmProjects/BW1'
SaveFile_Name = 'KLC_test_1.csv'

os.chdir(Folder_Path)

file_list = os.listdir(Folder_Path)


#df = pd.read_csv(Folder_Path + '/' + file_list[0])
#df.to_csv(SaveFile_Path + '/' + SaveFile_Name, index=False)

for i in file_list:
    df = pd.read_csv(Folder_Path + '/' + i)
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    df.to_csv(SaveFile_Path + '/' + SaveFile_Name, index=False, header=False, mode='a+')

summary = pd.read_csv(SaveFile_Path + '/' + SaveFile_Name)
'''



# visualisation of cwt data
klc_typical = pd.read_csv(r'C:\Users\simpl\PycharmProjects\BW1\spliced_2\KLC_20181030_CWA-DATAEpoch2018-10-17 17.52.05to2018-10-17 17.52.34.csv')
klc_typical.drop(klc_typical.columns[klc_typical.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
MW_typical = pd.read_csv(r'C:\Users\simpl\PycharmProjects\BW1\spliced\MW20181107_LEFT_CWA-DATAEpoch\MW20181107_LEFT_CWA-DATAEpoch2018-10-18 21.46.04to2018-10-18 21.46.33.csv')
MW_typical.drop(MW_typical.columns[MW_typical.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

# fig = plt.figure(1)

scale = np.arange(1, 100)

coef1, freq1 = pywt.cwt(np.array(klc_typical.x), scale, 'morl')
coef2, freq2 = pywt.cwt(np.array(MW_typical.x), scale, 'morl')
# 2D visualisation
plt.figure(1)
plt.subplot(121)
plt.imshow(coef1, cmap='coolwarm', aspect='auto')

plt.subplot(122)
plt.imshow(coef2, cmap='coolwarm', aspect='auto')

plt.show()
# 3D visualisation
fig = plt.figure(figsize=(40, 15))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')

Y = np.arange(1, 150, 1)
X = np.arange(1, 501, 1)

X, Y = np.meshgrid(X, Y)

ax1.plot_surface(X, Y, coef1, cmap=cm.coolwarm, linewidth=2, antialiased=True)

ax1.set_xlabel("Time", fontsize=15)
ax1.set_ylabel("Scale", fontsize=15)
ax1.set_zlabel("Amplitude", fontsize=15)
ax1.set_zlim3d(-1, 2)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')

Y = np.arange(1, 150, 1)
X = np.arange(1, 500, 1)

X, Y = np.meshgrid(X, Y)

ax2.plot_surface(X, Y, coef2, cmap=cm.coolwarm, linewidth=2, antialiased=True)

ax2.set_xlabel("Time", fontsize=15)
ax2.set_ylabel("Scale", fontsize=15)
ax2.set_zlabel("Amplitude", fontsize=15)
ax2.set_zlim3d(-1, 2)

plt.show()

# getting data
files_path = r'C:\Users\simpl\Documents\VM_Ubuntu\5s_spliced\BW_011_20190131'
gait_data = []
labels = []
file_names = []
tem_label = 0
for folder_name in os.listdir(files_path):
    gait_file = None
    tem_path = files_path + '\\' + folder_name
    for file_name in os.listdir(tem_path):
        tem_tem_path = tem_path + '\\' + file_name
        gait_file = pd.read_csv(tem_tem_path)
        gait_file.drop(gait_file.columns[gait_file.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
        labels.append(tem_label)
        file_names.append(file_name)
        gait_data.append(gait_file)
    tem_label += 1


''' 
files_path = r'C:\Users\simpl\PycharmProjects\BW1\spliced_2'
gait_data = []
labels = []
file_names = []

os.chdir(files_path)

for file_name in os.listdir(files_path):
    gait_file = None
    if file_name.startswith("BYB-CWA-DATAEpoch"):
        labels.append(0)
        gait_file = pd.read_csv(file_name)
        gait_file.drop(gait_file.columns[gait_file.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    elif file_name.startswith("KLC_20181030_CWA-DATAEpoch"):
        labels.append(1)
        gait_file = pd.read_csv(file_name)
        gait_file.drop(gait_file.columns[gait_file.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    if gait_file is not None:
        file_names.append(file_name)
        gait_data.append(gait_file)
'''
# Using cwt to generating features
pca = PCA(n_components=1)
n = 0
features = np.empty((0, 149))
scale = np.arange(1, 150)
for ind in range(len(gait_data)):
    n += 1
    print(str(n) + '/' + str(len(gait_data)))  # progress indicator
    coef, freq = pywt.cwt(gait_data[ind][:500], scale, 'morl')
    features = np.vstack([features, pca.fit_transform(coef).flatten()])

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20)

clf = SVC(gamma='scale')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# trying if dwt will be better

pca = PCA(n_components=4)

features = np.empty((0, 596))
n = 0
for ind in range(len(gait_data)):
    n += 1
    print(str(n) + '/' + str(len(gait_data)))
    cA, cD = pywt.dwt(gait_data[ind]['x^y^z'][:1400], 'db5')
    features = np.vstack([features, pca.fit_transform(cA).flatten()])
