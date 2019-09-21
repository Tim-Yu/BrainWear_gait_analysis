import pandas as pd
import os
import numpy as np
import pywt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks

# getting data
current_dir = os.path.dirname(__file__)
os.chdir(current_dir)
files_path = './5s_spliced/'
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
# fft filter
'''
files_path = r'C:\Users\simpl\Data_folder\ratio_among_all'
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
'''

# the filter for getting the stationary walking data
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
