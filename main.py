import numpy as np
import streamlit as st
import pandas as pd 
import seaborn as sns

st.write("""
# Pendeteksi Dini Penyakit Kardiovaskular
Dengan Metode **K-Nearest Neighbour**

Silahkan Menginput Data Pasien
"""
)

col1, col2 = st.columns(2)
with col1:
    age = st.number_input('Usia Pasien', 0, 1000)
with col2:
    sex = st.number_input('Jenis Kelamin | 0 : Laki-Laki, 1 : Perempuan', 0, 1)
with col1:
    trestbps = st.number_input('Tekanan Darah', 0, 300)
with col2:
    chol = st.number_input('konsentrasi kolesterol *mg/dl*', 0, 1000)
with col1:
    cp = st.number_input('nyeri dada | 0(tidak ada), 1(Ringan), 2(Signifikan), dan 3(Hebat)', 0, 3)
with col2:
    restecg = st.number_input('hasil ECG | 0(tidak ada kelainan), 1(kelainan minor), 2(kelainan signifikan)', 0, 2)
with col1:
    oldpeak = st.number_input('oldpeak (penurunan segmen ST pada elektrokardiogram (EKG) setelah latihan fisik)', 0, 1000)
with col2:
    slope = st.number_input('slope (Tingkat penurunan ST segment dengan nilai 0(tidak ada), 1(lambat), 2(cepat))', 0, 2)
with col1:
    ca = st.number_input('ca (jumlah sumbatan arteri koroner utama dengan nilai 0(tidak ada), 1(1 arteri), 2(2 arteri), 3(3 arteri)', 0, 3)
with col2:
    thal = st.number_input('thal (aliran darah ke otot jantung dengan nilai 0 = normal, 1 = fixed defect, 2 = reversable defect)', 0, 2)
fbs = st.number_input('konsentrasi gula setelah puasa dalam 8 jam | **1(> 120 mg/dl) dan 0(< 120 mg/dL)**', 0, 1)
thalach = st.number_input('thalach (detak jantung maksimal setelah latihan fisik intens)', 0, 1000)
exang = st.number_input('exang ( ada tidaknya rasa sakit atau tekanan pada dada akibat kurangnya pasokan darah dan oksigen yang cukup ke jantung)', 0, 1000)


data = pd.read_csv('heart.csv')
X = data.drop('target', axis=1)
Y = data['target']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 5)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, Y_train)

X_baru = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
prediksi = knn.predict(X_baru)

if st.button('Prediksi Penyakit Jantung'):
    if (prediksi[0]==0):
        prediksi = 'Pasien Tidak Berpotensi Terkena Penyakit Jantung'
        st.success(prediksi)
    else:
        prediksi = 'Pasien Berpotensi Terkena Penyakit Jantung'
        st.error(prediksi)
        
