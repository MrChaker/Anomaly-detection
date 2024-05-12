import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import random
from faker import Faker 
import scipy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pywt
faker = Faker()

def predict(data):
    return random.randint(0, 1)

# Fonction pour charger le modèle
def load_anomaly_detection_model(model_path):
    model = load_model(model_path)
    return model

# Fonction pour détecter les anomalies
def detect_anomalies(model, data):
    data = data.reshape((-1, 19, 1))
    # Votre logique de prédiction d'anomalies ici
    prediction = model.predict(data)
    

    #prediction = [p for p in prediction]
    return prediction

def plot_sample( sample):
    
    plt.plot(sample)
    plt.legend(shadow=True, frameon=True, facecolor="inherit", loc=1, fontsize=9)
    plt.title("Sample")
    
    plt.show()


def get_input(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    ecg_data = scaler.fit_transform(data.iloc[:,:-1])

    coeffs = pywt.wavedec(ecg_data, 'db4', level=1)
    ecg_wavelet = pywt.waverec(coeffs, 'db4') 
    # Initializing an empty list to store the features
    features = []
    # Extracting features for each sample
    for i in range(ecg_wavelet.shape[0]):
        #Finding the R-peaks
        r_peaks = scipy.signal.find_peaks(ecg_wavelet[i])[0]

        #Initialize lists to hold R-peak and T-peak amplitudes
        r_amplitudes = []
        t_amplitudes = []

        # Iterate through R-peak locations to find corresponding T-peak amplitudes
        for r_peak in r_peaks:
            # Find the index of the T-peak (minimum value) in the interval from R-peak to R-peak + 200 samples
            t_peak = np.argmin(ecg_wavelet[i][r_peak:r_peak+200]) + r_peak
            #Append the R-peak amplitude and T-peak amplitude to the lists
            r_amplitudes.append(ecg_wavelet[i][r_peak])
            t_amplitudes.append(ecg_wavelet[i][t_peak])

        # extracting singular value metrics from the r_amplitudes
        std_r_amp = np.std(r_amplitudes)
        mean_r_amp = np.mean(r_amplitudes)
        median_r_amp = np.median(r_amplitudes)
        sum_r_amp = np.sum(r_amplitudes)
        # extracting singular value metrics from the selected_datat_amplitudes
        std_t_amp = np.std(t_amplitudes)
        mean_t_amp = np.mean(t_amplitudes)
        median_t_amp = np.median(t_amplitudes)
        sum_t_amp = np.sum(t_amplitudes)

        # Find the time between consecutive R-peaks
        rr_intervals = np.diff(r_peaks)

        # Calculate the time duration of the data collection
        time_duration = (len(ecg_wavelet[i]) - 1) / 1000 # assuming data is in ms
        
        # Calculate the sampling rate
        sampling_rate = len(ecg_wavelet[i]) / time_duration

        # Calculate heart rate
        duration = len(ecg_wavelet[i]) / sampling_rate
        heart_rate = (len(r_peaks) / duration) * 60

        # QRS duration
        qrs_duration = []
        for j in range(len(r_peaks)):
            qrs_duration.append(r_peaks[j]-r_peaks[j-1])
        # extracting singular value metrics from the qrs_durations
        std_qrs = np.std(qrs_duration)
        mean_qrs = np.mean(qrs_duration)
        median_qrs = np.median(qrs_duration)
        sum_qrs = np.sum(qrs_duration)

        # Extracting the singular value metrics from the RR-interval
        std_rr = np.std(rr_intervals)
        mean_rr = np.mean(rr_intervals)
        median_rr = np.median(rr_intervals)
        sum_rr = np.sum(rr_intervals)

        # Extracting the overall standard deviation 
        std = np.std(ecg_wavelet[i])
        
        # Extracting the overall mean 
        mean = np.mean(ecg_wavelet[i])

        # Appending the features to the list
        features.append([mean, std, std_qrs, mean_qrs,median_qrs, sum_qrs, std_r_amp, mean_r_amp, median_r_amp, sum_r_amp, std_t_amp, mean_t_amp, median_t_amp, sum_t_amp, sum_rr, std_rr, mean_rr,median_rr, heart_rate])

        # Converting the list to a numpy array

    return np.array(features)

def main():
    st.title("Détection d'anomalies ECG")

    # Charger les cinq premières lignes du fichier CSV contenant les données ECG
    data = pd.read_csv("data.csv").iloc[:5]
    data.drop(columns=["a"], inplace=True)

    # Afficher les données dans un DataFrame
    st.write("Nos données: ")
    st.write(data)

    # Sélectionner une ligne spécifique
    selected_row = st.selectbox("Sélectionnez un patient :", options=data.index, format_func=lambda x: data["Name"][x] )
    plot_sample(selected_row)
    # Charger le modèle
    model = load_anomaly_detection_model("LSTM.h5")

    config = model.get_config()

    data.drop(columns=["Name"], inplace=True)

    data = get_input(data)
   

    # Préparer les données sélectionnées pour la prédiction
    selected_data = np.asarray(data[selected_row]).astype('float32')

    # Détecter les anomalies pour la ligne sélectionnée
    if st.button("Prédire les anomalies"):
        anomalies = detect_anomalies(model, selected_data)

        if anomalies == 1:
            st.image("normal.png")
        else:
            st.image("anomaly.webp")


if __name__ == "__main__":
    main()

