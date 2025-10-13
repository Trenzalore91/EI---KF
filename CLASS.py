from math import *
from scipy.signal import *
from scipy.fft import fft, fftfreq
from pykalman import KalmanFilter as pyKalmanFilter
from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm #pour réaliser la loi normale (seulement en test, devra être enlevé)

class Signal_Original:
    def __init__(self, Signal_Type, Amplitude, Frequence, Durée, Fs, Abscisse_Graph):
        self.Signal_Type = Signal_Type.lower()
        if self.Signal_Type not in ['sinus', 'carré']:
            raise ValueError("Type de signal non reconnu. Choisissez 'sinus' ou 'carré'.")
        self.Amplitude = Amplitude
        self.Frequence = Frequence
        self.Durée = Durée
        self.Fs = Fs
        self.Abscisse_Graph = Abscisse_Graph
    
    def GenerationSignal(self):
        if (self.Signal_Type == 'sinus'):
            signal = self.Amplitude * np.sin(2 * np.pi * self.Frequence * self.Abscisse_Graph)
        elif (self.Signal_Type == 'carré'):
            signal = self.Amplitude * square(2 * np.pi * self.Frequence * self.Abscisse_Graph)
        return signal
    
class SignalFilter:
    def __init__(self, signal, filter_type, gain, fc, ordre, fs, Noise_STD, Entrée_Pure):
        self.signal = signal
        self.filter_type = filter_type.lower()
        self.gain = gain
        self.ordre = ordre
        self.fc = fc
        self.fs = fs
        #self.Q = Q
        #self.P = P
        self.Noise_STD = Noise_STD
        self.Entrée_Pure = Entrée_Pure

        if self.filter_type not in ['passe-bas', 'kalman']:
            raise ValueError("Type de filtre non reconnu. Choisissez 'passe-bas' ou 'kalman'.")

    def Filtre_passe_bas(self):
        freq_nyquist = 0.5 * self.fs
        normal_freq_coupure = self.fc / freq_nyquist
        b, a = butter(self.ordre, normal_freq_coupure, btype='low', analog=False)
        y = self.gain * lfilter(b, a, self.signal)
        return y
    
    def Filtre_kalman(self, Sortie_Filtre, R_Kalman, P_Kalman, X_Kalman):
        tau = 1 / (2 * pi * self.fc)
        Kalman_A = exp(-(self.fs*1e-12)/tau)
        Kalman_B = self.gain*(1-exp(-(self.fs*1e-12)/tau))
        Kalman_C = 1.

        Kalman_Q0 = self.Noise_STD**2
        Kalman_R0 = R_Kalman
        Kalman_P0 = P_Kalman
        Kalman_x0 = X_Kalman

        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.F = np.array([[Kalman_A]])
        kf.H = np.array([[Kalman_C]])
        kf.B = np.array([[Kalman_B]])
        kf.Q = np.array([[Kalman_Q0]])
        kf.R = np.array([[Kalman_R0]])
        kf.P = np.array([[Kalman_P0]])
        kf.x = np.array([[Kalman_x0]])

        filtered = []
        kalman_gains = []
        for z, u_k in zip(Sortie_Filtre, self.Entrée_Pure):
            kf.predict(u=u_k)
            kf.update(z)
            kalman_gains.append(kf.K[0, 0])
            filtered.append(kf.x[0, 0])
        
        return np.array(filtered), np.array(kalman_gains)

def plot_graph(x, y, title, xlabel, ylabel, legend):
    plt.plot(x, y, label=legend)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if legend == "FFT":
        max_y = np.max(y)
        max_x = x[np.argmax(y)]
        plt.annotate(f'{max_x:.2f} Hz', xy=(max_x, max_y), xytext=(max_x, max_y + 0.1*max_y),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    horizontalalignment='center')
    plt.grid(True)
    plt.show()

def zoom_graph(x, y, xlim, ylim, title, xlabel, ylabel):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if title == "Zoom sur le spectre de fréquence":
        max_y = np.max(y)
        max_x = x[np.argmax(y)]
        plt.annotate(f'{max_x:.2f} Hz', xy=(max_x, max_y), xytext=(max_x, max_y + 0.1*max_y),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    horizontalalignment='center')
    plt.grid(True)
    plt.show()

def FFT_Signal(signal, fs, xlimit, abs_graph_FFT, ord_graph_FFT):
    n = len(signal)
    signal_fft = fft(signal)
    frequencies = fftfreq(n, 1/fs)
    positive_frequencies = frequencies[:n // 2]
    amplitudes = 2.0 / n * np.abs(signal_fft[:n // 2])
    plot_graph(positive_frequencies, amplitudes, "Spectre de fréquence du signal", abs_graph_FFT, ord_graph_FFT, "FFT")
    if xlimit is not None:
        zoom_graph(positive_frequencies, amplitudes, (0, xlimit), (0, max(amplitudes)+(0.1 * max(amplitudes))), "Zoom sur le spectre de fréquence", abs_graph_FFT, ord_graph_FFT)

def Bode_Diagram(fc, gain, titre, ord_graph_bode_gain, abs_graph_bode_phase, ord_graph_bode_phase):
    wc = 2 * np.pi * fc 
    num = [gain * wc]
    den = [1, wc]

    system = TransferFunction(num, den)

    w, mag, phase = bode(system)

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.semilogx(w/(2*np.pi), mag)
    plt.title(titre)
    plt.ylabel(ord_graph_bode_gain)
    plt.grid(which='both', linestyle='--')
    plt.axvline(x=fc, color='r', linestyle='--', label='Fréquence de coupure ' +str(fc)+' Hz')
    plt.axhline(y=-3, color='g', linestyle='--', label='-3 dB')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.semilogx(w/(2*np.pi), phase)
    plt.xlabel(abs_graph_bode_phase)
    plt.ylabel(ord_graph_bode_phase)
    plt.grid(which='both', linestyle='--')
    plt.axvline(x=fc, color='r', linestyle='--', label='Fréquence de coupure ' +str(fc)+' Hz')
    plt.axhline(y=-45, color='g', linestyle='--', label='-45 °')
    plt.legend()

    plt.tight_layout()
    plt.show()

def Moyenne_Glissante(signal, window_size):
    cumsum = np.cumsum(np.insert(signal, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

def Calc_retard(Signal_Pur, Signal_a_comparer, Freq_Signal_Pur, dt, Titre_Signal_a_comparer):
    correlation = correlate(Signal_Pur, Signal_a_comparer, mode='full')
    lags = np.arange(-len(Signal_a_comparer) + 1, len(Signal_Pur))
    idx = np.argmax(correlation)
    #delay_samples = lags[idx]
    delay_sec = lags[idx] * dt #delay_samples = lags[idx] * dt
    phase_rad = 2 * np.pi * Freq_Signal_Pur * delay_sec
    phase_deg = np.degrees(phase_rad)
    phase_deg = ((phase_deg + 180)%360)-180
    print(f"Le retard temporel du signal pur avec {Titre_Signal_a_comparer} est de {delay_sec} s.")
    print(f"Le retard degré du signal pur avec {Titre_Signal_a_comparer} est de {phase_deg} °.")

def Calc_SNR(Signal_pur_SNR, Signal_Bruité_SNR, Unité_SNR, Titre_SNR):
    Signal_RMS = np.sqrt(np.mean(Signal_pur_SNR**2))
    Bruité_RMS = np.sqrt(np.mean(Signal_Bruité_SNR**2))
    print(f"RMS Signal d'entrée : {Signal_RMS} {Unité_SNR}")
    print(f"RMS Signal bruité : {Bruité_RMS} {Unité_SNR}")
    
    SNR_Signal = 20 * log10(Signal_RMS / Bruité_RMS)
    print(f"SNR {Titre_SNR} : {SNR_Signal} dB")

