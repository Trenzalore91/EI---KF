from math import *
from scipy.signal import *
from scipy.fft import fft, fftfreq
from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm #pour réaliser la loi normale (seulement en test, devra être enlevé)
import csv

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
    
    def Somme_Filtre(self, Liste_Filtre, Signal_Bruité, Freq_Sample, W_Noise_STD, Signal_Entree):
        Somme_Filtre=0
        for i in range(len(Liste_Filtre)):
            Filtre_fc = Liste_Filtre[i][1]
            Filtre_Ordre = Liste_Filtre[i][2]
            Filtre_type = Liste_Filtre[i][3]
            Filtre_gain = Liste_Filtre[i][4]
            NewFiltre = SignalFilter(Signal_Bruité, Filtre_type, Filtre_gain, Filtre_fc, Filtre_Ordre, Freq_Sample, W_Noise_STD, Signal_Entree)
            NewFiltre.Filtre_passe_bas()
            Somme_Filtre += NewFiltre.Filtre_passe_bas()
        return Somme_Filtre
    
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

def plot_graph(x, y, title, xlabel, ylabel, legend, y2=None, legend2=None, y2color=None, y3=None, legend3=None, y3color=None):
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
    if y2 is not None:
        plt.plot(x, y2, label=legend2, color=y2color)
        plt.legend()
    if y3 is not None:
        plt.plot(x, y3, label=legend3, color=y3color)
        plt.legend()
    plt.grid(True)
    plt.show()

def plot_hist_gaussienne(gaussian_ns, std_dev, abs_graph, abs_graph_temp, ord_graph_temp, abs_graph_Gauss, ord_graph_Gauss):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(abs_graph, gaussian_ns)
    plt.title('Bruit Blanc Gaussien')
    plt.xlabel(abs_graph_temp)
    plt.ylabel(ord_graph_temp)

    plt.subplot(1, 2, 2)
    count, bins, ignored = plt.hist(gaussian_ns, bins=1000, density=True, alpha=0.5, color='g', label='Histogramme')

    mu, std = norm.fit(gaussian_ns)

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label='Gaussienne théorique')

    x_position1 = -3*std_dev
    x_position2 = 3*std_dev

    plt.axvline(x=x_position1, color='orange', linestyle='dashed', linewidth=1, label=f'Ligne 1 à -3 sigma: {x_position1:.2f}')
    plt.axvline(x=x_position2, color='orange', linestyle='dashed', linewidth=1, label=f'Ligne 2 à 3 sigma: {x_position2:.2f}')


    plt.title('Distribution de probabilité')
    plt.xlabel(abs_graph_Gauss)
    plt.ylabel(ord_graph_Gauss)
    plt.legend()

    plt.tight_layout()
    plt.show()

def zoom_graph(x, y, xlim, ylim, title, xlabel, ylabel, y2=None, legend2=None, y2color=None, y3=None, legend3=None, y3color=None):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if y2 is not None:
        plt.plot(x, y2, label=legend2, color=y2color)
        plt.legend()
    if y3 is not None:
        plt.plot(x, y3, label=legend3, color=y3color)
        plt.legend()
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

def Bode_Diagram(Liste_Filtre, ord_graph_bode_gain, abs_graph_bode_phase, ord_graph_bode_phase):
    for i in range(len(Liste_Filtre)):
        id = Liste_Filtre[i][0]
        fc = Liste_Filtre[i][1]
        gain = Liste_Filtre[i][4]

        wc = 2 * np.pi * fc 
        num = [gain * wc]
        den = [1, wc]

        system = TransferFunction(num, den)

        w, mag, phase = bode(system)

        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.semilogx(w/(2*np.pi), mag)
        plt.title('Diagramme de Bode du capteur ' + str(id))
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

def csv_export(File_Name, Abs_Graph, Signal_In, Signal_Ns, Signal_Filt, Signal_KF):
    with open(File_Name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time (s)', 'Signal d\entrée', 'Signal Bruité', 'Signal filtré par le capteur', 'Signal filtré par Kalman'])
        for t, s_in, s_bruite, s_filtre_capteur, s_filtre_kalman in zip(Abs_Graph, Signal_In, Signal_Ns, Signal_Filt, Signal_KF):
            writer.writerow([t, s_in, s_bruite, s_filtre_capteur, s_filtre_kalman])

    print("Exportation des données terminée dans '", File_Name, "'.")

def Calc_Stable_Gain_KF(KF_Gains, dt, Filtre_fc):
    for i in range(len(KF_Gains)):
        if KF_Gains[i] == KF_Gains[i-1] and i != 0:
            print("Le gain de Kalman devient stable après", i, "échantillons, soit", i*dt, "secondes.")
            print("Valeur stable du gain de Kalman :", KF_Gains[i], "V")
            break

    tau = 1/(2*pi*Filtre_fc)
    print("Constante de temps du filtre passe bas:", tau, "s")

    num_tau = (i*dt)/tau
    print("Le gain de Kalman devient stable après", num_tau, "constantes de temps.")