from math import *
from scipy.signal import *
from pykalman import KalmanFilter as pyKalmanFilter
from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import csv
import tkinter as tk

class Signal_Original:
    def __init__(self, Signal_Type='sinus', Amplitude, Frequence, Durée, Fs, Absicisse_Graph):
        self.Signal_Type = Signal_Type.lower()
        if self.Signal_Type not in ['sinus', 'carré', 'Sinus', 'Carré', 'SINUS', 'CARRE']:
            raise ValueError("Type de signal non reconnu. Choisissez 'sinus' ou 'carré'.")
        self.Amplitude = Amplitude
        self.Frequence = Frequence
        self.Durée = Durée
        self.Fs = Fs
        self.Absicisse_Graph = Absicisse_Graph

    
    def __GenerationSignal__(self):
        if (self.Signal_Type == 'sinus') or (self.Signal_Type == 'Sinus') or (self.Signal_Type == 'SINUS'):
            signal = self.Amplitude * np.sin(2 * np.pi * self.Frequence * self.Durée)
        elif (self.Signal_Type == 'carré') or (self.Signal_Type == 'Carré') or (self.Signal_Type == 'CARRE'):
            signal = self.Amplitude * square(2 * np.pi * self.Frequence * self.Durée)

        return t, signal
        
class SignalFilter:
    def __init__(self, filter_type, cutoff=None, order=1, dt=1.0, process_variance=1e-5, measurement_variance=1e-1):
        
        self.filter_type = filter_type.lower()
        self.order = order
        self.cutoff = cutoff
        self.dt = dt
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

        if self.filter_type in ['passe-bas', 'low-pass']:
            if cutoff is None:
                raise ValueError("Pour un filtre passe-bas, il faut définir la fréquence de coupure 'cutoff'.")
            # Conception d'un filtre Butterworth passe-bas
            self.b, self.a = butter(self.order, self.cutoff, btype='low', analog=False)

        elif self.filter_type in ['passe-haut', 'high-pass']:
            if cutoff is None:
                raise ValueError("Pour un filtre passe-haut, il faut définir la fréquence de coupure 'cutoff'.")
            # Conception d'un filtre Butterworth passe-haut
            self.b, self.a = butter(self.order, self.cutoff, btype='high', analog=False)

        elif self.filter_type in ['kalman']:
            # Pour le filtre de Kalman, aucun filtrage pré-calculé n'est nécessaire ici.
            # L'initialisation se fera dans la fonction interne au moment de filtrer.
            pass
        else:
            raise ValueError("Type de filtre non reconnu. Choisissez 'passe-bas', 'passe-haut' ou 'kalman'.")

    def filter(self, signal):
        """
        Applique le filtre au signal donné et retourne le signal filtré.
        """
        if self.filter_type in ['passe-bas', 'low-pass', 'passe-haut', 'high-pass']:
            # Utilisation de filtfilt permet d'éliminer le déphasage
            return filtfilt(self.b, self.a, signal)
        elif self.filter_type in ['kalman']:
            return self._kalman_filter(signal)

    def _kalman_filter(self, signal):
        """
        Implémentation simple d'un filtre de Kalman pour un signal 1D.
        """
        n = len(signal)
        x_est = np.zeros(n)  # estimations du signal
        P = 1.0              # covariance initiale
        Q = self.process_variance      # bruit de processus
        R = self.measurement_variance  # bruit de mesure

        # Initialisation par la première valeur mesurée
        x_est[0] = signal[0]

        for k in range(1, n):
            # Étape de prédiction
            x_pred = x_est[k - 1]   # modèle simple : x[k] = x[k-1]
            P = P + Q

            # Étape de correction (mise à jour)
            K = P / (P + R)         # gain de Kalman
            x_est[k] = x_pred + K * (signal[k] - x_pred)
            P = (1 - K) * P

        return x_est

# -------------------------------------------------------------------------
# Exemple d'utilisation
if __name__ == "__main__":
    # Génération d'un signal bruité : une sinusoïde avec bruit
    fs = 1000                        # Fréquence d'échantillonnage (Hz)
    T = 1.0                          # Durée en secondes
    t = np.linspace(0, T, int(fs*T), endpoint=False)
    freq_signal = 5                  # Fréquence de la sinusoïde (Hz)
    signal_clean = np.sin(2 * np.pi * freq_signal * t)
    bruit = 0.3 * np.random.randn(len(t))
    signal_bruite = signal_clean + bruit

    # Création d'instances de filtre :
    # 1. Filtre passe-bas Butterworth d'ordre 3, avec une fréquence de coupure à 0.1 (pour fs=1000, cela représente 50Hz si Nyquist=0.5*fs)
    filtre_pb = SignalFilter(filter_type='passe-bas', cutoff=0.1, order=3)
    signal_pb = filtre_pb.filter(signal_bruite)

    # 2. Filtre passe-haut Butterworth d'ordre 3, fréquence de coupure à 0.05 
    filtre_ph = SignalFilter(filter_type='passe-haut', cutoff=0.05, order=3)
    signal_ph = filtre_ph.filter(signal_bruite)

    # 3. Filtre de Kalman
    filtre_kalman = SignalFilter(filter_type='kalman', process_variance=1e-5, measurement_variance=0.1)
    signal_kalman = filtre_kalman.filter(signal_bruite)

    # Affichage des résultats
    plt.figure(figsize=(10, 8))
    plt.subplot(411)
    plt.plot(t, signal_bruite, label="Signal bruité", color='gray')
    plt.plot(t, signal_clean, label="Signal propre", color='black', linewidth=1.5)
    plt.title("Signal de base")
    plt.legend()
    plt.grid(True)

    plt.subplot(412)
    plt.plot(t, signal_pb, label="Passe-bas", color='blue')
    plt.title("Filtrage Passe-bas (Butterworth)")
    plt.legend()
    plt.grid(True)

    plt.subplot(413)
    plt.plot(t, signal_ph, label="Passe-haut", color='red')
    plt.title("Filtrage Passe-haut (Butterworth)")
    plt.legend()
    plt.grid(True)

    plt.subplot(414)
    plt.plot(t, signal_kalman, label="Filtre de Kalman", color='green')
    plt.title("Filtrage par Kalman")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


"""class Filtre():
    def __init__(self, id, fc, fs, ordre, signal_a_filtrer):
        self.id = id
        self.fc = fc
        self.fs = fs
        self.ordre = ordre
        self.signal_a_filtrer = signal_a_filtrer
        self.__sortiefiltre__()

    def __sortiefiltre__(self):
        freq = 0.5 * self.fs
        normal_freq_coupure = self.fc / freq
        b, a = butter(self.ordre, normal_freq_coupure, btype=self.type_filtre, analog=False)
        return lfilter(b, a, self.signal_a_filtrer)

class Filtres_standard(Filtre):
    def __init__(self, type_filtre):
        super().__init__(type_filtre)

    def __ReturnTypeFiltre__(self):
        super().Filtres_standard(type_filtre)
        freq = 0.5 * self.fs
        normal_freq_coupure = self.fc / freq
        b, a = butter(self.ordre, normal_freq_coupure, btype=self.type_filtre, analog=False)
        return lfilter(b, a, self.signal_a_filtrer)
        
class Filtre_Kalman():
    def __init__(self, signal_a_filtrer):
        self.signal_a_filtrer = signal_a_filtrer
    """