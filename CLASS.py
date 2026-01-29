from math import sin, pi, log10, exp #importation des fonctions mathématiques nécessaires
from scipy.signal import butter, lfilter, square, bode, TransferFunction, correlate #importation des fonctions de traitement du signal
from scipy.fft import fft, fftfreq  #importation des fonctions pour la FFT
from scipy.stats import norm    #importation des fonctions pour la distribution normale
from filterpy.kalman import KalmanFilter    #importation de la classe KalmanFilter de la bibliothèque filterpy
import numpy as np  #importation de la bibliothèque numpy pour les calculs numériques
from matplotlib.animation import FuncAnimation, PillowWriter #importation des fonctions pour l'animation des graphiques
import matplotlib.pyplot as plt #importation de la bibliothèque matplotlib pour le tracé des graphiques
import csv  #importation de la bibliothèque csv pour l'export des données en csv

class Signal_Original:
    def __init__(self, 
                 signal_type, 
                 amplitude, 
                 frequence, 
                 durée, 
                 fs, 
                 abscisse_graph):
        """Initialisation de la classe Signal_Original."""
        self.signal_type = signal_type.lower()  #type de signal: 'sinus' ou 'carré'
        if self.signal_type not in ['sinus', 'carré']:  #vérification du type de signal
            raise ValueError("Type de signal non reconnu. Choisissez 'sinus' ou 'carré'.")  #erreur si type non reconnu
        self.amplitude = amplitude  #amplitude du signal
        self.frequence = frequence  #fréquence du signal
        self.durée = durée  #durée du signal
        self.fs = fs #frequence d'"échantillonnage"
        self.abscisse_graph = abscisse_graph    #abscisse pour le graphe
    
    def GenerationSignal(self):
        """Génération du signal en fonction des paramètres donnés."""
        if (self.signal_type == 'sinus'):   #génération d'un signal sinusoidal si demande d'un signal sinusoïdal
            signal = self.amplitude * np.sin(2 * np.pi * self.frequence * self.abscisse_graph)  #calcul du signal sinusoidal
        elif (self.signal_type == 'carré'): #génération d'un signal carré si demande d'un signal carré
            signal = self.amplitude * square(2 * np.pi * self.frequence * self.abscisse_graph)  #calcul du signal carré
        return signal   #retourne le signal généré = signal d'entrée
    
class SignalFilter:
    """Classe pour appliquer des filtres à un signal donné."""
    def __init__(self, 
                 signal, 
                 filter_type, 
                 gain, 
                 fc,
                 ordre, 
                 fs, 
                 noise_STD, 
                 entrée_pure):
        self.signal = signal    #signal à filtrer
        self.filter_type = filter_type.lower()  #type de filtre: 'passe-bas' ou 'kalman'
        self.gain = gain    #gain du filtre passe-bas
        self.fc = fc    #fréquence de coupure passe-bas
        self.ordre = ordre    #ordre du filtre passe-bas
        self.fs = fs    #fréquence d'échantillonnage
        self.noise_STD = noise_STD  #écart-type du bruit
        self.entrée_pure = entrée_pure  #signal d'entrée pur pour le filtre de Kalman

        if self.filter_type not in ['passe-bas', 'kalman']: #vérification du type de filtre
            raise ValueError("Type de filtre non reconnu. Choisissez 'passe-bas' ou 'kalman'.") #erreur si type non reconnu

    def Sortie_Filtre_passe_bas(self):
        """Application d'un filtre passe-bas au signal."""
        freq_nyquist = 0.5 * self.fs    #calcul de la fréquence de Nyquist
        normal_freq_coupure = self.fc / freq_nyquist    #fréquence de coupure normalisée
        b, a = butter(self.ordre, normal_freq_coupure, btype='low', analog=False)   #conception du filtre passe-bas
        sortie_passe_bas = self.gain * lfilter(b, a, self.signal)  #application du filtre au signal
        return sortie_passe_bas
        
    def Filtre_kalman(self, 
                      sortie_filtre, 
                      a_kalman,
                      b_kalman,
                      c_kalman,
                      p_kalman, 
                      q_kalman,
                      r_kalman, 
                      x_kalman):
        """Application d'un filtre de Kalman au signal."""

        ### Filtre de Kalman monovariable ###
        # tau = 1 / (2 * pi * self.fc) #constante de temps du filtre passe-bas
        # kalman_a = exp(-(1/self.fs)/tau) #calcul du scalaire transition d'état
        # kalman_b = self.gain*(1-exp(-(1/self.fs)/tau)) #calcul du scalaire de contrôle
        # kalman_c = 1. #calcul du scalaire d'observation

        # kalman_q0 = self.noise_STD**2   #variance du bruit de processus
        # kalman_r0 = r_kalman #variance du bruit de mesure
        # kalman_p0 = p_kalman #estimation initiale de l'erreur de covariance
        # kalman_x0 = x_kalman #estimation initiale de l'état

        # kf = KalmanFilter(dim_x=1, dim_z=1)  #initialisation du filtre de Kalman monovariable
        # kf.F = np.array([[kalman_a]])   #matrice de transition d'état
        # kf.H = np.array([[kalman_c]])   #matrice d'observation
        # kf.B = np.array([[kalman_b]])   #matrice de contrôle
        # kf.Q = np.array([[kalman_q0]])  #matrice de covariance du bruit de processus
        # kf.R = np.array([[kalman_r0]])  #matrice de covariance du bruit de mesure
        # kf.P = np.array([[kalman_p0]])  #matrice de covariance de l'erreur
        # kf.x = np.array([[kalman_x0]])  #état initial

        ### Filtre de Kalman multivariable ###
        #tau = 1 / (2 * pi * self.fc) #constante de temps du filtre passe-bas
        kalman_a = a_kalman #exp(-(1/self.fs)/tau) #calcul du scalaire transition d'état
        kalman_b = b_kalman #self.gain*(1-exp(-(1/self.fs)/tau))  #calcul du scalaire de contrôle
        kalman_c = c_kalman #1.   #calcul du scalaire d'observation

        kalman_q0 = q_kalman #self.noise_STD**2   #variance du bruit de processus
        kalman_r0 = r_kalman    #variance du bruit de mesure
        kalman_p0 = p_kalman    #estimation initiale de l'erreur de covariance
        kalman_x0 = x_kalman    #estimation initiale de l'état
        kf = KalmanFilter(dim_x=1, dim_z=2)  #initialisation du filtre de Kalman multivariable
        kf.F = kalman_a   #matrice de transition d'état
        kf.H = kalman_c   #matrice d'observation
        kf.B = kalman_b   #matrice de contrôle
        kf.Q = kalman_q0  #matrice de covariance du bruit de processus
        kf.R = kalman_r0  #matrice de covariance du bruit de mesure
        kf.P = kalman_p0  #matrice de covariance de l'erreur
        kf.x = kalman_x0  #état initial

        filtered = []   #liste pour stocker les valeurs filtrées
        kalman_gains = []   #liste pour stocker les gains de Kalman
        x_k = []  #liste pour stocker les états estimés
        p_k = []  #liste pour stocker les covariances d'erreur
        innovations = []  #liste pour stocker les innovations
        s_k = []  #liste pour stocker les covariances des innovations

        for sortie_kalman, u_k in zip(sortie_filtre, self.entrée_pure): #itération sur les mesures et les entrées de contrôle
            # z_k = sortie_kalman #monovariable
            z_k = sortie_kalman.reshape(2, 1)  #remodelage de la mesure pour correspondre à la dimension attendue - multivariable
            kf.predict(u=u_k)   #étape de prédiction
            kf.update(z_k)    #étape de mise à jour
            kalman_gains.append(kf.K[0, 0]) #stockage du gain de Kalman
            filtered.append(kf.x[0, 0]) #stockage de la valeur filtrée 
            x_k.append(kf.x.copy())  #stockage de l'état estimé
            p_k.append(kf.P.copy())  #stockage de la covariance d'erreur
            innovations.append(kf.y.copy()) #stockage de l'innovation
            s_k.append(kf.S.copy())  #stockage de la covariance de l'innovation
        
        x_k = np.array(x_k).squeeze()  #conversion en tableau numpy et suppression des dimensions inutiles
            
        return np.array(filtered), np.array(kalman_gains), np.array(x_k), np.array(p_k), np.array(innovations), np.array(s_k) #retourne le signal filtré et les gains de Kalman
        
class Graphiques:
    """Classe pour tracer des graphiques avec différentes options."""
    def __init__(self):
        pass

    def plot_graph(self, 
                   x, 
                   y, 
                   title, 
                   xlabel, 
                   ylabel, 
                   legend, 
                   xlim=None, 
                   ylim=None, 
                   y2=None, 
                   legend2=None, 
                   y2color=None,
                   y3=None, 
                   legend3=None, 
                   y3color=None):
        """Fonction pour tracer des graphiques avec des options supplémentaires."""
        plt.plot(x, y, label=legend)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        if legend == "FFT": #annotation du maximum pour le graphique FFT
            max_y = np.max(y)
            max_x = x[np.argmax(y)]
            plt.annotate(f'{max_x:.2f} Hz',
                         xy=(max_x, max_y),
                         xytext=(max_x, max_y + 0.1*max_y),
                        arrowprops=dict(facecolor='black', shrink=0.05),
                        horizontalalignment='center')
        
        if xlim is not None:
            plt.xlim(xlim)
        
        if ylim is not None:
            plt.ylim(ylim)
            
        if y2 is not None:  #ajout d'une deuxième courbe si nécessaire
            plt.plot(x, y2, label=legend2, color=y2color)
            plt.legend()

        if y3 is not None:  #ajout d'une troisième courbe si nécessaire
            plt.plot(x, y3, label=legend3, color=y3color)
            plt.legend()
        plt.grid(True)
        plt.show()

    def plot_hist_gaussienne(self, 
                             gaussian_ns, 
                             std_dev, 
                             abs_graph, 
                             abs_graph_temp, 
                             ord_graph_temp, 
                             abs_graph_Gauss, 
                             ord_graph_Gauss):
        """Fonction pour tracer l'histogramme d'un signal bruité et la gaussienne théorique."""
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(abs_graph, gaussian_ns)
        plt.title('Bruit Blanc Gaussien')
        plt.xlabel(abs_graph_temp)
        plt.ylabel(ord_graph_temp)

        plt.subplot(1, 2, 2)
        count, bins, ignored = plt.hist(gaussian_ns,
                                        bins=1000,
                                        density=True,
                                        alpha=0.5,
                                        color='g',
                                        label='Histogramme')    #tracé de l'histogramme

        mu, std = norm.fit(gaussian_ns)  #calcul de la moyenne et de l'écart-type

        xmin, xmax = plt.xlim() #définition des limites de l'axe x
        x = np.linspace(xmin, xmax, 100)    #création d'un vecteur x pour la gaussienne théorique
        p = norm.pdf(x, mu, std)    #calcul de la gaussienne théorique
        plt.plot(x, p, 'k', linewidth=2, label='Gaussienne théorique')

        x_position1 = -3*std_dev
        x_position2 = 3*std_dev

        plt.axvline(x=x_position1, 
                    color='orange',
                    linestyle='dashed',
                    linewidth=1,
                    label=f'Ligne 1 à -3 sigma: {x_position1:.2f}')
        plt.axvline(x=x_position2,
                    color='orange',
                    linestyle='dashed',
                    linewidth=1,
                    label=f'Ligne 2 à 3 sigma: {x_position2:.2f}')

        plt.title('Distribution de probabilité')
        plt.xlabel(abs_graph_Gauss)
        plt.ylabel(ord_graph_Gauss)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def FFT_Signal(self,
                   signal,
                   fs, 
                   xlimit, 
                   abs_graph_FFT, 
                   ord_graph_FFT):
        """Fonction pour calculer et tracer la FFT d'un signal."""
        n = len(signal)  #nombre d'échantillons
        signal_fft = fft(signal)    #calcul de la FFT
        frequencies = fftfreq(n, 1/fs)  #calcul des fréquences associées
        positive_frequencies = frequencies[:n // 2] #fréquences positives
        amplitudes = 2.0 / n * np.abs(signal_fft[:n // 2])  #amplitudes correspondantes
        self.plot_graph(positive_frequencies,
                        amplitudes,
                        "Spectre de fréquence du signal", 
                        abs_graph_FFT, ord_graph_FFT,
                        "FFT",
                        xlimit) #tracé du spectre de fréquence

    def Bode_Diagram(self,
                     Liste_Filtre, 
                     ord_graph_bode_gain, 
                     abs_graph_bode_phase, 
                     ord_graph_bode_phase):
        """Fonction pour tracer le diagramme de Bode de plusieurs filtres."""
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
            plt.axvline(x=fc,
                        color='r',
                        linestyle='--',
                        label='Fréquence de coupure ' +str(fc)+' Hz')
            plt.axhline(y=-3, 
                        color='g',
                        linestyle='--',
                        label='-3 dB')
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.semilogx(w/(2*np.pi), phase)
            plt.xlabel(abs_graph_bode_phase)
            plt.ylabel(ord_graph_bode_phase)
            plt.grid(which='both', linestyle='--')
            plt.axvline(x=fc,
                        color='r',
                        linestyle='--',
                        label='Fréquence de coupure ' +str(fc)+' Hz')
            plt.axhline(y=-45,
                        color='g', 
                        linestyle='--',
                        label='-45 °')
            plt.legend()

            plt.tight_layout()
            plt.show()

class Monte_Carlo:
    """Classe pour effectuer des simulations de Monte Carlo."""
    def __init__(self,
                 signal_in,
                 abs_graph,
                 w_ns_mean,
                 w_ns_std,
                 capteur_out,
                 kalman_out,
                 n_mc):
        self.signal_in = signal_in  #variable interne pour le signal d'entrée
        self.abs_graph = abs_graph  #abscisse pour le graphe
        self.w_ns_mean = w_ns_mean  #moyenne du bruit
        self.w_ns_std = w_ns_std    #écart-type du bruit
        self.capteur_out = capteur_out  #sortie du filtre capteur
        self.kalman_out = kalman_out    #sortie du filtre de Kalman
        self.n_mc = n_mc    #nombre d'itérations Monte Carlo
    
    def Monte_Carlo_Simu(self):
        """Fonction pour exécuter une itération pour la méthode Monte Carlo."""
        #Génération du signal pur
        x_true = self.signal_in
        
        #Ajout du bruit
        x_noisy = self.signal_in + np.random.normal(self.w_ns_mean, self.w_ns_std, len(x_true))
        
        #Filtrage capteur
        x_sensor_filt = self.capteur_out
        
        #Filtre de Kalman
        x_kf = self.kalman_out
        
        return x_true, x_noisy, x_sensor_filt, x_kf

    def Monte_Carlo_Method(self):
        """Fonction pour exécuter la méthode de Monte Carlo sur plusieurs simulations."""
        x_true, x_noisy, x_sensor_filt, x_kf = self.Monte_Carlo_Simu()  #exécution d'une simulation pour obtenir la taille du signal
        N = len(x_true) #longueur du signal

        errors_noisy = np.zeros((self.n_mc, N)) #initialisation des matrices d'erreurs
        errors_sensor_filt = np.zeros((self.n_mc, N))   #initialisation des matrices d'erreurs
        errors_kf = np.zeros((self.n_mc, N))    #initialisation des matrices d'erreurs

        rmse_noisy_per_run = np.zeros(self.n_mc)    #initialisation des RMSE par réalisation
        rmse_sensor_filt_per_run = np.zeros(self.n_mc)  #initialisation des RMSE par réalisation
        rmse_kf_per_run = np.zeros(self.n_mc)   #initialisation des RMSE par réalisation

        # Boucle Monte Carlo
        for k in range(self.n_mc):  #itération sur le nombre de simulations
            x_true_k, x_noisy_k, x_sensor_filt_k, x_kf_k = self.Monte_Carlo_Simu()  #exécution d'une simulation
            
            #Vérification que la longueur est cohérente
            if len(x_true_k) != N:
                raise ValueError("Longueur du signal incohérente entre itérations")
            
            # Erreurs instantanées par rapport au signal pur
            errors_noisy[k, :] = x_noisy_k - x_true_k
            errors_sensor_filt[k, :] = x_sensor_filt_k - x_true_k
            errors_kf[k, :] = x_kf_k - x_true_k

            # RMSE global pour cette réalisation
            rmse_noisy_per_run[k] = np.sqrt(np.mean(errors_noisy**2))
            rmse_sensor_filt_per_run[k] = np.sqrt(np.mean(errors_sensor_filt**2))
            rmse_kf_per_run[k] = np.sqrt(np.mean(errors_kf**2))

        # Statistiques Monte Carlo
        # Moyenne et variance de l'erreur à chaque instant
        mean_err_noisy = np.mean(errors_noisy, axis=0)
        mean_err_sensor_filt = np.mean(errors_sensor_filt, axis=0)
        mean_err_kf = np.mean(errors_kf, axis=0)

        var_err_noisy = np.var(errors_noisy, axis=0)
        var_err_sensor_filt = np.var(errors_sensor_filt, axis=0)
        var_err_kf = np.var(errors_kf, axis=0)

        # RMSE global pour résumé global
        rmse_noisy = np.sqrt(np.mean(errors_noisy**2))
        rmse_sensor_filt = np.sqrt(np.mean(errors_sensor_filt**2))
        rmse_kf = np.sqrt(np.mean(errors_kf**2))

        # Écart-type des RMSE entre réalisations
        rmse_noisy_std = np.std(rmse_noisy_per_run, ddof=1)
        rmse_sensor_filt_std = np.std(rmse_sensor_filt_per_run, ddof=1)
        rmse_kf_std = np.std(rmse_kf_per_run, ddof=1)

        # MCSE = écart-type / sqrt(N_MC)
        mcse_noisy = rmse_noisy_std / np.sqrt(self.n_mc)
        mcse_sensor_filt = rmse_sensor_filt_std / np.sqrt(self.n_mc)
        mcse_kf = rmse_kf_std / np.sqrt(self.n_mc)

        # Affichage des résultats
        print("\nRMSE moyen par réalisation (Monte Carlo) + MCSE :")
        print(f"  - Signal bruité : {rmse_noisy:.4f} +/- {mcse_noisy:.4f} (RMSE +/- MCSE)")
        print(f"  - Filtre capteur : {rmse_sensor_filt:.4f} +/- {mcse_sensor_filt:.4f} (RMSE +/- MCSE)")
        print(f"  - Filtre de Kalman : {rmse_kf:.4f} +/- {mcse_kf:.4f} (RMSE +/- MCSE)")

        # Visualisation des résultats
        t = self.abs_graph 

        # Intervalle de confiance 95% ≈ moyenne ± 2*écart-type
        std_err_noisy = np.sqrt(var_err_noisy)
        std_err_sensor_filt = np.sqrt(var_err_sensor_filt)
        std_err_kf = np.sqrt(var_err_kf)

        plt.figure(figsize=(10, 6))
        plt.plot(t, x_true, label="Signal vrai", linewidth=2)

        # Signal bruité
        plt.plot(t, mean_err_noisy + x_true, label="Bruit moyen", linestyle="--")
        plt.fill_between(t,
                        x_true + mean_err_noisy - 2*std_err_noisy,
                        x_true + mean_err_noisy + 2*std_err_noisy,
                        alpha=0.2, label="IC 95% bruité")

        # Filtre capteur
        plt.plot(t, mean_err_sensor_filt + x_true, label="Capteur filtré moyen")
        plt.fill_between(t,
                        x_true + mean_err_sensor_filt - 2*std_err_sensor_filt,
                        x_true + mean_err_sensor_filt + 2*std_err_sensor_filt,
                        alpha=0.2, label="IC 95% capteur filt.")

        # Filtre de Kalman
        plt.plot(t, mean_err_kf + x_true, label="Kalman moyen")
        plt.fill_between(t,
                        x_true + mean_err_kf - 2*std_err_kf,
                        x_true + mean_err_kf + 2*std_err_kf,
                        alpha=0.2, label="IC 95% Kalman")

        plt.xlabel("Temps")
        plt.ylabel("Amplitude")
        plt.title(f"Monte Carlo ({self.n_mc} itérations)")
        plt.legend()
        plt.xlim(1., 1.25)
        plt.grid(True)
        plt.show()

def Moyenne_Glissante(signal, 
                      window_size):
    """Fonction pour appliquer une moyenne glissante à un signal."""
    cumsum = np.cumsum(np.insert(signal, 0, 0)) #calcul de la somme cumulée
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size #retourne le signal moyenné

def Calc_retard(signal_pur, 
                signal_a_comparer, 
                freq_signal_pur, 
                dt, 
                titre_signal_a_comparer):
    """Fonction pour appliquer un calcul de retard entre deux signaux"""
    correlation = correlate(signal_pur, signal_a_comparer, mode='full') #synchronise les signaux
    lags = np.arange(-len(signal_a_comparer) + 1, len(signal_pur))
    idx = np.argmax(correlation)
    #delay_samples = lags[idx]
    delay_sec = lags[idx] * dt #delay_samples = lags[idx] * dt
    phase_rad = 2 * np.pi * freq_signal_pur * delay_sec
    phase_deg = np.degrees(phase_rad)
    phase_deg = ((phase_deg + 180)%360)-180
    print(f"Le retard temporel du signal pur avec {titre_signal_a_comparer} est de {delay_sec} s.")
    print(f"Le retard degré du signal pur avec {titre_signal_a_comparer} est de {phase_deg} °.")

def Calc_SNR(signal_pur_SNR, 
             signal_bruité_SNR, 
             unité_SNR, 
             titre_SNR):
    """Fonction pour appliquer un calcul de SNR sur les deux signaux"""
    signal_RMS = np.sqrt(np.mean(signal_pur_SNR**2))
    bruité_RMS = np.sqrt(np.mean(signal_bruité_SNR**2))
    print(f"RMS Signal d'entrée : {signal_RMS} {unité_SNR}")
    print(f"RMS Signal bruité : {bruité_RMS} {titre_SNR}")
    
    SNR_Signal = 20 * log10(signal_RMS / bruité_RMS)
    print(f"SNR {titre_SNR} : {SNR_Signal} dB")

def CSV_Export(file_name, 
               abs_graph, 
               signal_in, 
               signal_ns, 
               signal_filt, 
               signal_KF):
    """Fonction pour exporter les donner dans un fichier .csv"""
    with open(file_name, mode='w', newline='') as file: #Ouvre le fichier, si n'existe pas, le créé
        writer = csv.writer(file) 
        writer.writerow(['Time (s)', 'Signal d\entrée', 'Signal Bruité', 'Signal filtré par le capteur', 'Signal filtré par Kalman'])   #écriture de l'en-tête
        for t, s_in, s_bruite, s_filtre_capteur, s_filtre_kalman in zip(abs_graph, signal_in, signal_ns, signal_filt, signal_KF):
            writer.writerow([t, s_in, s_bruite, s_filtre_capteur, s_filtre_kalman]) #écriture des données ligne par ligne
    print("Exportation des données terminée dans '", file_name, "'.")

def Calc_Stable_Gain_KF(KF_gains, 
                        dt, 
                        filtre_fc):
    """Fonction pour calculer le gain stable du filtre de Kalman et le nombre de constantes de temps nécessaires."""
    if len(KF_gains) == 0:
        print("La liste des gains de Kalman est vide.")
        return
    for i in range(len(KF_gains)): #attention bug si KF_Gains = 0
        if KF_gains[i] == KF_gains[i-1] and i != 0: #vérifie la stabilité du gain de Kalman
            print("Le gain de Kalman devient stable après", i, "échantillons, soit", i*dt, "secondes.")
            print("Valeur stable du gain de Kalman :", KF_gains[i], "V")
            break

    tau = 1/(2*pi*filtre_fc)    #calcul de la constante de temps du filtre passe-bas
    print("Constante de temps du filtre passe bas:", tau, "s")

    num_tau = (i*dt)/tau    #calcul du nombre de constantes de temps nécessaires pour atteindre la stabilité
    print("Le gain de Kalman devient stable après", num_tau, "constantes de temps.")