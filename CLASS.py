#importation des fonctions mathématiques nécessaires
from math import sin, pi, log10, exp 
#importation des fonctions de traitement du signal
from scipy.signal import butter, lfilter, square, bode
from scipy.signal import TransferFunction, correlate
#importation des fonctions pour la FFT
from scipy.fft import fft, fftfreq
#importation des fonctions pour la distribution normale
from scipy.stats import norm
#importation de la classe KalmanFilter de la bibliothèque filterpy
from filterpy.kalman import KalmanFilter
#importation de la bibliothèque numpy pour les calculs numériques
import numpy as np
#importation des fonctions pour l'animation des graphiques
from matplotlib.animation import FuncAnimation, PillowWriter
#importation de la bibliothèque matplotlib pour le tracé des graphiques
import matplotlib.pyplot as plt
#importation de la bibliothèque csv pour l'export des données en csv
import csv

class Signal_Original:
    def __init__(self, 
                 signal_type, 
                 amplitude, 
                 frequence, 
                 durée, 
                 fs, 
                 abscisse_graph):
        """Initialisation de la classe Signal_Original."""
        #type de signal: 'sinus' ou 'carré'
        self.signal_type = signal_type.lower()
        #vérification du type de signal
        if self.signal_type not in ['sinus', 'carré']:
            #erreur si type non reconnu
            raise ValueError(
                "Type de signal non reconnu. Choisissez 'sinus' ou 'carré'."
                )
        self.amplitude = amplitude  #amplitude du signal
        self.frequence = frequence  #fréquence du signal
        self.durée = durée  #durée du signal
        self.fs = fs #frequence d'"échantillonnage"
        self.abscisse_graph = abscisse_graph    #abscisse pour le graphe
    
    def GenerationSignal(self):
        """Génération du signal en fonction des paramètres donnés."""
        #génération d'un signal sinusoidal si demande d'un signal sinusoïdal
        if (self.signal_type == 'sinus'):
            #calcul du signal sinusoidal
            signal = self.amplitude * np.sin(
                2 * np.pi * self.frequence * self.abscisse_graph
                )
        #génération d'un signal carré si demande d'un signal carré
        elif (self.signal_type == 'carré'):
            #calcul du signal carré
            signal = self.amplitude * square(
                2 * np.pi * self.frequence * self.abscisse_graph
                )
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
        #signal à filtrer
        self.signal = signal
        #type de filtre: 'passe-bas' ou 'kalman'
        self.filter_type = filter_type.lower()
        #gain du filtre passe-bas
        self.gain = gain
        #fréquence de coupure passe-bas
        self.fc = fc
        #ordre du filtre passe-bas
        self.ordre = ordre
        #fréquence d'échantillonnage
        self.fs = fs
        #écart-type du bruit
        self.noise_STD = noise_STD
        #signal d'entrée pur pour le filtre de Kalman
        self.entrée_pure = entrée_pure

        #vérification du type de filtre
        if self.filter_type not in ['passe-bas', 'kalman']:
            #erreur si type non reconnu
            raise ValueError(
                "Type de filtre non reconnu." &
                "Choisissez 'passe-bas' ou 'kalman'."
                )

    def Sortie_Filtre_passe_bas(self):
        """Application d'un filtre passe-bas au signal."""
        #calcul de la fréquence de Nyquist
        freq_nyquist = 0.5 * self.fs
        #fréquence de coupure normalisée
        normal_freq_coupure = self.fc / freq_nyquist
        #conception du filtre passe-bas
        b, a = butter(
            self.ordre,
            normal_freq_coupure,
            btype='low',
            analog=False
            )
        #application du filtre au signal
        sortie_passe_bas = self.gain * lfilter(b, a, self.signal)
        #retourne le signal filtré par le filtre passe-bas
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

        ### Filtre de Kalman monovariable - début ###
        #constante de temps du filtre passe-bas
        tau = 1 / (2 * np.pi * self.fc)
        #calcul du scalaire transition d'état
        kalman_a = exp(-(1/self.fs)/tau)
        #calcul du scalaire de contrôle
        kalman_b = self.gain*(1-exp(-(1/self.fs)/tau))
        #calcul du scalaire d'observation
        kalman_c = 1.

        kalman_q0 = self.noise_STD**2 #variance du bruit de processus
        kalman_r0 = r_kalman #variance du bruit de mesure
        kalman_p0 = p_kalman #estimation initiale de l'erreur de covariance
        kalman_x0 = x_kalman #estimation initiale de l'état

        #initialisation du filtre de Kalman monovariable
        kf = KalmanFilter(dim_x=1, dim_z=1)
        #matrice de transition d'état
        kf.F = np.array([[kalman_a]])
        #matrice d'observation
        kf.H = np.array([[kalman_c]])
        #matrice de contrôle
        kf.B = np.array([[kalman_b]])
        #matrice de covariance du bruit de processus
        kf.Q = np.array([[kalman_q0]])
        #matrice de covariance du bruit de mesure
        kf.R = np.array([[kalman_r0]])
        #matrice de covariance de l'erreur
        kf.P = np.array([[kalman_p0]])
        #état initial
        kf.x = np.array([[kalman_x0]])
        ### Filtre de Kalman monovariable - fin ###

        ### Filtre de Kalman multivariable - début###
        # #constante de temps du filtre passe-bas
        # tau = 1 / (2 * np.pi * self.fc)
        # kalman_a = a_kalman #calcul du scalaire transition d'état
        # kalman_b = b_kalman #calcul du scalaire de contrôle
        # kalman_c = c_kalman #calcul du scalaire d'observation

        # kalman_q0 = q_kalman #variance du bruit de processus
        # kalman_r0 = r_kalman #variance du bruit de mesure
        # kalman_p0 = p_kalman #estimation initiale de l'erreur de covariance
        # kalman_x0 = x_kalman #estimation initiale de l'état

        # #initialisation du filtre de Kalman multivariable
        # kf = KalmanFilter(dim_x=1, dim_z=2)
        # kf.F = kalman_a #matrice de transition d'état
        # kf.H = kalman_c #matrice d'observation
        # kf.B = kalman_b #matrice de contrôle
        # kf.Q = kalman_q0 #matrice de covariance du bruit de processus
        # kf.R = kalman_r0 #matrice de covariance du bruit de mesure
        # kf.P = kalman_p0 #matrice de covariance de l'erreur
        # kf.x = kalman_x0 #état initial
        ### Filtre de Kalman multivariable - fin###

        filtered = [] #liste pour stocker les valeurs filtrées
        kalman_gains = [] #liste pour stocker les gains de Kalman
        x_k = [] #liste pour stocker les états estimés
        p_k = [] #liste pour stocker les covariances d'erreur
        innovations = [] #liste pour stocker les innovations
        s_k = [] #liste pour stocker les covariances des innovations

        #itération sur les mesures et les entrées de contrôle
        for sortie_kalman, u_k in zip(sortie_filtre, self.entrée_pure):
            #monovariable
            z_k = sortie_kalman
            # #reshape mesure pour la dimension attendue - multivariable
            # z_k = sortie_kalman.reshape(2, 1)
            kf.predict(u = u_k) #étape de prédiction
            kf.update(z_k) #étape de mise à jour
            kalman_gains.append(kf.K[0, 0]) #stockage du gain de Kalman
            filtered.append(kf.x[0, 0]) #stockage de la valeur filtrée 
            x_k.append(kf.x.copy()) #stockage de l'état estimé
            p_k.append(kf.P.copy()) #stockage de la covariance d'erreur
            innovations.append(kf.y.copy()) #stockage de l'innovation
            s_k.append(kf.S.copy()) #stockage de la covariance innovation
        
        #conversion en tableau numpy et suppression des dimensions inutiles
        x_k = np.array(x_k).squeeze()
        
        #retourne le signal filtré et les gains de Kalman
        return (
            np.array(filtered),
            np.array(kalman_gains), 
            np.array(x_k),
            np.array(p_k), 
            np.array(innovations),
            np.array(s_k)
        )
        
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
                   xlim = None, 
                   ylim = None, 
                   y2 = None, 
                   legend2 = None, 
                   y2color = None,
                   y3 = None, 
                   legend3 = None, 
                   y3color = None):
        """Fonction pour tracer des graphiques
        avec des options supplémentaires."""
        #tracé de la courbe principale
        plt.plot(x, y, label = legend)
        #configuration du titre du graphique
        plt.title(title)
        #configuration des axes et de la légende
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()

        if legend == "FFT": #annotation du maximum pour le graphique FFT
            #calcul du maximum de la FFT
            max_y = np.max(y)
            max_x = x[np.argmax(y)]
            #annotation du maximum sur le graphique
            plt.annotate(f'{max_x:.2f} Hz',
                         xy = (max_x, max_y),
                         xytext = (max_x, max_y + 0.1 * max_y),
                        arrowprops = dict(facecolor = 'black', shrink = 0.05),
                        horizontalalignment = 'center')
        
        #configuration des limites des axes si présentes
        if xlim is not None:
            plt.xlim(xlim)
        
        if ylim is not None:
            plt.ylim(ylim)
            
        if y2 is not None: #ajout d'une deuxième courbe si présente
            plt.plot(x, y2, label = legend2, color = y2color)
            plt.legend()

        if y3 is not None: #ajout d'une troisième courbe si présente
            plt.plot(x, y3, label = legend3, color = y3color)
            plt.legend()
        
        #configuration de la grille et affichage du graphique
        plt.grid(True)
        #affichage du graphique
        plt.show()

    def plot_hist_gaussienne(self, 
                             gaussian_ns, 
                             std_dev, 
                             abs_graph, 
                             abs_graph_temp, 
                             ord_graph_temp, 
                             abs_graph_Gauss, 
                             ord_graph_Gauss):
        """Fonction pour tracer l'histogramme d'un signal bruité
        et la gaussienne théorique."""
        #configuration de la figure pour afficher
        #les deux graphiques côte à côte
        plt.figure(figsize = (12, 6))

        #subdivision de la figure en deux sous-graphes
        plt.subplot(1, 2, 1)
        #tracé du signal bruité en fonction de l'abscisse donnée
        plt.plot(abs_graph, gaussian_ns)
        #configuration du titre du graphique
        plt.title('Bruit Blanc Gaussien')
        #configuration des axes
        plt.xlabel(abs_graph_temp)
        plt.ylabel(ord_graph_temp)

        #passage au deuxième sous-graphe pour tracer 
        #l'histogramme et la gaussienne théorique
        plt.subplot(1, 2, 2)
        #tracé de l'histogramme
        count, bins, ignored = plt.hist(gaussian_ns,
                                        bins = 1000,
                                        density = True,
                                        alpha = 0.5,
                                        color = 'g',
                                        label = 'Histogramme')

        #calcul de la moyenne et de l'écart-type
        mu, std = norm.fit(gaussian_ns)

        #définition des limites de l'axe x
        xmin, xmax = plt.xlim()
        #création d'un vecteur x pour la gaussienne théorique
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)    #calcul de la gaussienne théorique
        plt.plot(x, p, 'k', linewidth = 2, label = 'Gaussienne théorique')

        #tracé des lignes verticales à -3 sigma et +3 sigma
        x_position1 = -3 * std_dev
        x_position2 = 3 * std_dev

        #tracé des lignes verticales à -3 sigma et +3 sigma avec annotations
        plt.axvline(x = x_position1, 
                    color = 'orange',
                    linestyle = 'dashed',
                    linewidth = 1,
                    label = f'Ligne 1 à -3 sigma: {x_position1:.2f}')
        plt.axvline(x = x_position2,
                    color = 'orange',
                    linestyle = 'dashed',
                    linewidth = 1,
                    label = f'Ligne 2 à 3 sigma: {x_position2:.2f}')

        #configuration du titre et des axes du graphique
        plt.title('Distribution de probabilité')
        plt.xlabel(abs_graph_Gauss)
        plt.ylabel(ord_graph_Gauss)
        plt.legend()

        #configuration de la grille et affichage du graphique
        plt.tight_layout()
        plt.show()

    def FFT_Signal(self,
                   signal,
                   fs, 
                   xlimit, 
                   abs_graph_FFT, 
                   ord_graph_FFT):
        """Fonction pour calculer et tracer la FFT d'un signal."""
        #nombre d'échantillons
        n = len(signal)
        #calcul de la FFT
        signal_fft = fft(signal)
         #calcul des fréquences associées
        frequencies = fftfreq(n, 1/fs)
        #fréquences positives
        positive_frequencies = frequencies[:n // 2]
        #amplitudes correspondantes
        amplitudes = 2.0 / n * np.abs(signal_fft[:n // 2])
        #tracé du spectre de fréquence avec les options données
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
        #itération sur la liste des filtres pour calculer 
        #et tracer le diagramme de Bode de chacun
        for i in range(len(Liste_Filtre)):
            #extraction des paramètres du filtre dans la liste
            id = Liste_Filtre[i][0]
            fc = Liste_Filtre[i][1]
            gain = Liste_Filtre[i][4]

            #calcul de la fonction de transfert du filtre passe-bas
            wc = 2 * np.pi * fc 
            num = [gain * wc]
            den = [1, wc]

            #création du système linéaire à partir de la fonction de transfert
            system = TransferFunction(num, den)

            #calcul des données pour le diagramme de Bode
            w, mag, phase = bode(system)

            #tracé du diagramme de Bode avec les options données
            plt.figure(figsize = (10, 6))

            #subdivision de la figure en deux sous-graphiques (gain et phase)
            plt.subplot(2, 1, 1)
            #tracé du gain en fonction de la fréquence
            #sur une échelle logarithmique
            plt.semilogx(w / (2 * np.pi), mag)
            #configuration du titre et des axes du graphique
            plt.title('Diagramme de Bode du capteur ' + str(id))
            plt.ylabel(ord_graph_bode_gain)
            #configuration de la grille
            plt.grid(which = 'both', linestyle = '--')
            #tracé des lignes verticales et horizontales
            #pour la fréquence de coupure et le gain à -3 dB
            plt.axvline(x = fc,
                        color = 'r',
                        linestyle = '--',
                        label = 'Fréquence de coupure ' + str(fc) + ' Hz')
            plt.axhline(y = -3, 
                        color = 'g',
                        linestyle = '--',
                        label = '-3 dB')
            #configuration de la légende
            plt.legend()
            
            #passage au deuxième sous-graphique pour tracer la phase
            plt.subplot(2, 1, 2)
            #tracé de la phase en fonction de la fréquence
            plt.semilogx(w / (2 * np.pi), phase)
            #configuration du titre et des axes du graphique
            plt.xlabel(abs_graph_bode_phase)
            plt.ylabel(ord_graph_bode_phase)
            #configuration de la grille
            plt.grid(which = 'both', linestyle = '--')
            #tracé des lignes verticales et horizontales
            #pour la fréquence de coupure et le gain à -45°
            plt.axvline(x = fc,
                        color = 'r',
                        linestyle = '--',
                        label = 'Fréquence de coupure ' + str(fc) + ' Hz')
            plt.axhline(y = -45,
                        color = 'g', 
                        linestyle = '--',
                        label = '-45 °')
            #configuration de la légende
            plt.legend()

            #configuration de la mise en page et affichage du graphique
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
        self.signal_in = signal_in #variable interne pour le signal d'entrée
        self.abs_graph = abs_graph #abscisse pour le graphe
        self.w_ns_mean = w_ns_mean #moyenne du bruit
        self.w_ns_std = w_ns_std #écart-type du bruit
        self.capteur_out = capteur_out #sortie du filtre capteur
        self.kalman_out = kalman_out #sortie du filtre de Kalman
        self.n_mc = n_mc #nombre d'itérations Monte Carlo
    
    def Monte_Carlo_Simu(self):
        """Fonction pour exécuter une itération
        pour la méthode Monte Carlo."""
        #Génération du signal pur
        x_true = self.signal_in
        
        #Ajout du bruit
        x_noisy = self.signal_in + np.random.normal(
            self.w_ns_mean, 
            self.w_ns_std, 
            len(x_true)
            )
        
        #Filtrage capteur
        x_sensor_filt = self.capteur_out
        
        #Filtre de Kalman
        x_kf = self.kalman_out
        
        #retourne les différents signaux pour cette itération de Monte Carlo
        return x_true, x_noisy, x_sensor_filt, x_kf

    def Monte_Carlo_Method(self):
        """Fonction pour exécuter la méthode 
        de Monte Carlo sur plusieurs simulations."""
        #exécution d'une simulation pour obtenir la taille du signal
        x_true, x_noisy, x_sensor_filt, x_kf = self.Monte_Carlo_Simu()
        N = len(x_true) #longueur du signal

        #initialisation des matrices d'erreurs
        errors_noisy = np.zeros((self.n_mc, N))
        #initialisation des matrices d'erreurs
        errors_sensor_filt = np.zeros((self.n_mc, N))
        #initialisation des matrices d'erreurs
        errors_kf = np.zeros((self.n_mc, N))

        #initialisation des RMSE par réalisation
        rmse_noisy_per_run = np.zeros(self.n_mc)
        #initialisation des RMSE par réalisation
        rmse_sensor_filt_per_run = np.zeros(self.n_mc)
        #initialisation des RMSE par réalisation
        rmse_kf_per_run = np.zeros(self.n_mc)

        # Boucle Monte Carlo
        for k in range(self.n_mc): #itération sur le nombre de simulations
            #exécution d'une itération de la simulation Monte Carlo
            (
                x_true_k,
                x_noisy_k,
                x_sensor_filt_k,
                x_kf_k
            ) = self.Monte_Carlo_Simu()
            
            #Vérification que la longueur est cohérente
            if len(x_true_k) != N:
                raise ValueError(
                    "Longueur du signal incohérente entre itérations"
                    )
            
            # Erreurs instantanées par rapport au signal pur
            errors_noisy[k, :] = x_noisy_k - x_true_k
            errors_sensor_filt[k, :] = x_sensor_filt_k - x_true_k
            errors_kf[k, :] = x_kf_k - x_true_k

            # RMSE global pour cette réalisation
            rmse_noisy_per_run[k] = np.sqrt(np.mean(errors_noisy**2))
            rmse_sensor_filt_per_run[k] = np.sqrt(
                np.mean(errors_sensor_filt**2)
                )
            rmse_kf_per_run[k] = np.sqrt(np.mean(errors_kf**2))

        # Statistiques Monte Carlo
        # Moyenne et variance de l'erreur à chaque instant
        mean_err_noisy = np.mean(errors_noisy, axis = 0)
        mean_err_sensor_filt = np.mean(errors_sensor_filt, axis = 0)
        mean_err_kf = np.mean(errors_kf, axis = 0)

        # Variance de l'erreur à chaque instant
        var_err_noisy = np.var(errors_noisy, axis = 0)
        var_err_sensor_filt = np.var(errors_sensor_filt, axis = 0)
        var_err_kf = np.var(errors_kf, axis = 0)

        # RMSE global pour résumé global
        rmse_noisy = np.sqrt(np.mean(errors_noisy**2))
        rmse_sensor_filt = np.sqrt(np.mean(errors_sensor_filt**2))
        rmse_kf = np.sqrt(np.mean(errors_kf**2))

        # Écart-type des RMSE entre réalisations
        rmse_noisy_std = np.std(rmse_noisy_per_run, ddof = 1)
        rmse_sensor_filt_std = np.std(rmse_sensor_filt_per_run, ddof = 1)
        rmse_kf_std = np.std(rmse_kf_per_run, ddof = 1)

        # MCSE = écart-type / sqrt(N_MC)
        mcse_noisy = rmse_noisy_std / np.sqrt(self.n_mc)
        mcse_sensor_filt = rmse_sensor_filt_std / np.sqrt(self.n_mc)
        mcse_kf = rmse_kf_std / np.sqrt(self.n_mc)

        # Affichage des résultats
        print("\nRMSE moyen par réalisation (Monte Carlo) + MCSE :")
        print(
            f"- Signal bruité : {rmse_noisy:.4f} "
            f"+/- {mcse_noisy:.4f} (RMSE +/- MCSE)"
            )
        print(
            f"- Filtre capteur : {rmse_sensor_filt:.4f} "
            f"+/- {mcse_sensor_filt:.4f} (RMSE +/- MCSE)"
            )
        print(
            f"- Filtre de Kalman : {rmse_kf:.4f} "
            f"+/- {mcse_kf:.4f} (RMSE +/- MCSE)"
            )

        # Visualisation des résultats
        t = self.abs_graph 

        # Intervalle de confiance 95% ≈ moyenne ± 2 * écart-type
        std_err_noisy = np.sqrt(var_err_noisy)
        std_err_sensor_filt = np.sqrt(var_err_sensor_filt)
        std_err_kf = np.sqrt(var_err_kf)

        # Tracé des résultats
        plt.figure(figsize = (10, 6))
        plt.plot(t, x_true, label="Signal vrai", linewidth = 2)

        # Signal bruité
        plt.plot(
            t,
            mean_err_noisy + x_true,
            label = "Bruit moyen",
            linestyle = "--")
        #remplissage de l'intervalle de confiance
        plt.fill_between(t,
                        x_true + mean_err_noisy - 2 * std_err_noisy,
                        x_true + mean_err_noisy + 2 * std_err_noisy,
                        alpha=0.2, label = "IC 95% bruité")

        # Filtre capteur
        plt.plot(t,
                 mean_err_sensor_filt + x_true,
                 label="Capteur filtré moyen")
        plt.fill_between(t,
                        x_true + mean_err_sensor_filt - 2*std_err_sensor_filt,
                        x_true + mean_err_sensor_filt + 2*std_err_sensor_filt,
                        alpha=0.2, label="IC 95% capteur filt.")

        # Filtre de Kalman
        plt.plot(t, mean_err_kf + x_true, label = "Kalman moyen")
        plt.fill_between(t,
                        x_true + mean_err_kf - 2 * std_err_kf,
                        x_true + mean_err_kf + 2 * std_err_kf,
                        alpha = 0.2, label = "IC 95% Kalman")

        #configuration du titre et des axes du graphique
        plt.xlabel("Temps")
        plt.ylabel("Amplitude")
        plt.title(f"Monte Carlo ({self.n_mc} itérations)")
        #configuration de la légende
        plt.legend()
        #configuration de la limite de l'axe x
        plt.xlim(1., 1.10)
        #configuration de la grille et affichage du graphique
        plt.grid(True)
        plt.show()

def Moyenne_Glissante(signal, 
                      window_size):
    """Fonction pour appliquer une moyenne glissante à un signal."""
    #calcul de la somme cumulée
    cumsum = np.cumsum(np.insert(signal, 0, 0))
    #retourne le signal moyenné
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

def Calc_retard(signal_pur, 
                signal_a_comparer, 
                freq_signal_pur, 
                dt, 
                titre_signal_a_comparer):
    """Fonction pour appliquer un calcul de retard entre deux signaux"""
    #synchronise les signaux
    correlation = correlate(signal_pur, signal_a_comparer, mode='full')
    lags = np.arange(-len(signal_a_comparer) + 1, len(signal_pur))
    idx = np.argmax(correlation)
    #calcul du délai en seconde
    delay_sec = lags[idx] * dt
    #calcul de la phase entre les signaux
    phase_rad = 2 * np.pi * freq_signal_pur * delay_sec
    #conversion en degré
    phase_deg = np.degrees(phase_rad)
    phase_deg = ((phase_deg + 180)%360) - 180
    #affichage des résultats
    print(
        f"Le retard temporel du signal pur avec "
        f"{titre_signal_a_comparer} est de {delay_sec} s."
        )
    print(
        f"Le retard degré du signal pur avec "
        f"{titre_signal_a_comparer} est de {phase_deg} °."
        )

def Calc_SNR(signal_pur_SNR, 
             signal_bruité_SNR, 
             unité_SNR, 
             titre_SNR):
    """Fonction pour appliquer un calcul de SNR sur les deux signaux"""
    #calcul de la valeur RMS du signal "pur"
    signal_RMS = np.sqrt(np.mean(signal_pur_SNR**2))
    #calcul de la valeur RMS du signal "bruité"
    bruité_RMS = np.sqrt(np.mean(signal_bruité_SNR**2))
    #affichage des résultats
    print(f"RMS Signal d'entrée : {signal_RMS} {unité_SNR}")
    print(f"RMS Signal bruité : {bruité_RMS} {titre_SNR}")
    
    #calcul du SNR entre les deux signaux
    SNR_Signal = 20 * log10(signal_RMS / bruité_RMS)
    #affichage du résultat
    print(f"SNR {titre_SNR} : {SNR_Signal} dB")

def CSV_Export(file_name, 
               abs_graph, 
               signal_in, 
               signal_ns, 
               signal_filt, 
               signal_KF):
    """Fonction pour exporter les donner dans un fichier .csv"""
    #Ouvre le fichier, si n'existe pas, le créé
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file) 
        writer.writerow([
                'Time (s)', 
                'Signal d\entrée', 
                'Signal Bruité', 
                'Signal filtré par le capteur', 
                'Signal filtré par Kalman'
            ])   #écriture de l'en-tête
        
        for (
            t,
            s_in,
            s_bruite,
            s_filtre_capteur,
            s_filtre_kalman
            ) in zip(abs_graph, signal_in, signal_ns, signal_filt, signal_KF):
            writer.writerow([
                t,
                s_in,
                s_bruite,
                s_filtre_capteur,
                s_filtre_kalman
                ]) #écriture des données ligne par ligne
            
    print("Exportation des données terminée dans '", file_name, "'.")

def Calc_Stable_Gain_KF(KF_gains, 
                        dt, 
                        filtre_fc):
    """Fonction pour calculer le gain stable du filtre de Kalman
    et le nombre de constantes de temps nécessaires."""
    #cas où la liste des gains de Kalman est vide
    if len(KF_gains) == 0:
        print("La liste des gains de Kalman est vide.")
        return
    #itération sur les gains de Kalman
    for i in range(len(KF_gains)): #attention bug si KF_Gains = 0
        #vérifie la stabilité du gain de Kalman
        if KF_gains[i] == KF_gains[i-1] and i != 0:
            print("Le gain de Kalman devient stable après",
                  i,
                  "échantillons, soit",
                  i*dt,
                  "secondes.")
            print("Valeur stable du gain de Kalman :", KF_gains[i], "V")
            break

    #calcul de la constante de temps du filtre passe-bas
    tau = 1/(2 * np.pi * filtre_fc)
    print("Constante de temps du filtre passe bas:", tau, "s")

    #calcul du nombre de constantes de temps nécessaires
    #pour atteindre la stabilité
    num_tau = (i * dt)/tau
    print("Le gain de Kalman devient stable après",
          num_tau,
          "constantes de temps.")