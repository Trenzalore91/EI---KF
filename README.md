# EI - KF
 EI_KF

"Implémentation d'un filtre de Kalman afin de réduire le bruit d'un capteur".
Réalisé par Alessandro Farina (alessandro.farina@hainaut-ea.be)
Année 2025-2026
(Programmé sous VSCode)

Ce notebook a pour objectif d'étudier le fonctionnement d'un filtre de Kalman appliqué sur un capteur mesurant un signal bruité par un bruit blanc gaussien. Les données utiles sont générées directement via le programme. Le code peut générer une animation .gif ou un set de donnée sous .csv (à commenter ou non).

Ce notebook contient:
Section 1 : Librairies
Section 2 : Variables
Section 3 : Signal d'entrée
    3.1. Signal sinusoïdal / Génération d'un signal sinusoïdal ou carré en fonction du paramètre de l'objet
    3.2. Capteur Passe-bas
Section 4 : Filtre de Kalman
Section 5 : Graphiques
    5.1. Mise en graphique du signal d'entrée pur
    5.2. Mise en graphique du bruit blanc gaussien
    5.3. Mise en graphique du signal d'entrée bruité
    5.4. Graphique Bode Nyquist du/des filtres passe-bas
    5.5. Mise en graphique des signaux de sortie du/des passe-bas
    5.6. Mise en graphique de la sortie du filtre de Kalman
    5.7. Mise en graphique du ripple pré/post Kalman
    5.8. Mise en graphique de la moyenne glissante et du filtre de Kalman
Section 6 : Calcul déphasage des signaux (bruité, Kalman et moyenne glissante)
Section 7 : Export en .csv
Section 8 : Export de l'animation en .gif
Section 9 : Monte Carlo

Pré-requis:
- Python 3.12.1
- Jupyter Notebook
- Sources:
    - math : https://docs.python.org/3/library/math.html
    - scipy : https://pypi.org/project/scipy/
    - fft : https://docs.scipy.org/doc/scipy/tutorial/fft.html#fourier-transforms-scipy-fft
    - filterpy.kalman : https://filterpy.reaDThedocs.io/en/latest/kalman/KalmanFilter.html
    - numpy : https://pypi.org/project/numpy/
    - matplotlib : https://pypi.org/project/matplotlib/
    - csv : https://docs.python.org/fr/3/library/csv.html

