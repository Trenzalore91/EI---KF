class FILTRE:
    def __init__(self, fc, fs, ordre, type_filtre):
        self.fc = fc
        self.fs = fs
        self.ordre = ordre
        self.type_filtre = type_filtre

    def __calculfiltre__(self, data):
        nyquist = 0.5 * self.fs  #fréquence Nyquist
        normal_cutoff = self.fc / nyquist    #fréquence de coupure normalisée
        b, a = butter(self.ordre, normal_cutoff, btype=self.type_filtre)  #calcul des coefficients du filtre
        y = lfilter(b, a, data) #application du filtre
        return y