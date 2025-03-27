class FILTRE():
    def __init__(self, id, fc, fs, ordre, type_filtre, signal_a_filtrer):
        self.id = id
        self.fc = fc
        self.fs = fs
        self.ordre = ordre
        self.type_filtre = type_filtre
        self.signal_a_filtrer = signal_a_filtrer
        self.__calculfiltre__()

    def __calculfiltre__(self):
        freq_nyquist = 0.5 * self.fs
        normal_freq_coupure = self.fc / freq_nyquist
        b, a = butter(self.ordre, normal_freq_coupure, btype=self.type_filtre, analog=False)
        return lfilter(b, a, self.signal_a_filtrer)
