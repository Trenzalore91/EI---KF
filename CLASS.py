class FILTRE:
    def __init__(self, id, fc, fs, ordre, type_filtre):
        self.id = id
        self.fc = fc
        self.fs = fs
        self.ordre = ordre
        self.type_filtre = type_filtre

    def __calculfiltre__(self, signal_a_filtrer):
        freq_nyquist = 0.5 * self.fs
        normal_freq_coupure = self.fc / freq_nyquist
        b, a = butter(self.ordre, normal_freq_coupure, btype=self.type_filtre, analog=False)
        y = lfilter(b, a, self.signal_a_filtrer)