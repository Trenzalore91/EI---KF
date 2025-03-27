class Filtre():
    def __init__(self, id, fc, fs, ordre, signal_a_filtrer):
        self.id = id
        self.fc = fc
        self.fs = fs
        self.ordre = ordre
        self.signal_a_filtrer = signal_a_filtrer
        self.__calculfiltre__()

    def __calculfiltre__(self):
        freq = 0.5 * self.fs
        normal_freq_coupure = self.fc / freq
        b, a = butter(self.ordre, normal_freq_coupure, btype=self.type_filtre, analog=False)
        return lfilter(b, a, self.signal_a_filtrer)

class Filtres_standard(Filtre):
    def __init__(self, type_filtre):
        super().__init__(type_filtre)
        
class Filtre_Kalman():
    def __init__(self, signal_a_filtrer):
        self.signal_a_filtrer = signal_a_filtrer
    