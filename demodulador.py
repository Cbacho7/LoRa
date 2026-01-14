import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt
from pathlib import Path
from modulador import Modulador

class Demodulador:
    def __init__(self, SF: int, BW: float ,Fs: float , f0: float):
        """
        Parameters
        ----------
        SF : int
            Spreading Factor (mod)
        BW : float
            Ancho de banda del chirp [Hz]
        Fs : float
            Frecuencia de muestreo [Hz]
        f0 : float
            Frecuencia inicial del canal [Hz]
        """
        self.SF = SF
        self.BW = BW
        self.Fs = Fs
        self.f0 = f0

        self.M = 2 ** SF            # número de símbolos
        self.Ts = self.M / BW       # duración del símbolo
        self.Ns = int(self.Ts * Fs) # muestras por símbolo

        self.t = np.arange(self.Ns) / Fs
        self.k = BW / self.Ts       # pendiente del chirp

        self.upchirp = self.generate_upchirp()
        self.downchirp = self.generate_downchirp()
        
        
    def generate_upchirp(self):
        """
        Genera un upchirp LoRa de referencia (sin símbolo).
        """
        f_inst = (self.k * self.t) % self.BW
        phase = 2 * np.pi * np.cumsum(f_inst) / self.Fs
        return np.exp(1j * phase)

    
    def generate_downchirp(self) -> np.ndarray:
        """
        Genera un downchirp LoRa de referencia (sin símbolo).
        """
        f_inst = ((-self.k * self.t) % self.BW)
        phase = 2 * np.pi * np.cumsum(f_inst) / self.Fs
        return np.exp(1j * phase)
    
    def generate_preamble(self) -> np.ndarray:
        preamble = []

        # 8 upchirps
        for _ in range(8):
            preamble.append(self.upchirp)

        # 2 downchirps completos
        for _ in range(2):
            preamble.append(self.downchirp)

        # 0.25 downchirp 
        quarter_down = self.downchirp[: self.Ns // 4]
        preamble.append(quarter_down)

        return np.concatenate(preamble)
    
    def find_preamble(self, rx_signal):
        for i in range(0, len(rx_signal) - self.Ns, self.Ns // 4):
            block = rx_signal[i:i + self.Ns]
            dechirped = block * np.conj(self.upchirp)
            spectrum = np.abs(np.fft.fft(dechirped))[:self.M]

            if np.argmax(spectrum) == 0:
                return i
        return None

    def dechirp(self, symbol_signal):
        # bajar a banda base
        bb = symbol_signal * np.exp(-1j * 2 * np.pi * self.f0 * self.t)

        # quitar chirp
        dechirped = bb * np.conj(self.upchirp)
        return dechirped

    # def fft_symbol(self, dechirped: np.ndarray) -> np.ndarray:
    #     """
    #     Calcula la FFT del símbolo dechirpeado y devuelve
    #     la magnitud de los bins útiles.
    #     """
    #     if len(dechirped) != self.Ns:
    #         raise ValueError("La señal dechirpeada debe tener Ns muestras")

    #     fft_result = np.fft.fft(dechirped)

    #     # Solo los bins útiles 
    #     spectrum = np.abs(fft_result[:self.M])

        return spectrum
    def fft_symbol(self, dechirped: np.ndarray) -> np.ndarray:
        """
        Calcula la FFT y suma la energía de las frecuencias positivas y 
        las aliadas negativas (necesario cuando Fs >> BW).
        """
        if len(dechirped) != self.Ns:
            raise ValueError("La señal dechirpeada debe tener Ns muestras")

        fft_result = np.fft.fft(dechirped)
        
        # 1. Parte Positiva (bins 0 a M)
        part_pos = np.abs(fft_result[:self.M])
        
        # 2. Parte Negativa (bins Ns-M a Ns)
        # Estas son las frecuencias S - BW que aparecen al final por aliasing
        part_neg = np.abs(fft_result[-self.M:])
        
        # Sumamos ambas contribuciones para capturar toda la energía del símbolo
        spectrum = part_pos + part_neg

        return spectrum
    

    def detect_symbol(self, spectrum: np.ndarray) -> int:
        """
        Detecta el símbolo LoRa con corrección de resolución.
        """
        idx = np.argmax(spectrum)
        return int(idx % self.M)
    

    def demodulate_symbol(self, symbol_signal: np.ndarray) -> int:
        """
        Demodula un símbolo LoRa (Ns muestras).
        """
        if len(symbol_signal) != self.Ns:
            raise ValueError("El símbolo debe tener Ns muestras")

        dechirped = self.dechirp(symbol_signal)
        spectrum = self.fft_symbol(dechirped)
        symbol_sup = self.detect_symbol(spectrum)

        return symbol_sup

    def signal_to_symbols(self, rx_signal: np.ndarray) -> list[int]:
        """
        Demodula una señal LoRa completa en una lista de símbolos.
        Asume símbolos perfectamente alineados.
        """
        symbols = []

        num_symbols = len(rx_signal) // self.Ns

        for i in range(num_symbols):
            start = i * self.Ns
            end = (i + 1) * self.Ns
            symbol_block = rx_signal[start:end]

            symbol_sup = self.demodulate_symbol(symbol_block)
            symbols.append(symbol_sup)

        return symbols

    def symbols_to_msg(self, symbols: list[int], encoding="latin-1") -> str:
        """
        Convierte una lista de símbolos LoRa a texto, revirtiendo el empaquetado 
        de bits (Bit Packing) para cualquier SF.
        """
        # Convertir cada símbolo a una cadena de bits de longitud SF
        bit_string = "".join(f"{s:0{self.SF}b}" for s in symbols)
        
        byte_list = []
        
        # Tomar la cadena de bits y cortarla en trozos de 8 bits 
        for i in range(0, len(bit_string), 8):
            byte_chunk = bit_string[i:i + 8]
            
            # Si el último trozo no completa 8 bits (por el padding del modulador),
            # lo ignoramos o verificamos que sean solo ceros.
            if len(byte_chunk) == 8:
                byte_list.append(int(byte_chunk, 2))
                
        # 3. Convertir la lista de bytes a texto
        # Se usa latin-1 para evitar errores con caracteres como la 'ñ'
        try:
            return bytes(byte_list).decode(encoding).rstrip('\x00')
        except UnicodeDecodeError:
            return "Error de decodificación: los bits recibidos no forman texto válido."
        



def main():
    # Parámetros
    SF = 8
    BW = 400
    Fs = 48000
    f0 = 3000

    mod = Modulador(SF, BW, Fs, f0)
    demod = Demodulador(SF, BW, Fs, f0)

    msg_tx = "Love me love me Say that you love me Fool me fool me Go on and fool me Love me love me Pretend that you love me Leave me leave me Just say that you need me"
    
    symbols_tx = mod.msg_to_symbols(msg_tx)

    preamble = demod.generate_preamble()
    payload = mod.symbols_to_signal(symbols_tx)

    signal_tx = np.concatenate([preamble, payload])

    # (opcional ruido)
    noise = mod.make_noise(ruido_dB=-34, signal=signal_tx)
    filtered_noise = mod.bandpass_filter(noise, mod.f0, mod.f0 + mod.BW)

    # Para debug limpio:
    #signal_rx = signal_tx
    # Para ruido:
    signal_rx = signal_tx + filtered_noise
    #signal_rx = np.roll(signal_tx, 10)
    #t_global = np.arange(len(signal_tx)) / Fs
    #signal_rx = signal_tx * np.exp(1j * 2 * np.pi * 20 * t_global)




    # RECEPTOR 
    start = demod.find_preamble(signal_rx)

    if start is None:
        print("No se detectó preámbulo")
        return

    preamble_len = int((8 + 2.25) * demod.Ns)
    payload_start = start + preamble_len

    symbols_rx = demod.signal_to_symbols(signal_rx[payload_start:])
    
    msg_rx = demod.symbols_to_msg(symbols_rx)

    # RESULTADOS 
    print(f"TX: {msg_tx}")
    print(f"Símbolos TX: {symbols_tx}")
    print(f"Símbolos RX: {symbols_rx}")
    print(f"RX: {msg_rx}")



if __name__ == "__main__":
    main()