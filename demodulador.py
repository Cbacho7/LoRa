# /demodulador.py
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

        self.t = np.arange(0, self.Ns) / Fs
        self.k = BW / self.Ts       # pendiente del chirp
        
        self.upchirp = self.generate_upchirp()
        self.downchirp = self.generate_downchirp()
        
    def generate_upchirp(self):
        """
        Genera un Up-Chirp de referencia en la banda.
        Sube desde f0 hasta f0+BW.
        """
        phase_up = 2 * np.pi * (self.f0 * self.t + 0.5 * self.k * self.t**2)
        return np.exp(1j * phase_up)


    def generate_downchirp(self) -> np.ndarray:
        """
        Genera un Down-Chirp de referencia en la banda.
        Baja desde f0+BW hasta f0.
        """
        phase_down = 2 * np.pi * ((self.f0 + self.BW) * self.t - 0.5 * self.k * self.t**2)
        return np.exp(1j * phase_down)
    
    def dechirp(self, symbol_signal):
        dechirp = symbol_signal * np.conj(self.upchirp)
        return dechirp

   
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

    # TX
    symbols_tx = mod.msg_to_symbols(msg_tx)
    signal_tx = mod.symbols_to_signal(symbols_tx)

    # Canal
    noise = mod.make_noise(ruido_dB=-40, signal=signal_tx)
    filtered_noise = mod.bandpass_filter(noise, f0, f0 + BW)
    signal_rx = signal_tx + filtered_noise

    # RX
    symbols_rx = demod.signal_to_symbols(signal_rx)
    msg_rx = demod.symbols_to_msg(symbols_rx)

    # Resultados
    print(f"TX: {msg_tx}")
    print(f"Símbolos TX: {symbols_tx}")
    print(f"Símbolos RX: {symbols_rx}")
    print(f"RX: {msg_rx}")

if __name__ == "__main__":
    main()