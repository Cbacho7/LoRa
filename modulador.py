# .\modulador.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt
from pathlib import Path

class Modulador:
    def __init__(self, SF: int, BW: float ,Fs: float , f0: float):
        """
        Parameters
        ----------
        SF : int
            Spreading Factor
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

        phase_up = 2 * np.pi * (self.f0 * self.t + 0.5 * self.k * self.t**2)
        self.upchirp = np.exp(1j * phase_up) 

        phase_down = 2 * np.pi * ((self.f0 + self.BW) * self.t - 0.5 * self.k * self.t**2)
        self.downchirp = np.exp(1j * phase_down)  

    def generate_upchirps(self, n=8) -> np.ndarray:
        """
        Genera n upchirps consecutivos
        """
        upchirps = [self.upchirp for _ in range(n)]
        return np.concatenate(upchirps)

        
    def generate_downchirps(self, n=2) -> np.ndarray:
        """
        Genera n downchirps consecutivos
        """
        downchirps = [self.downchirp for _ in range(n)]
        return np.concatenate(downchirps)

        
    def generate_preamble(self) -> np.ndarray:
        """
        Genera: 7 Upchirps + 3 Downchirps
        Prueba Header2: 9 Upchirps + 1 Downchirp 
        """
        up = self.generate_upchirps(7)

        """Descomentar para pruebas de rutas en la máquina de estados y cambiar el return por el que se quiera probar"""
        # up1 = self.generate_upchirps(4)
        # msg = "ab"
        # symbol = self.msg_to_symbols(msg) # Símbolo de transición (puede ser cualquier otro)
        # SYMBOLRND = self.symbols_to_signal(symbol)
        # up2 = self.generate_upchirps(1)
        
        down = self.generate_downchirps(3)

        # # 0.25 downchirp
        # shift = self.Ns // 4
        # quarter_down = self.downchirp[shift:]

        return np.concatenate([up,down])
        # return np.concatenate([up,SYMBOLRND,up2,down])
   
    def msg_to_symbols(self, msg: str) -> list[int]:
        # Convertir todo el mensaje a una sola cadena de bits
        data = msg.encode('latin1')  # Convertir a bytes usando latin1
        bit_string = "".join(f"{b:08b}" for b in data) # Cada letra a 8 bits
        
        symbols = []
        # Ir tomando trozos de tamaño SF
        for i in range(0, len(bit_string), self.SF):
            chunk = bit_string[i:i + self.SF]
            
            # Si el último trozo es más corto que el SF, rellenar con ceros (Padding)
            if len(chunk) < self.SF:
                chunk = chunk.ljust(self.SF, '0')
                
            symbols.append(int(chunk, 2)) # Convertir esos bits a un número (símbolo)
            
        return symbols
    
    def generate_header(self, payload_len: int) -> list[int]:
        if not (0 <= payload_len <= 255):
            raise ValueError("Payload debe ser 0-255 bytes")
        
        checksum = 255 - payload_len

        return [payload_len, payload_len, payload_len]

    def symbol_to_chirp(self, symbol: int) -> np.ndarray:
        if symbol < 0 or symbol >= self.M:
            raise ValueError("Símbolo fuera de rango")
        # frecuencia instantánea
        f_inst = ((self.k * self.t + symbol * self.BW / self.M) % self.BW) + self.f0
        # integrar frecuencia → fase
        phase = 2 * np.pi * np.cumsum(f_inst) / self.Fs
        return np.exp(1j * phase)


    def symbols_to_signal(self, symbols: list[int]) -> np.ndarray:
        """
        Convierte una lista de símbolos lora en la señal transmitida.
        """
        signal = []
        for symbol in symbols:
            chirp = self.symbol_to_chirp(symbol)
            signal.append(chirp)
        return np.concatenate(signal)

    def save_spectrogram(self, signal, symbols, msg=None, save_as=None):
        """
        Guarda o muestra el espectrograma de la señal Lora.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        spec = ax.specgram(np.real(signal), NFFT=2000, Fs=self.Fs, noverlap=1536)
        for i in range(len(symbols) + 1):
            ax.axvline(x=i * self.Ts, color='black', linestyle='--', alpha=0.5)

        ax.set_xlim(0, len(signal) / self.Fs)
        ax.set_ylim(2800, 3600)
        ax.set_xlabel('Tiempo [s]')
        ax.set_ylabel('Frecuencia [Hz]')

        if msg:
            title = f'Espectrograma - {len(symbols)} símbolos- Mensaje: "{msg}", SF = {self.SF}'
        else:
            title = f'Espectrograma - {len(symbols)} símbolos'

        ax.set_title(title)
        plt.colorbar(spec[3], ax=ax, label='Potencia [dB]')

        if save_as:
            save_path = Path(save_as)
            # crear carpeta si no existe
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Figura guardada como: {save_path}")
        else:
            plt.show()

        return fig

    def make_noise(self, ruido_dB: float, signal: np.ndarray):
        """
        Genera ruido gaussiano real con un SNR deseado
        respecto a una señal LoRa (compleja).
        """
        # SNR lineal
        SNR = 10**(ruido_dB / 10)
        largo = len(signal)
        # Potencia media de la señal (correcta para señal compleja)
        Ps = np.mean(np.abs(signal)**2)
        # Potencia del ruido requerida
        Pn = Ps / SNR
        # Desviación estándar del ruido real
        sigma = np.sqrt(Pn)
        # Ruido AWGN real
        ruido = np.random.normal(0.0, sigma, largo)
        return ruido
    
    def bandpass_filter(self, signal, f_inicial, f_final, order=6):
        """
        Filtro pasabanda Butterworth aplicado a una señal real.
        """
        nyq = self.Fs / 2
        b, a = butter(order, [f_inicial/nyq, f_final/nyq], btype='band')
        return filtfilt(b, a, signal)

    def SNR_cal(self, Ruido, Signal):
        assert len(Ruido) == len(Signal)

        Pot_Ruido = np.mean(Ruido**2)
        Pot_signal = np.mean(np.abs(Signal)**2)

        SNR = Pot_signal / Pot_Ruido
        return 10 * np.log10(SNR)

def main():
    # Parámetros
    SF = 8
    BW = 400
    Fs = 48000
    f0 = 3000

    mod = Modulador(SF, BW, Fs, f0)
    msg = "Hola mundo"#"Love me love me Say that you love me Fool me fool me Go on and fool me Love me love me Pretend that you love me Leave me leave me Just say that you need me"
    symbols = mod.msg_to_symbols(msg)
    signal = mod.symbols_to_signal(symbols)
    noise = mod.make_noise(ruido_dB=-31.5, signal=signal)
    filtered_noise = mod.bandpass_filter(noise, mod.f0, mod.f0 + mod.BW)
    signal_noisy = signal + filtered_noise
    filtered_signal_noisy = mod.bandpass_filter(signal_noisy, mod.f0, mod.f0 + mod.BW)
    # mod.save_spectrogram(signal, symbols, msg, save_as=f'Imagenes/fotos_sf{mod.SF}/signal_sf{mod.SF}.png')
    # mod.save_spectrogram(noise, symbols, msg, save_as=f'Imagenes/fotos_sf{mod.SF}/noise.png')
    # mod.save_spectrogram(filtered_noise, symbols, msg, save_as=f'Imagenes/fotos_sf{mod.SF}/filtered_noise.png')
    # mod.save_spectrogram(signal_noisy, symbols, msg, save_as=f'Imagenes/fotos_sf{mod.SF}/signal_noisy.png')
    # mod.save_spectrogram(filtered_signal_noisy, symbols, msg, save_as=f'Imagenes/fotos_sf{mod.SF}/filtered_signal_noisy.png')
    snr = mod.SNR_cal(filtered_noise, signal)
    print(f"SNR calculado: {snr:.2f} dB")

if __name__ == "__main__":
    main()