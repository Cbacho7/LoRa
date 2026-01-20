# .\pyaudio_tx.py
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scp
import pyaudio
from scipy.io.wavfile import write
from modulador import Modulador

"""Transmisión de señal LoRa usando PyAudio"""
""" 
Que debe tener:
- Modulador LoRa
- Generación de preámbulo
- Header
- Generación de símbolos
- Generación de señal transmitida
- Transmisión usando PyAudio

"""

def main():
    # Parámetros de la señal
    SF = 8
    BW = 400  # Ancho de banda en Hz
    Fs = 48000  # Frecuencia de muestreo en Hz
    f0 = 3000   # Frecuencia inicial en Hz

    # Crear modulador
    mod = Modulador(SF, BW, Fs, f0)

    # Generar preambulo
    preamble = mod.generate_preamble()

    print("Escriba el mensaje a transmitir:")
    msg = input()
    symbols = mod.msg_to_symbols(msg)
    preamble_and_symbols = np.concatenate((preamble,symbols))
    signal_tx = mod.symbols_to_signal(preamble_and_symbols)

    # Create PyAudio instance
    py_audio = pyaudio.PyAudio()
    stream = py_audio.open(
    format=pyaudio.paFloat32,
    channels=1,
    rate=Fs,
    output=True,
        )
    
    # Transmitir señal
    stream.write(signal_tx.astype(np.float32).tobytes())

    # Stop stream and close everything
    stream.stop_stream()
    stream.close()
    py_audio.terminate()
    print('Mensaje enviado')

# Run main function
if __name__ == "__main__":
    main()
