# .\pyaudio_rx.py
import numpy as np
import pyaudio
from demodulador import Demodulador
from maquina import StateMachine, Machine

"""Recepción de señal LoRa usando PyAudio"""
def main():
    # Parámetros de la señal
    SF = 8
    BW = 400  # Ancho de banda en Hz
    Fs = 48000  # Frecuencia de muestreo en Hz
    f0 = 3000   # Frecuencia inicial en Hz

    # Crear demodulador
    demod = Demodulador(SF, BW, Fs, f0)

    # Create PyAudio instance
    py_audio = pyaudio.PyAudio()
    stream = py_audio.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=Fs,
        input=True,
        frames_per_buffer=demod.Ns
    )

    print("Escuchando... Presione Ctrl+C para detener.")

    try:
        while True:
            # Leer datos del stream
            data = stream.read(demod.Ns)
            signal_rx = np.frombuffer(data, dtype=np.float32)

            # Demodular la señal recibida
            symbols_rx = demod.signal_to_symbols(signal_rx)

            print("Símbolos recibidos:", symbols_rx)

    except KeyboardInterrupt:
        print("Deteniendo recepción.")

    # Stop stream and close everything
    stream.stop_stream()
    stream.close()
    py_audio.terminate()
    print('Recepción terminada')