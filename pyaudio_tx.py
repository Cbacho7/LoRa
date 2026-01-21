# .\pyaudio_tx.py
import numpy as np
import pyaudio
from modulador import Modulador

"""Transmisión de señal LoRa usando PyAudio"""

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
    if not msg:
        print("Mensaje vacío. Saliendo.")
        return
    
    payload_bytes = msg.encode('latin-1')
    header_bytes = mod.generate_header(len(payload_bytes))

    frame_symbols = []

    # Header → símbolos LoRa
    for b in header_bytes:
        frame_symbols.extend(
            mod.msg_to_symbols(chr(b))
        )

    # Payload → símbolos LoRa
    frame_symbols.extend(
        mod.msg_to_symbols(msg)
    )

    frame_symbols = np.array(frame_symbols)

    #frame_symbols = np.concatenate((header, payload))
    signal = mod.symbols_to_signal(frame_symbols)
    signal_tx = np.concatenate((preamble, signal))


    # Normalizar señal
    signal_tx = np.real(signal_tx)
    # Normalizar y dejar margen de seguridad
    max_val = np.max(np.abs(signal_tx))
    if max_val > 0:
        signal_tx = (signal_tx / max_val) * 0.8  # Volumen al 80% para evitar distorsión

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
