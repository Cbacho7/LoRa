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

    payload_bytes = msg.encode('latin1')  # Usamos latin1 para soportar caracteres especiales
    header_symbols = mod.generate_header(len(payload_bytes))
    frame_symbols = []

    # Header → símbolos LoRa
    frame_symbols.extend(header_symbols)

    # Payload → símbolos LoRa
    frame_symbols.extend(mod.msg_to_symbols(msg))
    signal = mod.symbols_to_signal(frame_symbols)
    signal_tx = np.concatenate((preamble, signal))

    # ruido_dB=-30
    
    # noise = mod.make_noise(ruido_dB=ruido_dB, signal=signal_tx)
    # filtered_noise = mod.bandpass_filter(noise, mod.f0, mod.f0 + mod.BW)
    # snr_real = mod.SNR_cal(filtered_noise, signal_tx)
    # print(f"--- REPORTE DE RUIDO ---")
    # print(f"SNR Solicitado (Broadband): -{ruido_dB} dB")
    # print(f"SNR Real (In-Band): {snr_real:.2f} dB")
    # print(f"------------------------")
    # signal_tx = signal_tx + filtered_noise

    print(f"Payload length: {len(payload_bytes)} bytes")
    print(f"Símbolos TX: {len(frame_symbols)}")
    print(f"Muestras TX: {len(signal_tx)}")



    # Normalizar señal
    signal_tx = np.real(signal_tx)
    # Normalizar y dejar margen de seguridad
    max_val = np.max(np.abs(signal_tx))
    if max_val > 0:
        signal_tx = (signal_tx / max_val) 

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
