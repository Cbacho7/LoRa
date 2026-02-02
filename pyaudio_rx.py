import time
import numpy as np
import pyaudio
from maquina import Machine, StateMachine

RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
CYAN    = "\033[96m"
RESET   = "\033[0m"
BOLD    = "\033[1m"

def main():
    # --- Parámetros LoRa (Los mismos de tu código original) ---
    SF = 8
    BW = 400
    Fs = 48000
    f0 = 3000

    # --- Inicialización de la Máquina ---
    machine = Machine(SF, BW, Fs, f0)
    
    # Definimos el tamaño del buffer igual que en tu lógica original
    buffer_size = machine.Ns // 4 

    # --- Definición del Callback ---
    # Esta función se ejecutará automáticamente cada vez que la tarjeta de sonido
    # llene el buffer.
    def callback(in_data, frame_count, time_info, status):
        # 1. Leer muestras y convertir a float32
        data = np.frombuffer(in_data, dtype=np.float32)
        
        # 2. Convertir a complejo (Requisito de tu código original)
        rx_complex = data.astype(np.complex64)

        # 3. Pasar muestras a la FSM
        machine.process(rx_complex)

        # 4. Verificar si el paquete está completo
        if machine.current_state == StateMachine.DONE:
            # Extraer y decodificar
            msg = machine.demod.symbols_to_msg(machine.payload)
            print(f"{RED}Mensaje recibido: {CYAN}{msg}{RESET}")
            # Reiniciar FSM para el próximo frame
            machine.reset_machine()
        
        # Retornamos None (porque no estamos reproduciendo audio, solo grabando)
        # y paContinue para que siga escuchando.
        return (None, pyaudio.paContinue)

    # --- Configuración de PyAudio ---
    py_audio = pyaudio.PyAudio()

    print(f"Iniciando escucha (Callback)... Buffer: {buffer_size}")
    print("Presiona Ctrl+C para detener.")

    stream = py_audio.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=Fs,
        input=True,
        frames_per_buffer=buffer_size,
        stream_callback=callback  # <--- Aquí vinculamos la función
    )

    # --- Bucle Principal ---
    # Iniciamos el stream
    stream.start_stream()

    try:
        # Mantenemos el programa vivo mientras el stream esté activo.
        # Usamos time.sleep para no quemar CPU innecesariamente en este bucle vacío.
        while stream.is_active():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nDeteniendo recepción...")

    finally:
        # Limpieza de recursos
        stream.stop_stream()
        stream.close()
        py_audio.terminate()
        print("Finalizado.")

if __name__ == "__main__":
    main()