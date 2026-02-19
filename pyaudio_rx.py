import time
import numpy as np
import pyaudio
import sys
# Asegúrate de que tu archivo con la clase se llama 'maquina.py'
from maquina import Machine, StateMachine 

# Colores
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
CYAN    = "\033[96m"
RESET   = "\033[0m"

# --- CLASE LOGGER (Tu código original - Está perfecto) ---
class DualLogger(object):
    def __init__(self, filename="registro_lora.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Activar Logger
sys.stdout = DualLogger()

def main():
    # --- Parámetros LoRa ---
    SF = 8
    BW = 400
    Fs = 48000
    f0 = 3000

    # Inicializar la Máquina
    machine = Machine(SF, BW, Fs, f0)
    
    # Tamaño del chunk para PyAudio
    # IMPORTANTE: Al usar deque, no necesitamos un buffer gigante aquí. 
    # Ns (un símbolo) es una buena medida de actualización.
    buffer_size = machine.Ns

    # --- CALLBACK DE AUDIO ---
    # Esta función es el "CORAZÓN" que conecta el hardware con tu lógica
    def callback(in_data, frame_count, time_info, status):
        # 1. Convertir bytes a numpy array
        data = np.frombuffer(in_data, dtype=np.float32)
        
        # 2. Convertir a complejo (Requerido por la FSM)
        #rx_complex = data.astype(np.complex64)

        # ---------------------------------------------------------
        # AQUÍ ESTÁ LA CORRECCIÓN: CONEXIÓN CON TU MÁQUINA DE ESTADOS
        # ---------------------------------------------------------
        
        # Paso A: Alimentar el buffer circular (deque)
        machine.add_data(data)

        # Paso B: "Girar la manivela" de la máquina según el estado actual
        # Como estamos en un callback (tiempo real), ejecutamos un paso lógico
        
        state = machine.current_state

        if state == StateMachine.SEARCH_PREAMBLE:
            # Buscamos preámbulo (Overlapping FFT)
            # Esta función ya la definimos y funciona con el deque
            machine.search_preamble()
        
        elif state == StateMachine.LOCKED:
            # Aquí irá la lógica de seguimiento (aún debemos definir process_locked_step)
            machine.locked()
            
            
        
        elif state == StateMachine.SEARCH_SFD:
            machine.search_sfd()
            
            
            
        elif state == StateMachine.SYNC_SFD:
            machine.sync_sfd()
            
            

        elif state == StateMachine.HEADER:
            machine.header()

        
        elif state == StateMachine.HEADER2:
            machine.header2()
            
            
            

        elif state == StateMachine.PAYLOAD:
            machine.payload()

        elif state == StateMachine.PAYLOAD2:
            machine.payload2()
            
        
        elif state == StateMachine.DONE:
            # Si la máquina marca DONE, es que terminó de procesar un paquete
            print(f"\n{GREEN}--- PAQUETE FINALIZADO ---{RESET}")
            
            try:
                # Intenta decodificar lo que haya en machine.datos_rx
                if hasattr(machine, 'datos_rx') and len(machine.datos_rx) > 0:
                    msg = machine.demod.symbols_to_msg(machine.datos_rx)
                    print(f"{MAGENTA}MENSAJE RECIBIDO: {msg}{RESET}")
                else:
                    print(f"{YELLOW}Paquete vacío o error de datos{RESET}")
            except Exception as e:
                print(f"{RED}Error decodificando: {e}{RESET}")

            # Reiniciar para el siguiente paquete
            machine.reset_machine()

        # ---------------------------------------------------------

        return (None, pyaudio.paContinue)

    # --- CONFIGURACIÓN PYAUDIO ---
    p = pyaudio.PyAudio()

    print(f"{YELLOW}Iniciando escucha LoRa (SF{SF} BW{BW})...{RESET}")
    print(f"Buffer Size: {buffer_size} muestras")
    print("Presiona Ctrl+C para salir.")

    try:
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=Fs,
            input=True,
            frames_per_buffer=buffer_size,
            stream_callback=callback
        )

        stream.start_stream()

        # Bucle principal para mantener vivo el script
        while stream.is_active():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print(f"\n{RED}Deteniendo recepción...{RESET}")
    
    except Exception as e:
        print(f"\n{RED}Error fatal: {e}{RESET}")

    finally:
        # Cerrar todo limpio
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()
        print("Bye.")

if __name__ == "__main__":
    main()
