#.\maquina.py
import numpy as np
from enum import Enum, auto
from demodulador import Demodulador

RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
CYAN    = "\033[96m"
RESET   = "\033[0m"
BOLD    = "\033[1m"


class StateMachine(Enum):
    SEARCH_PREAMBLE = auto()
    SYNC = auto()
    READ_HEADER = auto()
    READ_PAYLOAD = auto()
    DONE = auto()

class Machine:
    def __init__(self, SF: int, BW: float, Fs: float, f0: float):
        # Parámetros LoRa
        self.SF = SF
        self.BW = BW
        self.Fs = Fs
        self.f0 = f0

        self.M = 2 ** SF
        self.Ts = self.M / BW
        self.Ns = int(self.Ts * Fs)

        # Tiempo y chirps
        self.t = np.arange(self.Ns) / Fs
        self.k = BW / self.Ts

        self.upchirp = np.exp(1j * 2*np.pi * (f0*self.t + 0.5*self.k*self.t**2))
        self.downchirp = np.exp(1j * 2*np.pi * ((f0+BW)*self.t - 0.5*self.k*self.t**2))

        # FSM
        self.current_state = StateMachine.SEARCH_PREAMBLE
        self.rx_buffer = np.array([], dtype=float)

        # Sync
        self.sync_count = 0

        # Payload
        self.payload_len = None
        self.payload = []

        # Demod
        self.demod = Demodulador(SF, BW, Fs, f0)

        self.in_preamble_sequence = False
        self.upchirps_detected = 0
        self.k_up = None
        self.preamble_peaks = []
        self.current_preamble_peak = None

    # =========================================================
    # SEARCH PREAMBLE
    # =========================================================

    def search_preamble(self):
        if len(self.rx_buffer) < self.Ns:
            return

        step = self.Ns // 4
        window = self.rx_buffer[:self.Ns]
        
        dechirped = window * np.conj(self.upchirp)
        spectrum = np.abs(np.fft.fft(dechirped))
        
        peak_val = np.max(spectrum)
        mean_val = np.mean(spectrum)
        
        # Obtenemos el bin crudo (el que te sale como 30666)
        raw_bin = np.argmax(spectrum)

        # --- CAMBIO 1: NORMALIZACIÓN A REAL (CON SIGNO) ---
        # Si raw_bin es 30666 y Ns es 30720, esto lo convierte en -54
        if raw_bin > self.Ns // 2:
            real_bin = raw_bin - self.Ns
        else:
            real_bin = raw_bin

        # --- CAMBIO 2: FILTRO DE FRECUENCIA (TU SOLICITUD) ---
        # Si el pico detectado está más allá de +/- 1000, es ruido.
        MAX_FREQ_OFFSET = 1000
        
        # Condición de validez
        is_frequency_valid = abs(real_bin) < MAX_FREQ_OFFSET

        BIN_TOLERANCE = 2 

        is_signal_strong = peak_val > 4 * mean_val
        is_bin_consistent = True
        
        if self.in_preamble_sequence and len(self.preamble_peaks) > 0:
            last_bin = self.preamble_peaks[-1] # Este ya será un bin con signo
            
            # Comparamos usando la matemática real
            diff = abs(real_bin - last_bin)
            
            # Ya no necesitamos lógica de wraparound manual porque 'real_bin' ya tiene signo
            if diff > BIN_TOLERANCE:
                is_bin_consistent = False

        # --- CAMBIO 3: AÑADIMOS EL FILTRO A LA CONDICIÓN ---
        if is_signal_strong and is_bin_consistent and is_frequency_valid:
            self.in_preamble_sequence = True
            self.upchirps_detected += 1
            
            # Guardamos el bin REAL (con signo) para que el Sync lo use fácil
            self.preamble_peaks.append(real_bin)
            
            # DEBUG: Mostramos ambos para que veas que está funcionando
            print(f"{GREEN}[DEBUG-SEARCH] Upchirp #{self.upchirps_detected} | Bin Real: {real_bin} (Raw: {raw_bin}) | Val: {peak_val:.1f}{RESET}")

            self.rx_buffer = self.rx_buffer[self.Ns:] 

        else:
            if self.in_preamble_sequence:
                if self.upchirps_detected >= 6:
                    print(f"[SEARCH] Fin de preámbulo detectado (Chirps: {self.upchirps_detected}).")
                    
                    # El promedio ahora será correcto (ej. promedio de -54, -54, -53 es -53.6)
                    avg_peak =np.rint(np.mean(self.preamble_peaks))
                    print(f"[INFO] Promedio P_up (Normalizado): {avg_peak:.2f}")
                    
                    self.current_preamble_peak = avg_peak
                    self.current_state = StateMachine.SYNC
                else:
                    # Si falló por frecuencia inválida, lo decimos
                    if not is_frequency_valid and is_signal_strong:
                        print(f"[DEBUG-RESET] Racha rota por Bin fuera de rango: {real_bin}")
                    else:
                        print(f"[DEBUG-RESET] Racha rota en {self.upchirps_detected}. Bin: {real_bin}. Reiniciando.")
                    
                    self.rx_buffer = self.rx_buffer[step:]

                self.in_preamble_sequence = False
                self.upchirps_detected = 0
                self.preamble_peaks = []
            else:
                self.rx_buffer = self.rx_buffer[step:]

    # =========================================================
    # SYNC SFD
    # =========================================================

    def sync_sfd(self):
        # --- PROTECCIÓN INICIAL ---
        if len(self.rx_buffer) < 3.5 * self.Ns:
            return

        print("\n--- INICIO DEBUG SYNC (VERIFICADOR INTELIGENTE) ---")

        # 1. Recuperar k_up
        k_up = self.current_preamble_peak 

        # 2. Selector Inteligente
        k_down_candidates = []
        for i in range(2):
            start = i * self.Ns 
            dn_window = self.rx_buffer[start : start + self.Ns]
            spec_dn = np.abs(np.fft.fft(dn_window * np.conj(self.downchirp)))
            
            raw_k = np.argmax(spec_dn)
            if raw_k > self.Ns // 2: k = raw_k - self.Ns
            else: k = raw_k
            
            k_down_candidates.append(k)

        diff1 = abs(k_up + k_down_candidates[0])
        diff2 = abs(k_up + k_down_candidates[1])
        k_down = k_down_candidates[0] if diff1 < diff2 else k_down_candidates[1]

        # 3. Cálculo Estándar
        sto_bins = (k_up - k_down) / 2
        print(f"{GREEN}[SYNC] STO Calculado: {sto_bins:.1f} bins{RESET}")

        base_sfd_skip = 2 * self.Ns
        sto_samples = int(sto_bins * (self.Fs / self.BW))
        
        # Opción A: El salto normal (Base - STO)
        option_A = int(base_sfd_skip - sto_samples)
        if option_A < 0: option_A = 0
        
        # Opción B: El "parche" (+1 Símbolo)
        option_B = option_A + self.Ns

        # --- 4. VERIFICACIÓN SEGURA ---
        max_idx = len(self.rx_buffer) - self.Ns
        
        if option_B > max_idx:
            print(f"{YELLOW}[WARN] Buffer insuficiente para verificar Opción B. Esperando datos...{RESET}")
            return 

        # Medimos energías
        win_A = self.rx_buffer[option_A : option_A + self.Ns]
        val_A = np.max(np.abs(np.fft.fft(win_A * np.conj(self.upchirp))))

        win_B = self.rx_buffer[option_B : option_B + self.Ns]
        val_B = np.max(np.abs(np.fft.fft(win_B * np.conj(self.upchirp))))
            
        print(f"[VERIFICADOR] Opción A: {val_A:.1f} | Opción B: {val_B:.1f}")

        # --- DECISIÓN CORREGIDA (AQUÍ ESTÁ EL ARREGLO) ---
        
        # Definimos qué tan estricto seremos.
        # Si el STO es positivo, sabemos que la matemática 'Base - STO' tiende a dejarnos cortos.
        # Por lo tanto, le damos ventaja a la Opción B (bajamos el umbral).
        if sto_bins > 0:
            threshold = 0.8 # Si B es al menos el 80% de A, lo tomamos.
            print(f"[DECISIÓN] STO Positivo detectado -> Favoreciendo Opción B (Umbral 0.8)")
        else:
            threshold = 1.2 # Si STO es negativo, A suele ser correcto. Exigimos mucho para cambiar.
            print(f"[DECISIÓN] STO Negativo detectado -> Favoreciendo Opción A (Umbral 1.2)")

        # Aplicamos el umbral dinámico
        if val_B > val_A * threshold:
            final_skip = option_B
            print(f"{YELLOW}[CORRECCIÓN] Avanzando 1 símbolo (B={val_B:.1f} > A={val_A:.1f}).{RESET}")
        else:
            final_skip = option_A
            print(f"{GREEN}[OK] Manteniendo Opción A.{RESET}")

        print(f"{GREEN}[SYNC] Salto Final: {final_skip} muestras.{RESET}")
        print("--- FIN DEBUG SYNC ---\n")
            
        self.rx_buffer = self.rx_buffer[final_skip:]
        self.current_state = StateMachine.READ_HEADER


    

    # =========================================================
    # SYMBOL READ (alineado)
    # =========================================================

    def read_symbol(self):
        if len(self.rx_buffer) < self.Ns:
            return None

        block = self.rx_buffer[:self.Ns]
        self.rx_buffer = self.rx_buffer[self.Ns:]

        return self.demod.demodulate_symbol(block)


    # =========================================================
    # HEADER
    # =========================================================

    def read_header(self):
        # Verificamos si hay suficientes datos
        if len(self.rx_buffer) < self.Ns:
            return

        # --- INICIO DEBUG AUTOPSIA ---
        # Miramos el buffer sin consumirlo todavía para ver qué bin crudo tiene.
        window = self.rx_buffer[:self.Ns]
        
        # Hacemos la desmodulación manual (Downchirp + FFT) para ver el pico
        dechirped = window * np.conj(self.upchirp)
        spectrum = np.abs(np.fft.fft(dechirped))
        
        raw_bin = np.argmax(spectrum)
        raw_power = np.max(spectrum)
        
        print(f"\n[DEBUG-HEADER] Autopsia antes de leer:")
        print(f"{GREEN} -> Bin Crudo Detectado: {raw_bin}{RESET}")
        print(f"{GREEN} -> Potencia (MaxVal): {raw_power:.1f}{RESET}")
        # --- FIN DEBUG AUTOPSIA ---

        # Ahora sí, leemos normalmente usando tu función
        symbol = self.read_symbol()
        
        if symbol is None:
            return

        self.payload_len = symbol
        print(f"{GREEN}[HEADER] Payload length decodificado = {self.payload_len}{RESET}")

        # Filtro de seguridad
        if self.payload_len > 0 and self.payload_len < 255:
            self.payload = []
            self.current_state = StateMachine.READ_PAYLOAD
        else:
            print("[ERROR] Header inválido o desalineado. Reset.")
            self.reset_machine()

    # # def read_header(self):
    # #     # --- DEBUG VISUAL DEL HEADER ---
    # #     # Antes de consumir el buffer, miramos qué hay ahí.
    # #     if len(self.rx_buffer) < self.Ns:
    # #         return

    # #     # 1. Tomamos la ventana sin borrarla aún
    # #     window = self.rx_buffer[:self.Ns]

    # #     # 2. Hacemos la FFT manual para ver el pico crudo (Raw Bin)
    # #     # Esto nos dirá EXACTAMENTE qué número está viendo la máquina
    # #     dechirped = window * np.conj(self.downchirp) # El header viene modulado con chirps
    # #     spectrum = np.abs(np.fft.fft(dechirped))
    # #     peak_bin = np.argmax(spectrum)
    # #     peak_val = np.max(spectrum)
        
    # #     print(f"\n[DEBUG-HEADER-AUTOPSIA]")
    # #     print(f" -> Bin Crudo Detectado: {peak_bin}")
    # #     print(f" -> Potencia (MaxVal): {peak_val:.1f}")
        
    # #     # Si el bin crudo es 153, es que la señal es fuerte pero estamos desalineados.
    # #     # Si la potencia es baja, es que saltamos al vacío (ruido).
        
    # #     # 3. Ahora sí, leemos normalmente
    # #     symbol = self.read_symbol()
        
    # #     if symbol is None:
    # #         return

    # #     self.payload_len = symbol
    # #     print(f"[HEADER] Payload length = {self.payload_len}")

    # #     # Filtro de seguridad
    # #     if self.payload_len > 0 and self.payload_len < 255:
    # #         self.payload = []
    # #         self.current_state = StateMachine.READ_PAYLOAD
    # #     else:
    # #         print("[ERROR] Header inválido. Reset.")
    # #         self.reset_machine()

    # def read_header(self):
    #     symbol = self.read_symbol()
    #     if symbol is None:
    #         return

    #     self.payload_len = symbol
    #     print(f"[HEADER] Payload length = {self.payload_len}")

    #     if self.payload_len > 0:
    #         self.payload = []
    #         self.current_state = StateMachine.READ_PAYLOAD
    #     else:
    #         self.reset_machine()

    # =========================================================
    # PAYLOAD
    # =========================================================

    def read_payload(self):
        symbol = self.read_symbol()
        if symbol is None:
            return

        self.payload.append(symbol)

        if len(self.payload) >= self.payload_len:
            print(f"{RED}[PAYLOAD] Completo:{RESET}", self.payload)
            self.current_state = StateMachine.DONE

    # =========================================================
    # FSM DRIVER
    # =========================================================

    def process(self, rx):
        # Asegurarse que es complejo si viene como float
        if rx.dtype != np.complex64 and rx.dtype != np.complex128:
             rx = rx.astype(np.complex64)

        self.rx_buffer = np.concatenate((self.rx_buffer, rx))

        if self.current_state == StateMachine.SEARCH_PREAMBLE:
            self.search_preamble()

        elif self.current_state == StateMachine.SYNC:
            self.sync_sfd()

        elif self.current_state == StateMachine.READ_HEADER:
            self.read_header()

        elif self.current_state == StateMachine.READ_PAYLOAD:
            self.read_payload()

    # =========================================================
    # RESET
    # =========================================================

    def reset_machine(self):
        self.current_state = StateMachine.SEARCH_PREAMBLE
        self.rx_buffer = np.array([], dtype=float)
        self.sync_count = 0
        self.payload = []
        self.payload_len = None
        self.in_preamble_sequence = False
        self.upchirps_detected = 0