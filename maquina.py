"""
Docstring for maquinaTerminada?
Completa hasta search sfd, con todo lo que me pide el diagrama, tambien le cambie el condicional de las vidas a una desigualdad solamente, pq si era igual se reiniciaba (no se si esta bien, quizas no y lo dejo como antes)
, falta agregar o ver bien el tramo 2 en sync sfd, y cuando va a header 2,

corrige el encontrar 0 en header

agrega para mas sf compatibles
"""
import time
import math
import numpy as np
from enum import Enum, auto
from demodulador import Demodulador 
from collections import deque, Counter
import json # NUEVO
import os   # NUEVO
from datetime import datetime # NUEVO

# Colores (Mantenemos tu estilo)
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
CYAN    = "\033[96m"
BG_WHITE = "\033[47m"
REVERSE = "\033[7m"
UNDERLINE = "\033[4m"
BOLD = "\033[1m"
RESET   = "\033[0m"

class StateMachine(Enum):
    SEARCH_PREAMBLE = auto() # Buscando energía bruta
    LOCKED          = auto() # E° LOCKED: Confirmando que no es ruido (El loop del diagrama)
    SEARCH_SFD      = auto() # E° SEARCH SFD: Buscando el cambio de modulación
    SYNC_SFD        = auto() # E° SYNC SFD: Ajuste fino de tiempo
    HEADER          = auto() # E° HEADER: Lectura con votación
    HEADER2         = auto() # E° HEADER2: Corrige la ventana del SFD, y hace lo mismo que header pero con la ventana corregida 
    PAYLOAD         = auto() # E° PAYLOAD
    PAYLOAD2        = auto() # E° PAYLOAD2: Igual que payload pero con corrección de ventana según las vidas gastadas en SYNC_SFD
    DONE            = auto()

class Machine:
    def __init__(self, SF: int, BW: float, Fs: float, f0: float):
        # --- PARÁMETROS ---
        self.SF = SF
        self.BW = BW
        self.Fs = Fs
        self.f0 = f0

        self.M = 2 ** SF
        self.Ts = self.M / BW
        self.Ns = int(self.Ts * Fs) # Muestras por símbolo

        # Tiempo y chirps pre-calculados 
        self.t = np.arange(self.Ns) / Fs
        self.k = BW / self.Ts # Pendiente del chirp

        # Chirps de referencia 
        self.downchirp = np.exp(1j * 2*np.pi * ((f0+BW)*self.t - 0.5*self.k*self.t**2))
        self.upchirp   = np.exp(1j * 2*np.pi * (f0*self.t + 0.5*self.k*self.t**2))

        self.demod = Demodulador(SF, BW, Fs, f0) 
        
        # --- VARIABLES DE LA MÁQUINA DE ESTADOS ---
        self.current_state = StateMachine.SEARCH_PREAMBLE
        #self.buffer_interno = np.array([], dtype=np.complex64) #No se si lo ocupo
        
        # Configuración de Umbrales y Tolerancias
        """ E° Search Preamble """
        self.count_upchirps = 0  # Cuenta los upchirps encontrados, preambulos validos -> E° Locked
        #self.umbral_1 = 6500     # PARA VIRTUAL CABLE SIN RUIDO
        self.umbral_1 = 1000     # Ajustar según ruido real, umbral para encontrar upchirps

        """ E° Locked """
        self.umbral_2 = self.umbral_1         # Ajustar según ruido real -> E° Sync SFD, umbral para encontrar downchirps (Para virtual cable no deberia haber ruido, y vi que el maximo esta sobre 30000 en señal compleja y elen señal real es la mitad, asi que 100 podria ser un buen punto de partida)
        self.tol_freq1 = 5           # Tolerancia en bins de FFT para el estado LOCKED
        self.min_upchirps = 5        # Mínimo para pasar a buscar SFD
        self.pup_idx = 0             # Compara el upchirp encontrado en search preamble (creo aca deberia guardarse)   # "Pupidx": Posición del pico 
        self.list_pup_idx = []       # Lista de índices de upchirps encontrados  lista.append(self.pup_idx)
        self.lim_pre_lifes = 6       # Vidas máximas en LOCKED
        self.pre_lifes = 0           # Contador actual vidas

        """ E° Search SFD """
        self.tol_freq2 = 5           # Tolerancia en bins de FFT 
        self.lim_sfd_lifes = 3       # Vidas para encontrar SFD
        self.count_downchirps = 0    # Cuenta los downchirps encontrados

        """ E° Sync SFD """
        self.sfd_lifes = 0              # Contador actual vidas
        self.pdown_idx = 0              # Compara el downchirp encontrado en search sfd o locked (creo aca deberia guardarse)   # "Pdownidx": Posición del pico
        self.list_pdown_idx = []        # Lista de índices de downchirps encontrados 
        self.min_downchirps = 3         # Mínimo para pasar a buscar SFD

        """ E° Header """
        self.valor_pup_promedio = 0          # Promedio de pup_idx para comparar con header
        self.valor_pdown_promedio = 0        # Promedio de pdown_idx para comparar con header
        self.header_candidates = []     # Cantidad de headers leídos (para votación)
        self.header_reads_count = 0     # Contador actual de lecturas de header
        self.lim_header_read = 3        # Cantidad de lecturas de header para votar
        self.sto = 0
        self.cfo = 0

        """ E° Payload """
        self.payload_len = 0           # Largo del payload en bytes (leído del header)
        self.total_payload_symbols = 0 # Cantidad total de símbolos a leer (calculado a partir del payload_len)
        self.payload_symbols_leidos = 0 # Contador de símbolos de payload leídos
        self.datos_rx = []            # Lista donde se van guardando los símbolos de payload leídos

        # --- BUFFER CIRCULAR ---
        self.buffer_capacidad = 7 * self.Ns
        self.buffer = deque(maxlen=self.buffer_capacidad)

        self.window_offset = 0

        # ==========================================
        # SISTEMA DE REGISTRO (LOGGER JSON)
        # ==========================================
        self.log_dir = "logs_recepcion"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.init_rx_log()

        # === AÑADIR ESTO PARA LAS FFT ===
        self.plot_dir = "debug_plots"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        

    # --- FUNCIONES DE REGISTRO ---
    def init_rx_log(self):
        self.rx_id = int(time.time() * 1000)

        self.plot_counter = 0
        self.current_plot_dir = None

        self.rx_log = {
            "id": self.rx_id,
            "estado_final": "INCOMPLETO",
            "inicio": None,
            "fin": None,
            "vidas_preamble_gastadas": 0,
            "vidas_sfd_gastadas": 0,
            "ruta_usada": "NORMAL",
            "sto": 0.0,
            "cfo": 0.0,
            "largo_detectado": 0,
            "mensaje": "",
            "eventos": []
        }

    def record_event(self, evento, detalles=""):
        t_exacto = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.rx_log["eventos"].append(f"[{t_exacto}] {evento}: {detalles}")

    def save_rx_log(self):
        # Solo guarda si realmente detectó al menos un Upchirp (para no llenar de basura)
        if self.rx_log["inicio"] is not None:
            self.rx_log["fin"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            filepath = os.path.join(self.log_dir, f"rx_{self.rx_id}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.rx_log, f, indent=4)
    # -----------------------------

    def save_spectrum_plot(self, spectrum, filename_suffix, peak_idx=None):
        """Guarda el espectro crudo en una subcarpeta específica para este mensaje"""
        self.plot_counter += 1
        
        # 1. Si la subcarpeta del mensaje actual no existe, la creamos
        if self.current_plot_dir is None:
            self.current_plot_dir = os.path.join(self.plot_dir, f"rx_{self.rx_id}")
            if not os.path.exists(self.current_plot_dir):
                os.makedirs(self.current_plot_dir)

        # 2. Generar nombre de archivo
        str_peak = f"_peak_{peak_idx}" if peak_idx is not None else ""
        filename = f"{self.plot_counter:03d}_{filename_suffix}{str_peak}.npy"
        
        # 3. Guardar dentro de la subcarpeta del mensaje
        filepath = os.path.join(self.current_plot_dir, filename)
        np.save(filepath, spectrum)

    # ================================================================
    #   FUNCIONES DE PROCESAMIENTO (Dechirp + FFT + Peak)
    # ================================================================

    def dechirp_fft(self, window, chirp_ref):
        dechirped = window * np.conj(chirp_ref)
        spectrum = np.abs(np.fft.fft(dechirped))
        return spectrum
    
    def get_peak(self, spectrum):
        """
        Busca el pico SOLO en los bordes y calcula el ruido usando el centro.
        Retorna: (indice_pico, valor_pico, valor_promedio_ruido)
        """
        Ns = len(spectrum)
        margen = self.M // 2 # 128 para SF=8, ajustable según SF

        # 1. Extraemos las zonas de interés (Señal posible)
        # Zona baja: 0 a 128
        low_region = spectrum[0 : margen]
        # Zona alta: 30592 a 30720
        high_region = spectrum[Ns - margen : Ns]

        # 2. Extraemos la zona central (Solo Ruido)
        # Esto es vital: si el pico en el borde no es mucho mayor que esto, es falsa alarma.
        noise_region = spectrum[margen : Ns - margen]
        noise_floor = np.mean(noise_region) 

        # 3. Buscamos el máximo en los bordes
        low_idx = np.argmax(low_region)
        low_peak = low_region[low_idx]

        high_idx = np.argmax(high_region)
        high_peak = high_region[high_idx]

        # 4. Decidimos el ganador
        if low_peak > high_peak:
            best_idx = low_idx             # Ya es el índice correcto (0..127)
            best_peak = low_peak
        else:
            best_idx = (Ns - margen) + high_idx# Ajustamos offset (30592..30719)
            best_peak = high_peak

        return best_idx, best_peak, noise_floor

    
    # def get_peak(self, spectrum):
    #     # Tomamos el máximo crudo
    #     idx = np.argmax(spectrum)
    #     peak = spectrum[idx]
    #     return idx, peak

    # ================================================================
    #   UTILIDADES DEL NÚCLEO
    # ================================================================

    def add_data(self, samples): 
        self.buffer.extend(samples)

    def consume(self, n_samples):
        """
        Borra (consume) 'n_samples' del inicio del buffer.
        Equivale a avanzar el tiempo en la máquina.
        """
        # Protección: No intentar borrar más de lo que existe
        limit = min(len(self.buffer), n_samples)
        
        for _ in range(limit):
            self.buffer.popleft()

    def get_window(self):
        """
        Extrae Ns muestras empezando desde buffer[window_offset].
        NO borra datos.
        """
        # Necesitamos tener suficientes datos para cubrir el offset + Ns
        if len(self.buffer) < self.Ns + self.window_offset:
            return None
            
        full_buffer = np.array(self.buffer)
        
        start = self.window_offset
        end   = start + self.Ns
        
        return full_buffer[start : end]
    

    def reset_machine(self):

        self.save_rx_log()

        """ E° Search Preamble """
        self.count_upchirps = 0

        """ E° Locked """
        self.pup_idx = 0           
        self.list_pup_idx = []       
        self.pre_lifes = 0  

        """ E° Search SFD """
        self.count_downchirps = 0

        """ E° Sync SFD """
        self.sfd_lifes = 0              
        self.pdown_idx = 0             
        self.list_pdown_idx = [] 

        """ E° Header """
        self.valor_pup_promedio = 0        
        self.valor_pdown_promedio = 0      
        self.header_candidates = []     
        self.header_reads_count = 0       
        self.sto = 0
        self.cfo = 0

        """ E° Payload """
        self.payload_len = 0          
        self.total_payload_symbols = 0 
        self.payload_symbols_leidos = 0 
        self.datos_rx = [] 

        self.current_state = StateMachine.SEARCH_PREAMBLE
        self.window_offset = 0

        self.init_rx_log()


    # ================================================================
    #   LÓGICA DE ESTADOS 
    # ================================================================ 

    def search_preamble(self):
        """
        E° SEARCH PREAMBLE:
        Busca energía (Upchirps) SOLO en las zonas seguras:
        - Zona Baja: bins 0 a 128
        - Zona Alta: bins 30592 a 30720
        """        
        # 1. Verificar buffer suficiente
        target_offset = 3 * self.Ns 
        muestras_necesarias = target_offset + self.Ns
        if len(self.buffer) < muestras_necesarias:
            return
        
        # Queremos que 'start' sea donde empieza el 3er símbolo
        self.window_offset = target_offset

        # Ahora llamamos a la función
        window = self.get_window()
        
        # Dechirp con DOWNCHIRP (porque buscamos un Upchirp)
        #t_i = time.perf_counter()
        spectrum = self.dechirp_fft(window, self.upchirp)
        #t_f = time.perf_counter()
        #print(f"{YELLOW}[SEARCH PREAMBLE] Tiempo de procesamiento: {(t_f - t_i)*1000:.2f} ms{RESET}")

        best_idx, best_peak, noise_floor = self.get_peak(spectrum)
        # Ajustamos el índice para que sea relativo a 0 (en vez de 30592..30719 -128 ... -1)
        if best_idx > self.Ns // 2: 
            best_bin = best_idx - self.Ns
        else: 
            best_bin = best_idx

        # Verificamos si el pico es suficientemente alto respecto al ruido
        if best_peak > self.umbral_1: #and best_peak > 5 * noise_floor: # El factor 5 es un margen de seguridad, ajustable
            if self.count_upchirps == 0:
                self.rx_log["inicio"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.record_event("SEARCH_PREAMBLE", f"Primer Upchirp detecado en bin {best_bin} (Peak: {best_peak:.0f})")
            self.count_upchirps += 1
            self.pup_idx = best_bin # Guardamos el índice del pico encontrado (el normalizado, no el crudo)
            self.list_pup_idx.append(self.pup_idx)
            print(f"{GREEN}[SEARCH PREAMBLE] Upchirp detectado en Pico: {best_peak:.2f}, Ruido: {noise_floor:.2f}, Índice: {best_bin}{RESET}")
            print(f"{RESET}{BOLD}{REVERSE} [SEARCH PREAMBLE] Ruta 1 {RESET}")
            print(f"{RED}[SEARCH PREAMBLE] Pasando a LOCKED... (Count Upchirps: {self.count_upchirps}){RESET}")
            self.consume(self.Ns)
            self.current_state = StateMachine.LOCKED # Avanzamos para no leer lo mismo

        else:
            print(f"{RESET}{BOLD}{REVERSE} [SEARCH PREAMBLE] Ruta 2 {RESET}")
            #print(f"{GREEN}[SEARCH PREAMBLE] Upchirp NO det en Pico: {best_peak:.2f}, Ruido: {noise_floor:.2f}, Índice: {best_bin}{RESET}")
            step = self.Ns // 4
            self.consume(step) # Avanzamos un cuarto de símbolo para no perder la sincronización

    def locked(self):
        """
        E° LOCKED: 
        1. Busca pico SOLO en Zonas Seguras.
        2. Verifica si el pico está cerca del anterior.
        3. Si es un Upchirp válido, avanza el buffer y sigue buscando hasta el minimo y pasa a SEARCH_SFD.
        4. Si no encuentra un Upchirp pero encuentra un Downchirp fuerte, puede ser el SFD, así que pasa a SYNC_SFD.
        5. Si no encuentra nada, incrementa "vidas". Si se acaban, vuelve a SEARCH_PREAMBLE. (AUN NO IMPLEMENTADO)
        """
         # 1. Verificar buffer suficiente
        target_offset = 3 * self.Ns 
        muestras_necesarias = target_offset + self.Ns
        if len(self.buffer) < muestras_necesarias:
            return
        
        # Queremos que 'start' sea donde empieza el 3er símbolo
        self.window_offset = target_offset

        window = self.get_window() # Ventana en la mitad
        
        # 3. Dechirp (Seguimos buscando Upchirps)
        spectrum = self.dechirp_fft(window, self.upchirp)
        
        # 4. Análisis
        # Aquí usamos get_peak, pero ahora esperamos que el pico esté 
        # EN EL MISMO LUGAR que el anterior (self.pup_idx) con pequeña tolerancia.
        best_idx, best_peak, noise_floor = self.get_peak(spectrum)
        self.save_spectrum_plot(spectrum, "LOCKED", best_idx) # <-- AÑADIR


        # Valores entre -128..128 
        if best_idx > self.Ns // 2: 
            best_bin = best_idx - self.Ns
        else: 
            best_bin = best_idx
        
        # Calculamos la diferencia con el pico original (Tracking)
        diff = abs(int(self.pup_idx) - int(best_bin))

        # 5. Lógica de Decisión
        es_upchirp =  (diff <= self.tol_freq1) # (best_peak > self.umbral_1) and
        
        if es_upchirp:
            print(f"{RESET}{BLUE}{BOLD}{REVERSE} [LOCKED] Ruta 1 {RESET}")
            self.count_upchirps += 1
            # Actualizamos la referencia (tracking suave) (No se si deba actualizarlo)
            # self.pup_idx = best_bin
            self.list_pup_idx.append(best_bin) # Guardamos el índice del pico encontrado Lista de pup_idx para análisis posterior
            print(f"{CYAN}[LOCKED] Upchirp #{self.count_upchirps} confirmado. Bin: {best_bin} Peak: {best_peak:.0f}{RESET}")

                
            # ¿Ya tenemos suficientes para buscar el SFD?
            if self.count_upchirps >= self.min_upchirps:
                print(f"{RESET}{BLUE}{BOLD}{REVERSE} [LOCKED] Ruta 1.1 {RESET}")
                promedio = sum(self.list_pup_idx) / len(self.list_pup_idx) #np.rint(sum(self.list_pup_idx) / len(self.list_pup_idx))  ## CAMBIO
                self.valor_pup_promedio = promedio

                self.record_event("LOCKED_EXITO", f"Preámbulo sólido. Promedio: {promedio:.2f}")

                print(f"{GREEN}[LOCKED] ---> Preámbulo sólido. Promedio de Upchirps: {promedio}{RESET}")
                print(f"{RED}[LOCKED] Pasando a SEARCH_SFD... (Count Upchirps: {self.count_upchirps}){RESET}")
                # AVANZAMOS:
                self.consume(self.Ns) # Avanzamos una ventana completa (Ns) para buscar el siguiente símbolo
                self.current_state = StateMachine.SEARCH_SFD
            else:
                print(f"{RESET}{BLUE}{BOLD}{REVERSE} [LOCKED] Ruta 1.2 {RESET}")
                # AVANZAMOS:
                self.consume(self.Ns) # Avanzamos una ventana completa (Ns) para buscar el siguiente símbolo
                self.current_state = StateMachine.LOCKED  # Seguimos en Locked buscando más upchirps para confirmar el preámbulo

        else:
            print(f"{RESET}{BLUE}{BOLD}{REVERSE} [LOCKED] Ruta 2 {RESET}")
            # ¿Y si no es un Upchirp?
            # Puede ser el SFD (Downchirp) o que perdimos la señal.
            self.pre_lifes += 1

            self.rx_log["vidas_preamble_gastadas"] = self.pre_lifes
            self.record_event("LOCKED_VIDA_PERDIDA", f"Vida {self.pre_lifes}/{self.lim_pre_lifes} gastada")

            print(f"{YELLOW}[LOCKED] No se detectó un Upchirp válido. Vidas: {self.pre_lifes}/{self.lim_pre_lifes}{RESET}")
            if self.pre_lifes >= self.lim_pre_lifes:
                print(f"{RESET}{BLUE}{BOLD}{REVERSE} [LOCKED] Ruta 2.2 {RESET}")
                self.record_event("LOCKED_FALLO", "Límite de vidas alcanzado. Reiniciando.")
                self.reset_machine()
                print(f"{RED}[LOCKED] Señal perdida. Volviendo a Search Preamble.{RESET}")

            else:
                print(f"{RESET}{BLUE}{BOLD}{REVERSE} [LOCKED] Ruta 2.1 {RESET}")
                # Verificamos si es SFD (Downchirp)
                spec_down = self.dechirp_fft(window, self.downchirp) # Invertimos referencia
                best_idx_sfd, best_peak_sfd, _ = self.get_peak(spec_down)

                # Valores entre -128..128
                if best_idx_sfd > self.Ns // 2: 
                    best_bin_sfd = best_idx_sfd - self.Ns
                else: 
                    best_bin_sfd = best_idx_sfd
                
                if best_peak_sfd > self.umbral_2:
                    print(f"{RESET}{BLUE}{BOLD}{REVERSE} [LOCKED] Ruta 2.1.1 {RESET}")
                    self.record_event("LOCKED_SFD_PREMATURO", f"SFD detectado en bin {best_bin_sfd}")
                    print(f"{MAGENTA}[LOCKED] ¡Energía Downchirp detectada! Posible SFD {best_bin_sfd} Peak: {best_peak_sfd:.0f}{RESET}")
                    self.pdown_idx = best_bin_sfd # Guardamos el índice del pico encontrado (el normalizado, no el crudo)
                    promedio = sum(self.list_pup_idx) / len(self.list_pup_idx) #np.rint(sum(self.list_pup_idx) / len(self.list_pup_idx)) ## CAMBIO
                    self.valor_pup_promedio = promedio # Para comparar con el header
                    self.list_pdown_idx.append(self.pdown_idx)
                    self.count_downchirps += 1
                    self.consume(self.Ns) # Avanzamos para no leer lo mismo
                    print(f"{RED}[LOCKED] Pasando a SYNC_SFD... (Count Downchirps: {self.count_downchirps}){RESET}")
                    self.current_state = StateMachine.SYNC_SFD
                else:
                    print(f"{RESET}{BLUE}{BOLD}{REVERSE} [LOCKED] Ruta 2.1.2 {RESET}")
                    print(f"{RED}[LOCKED] Señal perdida. Volviendo a LOCKED.{RESET}")
                    self.consume(self.Ns) # Avanzamos para no leer lo mismo
                    self.current_state = StateMachine.LOCKED    
                    

    def search_sfd(self):
         # 1. Verificar buffer suficiente
        target_offset = 3 * self.Ns 
        muestras_necesarias = target_offset + self.Ns
        if len(self.buffer) < muestras_necesarias:
            return
        
        # Queremos que 'start' sea donde empieza el 3er símbolo
        self.window_offset = target_offset

        window = self.get_window() # Ventana en la mitad

        # Buscamoa a ver si hay upchirp
        sectrum = self.dechirp_fft(window, self.upchirp)
        best_idx_up, best_peak_up, _ = self.get_peak(sectrum)
        # Normalizamos el índice para que sea relativo a 0 (en vez de 30592..30719)
        if best_idx_up > self.Ns // 2: 
            best_bin_up = best_idx_up - self.Ns
        else: 
           best_bin_up = best_idx_up

        # Calculamos la diferencia con el pico original (Tracking)
        diff = abs(int(self.pup_idx) - int(best_bin_up))

        # 5. Lógica de Decisión
        es_upchirp =  (diff <= self.tol_freq1) # (best_peak_up > self.umbral_1) and
        
        if es_upchirp:
            print(f"{RESET}{MAGENTA}{BOLD}{REVERSE} Ruta 0.1 {RESET}")
            print(f"{RESET}{GREEN}{BOLD}{REVERSE} [SEARCH SFD] Upchirp detectado! {RESET}")
            self.list_pup_idx.append(self.pup_idx)
            self.sfd_lifes = 0 # Reiniciamos vidas porque encontramos un upchirp fuerte
            self.count_upchirps += 1
            self.list_pup_idx.append(best_bin_up) # Guardamos el índice del pico encontrado Lista de pup_idx para análisis posterior
            print(f"{CYAN}[SEARCH SFD] Upchirp #{self.count_upchirps} confirmado. Bin: {best_bin_up} Peak: {best_peak_up:.0f}{RESET}")
            promedio = sum(self.list_pup_idx) / len(self.list_pup_idx) #np.rint(sum(self.list_pup_idx) / len(self.list_pup_idx))  ## CAMBIO
            self.valor_pup_promedio = promedio
            print(f"{GREEN}[SEARCH SFD] ---> Promedio de Upchirps Actualizado: {promedio}{RESET}")
            self.consume(self.Ns) # Avanzamos para no leer lo mismo
            self.current_state = StateMachine.SEARCH_SFD 
        else:
            print(f"{RESET}{MAGENTA}{BOLD}{REVERSE} Ruta 0.2 {RESET}")
            # 3. Dechirp (Seguimos buscando Downchirps)
            spec_down = self.dechirp_fft(window, self.downchirp)
            best_idx_sfd, best_peak_sfd, _ = self.get_peak(spec_down)
            # Normalizamos el índice para que sea relativo a 0 (en vez de 30592..30719)
            if best_idx_sfd > self.Ns // 2: 
                best_bin_sfd = best_idx_sfd - self.Ns
            else: 
                best_bin_sfd = best_idx_sfd
                
            if best_peak_sfd > self.umbral_2:
                print(f"{RESET}{MAGENTA}{BOLD}{REVERSE} Ruta 1 {RESET}")

                self.record_event("SEARCH_SFD_EXITO", f"Downchirp 1 detectado en bin {best_bin_sfd}")
                
                print(f"{GREEN}[SEARCH SFD] Vidas gastadas {self.sfd_lifes}/{self.lim_sfd_lifes} {RESET}")
                print(f"{MAGENTA}[SEARCH SFD] ¡Energía Downchirp detectada! Posible SFD {best_bin_sfd} Peak: {best_peak_sfd:.0f}{RESET}")
                self.pdown_idx = best_bin_sfd # Guardamos el índice del pico encontrado (el normalizado, no el crudo)
                self.list_pdown_idx.append(self.pdown_idx)
                self.count_downchirps += 1
                self.sfd_lifes = 0 # Reiniciamos vidas porque encontramos un downchirp fuerte
                print(f"{RED}[SEARCH SFD] Pasando a SYNC_SFD... (Count Downchirps: {self.count_downchirps}){RESET}")
                self.current_state = StateMachine.SYNC_SFD
                self.consume(self.Ns) # Avanzamos para no leer lo mismo
            else:
                print(f"{RESET}{MAGENTA}{BOLD}{REVERSE} Ruta 2 {RESET}")
                self.sfd_lifes += 1
                self.rx_log["vidas_sfd_gastadas"] = self.sfd_lifes
                self.record_event("SEARCH_SFD_VIDA_PERDIDA", f"Vida {self.sfd_lifes}/{self.lim_sfd_lifes}")

                print(f"{YELLOW}[SEARCH SFD] No se detectó un Downchirp válido. Vidas: {self.sfd_lifes}/{self.lim_sfd_lifes}{RESET}")
                if self.sfd_lifes >= self.lim_sfd_lifes:
                    print(f"{RESET}{MAGENTA}{BOLD}{REVERSE} Ruta 2.1 {RESET}")
                    self.record_event("SEARCH_SFD_FALLO", "Límite de vidas SFD alcanzado")
                    print(f"{RED}[SEARCH SFD] No se encontró el SFD. Volviendo a Search Preamble.{RESET}")
                    self.reset_machine()
                else:
                    print(f"{RESET}{MAGENTA}{BOLD}{REVERSE} Ruta 2.2 {RESET}")
                    self.consume(self.Ns) 
                    self.current_state = StateMachine.SEARCH_SFD # Seguimos buscando el SFD
                
    ####################################################################################################################################

    # ARRIBA ESTA COMPLETO, SE PROBO PASAR POR TODAS LAS RUTAS (JUGANDO CON MINIMOS, O CON LA SEÑAL DE PREAMBULO)

    ######################################################################################################################################

    def sync_sfd(self):
         # 1. Verificar buffer suficiente
        target_offset = 3 * self.Ns 
        muestras_necesarias = target_offset + self.Ns
        if len(self.buffer) < muestras_necesarias:
            return
        
        # Queremos que 'start' sea donde empieza el 3er símbolo
        self.window_offset = target_offset

        window = self.get_window() # Ventana en la mitad

        # 3. Dechirp (Seguimos buscando Downchirps)
        spec_down = self.dechirp_fft(window, self.downchirp)
        best_idx_sfd, best_peak_sfd, _ = self.get_peak(spec_down)
        self.save_spectrum_plot(spec_down, "SYNC_SFD", best_idx_sfd) # <-- AÑADIR

        # Normalizamos el índice para que sea relativo a 0 (en vez de 30592..30719)
        if best_idx_sfd > self.Ns // 2: 
            best_bin_sfd = best_idx_sfd - self.Ns
        else: 
           best_bin_sfd = best_idx_sfd

        # Calculamos la diferencia con el pico original (Tracking)
        diff = abs(int(self.pdown_idx) - int(best_bin_sfd))

        # 5. Lógica de Decisión
        es_downchirp =  (diff <= self.tol_freq2) # (best_peak_sfd > self.umbral_2) and
            
        if es_downchirp:
            print(f"{RESET}{GREEN}{BOLD}{REVERSE} [SYNC SFD] Ruta 1 {RESET}")
            self.count_downchirps += 1
            # self.pdown_idx = best_bin_sfd # Actualizamos la referencia (tracking suave)
            self.list_pdown_idx.append(best_bin_sfd) # Guardamos el índice del pico
            print(f"{CYAN}[SYNC SFD] Downchirp confirmado. Bin: {best_bin_sfd} Peak: {best_peak_sfd:.0f}{RESET}")

            self.consume(self.Ns) # Avanzamos una ventana completa (Ns) para buscar el siguiente símbolo
            if self.count_downchirps >= self.min_downchirps:
                print(f"{RESET}{GREEN}{BOLD}{REVERSE} [SYNC SFD] Ruta 1.1 {RESET}")
                promedio = sum(self.list_pdown_idx) / len(self.list_pdown_idx) #np.rint(sum(self.list_pdown_idx) / len(self.list_pdown_idx)) ## CAMBIO
                self.valor_pdown_promedio = promedio
                self.record_event("SYNC_SFD_EXITO", f"SFD Sólido. Promedio: {promedio:.2f}")

                print(f"{GREEN}[SYNC SFD] ---> SFD sólido. Promedio de Downchirps: {promedio}{RESET}")
                print(f"{RED}[SYNC SFD] Pasando a HEADER... (Count Downchirps: {self.count_downchirps}){RESET}")
                self.current_state = StateMachine.HEADER
            else:
                print(f"{RESET}{GREEN}{BOLD}{REVERSE} [SYNC SFD] Ruta 1.2 {RESET}")
                self.current_state = StateMachine.SYNC_SFD # Seguimos en SYNC_SFD buscando más downchirps para confirmar el SFD

        else:
            print(f"{RESET}{GREEN}{BOLD}{REVERSE} [SYNC SFD] Ruta 2 {RESET}")
            # Para tomar esta ruta, colocar 9 upchirps y 1 downchirp
            self.sfd_lifes += 1
            self.rx_log["vidas_sfd_gastadas"] = self.sfd_lifes

            self.count_downchirps += 1 
            self.record_event("SYNC_SFD_VIDA_PERDIDA", f"Ruta 2 iniciada. Vida {self.sfd_lifes}")

            if self.count_downchirps >= self.min_downchirps:
                print(f"{RESET}{GREEN}{BOLD}{REVERSE} [SYNC SFD] Ruta 2.1 {RESET}")
                promedio = sum(self.list_pdown_idx) / len(self.list_pdown_idx) #np.rint(sum(self.list_pdown_idx) / len(self.list_pdown_idx)) ## CAMBIO
                self.valor_pdown_promedio = promedio
                print(f"{GREEN}[SYNC SFD] ---> SFD sólido. Promedio de Downchirps: {promedio}{RESET}")
                print(f"{RED}[SYNC SFD] Pasando a HEADER2... {RESET}")
                self.rx_log["ruta_usada"] = "RUTA 2 (HEADER2)"

                self.record_event("SYNC_SFD_EXITO_RUTA2", f"SFD Sólido por vida. Promedio: {promedio:.2f}")

                self.consume(self.Ns) # Avanzamos una ventana completa (Ns) para buscar el siguiente símbolo
                self.current_state = StateMachine.HEADER2
            else:
                print(f"{RESET}{GREEN}{BOLD}{REVERSE} [SYNC SFD] Ruta 2.2 {RESET}")
                self.consume(self.Ns) # Avanzamos una ventana completa (Ns) para buscar el siguiente símbolo
                self.current_state = StateMachine.SYNC_SFD # Seguimos buscando el SFD
                    

    def header(self):
        k_up = self.valor_pup_promedio
        k_down = self.valor_pdown_promedio

        sto_bins = (k_up - k_down) / 2
        self.sto = sto_bins
        self.rx_log["sto"] = sto_bins
        print(f"{GREEN}[HEADER] STO Calculado: {sto_bins:.1f} bins{RESET}")
        sto_samples = int(sto_bins * (self.Fs / self.BW))

        cfo_bins = (k_up + k_down) / 2 
        self.cfo = cfo_bins
        self.rx_log["cfo"] = cfo_bins
        print(f"{GREEN}[HEADER] CFO Calculado: {cfo_bins:.1f} bins{RESET}")

        # 1. Verificar buffer suficiente
        target_offset = 3 * self.Ns 
        muestras_necesarias = target_offset + self.Ns + abs(sto_samples) + 100 # Necesitamos espacio para ajustar la ventana con el STO
        if len(self.buffer) < muestras_necesarias:
            return
        
        # Queremos que 'start' sea donde empieza el 3er símbolo
        self.window_offset = target_offset - sto_samples # Ajustamos la ventana con el STO calculado

        window = self.get_window() # Ventana 
        if window is None: return

        # --- C. DECHIRP Y FFT ---
        spectrum = self.dechirp_fft(window, self.upchirp)
        
        raw_bin = np.argmax(spectrum)
        raw_power = np.max(spectrum)
        self.save_spectrum_plot(spectrum, "HEADER", raw_bin) # <-- AÑADIR

        val_corregido = int(round(raw_bin - cfo_bins)) % self.M

        # --- E. GUARDAR Y AVANZAR ---
        self.header_candidates.append(val_corregido)
        self.header_reads_count += 1
        
        print(f"{BLUE}[HEADER] Símbolo {self.header_reads_count}/{self.lim_header_read}: {val_corregido} (Power: {raw_power:.0f}){RESET}")
        
        self.consume(self.Ns) # Avanzamos un símbolo

        if self.header_reads_count >= self.lim_header_read:
            print(f"{RESET}{YELLOW}{BOLD}{REVERSE} [LOCKED] Ruta 1 {RESET}")
            print(f"{GREEN}[HEADER] Finalizando lectura. Candidatos: {self.header_candidates}{RESET}")
            
            # --- LÓGICA DEL DIAGRAMA (VOTACIÓN + MAX) ---
            
            # 1. Contamos las ocurrencias
            # Ej: Si candidates es [9, 9, 13], counts será {9: 2, 13: 1}
            conteo = Counter(self.header_candidates)
            
            # Obtenemos el más común (Moda)
            # most_common(1) devuelve una lista [(valor, repeticiones)]
            ganador, votos = conteo.most_common(1)[0]
            
            length_decidido = 0

            # 2. Árbol de decisión
            if ganador != 0: # Si el ganador no es 0 (que sería sospechoso, podría ser ruido)
                if votos == 2:
                    print(f"{RESET}{YELLOW}{BOLD}{REVERSE} [LOCKED] Ruta 1.1a {RESET}")
                    # CASO A: ¿Se repite alguno? -> SI -> Usamos la Moda
                    length_decidido = ganador
                    self.consume(self.Ns)
                    print(f"{GREEN}[HEADER] ¡Consenso detectado! Ganador por mayoría: {length_decidido}{RESET}")

                elif votos > 2:
                    print(f"{RESET}{YELLOW}{BOLD}{REVERSE} [LOCKED] Ruta 1.1b (Mayoría clara) {RESET}")
                    # CASO A: ¿Se repite alguno? -> SI -> Usamos la Moda
                    length_decidido = ganador 
                    print(f"{GREEN}[HEADER] ¡Consenso claro detectado! Ganador por mayoría: {length_decidido} (Votos: {votos}){RESET}")

                else:
                    print(f"{RESET}{YELLOW}{BOLD}{REVERSE} [LOCKED] Ruta 1.2 {RESET}")
                    # CASO B: ¿Se repite alguno? -> NO -> Usamos el MÁXIMO 
                    # Ej: [8, 9, 12] -> Gana 12
                    length_decidido = max(self.header_candidates)
                    print(f"{YELLOW}[HEADER] Sin consenso (todos distintos). Usando el MÁXIMO: {length_decidido}{RESET}")
            elif ganador == 0 and votos == 2:
                print(f"{RESET}{YELLOW}{BOLD}{REVERSE} [LOCKED] Ruta 1.3 {RESET}")
                length_decidido = max(self.header_candidates)
                self.consume(2*self.Ns)
                print(f"{YELLOW}[HEADER] Sin consenso (todos distintos). Usando el MÁXIMO: {length_decidido}{RESET}")

            else:
                    print(f"{RESET}{YELLOW}{BOLD}{REVERSE} [LOCKED] Ruta 1.2 {RESET}")
                    # CASO B: ¿Se repite alguno? -> NO -> Usamos el MÁXIMO (según tu dibujo)
                    # Ej: [8, 9, 12] -> Gana 12
                    length_decidido = max(self.header_candidates)
                

            # --- APLICACIÓN DEL RESULTADO ---
            
            # Asignamos el largo decidido
            self.payload_len = length_decidido
            bits_totales = self.payload_len * 8
            self.rx_log["largo"] = self.payload_len # <-- AÑADIR
            self.record_event("HEADER_DECODIFICADO", f"Largo: {self.payload_len}") # <-- AÑADIR
            
            # Verificación de seguridad (por si el largo es 0 o muy loco)
            if self.payload_len > 0 and self.payload_len < 255: # Límite razonable para LoRa
                print(f"{GREEN}[HEADER] ---> LARGO CONFIGURADO: {self.payload_len} bytes.{RESET}")
                
                # Configuración para Payload
                self.current_state = StateMachine.PAYLOAD
                self.total_payload_symbols = math.ceil(bits_totales / self.SF)
                self.payload_symbols_leidos = 0
                self.datos_rx = []
                
            else:
                print(f"{RED}[HEADER] Error: Largo inválido detectado ({self.payload_len}). Reiniciando.{RESET}")
                self.reset_machine()
            
            # Limpieza
            self.header_candidates = [] 
            self.header_reads_count = 0
            
        else:
            print(f"{RESET}{YELLOW}{BOLD}{REVERSE} [LOCKED] Ruta 2 {RESET}")
            # Seguimos leyendo hasta completar los intentos
            pass

    def header2(self):
        # Correccion sync window N SFD
        vidas = self.sfd_lifes

        correction = vidas * self.Ns # Cada vida perdida es un de símbolo de corrección
        
        k_up = self.valor_pup_promedio
        k_down = self.valor_pdown_promedio

        sto_bins = (k_up - k_down) / 2
        self.sto = sto_bins
        self.rx_log["sto"] = sto_bins
        print(f"{GREEN}[HEADER] STO Calculado: {sto_bins:.1f} bins{RESET}")
        sto_samples = int(sto_bins * (self.Fs / self.BW))

        cfo_bins = (k_up + k_down) / 2 
        self.cfo = cfo_bins
        self.rx_log["cfo"] = cfo_bins
        print(f"{GREEN}[HEADER] CFO Calculado: {cfo_bins:.1f} bins{RESET}")

        # 1. Verificar buffer suficiente
        target_offset = 3 * self.Ns 
        muestras_necesarias = target_offset + self.Ns + abs(sto_samples) + 100 + correction # Necesitamos espacio para ajustar la ventana con el STO
        if len(self.buffer) < muestras_necesarias:
            return
        
        # Queremos que 'start' sea donde empieza el 3er símbolo
        self.window_offset = target_offset - sto_samples - correction # Ajustamos la ventana con el STO calculado

        window = self.get_window() # Ventana 
        if window is None: return

        # --- C. DECHIRP Y FFT ---
        spectrum = self.dechirp_fft(window, self.upchirp)
        
        raw_bin = np.argmax(spectrum)
        raw_power = np.max(spectrum)
        self.save_spectrum_plot(spectrum, "HEADER", raw_bin) # <-- AÑADIR

        val_corregido = int(round(raw_bin - cfo_bins)) % self.M

        # --- E. GUARDAR Y AVANZAR ---
        self.header_candidates.append(val_corregido)
        self.header_reads_count += 1
        
        print(f"{BLUE}[HEADER] Símbolo {self.header_reads_count}/{self.lim_header_read}: {val_corregido} (Power: {raw_power:.0f}){RESET}")
        
        self.consume(self.Ns) # Avanzamos un símbolo

        if self.header_reads_count >= self.lim_header_read:
            print(f"{RESET}{YELLOW}{BOLD}{REVERSE} [LOCKED] Ruta 1 {RESET}")
            print(f"{GREEN}[HEADER] Finalizando lectura. Candidatos: {self.header_candidates}{RESET}")
            
            # --- LÓGICA DEL DIAGRAMA (VOTACIÓN + MAX) ---
            
            # 1. Contamos las ocurrencias
            # Ej: Si candidates es [9, 9, 13], counts será {9: 2, 13: 1}
            conteo = Counter(self.header_candidates)
            
            # Obtenemos el más común (Moda)
            # most_common(1) devuelve una lista [(valor, repeticiones)]
            ganador, votos = conteo.most_common(1)[0]
            
            length_decidido = 0

            # 2. Árbol de decisión
            if ganador != 0: # Si el ganador no es 0 (que sería sospechoso, podría ser ruido)
                if votos == 2:
                    print(f"{RESET}{YELLOW}{BOLD}{REVERSE} [LOCKED] Ruta 1.1a {RESET}")
                    # CASO A: ¿Se repite alguno? -> SI -> Usamos la Moda
                    length_decidido = ganador 
                    self.consume(self.Ns)
                    print(f"{GREEN}[HEADER] ¡Consenso detectado! Ganador por mayoría: {length_decidido}{RESET}")

                elif votos > 2:
                    print(f"{RESET}{YELLOW}{BOLD}{REVERSE} [LOCKED] Ruta 1.1b (Mayoría clara) {RESET}")
                    # CASO A: ¿Se repite alguno? -> SI -> Usamos la Moda
                    length_decidido = ganador 
                    print(f"{GREEN}[HEADER] ¡Consenso claro detectado! Ganador por mayoría: {length_decidido} (Votos: {votos}){RESET}")

                else:
                    print(f"{RESET}{YELLOW}{BOLD}{REVERSE} [LOCKED] Ruta 1.2 {RESET}")
                    # CASO B: ¿Se repite alguno? -> NO -> Usamos el MÁXIMO (según tu dibujo)
                    # Ej: [8, 9, 12] -> Gana 12
                    length_decidido = max(self.header_candidates)
                    print(f"{YELLOW}[HEADER] Sin consenso (todos distintos). Usando el MÁXIMO: {length_decidido}{RESET}")
            elif ganador == 0 and votos == 2:
                print(f"{RESET}{YELLOW}{BOLD}{REVERSE} [LOCKED] Ruta 1.3 {RESET}")
                length_decidido = max(self.header_candidates) 
                self.consume(2*self.Ns)
                print(f"{YELLOW}[HEADER] Sin consenso (todos distintos). Usando el MÁXIMO: {length_decidido}{RESET}")

            else:
                    print(f"{RESET}{YELLOW}{BOLD}{REVERSE} [LOCKED] Ruta 1.2 {RESET}")
                    # CASO B: ¿Se repite alguno? -> NO -> Usamos el MÁXIMO (según tu dibujo)
                    # Ej: [8, 9, 12] -> Gana 12
                    length_decidido = max(self.header_candidates)

            # --- APLICACIÓN DEL RESULTADO ---
            
            # Asignamos el largo decidido
            self.payload_len = length_decidido
            bits_totales = self.payload_len * 8
            self.rx_log["largo"] = self.payload_len # <-- AÑADIR
            self.record_event("HEADER_DECODIFICADO", f"Largo: {self.payload_len}") # <-- AÑADIR
            
            # Verificación de seguridad (por si el largo es 0 o muy loco)
            if self.payload_len > 0 and self.payload_len < 255: # Límite razonable para LoRa
                print(f"{GREEN}[HEADER] ---> LARGO CONFIGURADO: {self.payload_len} bytes.{RESET}")
                
                # Configuración para Payload
                self.current_state = StateMachine.PAYLOAD2
                self.total_payload_symbols = math.ceil(bits_totales / self.SF)
                self.payload_symbols_leidos = 0
                self.datos_rx = []
                
            else:
                print(f"{RED}[HEADER] Error: Largo inválido detectado ({self.payload_len}). Reiniciando.{RESET}")
                self.reset_machine()
            
            # Limpieza
            self.header_candidates = [] 
            self.header_reads_count = 0
            
        else:
            print(f"{RESET}{YELLOW}{BOLD}{REVERSE} [LOCKED] Ruta 2 {RESET}")
            # Seguimos leyendo hasta completar los intentos
            pass
         
    def payload(self):
        """
        E° PAYLOAD:
        Lee el mensaje usando la sincronización capturada en el Header.
        """
        # 1. Recuperamos la sincronización guardada
        sto_bins = self.sto
        sto_samples = int(sto_bins * (self.Fs / self.BW))
        cfo_bins    = self.cfo
        
        # 2. Configurar Ventana
        # Siempre apuntamos a 2*Ns (el "presente" del buffer)
        target_offset = 3 * self.Ns 
        
        # Verificamos buffer (+100 de margen por seguridad)
        if len(self.buffer) < target_offset + self.Ns + abs(sto_samples) + 100:
            return 
        
        # APLICAMOS LA CORRECCIÓN DE TIEMPO (RESTANDO, igual que en Header)
        self.window_offset = target_offset - sto_samples 

        window = self.get_window() 
        if window is None: return

        # 3. Dechirp (Datos son Upchirps)
        spectrum = self.dechirp_fft(window, self.upchirp)

        # int(bytes * paykloadlenght / sf) se tiene que tomar el techo si da decimal
        
        # 4. Extracción y Corrección
        raw_bin = np.argmax(spectrum)
        raw_power = np.max(spectrum)
        self.save_spectrum_plot(spectrum, "PAYLOAD", raw_bin) # <-- AÑADIR
        
        # Corrección de Frecuencia + Módulo
        val_ajustado = raw_bin - cfo_bins
        val_corregido = int(round(val_ajustado)) % self.M
        
        # 5. Guardar y Avanzar
        self.datos_rx.append(val_corregido)
        self.payload_symbols_leidos += 1
        
        print(f"{CYAN}[PAYLOAD] Símbolo {self.payload_symbols_leidos}/{self.total_payload_symbols}: {val_corregido} (Power: {raw_power:.0f}){RESET}")
        
        self.consume(self.Ns) # Avanzamos un símbolo

        # 6. ¿Terminamos?
        if self.payload_symbols_leidos >= self.total_payload_symbols:
            print(f"{GREEN}[PAYLOAD]---> Payload completo. Decodificando...{RESET}")
            try:
                # Decodificación Final
                mensaje_final = self.demod.symbols_to_msg(self.datos_rx)
                self.rx_log["mensaje"] = mensaje_final # <-- AÑADIR
                self.record_event("PAYLOAD_FIN", "Mensaje decodificado") # <-- AÑADIR
                
                print(f"\n{GREEN}{'='*40}")
                print(f" MENSAJE RECIBIDO: {mensaje_final}")
                print(f"{'='*40}{RESET}\n")
            except Exception as e:
                self.record_event("ERROR_DECODIFICACION", str(e)) # <-- AÑADIR
                print(f"Error: {e}")
            
            # Reiniciar Máquina
            self.reset_machine()
            print(f"{BLUE}Esperando nueva transmisión...{RESET}")

    def payload2(self):
        """
        E° PAYLOAD 2:
        Versión especial que mantiene la corrección de 'vidas' usada en header2.
        """
        # --- 1. RECUPERAR DATOS DE SINCRONIZACIÓN ---
        sto_bins = self.sto
        sto_samples = int(sto_bins * (self.Fs / self.BW))
        cfo_bins    = self.cfo
        
        # --- [CLAVE] RECALCULAR LA CORRECCIÓN DE VIDAS ---
        # Esto es lo que faltaba. Debe ser igual que en header2
        vidas = self.sfd_lifes
        correction = vidas * self.Ns 

        # --- 2. CONFIGURAR VENTANA ---
        target_offset = 3 * self.Ns 
        
        # A. Chequeo de Buffer (Incluyendo 'correction' para no salirnos del array)
        muestras_necesarias = target_offset + self.Ns + abs(sto_samples) + 100 + correction
        
        if len(self.buffer) < muestras_necesarias:
            return 
        
        # B. Aplicar Offset (Incluyendo 'correction')
        # Si no restas 'correction' aquí, la ventana mirará al futuro y perderá el símbolo.
        self.window_offset = target_offset - sto_samples - correction

        window = self.get_window() 
        if window is None: return

        # --- 3. DECHIRP Y FFT (Igual que siempre) ---
        spectrum = self.dechirp_fft(window, self.upchirp)
        
        # --- 4. EXTRACCIÓN Y CORRECCIÓN ---
        raw_bin = np.argmax(spectrum)
        raw_power = np.max(spectrum)
        self.save_spectrum_plot(spectrum, "PAYLOAD", raw_bin) # <-- AÑADIR
        
        # Corrección de Frecuencia (CFO)
        val_ajustado = raw_bin - cfo_bins
        val_corregido = int(round(val_ajustado)) % self.M
        
        # --- 5. GUARDAR Y AVANZAR ---
        self.datos_rx.append(val_corregido)
        self.payload_symbols_leidos += 1
        
        print(f"{CYAN}[PAYLOAD 2] Símbolo {self.payload_symbols_leidos}/{self.total_payload_symbols}: {val_corregido} (Power: {raw_power:.0f}){RESET}")
        
        self.consume(self.Ns) # Avanzamos un símbolo físicamente

        # --- 6. FINALIZACIÓN ---
        if self.payload_symbols_leidos >= self.total_payload_symbols:
            print(f"{GREEN}[PAYLOAD 2]---> Payload completo. Decodificando...{RESET}")
            
            try:
                # Decodificación Final
                mensaje_final = self.demod.symbols_to_msg(self.datos_rx)
                self.rx_log["mensaje"] = mensaje_final # <-- AÑADIR
                self.record_event("PAYLOAD_FIN", "Mensaje decodificado") # <-- AÑADIR
                
                print(f"\n{GREEN}{'='*40}")
                print(f" MENSAJE RECIBIDO: {mensaje_final}")
                print(f"{'='*40}{RESET}\n")
            except Exception as e:
                self.record_event("ERROR_DECODIFICACION", str(e)) # <-- AÑADIR
                print(f"Error: {e}")
            
            # Reiniciar
            self.reset_machine()
            print(f"{BLUE}Esperando nueva transmisión...{RESET}")