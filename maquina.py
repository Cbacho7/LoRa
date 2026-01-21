# .\maquina.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt
from pathlib import Path
from enum import Enum, auto

class StateMachine(Enum):
    IDLE = auto()
    PREAMBLE = auto()
    SYNC = auto()
    READ_HEADER = auto()
    READ_PAYLOAD = auto()
    DONE = auto()


class Machine:
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
        
        phase_up = 2 * np.pi * (self.f0 * self.t + 0.5 * self.k * self.t**2)
        self.upchirp = np.exp(1j * phase_up) 

        phase_down = 2 * np.pi * ((self.f0 + self.BW) * self.t - 0.5 * self.k * self.t**2)
        self.downchirp = np.exp(1j * phase_down)  

        self.current_state = StateMachine.IDLE
        
         # Memoria FSM 
        self.last_symbol = None
        self.repeat_count = 0

        self.preamble_symbols = []
        self.header_symbols = []
        self.payload_symbols = []

        self.payload_len = None

        # Parámetros FSM
        self.MIN_PREAMBLE = 6 # Mínimo número de símbolos iguales para detectar preámbulo


    def IDLE(self, symbol):
        if symbol == self.last_symbol:
            self.repeat_count += 1
        else:
            self.repeat_count = 1

        self.last_symbol = symbol

        if self.repeat_count >= 2:
            self.current_state = StateMachine.PREAMBLE
            self.preamble_symbols = [symbol, symbol]

    def detect_preamble(self, symbol):
        if symbol == self.last_symbol:
            self.repeat_count += 1
            self.preamble_symbols.append(symbol)
        else:
            self.reset_machine(symbol)
            return False

        self.last_symbol = symbol

        if self.repeat_count >= self.MIN_PREAMBLE:
            return True

        return False

    def synchronize(self, symbol):
        pass

    def read_header(self, symbol):
        pass

    def read_payload(self, symbol):
        pass

    def reset_machine(self, symbol):
        self.current_state = StateMachine.IDLE
        self.last_symbol = symbol
        self.repeat_count = 0

        self.preamble_symbols.clear()
        self.header_symbols.clear()
        self.payload_symbols.clear()

        self.payload_len = None

    def state_transition(self, symbol):
        """
        Lógica de transición de estados basada en la señal de entrada.
        """
        if self.current_state == StateMachine.IDLE:
            self.IDLE(symbol)

        elif self.current_state == StateMachine.PREAMBLE:
            if self.detect_preamble(symbol) == True:
                self.current_state = StateMachine.SYNC

        elif self.current_state == StateMachine.SYNC:
            self.synchronize(symbol)

        elif self.current_state == StateMachine.READ_HEADER:
            self.read_header(symbol)

        elif self.current_state == StateMachine.READ_PAYLOAD:
            self.read_payload(symbol)
        if self.current_state == StateMachine.DONE:
            self.reset_machine(symbol)



