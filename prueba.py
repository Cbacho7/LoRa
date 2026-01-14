import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt
from pathlib import Path
from modulador import Modulador
from demodulador import Demodulador
from tqdm import tqdm # Para la barra de progreso


def single_ser_test(mod, demod, snr_db, num_symbols=200):
    # símbolos aleatorios
    symbols_tx = np.random.randint(0, mod.M, size=num_symbols)

    # modular
    signal_tx = mod.symbols_to_signal(symbols_tx)

    # calcular potencia
    signal_power = np.mean(np.abs(signal_tx)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear

    # ruido AWGN complejo
    noise = np.sqrt(noise_power/2) * (
        np.random.randn(len(signal_tx)) +
        1j * np.random.randn(len(signal_tx))
    )

    signal_rx = signal_tx + noise

    # demodular
    symbols_rx = demod.signal_to_symbols(signal_rx)

    # contar errores
    errors = np.sum(symbols_tx != symbols_rx[:num_symbols])

    return errors / num_symbols

def run_ser_curve(SF, snr_range, num_symbols=200, num_trials=50):
    Fs = 48000
    BW = 400
    f0 = 3000

    mod = Modulador(SF, BW, Fs, f0)
    demod = Demodulador(SF, BW, Fs, f0)

    SER = []

    for snr_db in tqdm(snr_range, desc=f"SF={SF}"):
        errors = 0
        total = 0

        for _ in range(num_trials):
            # 1. Generar símbolos y señal
            symbols_tx = np.random.randint(0, mod.M, size=num_symbols)
            signal_tx = mod.symbols_to_signal(symbols_tx)

            # 2. USAR FUNCIÓN PARA GENERAR RUIDO
            # Esto calcula la potencia correcta automáticamente
            noise = mod.make_noise(ruido_dB=snr_db, signal=signal_tx)

            # 3. USAR FUNCIÓN PARA FILTRAR EL RUIDO 
            # El ruido generado es de banda ancha (48kHz). 
            # Hay que filtrarlo para que solo quede el ruido que ve el receptor (400Hz).
            noise_filtered = mod.bandpass_filter(noise, f0, f0 + BW)

            # 4. Sumar señal + ruido filtrado
            signal_rx = signal_tx + noise_filtered

            # 5. Demodular
            symbols_rx = demod.signal_to_symbols(signal_rx)

            # Contar errores
            limit = min(len(symbols_tx), len(symbols_rx))
            errors += np.sum(symbols_tx[:limit] != symbols_rx[:limit])
            total += limit

        if total > 0:
            SER.append(errors / total)
        else:
            SER.append(0)
    return np.array(SER)

def main():
    # Rango de SNR (El ruido que se fija sin pasar por el filtro pasabanda)
    snr_range = np.linspace(-50, -20, 14)
    plt.figure(figsize=(10, 6)) 
    # for SF in [7, 8, 9, 10, 11, 12]:
    #     ser = run_ser_curve(
    #         SF,
    #         snr_range,
    #         num_symbols=70,
    #         num_trials=10
    #     )
        #plt.plot(snr_range, ser * 100, marker='o', label=f"SF={SF}")
    ser = run_ser_curve(
            8,
            snr_range,
            num_symbols=70,
            num_trials=10)
    plt.plot(snr_range, ser * 100, marker='o', label=f"SF={8}")
    plt.xlabel("SNR [dB]")
    plt.ylabel("SER [%]") # Etiqueta correcta en porcentaje
    plt.title("SER vs SNR para distintos Spreading Factors")
    plt.grid(True, which="both", linestyle='--', alpha=0.7)
    plt.legend()
    plt.ylim(0, 100) # Limitar el eje Y de 0% a 100%
    plt.show()

if __name__ == "__main__":
    main()
