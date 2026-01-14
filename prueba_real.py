import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from modulador import Modulador
from demodulador import Demodulador

def run_ser_curve(SF, snr_range_input, num_symbols, num_trials):
    Fs = 48000
    BW = 400
    f0 = 3000

    mod = Modulador(SF, BW, Fs, f0)
    demod = Demodulador(SF, BW, Fs, f0)

    SER = []
    SNR_REAL_MEDIO = [] # Aquí guardaremos el promedio del SNR que realmente ocurrió

    # Iteramos sobre el rango de ruido "solicitado"
    for snr_input in tqdm(snr_range_input, desc=f"SF={SF}"):
        errors = 0
        total = 0
        acc_snr = 0 # Acumulador para promediar el SNR medido en estos trials

        for _ in range(num_trials):
            # 1. Generar señal
            symbols_tx = np.random.randint(0, mod.M, size=num_symbols)
            signal_tx = mod.symbols_to_signal(symbols_tx)

            # 2. Generar ruido 
            noise = mod.make_noise(ruido_dB=snr_input, signal=signal_tx)

            # 3. Filtrar ruido 
            noise_filtered = mod.bandpass_filter(noise, f0, f0 + BW)

            # 4. MEDIR EL SNR REAL
            # Calculamos la relación real entre la señal limpia y el ruido que quedó
            snr_medido = mod.SNR_cal(Ruido=noise_filtered, Signal=signal_tx)
            acc_snr += snr_medido # Lo guardamos para el promedio

            # 5. Sumar y Demodular
            signal_rx = signal_tx + noise_filtered
            symbols_rx = demod.signal_to_symbols(signal_rx)

            # 6. Contar errores
            limit = min(len(symbols_tx), len(symbols_rx))
            errors += np.sum(symbols_tx[:limit] != symbols_rx[:limit])
            total += limit

        # Guardamos resultados promedios de este punto
        if total > 0:
            SER.append(errors / total)
        else:
            SER.append(0)
            
        SNR_REAL_MEDIO.append(acc_snr / num_trials)

    return np.array(SER), np.array(SNR_REAL_MEDIO)


def main():
    # RANGO DE ENTRADA:
    # Como el filtro quita mucho ruido, pedimos valores muy bajos (ej: -50)
    # para que al medirlos den valores razonables (ej: -30).
    snr_range_input = np.linspace(-50, -15, 15)

    plt.figure(figsize=(10, 6))

    # Colores profesionales
    colors = {
    7: '#1f77b4',  # Azul (SF7)
    8: '#ff7f0e',  # Naranja (SF8)
    9: '#2ca02c',  # Verde (SF9)
    10: '#d62728', # Rojo (SF10)
    11: '#9467bd', # Morado (SF11)
    12: '#8c564b'  # Café (SF12)
}

    for SF in [7, 8, 9, 10, 11, 12]:
        ser, snr_real = run_ser_curve(
            SF,
            snr_range_input,
            num_symbols=70, 
            num_trials=15
        )
        
        # EJE X: Usamos 'snr_real'
        plt.plot(snr_real, ser * 100, marker='o', label=f"SF={SF}", color=colors[SF])

    # Etiquetas y Estilo
    plt.xlabel("SNR Calculado [dB] (Post-Filtro)")
    plt.ylabel("Tasa de Error de Símbolo (SER) [%]")
    plt.title("Rendimiento de LoRa: SER vs SNR Real")
    
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.legend(title="Spreading Factor")
    
    # Límites para que se vea limpio (0 a 100%)
    plt.ylim(-2, 105) 
    
    # Guardar gráfico para tu PPT
    plt.savefig("resultado_ser_lora.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()