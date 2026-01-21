import json
import matplotlib.pyplot as plt
import numpy as np

def cargar_datos(nombre_archivo):
    try:
        with open(nombre_archivo, "r") as f:
            datos = json.load(f)
        return np.array(datos["snr_real"]), np.array(datos["ser"])
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {nombre_archivo}")
        return None, None

def graficar_comparativa():
    # 1. Cargar los datos de ambos experimentos
    snr_lora, ser_lora = cargar_datos("datos_SF7_SF7.json")
    snr_magic, ser_magic = cargar_datos("MAGIC.JSON")

    if snr_lora is None or snr_magic is None:
        return

    # 2. Configurar el gráfico
    plt.figure(figsize=(10, 6))

    # Curva de (LoRa)
    # Se convierte el SER a porcentaje (* 100)
    plt.plot(snr_lora, ser_lora * 100, marker='o', linestyle='-', 
             linewidth=2, label=" (LoRa SF7 BW=140- Ns=43886)", color='#ff7f0e')

    # Curva de Magic (FSK)
    plt.plot(snr_magic, ser_magic * 100, marker='s', linestyle='--', 
             linewidth=2, label=" (FSK - Ns=48000)", color='#1f77b4')

    # 3. Estética del gráfico
    plt.xlabel("SNR Real [dB] (Post-Filtro)")
    plt.ylabel("Tasa de Error de Símbolo (SER) [%]")
    plt.title("Comparativa de Robustez: LoRa vs FSK")
    
    # Ajustar límites para mejor visualización
    plt.grid(True, which="both", linestyle='--', alpha=0.7)
    plt.legend(title="Implementación")
    
    # Límites del eje Y (0 a 105% para dejar espacio arriba)
    plt.ylim(-2, 105)
    
    # 4. Guardar y mostrar
    plt.savefig("Imagenes/comparativa_final_lora_fsk2.png", dpi=300)
    print("\n✅ Gráfico guardado como: comparativa_final_lora_fsk1.png")
    plt.show()

if __name__ == "__main__":
    graficar_comparativa()