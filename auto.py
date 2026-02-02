import subprocess
import time
import string
import random

# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------
TX_SCRIPT = "pyaudio_tx.py"   # nombre de tu TX
PYTHON = "python"             # o "python3" según tu sistema
PERIOD = 7.0                  # segundos entre mensajes
N_MESSAGES = 1000             # cantidad de pruebas

# --------------------------------------------------
def random_message(length=9):
    alphabet = string.ascii_lowercase
    return "".join(random.choice(alphabet) for _ in range(length))


# --------------------------------------------------
print("[AUTO_TX] Iniciando transmisión automática...")

for i in range(N_MESSAGES):
    msg = f"test_{i:04d}"#_" #+ random_message(8)

    print(f"[AUTO_TX] Enviando: {msg}")

    subprocess.run(
        [PYTHON, TX_SCRIPT],
        input=msg,
        text=True,
        shell=True
    )

    time.sleep(PERIOD)

print("[AUTO_TX] Fin de pruebas")
