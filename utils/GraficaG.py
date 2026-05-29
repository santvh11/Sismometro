import numpy as np
import matplotlib.pyplot as plt

archivo = "Datos60HZ.txt"
data = np.genfromtxt(archivo, skip_header=2)

# Extraer columnas (0: tiempo, 1: x, 2: y)
t = data[:, 0]
y = data[:, 2]

# Calculo intervalo tiempo promedio y número muestras
dt = np.mean(np.diff(t))
N = len(t)

# FFT posición Y
Y_fft = np.fft.fft(y)
freqs = np.fft.fftfreq(N, dt)

# Filtrar únicamente frecuencias positivas
mask = freqs > 0
freqs = freqs[mask]
Y_fft = Y_fft[mask]

# Calcular Amplitud real y Velocidad Angular (omega)
Y_amp = 2.0 / N * np.abs(Y_fft)
omega = 2 * np.pi * freqs

# Calcular Fuerza G
G_force = (Y_amp * (omega**2)) / 9.81

# Setteo de la gráfica
plt.figure(figsize=(10, 5))
plt.plot(omega, G_force, color="red", marker="o", markersize=3, linestyle="-")
plt.title("Espectro de Fuerza G vs Frecuencia Angular $\omega$ (Eje Y)")
plt.xlabel("$\omega$ (rad/s)")
plt.ylabel("Fuerza G")
plt.grid(True, linestyle="--", alpha=0.7)
plt.xlim(0, max(omega))  # Limita el eje X hasta la frecuencia máxima detectada
plt.tight_layout()

plt.show()
