import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, find_peaks

# ------------------------------------------------------------
# 1. Cargar datos del archivo (omitir primera fila de encabezado)
# ------------------------------------------------------------
data = np.loadtxt(
    "Datos60HZ copy.txt", skiprows=1, usecols=(1, 3)
)  # columnas t (1) y y (3)
t = data[:, 0]
y = data[:, 1]  # desplazamiento en metros

# Verificar que t sea creciente
dt = np.mean(np.diff(t))
fs = 1.0 / dt  # frecuencia de muestreo (Hz)
print(f"Frecuencia de muestreo estimada: {fs:.1f} Hz")

# ------------------------------------------------------------
# 2. Graficar posición original y filtrada
# ------------------------------------------------------------
# Diseño de filtro pasa bajas (eliminar ruido por encima de 3-4 veces la frecuencia esperada)
f_corte = 200.0  # Hz (por encima de 60 Hz, para dejar pasar hasta el tercer armónico)
nyquist = 0.5 * fs
normal_cutoff = f_corte / nyquist
b, a = butter(4, normal_cutoff, btype="low", analog=False)
y_filt = filtfilt(b, a, y)

plt.figure(figsize=(12, 4))
plt.plot(t, y, "b-", alpha=0.3, label="Original (ruidosa)")
plt.plot(t, y_filt, "r-", linewidth=1.5, label="Filtrada (low-pass 200 Hz)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Desplazamiento y (m)")
plt.title("Movimiento vertical de la mesa (60 Hz esperado)")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# 3. Análisis espectral (FFT) para verificar frecuencia dominante
# ------------------------------------------------------------
N = len(t)
T_muestreo = 1.0 / fs
yf = fft(y_filt)
xf = fftfreq(N, T_muestreo)[: N // 2]
amplitud_fft = 2.0 / N * np.abs(yf[0 : N // 2])

# Buscar pico máximo (descartar componente DC)
idx_max = np.argmax(amplitud_fft[1:]) + 1
f_dominante = xf[idx_max]
print(f"Frecuencia dominante detectada: {f_dominante:.2f} Hz (esperado 60 Hz)")

plt.figure(figsize=(10, 4))
plt.plot(xf, amplitud_fft, "b-")
plt.axvline(
    x=f_dominante, color="r", linestyle="--", label=f"Pico: {f_dominante:.1f} Hz"
)
plt.xlim(0, 200)
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud (m)")
plt.title("FFT del desplazamiento filtrado")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# 4. Extraer amplitud pico a pico y calcular aceleración
# ------------------------------------------------------------
# Encontrar picos positivos de la señal filtrada
peaks, _ = find_peaks(y_filt, height=np.std(y_filt))
valleys, _ = find_peaks(-y_filt, height=np.std(y_filt))

if len(peaks) > 0 and len(valleys) > 0:
    # Amplitud semipico = (promedio de picos - promedio de valles)/2
    mean_peak = np.mean(y_filt[peaks])
    mean_valley = np.mean(y_filt[valleys])
    A_semiamplitude = (mean_peak - mean_valley) / 2.0
else:
    # Si no encuentra picos, usar la desviación estándar * sqrt(2) como aproximación
    A_semiamplitude = np.std(y_filt) * np.sqrt(2)

print(f"Amplitud semipico del desplazamiento: {A_semiamplitude:.3e} m")

# Aceleración pico = (2πf)^2 * A
omega = 2 * np.pi * f_dominante
a_pico = (omega**2) * A_semiamplitude  # m/s²
a_pico_G = a_pico / 9.81
print(f"Aceleración pico: {a_pico:.3f} m/s² = {a_pico_G:.3f} G")

# Si se desea la fuerza (F = m_vibrante * a_pico)
m_vibrante = 0.077  # kg (ejemplo para bobina grande)
F_pico = m_vibrante * a_pico
print(f"Fuerza pico para m_vibrante={m_vibrante} kg: {F_pico:.2f} N")

# ------------------------------------------------------------
# 5. Guardar punto (frecuencia, aceleración_G) para construir la tabla
# ------------------------------------------------------------
# Este punto corresponde a f_dominante y a_pico_G. Repite para varias frecuencias.
# Puedes ir guardando en un archivo CSV.
with open("puntos_mesa.csv", "a") as f:
    f.write(f"{f_dominante:.2f}, {a_pico_G:.6f}\n")
print("Punto guardado en 'puntos_mesa.csv' (frecuencia_Hz, aceleración_G)")
