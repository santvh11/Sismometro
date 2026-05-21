import serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
import collections

# ==========================================
# 1. CONFIGURACIÓN DEL SISTEMA
# ==========================================
PUERTO = 'COM3'
BAUDIOS = 115200
TAMANO_VENTANA = 512  # Muestras para la FFT y el ajuste de curva
FS = 1000.0           # Frecuencia de muestreo teórica en Hz

# Parámetros de Filtros Digitales (en Hz)
FC_PASA_ALTAS = 0.5   # Elimina el DC y hace el "espejo" AC
FC_PASA_BAJAS = 40.0  # Limpia el ruido eléctrico agudo

# Constantes del Sistema Físico (¡Modifica estas variables con tus datos!)
MASA = 1.0            # kg
RESISTENCIA = 100.0   # Ohms
DPHI_DX = 0.5         # Cambio de flujo magnético respecto a x (T*m)
F0_EXCITACION = 10.0  # Amplitud de la fuerza sinusoidal (N)
OMEGA_EXCITACION = 2.0 * np.pi * 5.0 # Frecuencia angular de excitación (rad/s)

# Buffers circulares de alta velocidad
buffer_t = collections.deque(maxlen=TAMANO_VENTANA)
buffer_v = collections.deque(maxlen=TAMANO_VENTANA)

# ==========================================
# 2. DEFINICIÓN DE MATEMÁTICA Y FILTROS
# ==========================================
def aplicar_filtros(tiempo, voltaje):
    # Diseño de filtros Butterworth
    nyquist = 0.5 * FS
    b_altas, a_altas = butter(2, FC_PASA_ALTAS / nyquist, btype='high')
    b_bajas, a_bajas = butter(2, FC_PASA_BAJAS / nyquist, btype='low')
    
    # Aplicar filtros secuencialmente (filtfilt evita el desfase en el tiempo)
    v_filtrado = filtfilt(b_altas, a_altas, voltaje)
    v_filtrado = filtfilt(b_bajas, a_bajas, v_filtrado)
    
    return v_filtrado

def modelo_fisico(t, c):
    """
    Función para scipy.optimize. 
    Ajusta el coeficiente 'c' basándose en la relación V = v * (dphi/dx) * R
    y la fuerza F = F0 * sin(omega * t).
    (Aquí puedes expandir tu ecuación diferencial).
    """
    # Ejemplo simplificado de la relación de velocidad y voltaje esperado
    # v(t) teórica basada en la fuerza y el amortiguamiento
    velocidad_teorica = (F0_EXCITACION / c) * np.sin(OMEGA_EXCITACION * t)
    
    # Voltaje teórico: V = v * dphi_dx * R
    voltaje_teorico = velocidad_teorica * DPHI_DX * RESISTENCIA
    return voltaje_teorico

# ==========================================
# 3. PREPARACIÓN DE LA INTERFAZ GRÁFICA
# ==========================================
try:
    esp32 = serial.Serial(PUERTO, BAUDIOS)
    print(f"Conectado a {PUERTO}. Esperando sincronización...")
except Exception as e:
    print(f"Error conectando a {PUERTO}: {e}")
    exit()

fig, (ax_tiempo, ax_fft) = plt.subplots(2, 1, figsize=(12, 8))
fig.canvas.manager.set_window_title('DAQ Sismómetro EAFIT')

# Plot Dominio del Tiempo
linea_t, = ax_tiempo.plot([], [], lw=2, color='blue', label='Voltaje Filtrado')
ax_tiempo.set_title("Dominio del Tiempo (Onda y Filtros)")
ax_tiempo.set_ylabel("Voltaje (V)")
ax_tiempo.set_xlabel("Tiempo (s)")
ax_tiempo.grid(True)
texto_metricas = ax_tiempo.text(0.02, 0.85, '', transform=ax_tiempo.transAxes, 
                                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

# Plot Dominio de la Frecuencia (FFT)
linea_f, = ax_fft.plot([], [], lw=2, color='red')
ax_fft.set_title("Dominio de la Frecuencia (FFT)")
ax_fft.set_ylabel("Amplitud")
ax_fft.set_xlabel("Frecuencia (Hz)")
ax_fft.set_xlim(0, 100) # Muestra de 0 a 100 Hz por defecto
ax_fft.grid(True)

plt.tight_layout()

# Variables para control de tiempo
tiempo_inicial_micros = None

# ==========================================
# 4. MOTOR PRINCIPAL DE PROCESAMIENTO
# ==========================================
def actualizar(frame):
    global tiempo_inicial_micros
    
    # Leer datos del puerto serie rápidamente
    while esp32.in_waiting > 0:
        try:
            linea = esp32.readline().decode('utf-8').strip()
            if "," in linea:
                t_micros_str, v_str = linea.split(",")
                t_micros = int(t_micros_str)
                v_crudo = float(v_str)
                
                if tiempo_inicial_micros is None:
                    tiempo_inicial_micros = t_micros
                
                # Convertir microsegundos a segundos reales
                t_segundos = (t_micros - tiempo_inicial_micros) / 1000000.0
                
                buffer_t.append(t_segundos)
                buffer_v.append(v_crudo)
        except Exception:
            pass # Ignorar tramas corruptas

    # Solo procesar y graficar si tenemos la ventana llena (512 muestras)
    if len(buffer_t) == TAMANO_VENTANA:
        t_arr = np.array(buffer_t)
        v_arr = np.array(buffer_v)
        
        # 1. Aplicar Filtros (Pasa-Altas y Pasa-Bajas)
        v_filtrado = aplicar_filtros(t_arr, v_arr)
        
        # 2. Ingeniería Inversa: Ajuste de curva con scipy.optimize
        try:
            # P0 es la semilla inicial para 'c'
            popt, pcov = curve_fit(modelo_fisico, t_arr, v_filtrado, p0=[5.0])
            c_estimado = popt[0]
        except:
            c_estimado = 0.0 # Si no logra converger
            
        # 3. Transformada Rápida de Fourier (FFT)
        N = TAMANO_VENTANA
        T_muestreo = 1.0 / FS
        
        yf = fft(v_filtrado)
        xf = fftfreq(N, T_muestreo)[:N//2]
        amplitud_fft = 2.0/N * np.abs(yf[0:N//2])
        
        # 4. Actualizar Gráfica de Tiempo (Autoajustable)
        linea_t.set_data(t_arr, v_filtrado)
        ax_tiempo.set_xlim(t_arr[0], t_arr[-1])
        margen = max(np.abs(v_filtrado)) * 1.2 + 0.01
        ax_tiempo.set_ylim(-margen, margen) # Y dinámico y centrado en 0
        
        # Calcular y mostrar métricas
        v_pico = np.max(np.abs(v_filtrado))
        v_rms = np.sqrt(np.mean(v_filtrado**2))
        texto_metricas.set_text(f"Pico: {v_pico:.3f} V\nPromedio (RMS): {v_rms:.3f} V\nCoef. 'c' (Estimado): {c_estimado:.3f}")
        
        # 5. Actualizar Gráfica de Frecuencia
        linea_f.set_data(xf, amplitud_fft)
        ax_fft.set_ylim(0, max(amplitud_fft) * 1.2 + 0.01)

    return linea_t, linea_f, texto_metricas

# Iniciar la animación
ani = animation.FuncAnimation(fig, actualizar, interval=30, blit=False)
plt.show()

esp32.close()