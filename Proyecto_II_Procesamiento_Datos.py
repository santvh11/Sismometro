import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# ==========================================
# CONFIGURACIÓN PRINCIPAL
# ==========================================
PUERTO = 'COM3'       # El puerto que me mostraste en la imagen
BAUDIOS = 115200      # La velocidad (debe ser igual a Serial.begin en Arduino)
MAX_PUNTOS = 200      # Cuántos puntos de la onda queremos ver en pantalla al mismo tiempo

# ==========================================
# PREPARANDO LAS HERRAMIENTAS
# ==========================================
# 1. Conectamos con el ESP32 por el cable USB
try:
    esp32 = serial.Serial(PUERTO, BAUDIOS)
    print(f"Conectado exitosamente al puerto {PUERTO}")
except Exception as e:
    print(f"¡Error! No se pudo conectar al {PUERTO}. Verifica que no esté abierto en otro lado.")
    exit()

# 2. Creamos una "lista mágica" (deque) que guarda solo los últimos 200 datos.
# Si entra el dato 201, el dato 1 se borra automáticamente (como una cinta de correr).
datos_voltaje = deque([0.0] * MAX_PUNTOS, maxlen=MAX_PUNTOS)

# 3. Preparamos el lienzo de la gráfica
fig, ax = plt.subplots(figsize=(10, 5))
linea, = ax.plot(datos_voltaje, color='blue') # Dibujamos la línea inicial
ax.set_ylim(-0.5, 0.5) # Rango del eje Y (De 0V a 3.5V, ya que tu ESP32 lee hasta 3.3V)
ax.set_title("Onda del Sismómetro en Tiempo Real")
ax.set_ylabel("Voltaje (V)")
ax.set_xlabel("Muestras recientes")
ax.grid(True)

# ==========================================
# EL MOTOR DE LA GRÁFICA (Esta función se repite sola)
# ==========================================
def actualizar_grafica(frame):
    # Mientras haya datos esperando en el cable USB...
    while esp32.in_waiting > 0:
        try:
            # Leemos la línea que envió el ESP32, le quitamos espacios y la convertimos a texto
            linea_texto = esp32.readline().decode('utf-8').strip()
            
            # Convertimos ese texto a un número decimal (float)
            voltaje = float(linea_texto)
            
            # Añadimos el nuevo voltaje a nuestra lista mágica
            datos_voltaje.append(voltaje)
            
        except ValueError:
            # A veces el cable manda basura o un texto incompleto, lo ignoramos y seguimos
            pass 

    # Le pasamos los datos actualizados a la línea del dibujo
    linea.set_ydata(datos_voltaje)
    return linea,

# ==========================================
# ¡ACCIÓN!
# ==========================================
# FuncAnimation es el director de orquesta. Llama a "actualizar_grafica" muy rápido.
ani = animation.FuncAnimation(fig, actualizar_grafica, interval=20, blit=True)

# Mostramos la ventana
plt.tight_layout()
plt.show()

# Cuando cierres la ventana, cerramos el puerto USB por educación
esp32.close()
print("Desconectado.")