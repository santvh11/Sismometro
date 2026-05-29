import os
import csv
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def listar_archivos_logs(directorio="logs"):
    if not os.path.exists(directorio):
        print(f"El directorio '{directorio}' no existe.")
        return []
    
    archivos_csv = glob.glob(os.path.join(directorio, "*.csv"))
    archivos_csv.sort(key=os.path.getmtime, reverse=True) # Más recientes primero
    return archivos_csv

def procesar_csv(archivo):
    tiempos_relativos = []
    voltajes = []
    velocidades = []
    posiciones = []
    frecuencias = []
    
    tiempo_inicial_dt = None
    
    with open(archivo, mode='r', encoding='utf-8') as f:
        lector = csv.reader(f)
        try:
            encabezado = next(lector)
        except StopIteration:
            print("El archivo está vacío.")
            return None
        
        for fila in lector:
            if len(fila) < 6:
                continue
            
            # Formato de tiempo: %H:%M:%S.%f
            tiempo_str = fila[0]
            try:
                dt = datetime.strptime(tiempo_str, "%H:%M:%S.%f")
                if tiempo_inicial_dt is None:
                    tiempo_inicial_dt = dt
                
                # Calcular segundos relativos
                delta = (dt - tiempo_inicial_dt).total_seconds()
                
                # Manejar el salto de día si pasa de 23:59 a 00:00 (muy raro, pero seguro)
                if delta < 0:
                    delta += 86400
                    
                tiempos_relativos.append(delta)
                voltajes.append(float(fila[1]))
                velocidades.append(float(fila[2]))
                posiciones.append(float(fila[3]))
                frecuencias.append(float(fila[4]))
            except ValueError:
                continue
                
    return (np.array(tiempos_relativos), np.array(voltajes), 
            np.array(velocidades), np.array(posiciones), np.array(frecuencias))

def graficar_datos(tiempos, voltajes, velocidades, posiciones, frecuencias, nombre_archivo):
    # Crear figura con 4 subplots compartiendo el eje X
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f"Análisis de Log: {os.path.basename(nombre_archivo)}", fontsize=14, fontweight='bold')
    
    # 1. Voltaje Pico
    axs[0].plot(tiempos, voltajes, color='b', linewidth=1.5)
    axs[0].set_ylabel("Voltaje Pico (V)", fontweight='bold')
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    # 2. Velocidad
    axs[1].plot(tiempos, velocidades, color='orange', linewidth=1.5)
    axs[1].set_ylabel("Velocidad (m/s)", fontweight='bold')
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    # 3. Posición
    axs[2].plot(tiempos, posiciones, color='g', linewidth=1.5)
    axs[2].set_ylabel("Posición (m)", fontweight='bold')
    axs[2].grid(True, linestyle='--', alpha=0.7)
    
    # 4. Frecuencia Dominante FFT
    axs[3].plot(tiempos, frecuencias, color='r', linewidth=1.5)
    axs[3].set_ylabel("Frecuencia (Hz)", fontweight='bold')
    axs[3].set_xlabel("Tiempo Transcurrido (segundos)", fontweight='bold')
    axs[3].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def main():
    print("========================================")
    print("  VISOR DE LOGS DEL SISMÓMETRO EAFIT")
    print("========================================")
    
    archivos = listar_archivos_logs()
    if not archivos:
        print("No se encontraron archivos de log en la carpeta 'logs/'.")
        return
        
    print("Archivos disponibles (del más reciente al más antiguo):")
    for i, arch in enumerate(archivos):
        nombre = os.path.basename(arch)
        tamano = os.path.getsize(arch) / 1024 # KB
        print(f" [{i}] {nombre} ({tamano:.1f} KB)")
        
    seleccion = input("\nIngrese el número del archivo que desea graficar: ")
    try:
        idx = int(seleccion)
        if idx < 0 or idx >= len(archivos):
            print("Número fuera de rango.")
            return
    except ValueError:
        print("Entrada inválida. Debe ingresar un número.")
        return
        
    archivo_seleccionado = archivos[idx]
    print(f"\nProcesando '{os.path.basename(archivo_seleccionado)}'...")
    
    datos = procesar_csv(archivo_seleccionado)
    if datos is None or len(datos[0]) == 0:
        print("No se pudieron extraer datos del archivo. Puede que esté vacío o tenga un formato incorrecto.")
        return
        
    tiempos, voltajes, velocidades, posiciones, frecuencias = datos
    print(f"Se extrajeron {len(tiempos)} puntos de datos.")
    print("Abriendo gráficas (puede usar las herramientas de Matplotlib para hacer zoom o guardar la imagen)...")
    
    graficar_datos(tiempos, voltajes, velocidades, posiciones, frecuencias, archivo_seleccionado)

if __name__ == "__main__":
    main()
