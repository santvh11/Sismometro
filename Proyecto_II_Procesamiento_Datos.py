import requests
import matplotlib.pyplot as plt
import pandas as pd

# --- Configuración ---
CHANNEL_ID = 3343167
READ_API_KEY = "9AQFLP8BXGIXVMRS" 
RESULTS = 20 

url = f'https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results={RESULTS}'

def graficar_datos():
    response = requests.get(url)
    data = response.json()
    
    # Extraer datos a un DataFrame de Pandas
    feeds = data['feeds']
    df = pd.DataFrame(feeds)
    
    # Convertir a valores numéricos
    df['field1'] = pd.to_numeric(df['field1']) # Voltaje
    df['field2'] = pd.to_numeric(df['field2']) # Frecuencia FFT
    df['created_at'] = pd.to_datetime(df['created_at'])

    # Crear Gráfica Doble
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # 1. Gráfica de Muestreo (Histórico de voltaje)
    ax1.plot(df['created_at'], df['field1'], color='blue', marker='o')
    ax1.set_title('Monitoreo de Voltaje en el Tiempo (Directo)')
    ax1.set_ylabel('Voltaje (V)')
    ax1.grid(True)

    # 2. Gráfica de Frecuencia (Pico de Fourier)
    ax2.bar(df['created_at'].dt.strftime('%H:%M:%S'), df['field2'], color='red')
    ax2.set_title('Picos de Frecuencia Detectados (FFT)')
    ax2.set_ylabel('Frecuencia (Hz)')
    ax2.set_xlabel('Hora de Medición')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    graficar_datos()
