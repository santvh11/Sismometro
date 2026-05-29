# Documentación Técnica del Software del Sismómetro (Proyecto Gaia)

Esta documentación describe la arquitectura y los fundamentos matemáticos detrás de los cuatro scripts principales del proyecto: `main.py`, `params.py`, `models.py` y `daq.py`.

---

## 1. Arquitectura General y Archivos

El sistema está diseñado de forma modular, separando los parámetros físicos, la matemática teórica, la adquisición de datos y la interfaz de usuario.

### `params.py` (Parámetros y Constantes)
Actúa como la "fuente de la verdad" para todo el programa. Utiliza `dataclasses` para definir un objeto `SectionParams` que almacena:
- **Constantes Mecánicas:** Masa (m), constante del resorte (k), amortiguamiento (c).
- **Constantes Electromagnéticas:** Inductancia (L), capacitancia (C), campo magnético (m_mag), resistividad y dimensiones de la bobina.
- **Configuraciones de Software:** Tamaño de la ventana (`TAMANO_VENTANA`), frecuencia de muestreo (`FS`), y variables para graficación.

### `main.py` (Punto de Entrada)
Es el orquestador del programa.
1. Instancia los parámetros desde `params.py`.
2. Llama a la función teórica `Factores_Acople` para calcular las constantes geométricas y magnéticas base.
3. Despliega un menú interactivo en consola para enrutar la ejecución hacia simulaciones teóricas (MK1, MK2), adquisición en tiempo real, guardado headless, o experimentos de caracterización.

---

## 2. Traducción de la Matemática a Código (`models.py`)

Este módulo se encarga de las simulaciones teóricas ("digital twins" del sismómetro). 

### Factores de Acople Electromagnético
La función `Factores_Acople` convierte las integrales de línea del campo magnético (ley de Biot-Savart) en dos constantes estáticas mediante integrales analíticas pre-calculadas en el código:
- **G_A (Factor de Área / FEM):** Cuántos Voltios se inducen por cada 1 m/s de velocidad.
- **G_L (Factor de Longitud / Laplace):** Cuántos Newtons de fuerza de repulsión genera 1 Amperio de corriente en la bobina.

### Modelo MK1 (Acoplamiento Lineal Analítico)
El MK1 asume que la fricción (c) es lineal (F_c = c * v). El sistema (mecánico + eléctrico) se representa con 4 variables de estado: Posición (z), Velocidad (dz/dt), Carga (Q), y Corriente (I).
- **Matriz de Estado (A):** Se ensambla una matriz 4x4 en `numpy`.
- **Fasores y Álgebra Lineal:** Para la solución particular (estado estacionario ante una excitación F_0 * sen(omega * t)), el código usa números complejos: `Xp_complejo = np.linalg.solve(1j * omega * I - A, lado_derecho)`.
- **Eigenvalores:** La parte transitoria se resuelve hallando los eigenvalores y eigenvectores de la matriz A con `np.linalg.eig`, permitiendo construir la curva de evolución temporal exacta de forma puramente analítica.

### Modelo MK2 (Acoplamiento No-Lineal Numérico)
Si el número de Reynolds indica turbulencia, la fricción aerodinámica/viscosa se vuelve proporcional al cuadrado de la velocidad (F_c = c * |v| * v). 
- Al no tener solución analítica sencilla, el código define una función diferencial `state_matrix` que evalúa las derivadas en cada instante.
- Se utiliza `scipy.integrate.solve_ivp` con el método **LSODA** (o RK45) para resolver la ecuación diferencial ordinaria (ODE) iterativamente a lo largo del vector de tiempo.

---

## 3. Adquisición y Procesamiento de Datos (`daq.py`)

Este es el módulo que dialoga con el hardware (ESP32) mediante `pyserial`. Utiliza un **hilo asíncrono** (`threading.Thread`) para leer del puerto serie y guardar los datos en buffers (deques) para que la interfaz gráfica (`pyqtgraph`) no sufra de lag o se congele.

### Aplicación de Filtros y Cinemática
- **Filtros IIR:** Se utilizan filtros pasa-altas y pasa-bajas de Butterworth (`scipy.signal.butter`). Estos eliminan la deriva de corriente continua (DC Offset) térmica y el ruido de alta frecuencia del ADC.
- **Velocidad:** Sabiendo que FEM = G_A * R_% * v, el código despeja directamente v = -FEM / (G_A * R_%).
- **Posición (Integración en tiempo real):** Se toma el array de velocidad y se integra usando la regla del trapecio (`scipy.integrate.cumulative_trapezoid`) para estimar cuánto se movió físicamente el imán.

### Integración de la Transformada de Fourier (FFT)
Para detectar a qué frecuencia está oscilando la mesa vibratoria o el suelo:
1. El programa extrae una porción (ventana deslizante) de los últimos 512 datos de voltaje (`v_real`).
2. Se aplica `scipy.fft.fft` sobre esta ventana para pasar del dominio del tiempo al de la frecuencia.
3. Se generan los ejes de frecuencia (Hz) con `fftfreq(N, T_muestreo)`.
4. El programa busca el pico máximo (`np.argmax`) del valor absoluto del array complejo resultante. Esta frecuencia dominante y su amplitud se grafican en vivo y se guardan en el archivo `.csv`.

---

## 4. Caracterización del Coeficiente de Amortiguamiento (c)

La "Opción 5" en el menú es un experimento diseñado para capturar un sismo controlado y estimar el coeficiente físico de amortiguamiento viscoso (c) de la maqueta usando ventanas de datos deslizantes. Se emplean dos técnicas:

### Método 1: Newton-Raphson (Analítico Inverso)
1. **Integral del Voltaje:** Se calcula la integral numérica (`simpson`) de la ventana de voltaje, lo cual es proporcional a la integral de la velocidad (es decir, el desplazamiento real de la masa).
2. **Solución Teórica:** El código evalúa la función teórica de desplazamiento para un sismómetro subamortiguado.
3. **Búsqueda de Raíces:** Al tener una integral observada y una ecuación teórica que depende de c, el script aplica el algoritmo de **Newton-Raphson** iterativamente. Calcula las derivadas complejas con respecto a c para "acercarse" al valor exacto de c que haría que la teoría iguale la realidad.

### Método 2: Ajuste Senoidal por Mínimos Cuadrados
Asumiendo que el estado estacionario ya fue alcanzado:
1. La onda de voltaje se convierte a onda de velocidad empírica.
2. Usando `scipy.optimize.curve_fit`, el código obliga matemáticamente a una función de la forma v(t) = A * cos(omega * t - phi) + C a que se superponga sobre los datos experimentales reales, extrayendo la amplitud empírica (A_est) y el desfase empírico (phi_est).
3. Las ecuaciones teóricas de la respuesta a la frecuencia dictan que la amplitud máxima y la fase dependen estrictamente de m, k, omega y c. Sabiendo las primeras tres, el programa usa álgebra básica para despejar c a partir de la amplitud y, paralelamente, a partir del desfase.

#### Filtrado de Datos Basura (`np.nan`)
Puesto que en un sismo hay momentos de ruido plano o convergencias falsas de los algoritmos (p.ej., intentando ajustar un coseno a ruido eléctrico plano), el código filtra cualquier c calculado que sea menor a 0 o mayor a 3, reemplazándolo por `np.nan`. Matplotlib y NumPy ignoran nativamente los NaNs, logrando promedios precisos y gráficas limpias sin desincronizar los ejes del tiempo.
