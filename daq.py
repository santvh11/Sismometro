import serial
import time
import collections
import csv
from datetime import datetime
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
from scipy.integrate import cumulative_trapezoid, simpson
from params import SectionParams
from models import Factores_Acople
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
import threading
import os

def run_daq_realtime(params: SectionParams, G_sub_L: float, G_sub_A: float):
    m = params.m
    c = params.c
    c_sub_lambda = params.c_sub_lambda
    k = params.k
    L = params.L
    C = params.C
    z_0 = params.z
    z_dot_0 = params.z_dot
    Q_0 = params.Q
    Q_dot_0 = params.Q_dot
    t = params.t
    F_0 = params.F_0
    omega = params.omega
    delta_t = params.delta_t
    G_sub_L, G_sub_A = Factores_Acople(params)
    XC = params.XC
    R = params.R
    R_porcentaje = params.R_porcentaje
    Zeta = params.Zeta
    alpha = params.alpha
    omega_sub_0_phi_m = params.omega_sub_0_phi_m
    R_sub_e = params.R_sub_e
    r_sub_p = params.r_sub_p
    eta = params.eta
    rho = params.rho
    m_mag = params.m_mag
    omega_sub_n = params.omega_sub_n
    omega_sub_n_f = omega_sub_n * 2 * np.pi
    omega_sub_0_phi_m_f = omega_sub_0_phi_m * 2 * np.pi
    a = params.a
    b = params.b
    factor_amplificacion = params.factor_amplificacion
    subida_voltaje = params.subida_voltaje
    omega_filtro = params.omega_filtro
    factor_ESP_32 = params.factor_ESP_32
    ESCALA_TIEMPO = params.ESCALA_TIEMPO
    FACTOR_TIEMPO_ANIMACION = params.FACTOR_TIEMPO_ANIMACION
    RANGO_VOLTAJE = params.RANGO_VOLTAJE

    # Impresión de parámetros relevantes
    print(f"masa estimada en :{m:.3e} Kg")
    print(f"El amortiguamiento auxiliar es:{c_sub_lambda:.3e}")
    print(f"La constante de amortiguamiento  es de :{c:.3e} Kg/s")
    print(f"La constante elástica es de :{k:.3e} Kg/s^2")
    print(f"La resistencia es de :{R:.3e} Ohmnios")
    print(f"La inductancia es de :{L:.3e} H")
    print(f"La capacitancia es de :{C:.3e} F")
    print(f"Momento magnético configurado en: {m_mag:.3e}")
    print(f"El factor Magnético de Área es: {G_sub_A:.3e}")
    print(f"El factor Magnético de Longitud es: {G_sub_L:.3e}")
    print(f"El factor de amortiguamiento es de: {Zeta:.3e}")
    print(f"La frecuencia natural es de: {omega_sub_n_f:.3e} Hz")
    print(f"La inercia eléctrica es de: {alpha:.3e} ")
    print(f"La frecuencia eléctrica es de: {omega_sub_0_phi_m_f:.3e} Hz")
    print(f"La fuerza base de la mesa F_0 es de :{F_0:.3e} N")


    PUERTO = "COM3"
    BAUDIOS = 115200
    TAMANO_VENTANA = 512
    FS = 1000.0  # Frecuencia de muestreo supuesta (Hz)

    # Parámetros para filtros adaptativos
    MARGEN_FILTRO = 5.0  # Hz por encima/debajo de la frecuencia de excitación
    f_center = params.omega_Hz  # frecuencia de excitación actual (Hz)
    FC_PASA_ALTAS = max(0.5, f_center - MARGEN_FILTRO)
    FC_PASA_BAJAS = f_center + MARGEN_FILTRO
    # Límite de Nyquist
    if FC_PASA_BAJAS > FS / 2:
        FC_PASA_BAJAS = FS / 2 - 1
    print(
        f"Filtros adaptados: pasa-altas = {FC_PASA_ALTAS:.1f} Hz, pasa-bajas = {FC_PASA_BAJAS:.1f} Hz"
    )

    # Umbral de voltaje para descartar ruido (20 mV pico a pico)
    UMBRAL_VOLTAJE = 0.20  # voltios

    TIMEOUT_SEGUNDOS = 4
    MAX_REINTENTOS = 3

    # Opción de solo guardar sin gráficos (ahorro de recursos)
    MODO_SOLO_GUARDAR = bool(
        input("¿Desea ver gráficas (Enter) o guardar datos (escribir algo): ")
    )

    # Si se ven gráficas, seleccionar pestaña
    if not MODO_SOLO_GUARDAR:
        VISTA_SELECCIONADA = int(
            input(
                "¿Desea ver todo (0), FEM (1), posición (2), velocidad (3) o Transformada de Fourier (4): "
            )
        )

    # ---------------------------------------------------------------
    # Función para conectar al puerto serie con reintentos
    # ---------------------------------------------------------------
    def conectar_serial(puerto, baudios, intento=1):
        try:
            esp = serial.Serial(puerto, baudios, timeout=0.1)
            print(f"Conectado exitosamente a {puerto}. Recibiendo datos...")
            return esp
        except Exception as e:
            print(f"Error al conectar (intento {intento}): {e}")
            return None

    # Intento inicial de conexión
    reintentos = 0
    esp32 = None
    while reintentos < MAX_REINTENTOS and esp32 is None:
        esp32 = conectar_serial(PUERTO, BAUDIOS, reintentos + 1)
        if esp32 is None:
            reintentos += 1
            if reintentos < MAX_REINTENTOS:
                print(f"Reintentando en 2 segundos...")
                time.sleep(2)
    if esp32 is None:
        print(
            "No se pudo conectar al ESP32. Saliendo del modo 3."
        )
        return

    # Buffers circulares
    buffer_t = collections.deque(maxlen=TAMANO_VENTANA)
    buffer_v = collections.deque(maxlen=TAMANO_VENTANA)
    estado = {
        "tiempo_inicial": None,
        "ultima_actualizacion": time.time(),
        "conexion_activa": True,
    }

    # Pre-calcular los coeficientes de los filtros para no hacerlo en cada ciclo
    nyquist = 0.5 * FS
    b_altas, a_altas = butter(2, FC_PASA_ALTAS / nyquist, btype="high")
    b_bajas, a_bajas = butter(2, FC_PASA_BAJAS / nyquist, btype="low")

    def aplicar_filtros(tiempo, voltaje):
        """Filtro pasa-altas + pasa-bajas usando filtfilt (sin desfase)."""
        v_filt = filtfilt(b_altas, a_altas, voltaje)
        v_filt = filtfilt(b_bajas, a_bajas, v_filt)
        return v_filt * factor_ESP_32

    # ---------------------------------------------------------------
    # Función para reiniciar la conexión si se pierden los datos
    # ---------------------------------------------------------------
    def reiniciar_conexion():
        nonlocal esp32, reintentos
        print(
            f"\n⚠️ Sin datos durante {TIMEOUT_SEGUNDOS} s. Reintentando..."
        )
        try:
            esp32.close()
        except:
            pass
        time.sleep(1)
        reintentos += 1
        if reintentos < MAX_REINTENTOS:
            esp32 = conectar_serial(PUERTO, BAUDIOS, reintentos + 1)
            if esp32 is not None:
                # Limpiar buffers y estado
                buffer_t.clear()
                buffer_v.clear()
                estado["tiempo_inicial"] = None
                estado["ultima_actualizacion"] = time.time()
                estado["conexion_activa"] = True
                print("Conexión restablecida.")
            else:
                estado["conexion_activa"] = False
                print("Fallo en la reconexión.")
        else:
            print("Máximo de reintentos alcanzado. Cerrando modo de visualización.")
            estado["conexion_activa"] = False
            plt.close("all")
            # Salimos de la función Solver (opcional: podríamos solo salir del modo)
            raise SystemExit("Se perdió la comunicación con el ESP32.")

    # -----------------------------------------------
    # SUB-MODO: SOLO GUARDAR DATOS (HEADLESS)
    # -----------------------------------------------
    if MODO_SOLO_GUARDAR:
        os.makedirs("logs", exist_ok=True)
        nombre_archivo = (
            f"logs/log_sismometro_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        print("\n=======================================================")
        print("MODO LOGEO ACTIVO - VISUALIZACIÓN GRÁFICA APAGADA")
        print(f"Guardando datos en: {nombre_archivo}")
        print("Presione Ctrl+C en la consola para detener la captura.")
        print("=======================================================\n")

        try:
            with open(nombre_archivo, mode="w", newline="") as archivo_csv:
                escritor_csv = csv.writer(archivo_csv)
                escritor_csv.writerow(
                    [
                        "Marca_de_Tiempo",
                        "FEM_Real_Pico_(V)",
                        "Velocidad_Max_(m/s)",
                        "Posicion_Max_(m)",
                        "Frec_Dominante_FFT_(Hz)",
                        "Amplitud_Max_FFT",
                    ]
                )
                ultimo_log = time.time()
                while True:
                    if not estado["conexion_activa"]:
                        break
                    while esp32.in_waiting > 0:
                        try:
                            linea = (
                                esp32.readline()
                                .decode("utf-8", errors="ignore")
                                .strip()
                            )
                            if "," in linea:
                                t_micros_str, v_str = linea.split(",")
                                t_micros = int(t_micros_str)
                                v_crudo = float(v_str)
                                if abs(v_crudo) > 10.0:
                                    continue
                                if estado["tiempo_inicial"] is None:
                                    estado["tiempo_inicial"] = t_micros
                                t_segundos = (
                                    t_micros - estado["tiempo_inicial"]
                                ) / 1e6
                                buffer_t.append(t_segundos)
                                buffer_v.append(v_crudo)
                                estado["ultima_actualizacion"] = time.time()
                        except Exception:
                            pass
                    # Timeout
                    if (
                        time.time() - estado["ultima_actualizacion"]
                        > TIMEOUT_SEGUNDOS
                    ):
                        reiniciar_conexion()
                        if not estado["conexion_activa"]:
                            break
                        else:
                            continue
                    # Procesar cada 0.1 s si la ventana está llena
                    if (
                        len(buffer_t) == TAMANO_VENTANA
                        and (time.time() - ultimo_log) >= 0.1
                    ):
                        ultimo_log = time.time()
                        t_arr = np.array(buffer_t)
                        v_arr = np.array(buffer_v) - subida_voltaje
                        v_filtrado = aplicar_filtros(t_arr, v_arr)
                        v_real = v_filtrado * factor_amplificacion
                        v_real = np.clip(v_real, -0.8, 0.8) # Restringir los picos de voltaje a +-0.8V
                        # Umbral: si la señal es muy pequeña, se guardan ceros
                        if np.max(np.abs(v_real)) < UMBRAL_VOLTAJE:
                            escritor_csv.writerow(
                                [
                                    datetime.now().strftime("%H:%M:%S.%f")[:-3],
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                ]
                            )
                            continue
                        velocidad_real = -v_real / (G_sub_A * R_porcentaje)
                        posicion_iman = cumulative_trapezoid(
                            velocidad_real, t_arr, initial=0
                        )
                        N = TAMANO_VENTANA
                        T_muestreo = 1.0 / FS
                        yf = fft(v_real)
                        xf = fftfreq(N, T_muestreo)[: N // 2]
                        amplitud_fft = 2.0 / N * np.abs(yf[0 : N // 2])
                        indice_max_fft = np.argmax(amplitud_fft)
                        timestamp_log = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        escritor_csv.writerow(
                            [
                                timestamp_log,
                                np.max(np.abs(v_real)),
                                np.max(np.abs(velocidad_real)),
                                np.max(np.abs(posicion_iman)),
                                xf[indice_max_fft],
                                amplitud_fft[indice_max_fft],
                            ]
                        )
        except KeyboardInterrupt:
            print("\nCaptura detenida por el usuario. Archivo guardado.")
        except SystemExit:
            pass
        finally:
            if esp32 and esp32.is_open:
                esp32.close()
        return

    # -----------------------------------------------
    # SUB-MODO: VISUALIZACIÓN CON GRÁFICAS EN TIEMPO REAL (PYQTGRAPH)
    # -----------------------------------------------
    else:
        os.makedirs("logs", exist_ok=True)
        nombre_archivo = f"logs/log_sismometro_gui_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        archivo_csv = open(nombre_archivo, mode="w", newline="")
        escritor_csv = csv.writer(archivo_csv)
        escritor_csv.writerow([
            "Marca_de_Tiempo", "FEM_Real_Pico_(V)", "Velocidad_Max_(m/s)",
            "Posicion_Max_(m)", "Frec_Dominante_FFT_(Hz)", "Amplitud_Max_FFT"
        ])

        app = pg.mkQApp("Sismometro DAQ")
        win = pg.GraphicsLayoutWidget(title=f"DAQ Sismómetro EAFIT - Vista Modo {VISTA_SELECCIONADA}")
        win.resize(800, 600)
        # Asegurar que se posicione en un punto visible de la pantalla
        win.move(100, 100)
        win.setBackground('k')
        win.show()

        # Dictionary to store plot items and curves
        plots = {}
        curves = {}

        if VISTA_SELECCIONADA in [0, 1]:
            p_fem = win.addPlot(title="Dominio del Tiempo: Voltaje Inducido (FEM)")
            p_fem.setLabel('left', "Voltaje", units='V')
            p_fem.setLabel('bottom', "Tiempo", units='s')
            p_fem.showGrid(x=True, y=True)
            c_fem = p_fem.plot(pen=pg.mkPen('b', width=2), name="FEM Real")
            plots['fem'] = p_fem
            curves['fem'] = c_fem
            if VISTA_SELECCIONADA == 0: win.nextRow()

        if VISTA_SELECCIONADA in [0, 2]:
            p_vel = win.addPlot(title="Cinemática: Velocidad Estimada")
            p_vel.setLabel('left', "Velocidad", units='m/s')
            p_vel.setLabel('bottom', "Tiempo", units='s')
            p_vel.showGrid(x=True, y=True)
            c_vel = p_vel.plot(pen=pg.mkPen(color=(255,165,0), width=2), name="Velocidad del Imán")
            plots['vel'] = p_vel
            curves['vel'] = c_vel
            if VISTA_SELECCIONADA == 0: win.nextRow()

        if VISTA_SELECCIONADA in [0, 3]:
            p_pos = win.addPlot(title="Cinemática: Posición Estimada")
            p_pos.setLabel('left', "Desplazamiento", units='m')
            p_pos.setLabel('bottom', "Tiempo", units='s')
            p_pos.showGrid(x=True, y=True)
            c_pos = p_pos.plot(pen=pg.mkPen('g', width=2), name="Posición del Imán")
            plots['pos'] = p_pos
            curves['pos'] = c_pos
            if VISTA_SELECCIONADA == 0: win.nextRow()

        if VISTA_SELECCIONADA in [0, 4]:
            p_fft = win.addPlot(title="Dominio de la Frecuencia (FFT)")
            p_fft.setLabel('left', "Amplitud")
            p_fft.setLabel('bottom', "Frecuencia", units='Hz')
            p_fft.setXRange(0, 120)
            p_fft.showGrid(x=True, y=True)
            c_fft = p_fft.plot(pen=pg.mkPen('r', width=2))
            plots['fft'] = p_fft
            curves['fft'] = c_fft

        # Flag to control the background thread
        corriendo = True
        
        def hilo_lectura():
            while corriendo:
                if not estado["conexion_activa"]:
                    time.sleep(0.1)
                    continue

                try:
                    if esp32.in_waiting > 0:
                        linea = esp32.readline().decode("utf-8", errors="ignore").strip()
                        if "," in linea:
                            t_micros_str, v_str = linea.split(",")
                            t_micros = int(t_micros_str)
                            v_crudo = float(v_str)
                            if abs(v_crudo) > 10.0:
                                continue
                            if estado["tiempo_inicial"] is None:
                                estado["tiempo_inicial"] = t_micros
                            t_segundos = (t_micros - estado["tiempo_inicial"]) / 1e6
                            buffer_t.append(t_segundos)
                            buffer_v.append(v_crudo)
                            estado["ultima_actualizacion"] = time.time()
                    else:
                        time.sleep(0.001)
                except Exception:
                    pass

                # Check timeout in the background thread
                if time.time() - estado["ultima_actualizacion"] > TIMEOUT_SEGUNDOS:
                    reiniciar_conexion()
                    if not estado["conexion_activa"]:
                        break

        # Start the background thread
        hilo = threading.Thread(target=hilo_lectura, daemon=True)
        hilo.start()

        def actualizar():
            if not estado["conexion_activa"]:
                timer.stop()
                app.quit()
                return

            if len(buffer_t) == TAMANO_VENTANA:
                t_arr = np.array(buffer_t)
                t_escalado = t_arr * ESCALA_TIEMPO
                v_arr = np.array(buffer_v) - subida_voltaje
                v_filtrado = aplicar_filtros(t_arr, v_arr)
                v_real = v_filtrado * factor_amplificacion
                v_real = np.clip(v_real, -0.8, 0.8) # Restringir los picos de voltaje a +-0.8V

                if np.max(np.abs(v_real)) < UMBRAL_VOLTAJE:
                    v_real[:] = 0.0 # Silenciar el ruido (baja amplitud) en vez de congelar la gráfica
                    velocidad_real = np.zeros_like(v_real)
                    posicion_iman = np.zeros_like(v_real)
                    N = TAMANO_VENTANA
                    T_muestreo = 1.0 / FS
                    xf = fftfreq(N, T_muestreo)[: N // 2]
                    amplitud_fft = np.zeros_like(xf)
                    escritor_csv.writerow([datetime.now().strftime("%H:%M:%S.%f")[:-3], 0.0, 0.0, 0.0, 0.0, 0.0])
                else:
                    velocidad_real = -v_real / (G_sub_A * R_porcentaje)
                    posicion_iman = cumulative_trapezoid(velocidad_real, t_arr, initial=0)
                    N = TAMANO_VENTANA
                    T_muestreo = 1.0 / FS
                    yf = fft(v_real)
                    xf = fftfreq(N, T_muestreo)[: N // 2]
                    amplitud_fft = 2.0 / N * np.abs(yf[0 : N // 2])
                    
                    indice_max_fft = np.argmax(amplitud_fft)
                    escritor_csv.writerow([
                        datetime.now().strftime("%H:%M:%S.%f")[:-3],
                        np.max(np.abs(v_real)),
                        np.max(np.abs(velocidad_real)),
                        np.max(np.abs(posicion_iman)),
                        xf[indice_max_fft],
                        amplitud_fft[indice_max_fft]
                    ])

                if VISTA_SELECCIONADA in [0, 1]:
                    curves['fem'].setData(t_escalado, v_real)
                    margen = np.nanmax(np.abs(v_real)) * RANGO_VOLTAJE + 0.01
                    if np.isnan(margen) or np.isinf(margen): margen = 1.0
                    plots['fem'].setYRange(-margen, margen)

                if VISTA_SELECCIONADA in [0, 2]:
                    curves['vel'].setData(t_escalado, velocidad_real)
                    margen_vel = np.nanmax(np.abs(velocidad_real)) * 1.2 + 1e-6
                    if np.isnan(margen_vel) or np.isinf(margen_vel): margen_vel = 1.0
                    plots['vel'].setYRange(-margen_vel, margen_vel)

                if VISTA_SELECCIONADA in [0, 3]:
                    curves['pos'].setData(t_escalado, posicion_iman)
                    margen_pos = np.nanmax(np.abs(posicion_iman)) * 1.2 + 1e-7
                    if np.isnan(margen_pos) or np.isinf(margen_pos): margen_pos = 1.0
                    plots['pos'].setYRange(-margen_pos, margen_pos)

                if VISTA_SELECCIONADA in [0, 4]:
                    curves['fft'].setData(xf, amplitud_fft)

        timer = QtCore.QTimer()
        timer.timeout.connect(actualizar)
        timer.start(FACTOR_TIEMPO_ANIMACION)
        
        try:
            app.exec_()
        finally:
            corriendo = False # Detener el hilo de lectura
            if esp32 and esp32.is_open:
                esp32.close()
            archivo_csv.close() # Cerrar el log al terminar


def run_experimento_c(params: SectionParams, G_sub_L: float, G_sub_A: float):
    m = params.m
    c = params.c
    c_sub_lambda = params.c_sub_lambda
    k = params.k
    L = params.L
    C = params.C
    z_0 = params.z
    z_dot_0 = params.z_dot
    Q_0 = params.Q
    Q_dot_0 = params.Q_dot
    t = params.t
    F_0 = params.F_0
    omega = params.omega
    delta_t = params.delta_t
    G_sub_L, G_sub_A = Factores_Acople(params)
    XC = params.XC
    R = params.R
    R_porcentaje = params.R_porcentaje
    Zeta = params.Zeta
    alpha = params.alpha
    omega_sub_0_phi_m = params.omega_sub_0_phi_m
    R_sub_e = params.R_sub_e
    r_sub_p = params.r_sub_p
    eta = params.eta
    rho = params.rho
    m_mag = params.m_mag
    omega_sub_n = params.omega_sub_n
    omega_sub_n_f = omega_sub_n * 2 * np.pi
    omega_sub_0_phi_m_f = omega_sub_0_phi_m * 2 * np.pi
    a = params.a
    b = params.b
    factor_amplificacion = params.factor_amplificacion
    subida_voltaje = params.subida_voltaje
    omega_filtro = params.omega_filtro
    factor_ESP_32 = params.factor_ESP_32
    ESCALA_TIEMPO = params.ESCALA_TIEMPO
    FACTOR_TIEMPO_ANIMACION = params.FACTOR_TIEMPO_ANIMACION
    RANGO_VOLTAJE = params.RANGO_VOLTAJE

    # Impresión de parámetros relevantes
    print(f"masa estimada en :{m:.3e} Kg")
    print(f"El amortiguamiento auxiliar es:{c_sub_lambda:.3e}")
    print(f"La constante de amortiguamiento  es de :{c:.3e} Kg/s")
    print(f"La constante elástica es de :{k:.3e} Kg/s^2")
    print(f"La resistencia es de :{R:.3e} Ohmnios")
    print(f"La inductancia es de :{L:.3e} H")
    print(f"La capacitancia es de :{C:.3e} F")
    print(f"Momento magnético configurado en: {m_mag:.3e}")
    print(f"El factor Magnético de Área es: {G_sub_A:.3e}")
    print(f"El factor Magnético de Longitud es: {G_sub_L:.3e}")
    print(f"El factor de amortiguamiento es de: {Zeta:.3e}")
    print(f"La frecuencia natural es de: {omega_sub_n_f:.3e} Hz")
    print(f"La inercia eléctrica es de: {alpha:.3e} ")
    print(f"La frecuencia eléctrica es de: {omega_sub_0_phi_m_f:.3e} Hz")
    print(f"La fuerza base de la mesa F_0 es de :{F_0:.3e} N")


    print("Seleccione el método de estimación:")
    print("  1 - Newton-Raphson (basado en posición, requiere integración)")
    print("  2 - Ajuste senoidal (basado en velocidad, más rápido y robusto)")
    metodo_c = input("Opción (1/2): ").strip()

    PUERTO = "COM3"
    BAUDIOS = 115200
    FS = 1000.0  # Frecuencia de muestreo (Hz) consistente con ESP32
    UMBRAL_DISPARO = 0.05  # Voltaje de disparo (sobre el offset)
    TIEMPO_CAPTURA = 3.0  # Segundos de registro

    # --- Filtros adaptativos centrados en la frecuencia de excitación ---
    MARGEN_FILTRO = 5.0  # Hz
    f_center = params.omega_Hz  # frecuencia de excitación actual (Hz)
    FC_PASA_ALTAS = max(0.5, f_center - MARGEN_FILTRO)
    FC_PASA_BAJAS = f_center + MARGEN_FILTRO
    if FC_PASA_BAJAS > FS / 2:
        FC_PASA_BAJAS = FS / 2 - 1
    print(
        f"Filtros adaptados: pasa-altas = {FC_PASA_ALTAS:.1f} Hz, pasa-bajas = {FC_PASA_BAJAS:.1f} Hz"
    )

    # Umbral para descartar ruido (20 mV pico)
    UMBRAL_VOLTAJE = 0.02  # voltios

    # -----------------------------------------------------------------
    # Función de captura de datos (común a ambos métodos)
    # -----------------------------------------------------------------
    def capturar_evento_sismico():
        print(f"Conectando a {PUERTO}...")
        try:
            esp32 = serial.Serial(PUERTO, BAUDIOS)
        except Exception as e:
            print(f"Error de conexión: {e}")
            return None, None

        # Diseño de filtros digitales (usando las frecuencias adaptativas)
        nyquist = 0.5 * FS
        b_altas, a_altas = butter(2, FC_PASA_ALTAS / nyquist, btype="high")
        b_bajas, a_bajas = butter(2, FC_PASA_BAJAS / nyquist, btype="low")

        pre_trigger = 100
        historial_t = collections.deque(maxlen=pre_trigger)
        historial_v = collections.deque(maxlen=pre_trigger)
        datos_t, datos_v = [], []
        muestras_objetivo = int(TIEMPO_CAPTURA * FS)
        muestras_grabadas = 0
        tiempo_inicial_micros = None
        estado = "IDLE"
        print("Esperando perturbación...")

        while muestras_grabadas < muestras_objetivo:
            if esp32.in_waiting > 0:
                try:
                    linea = esp32.readline().decode("utf-8").strip()
                    if "," not in linea:
                        continue
                    t_str, v_str = linea.split(",")
                    t_micros = int(t_str)
                    v_crudo = float(v_str) - subida_voltaje  # aplicar offset
                    if abs(v_crudo) > 10.0:
                        continue
                    if tiempo_inicial_micros is None:
                        tiempo_inicial_micros = t_micros
                    t_segundos = (t_micros - tiempo_inicial_micros) / 1e6

                    # Aplicar filtros en tiempo real a la señal (ventana deslizante)
                    # Para simplificar, filtramos después de capturar, pero aquí aplicamos un filtrado básico
                    if estado == "IDLE":
                        historial_t.append(t_segundos)
                        historial_v.append(v_crudo)
                        # Detección de disparo sobre la señal filtrada (aproximación)
                        if abs(v_crudo) > UMBRAL_DISPARO:
                            print("¡Trigger! Capturando onda...")
                            datos_t.extend(historial_t)
                            datos_v.extend(historial_v)
                            estado = "RECORDING"
                    elif estado == "RECORDING":
                        datos_t.append(t_segundos)
                        datos_v.append(v_crudo)
                        muestras_grabadas += 1
                except Exception:
                    continue

        esp32.close()

        # Convertir a arrays y aplicar filtro completo (pasa-altas + pasa-bajas) sin desfase
        t_raw = np.array(datos_t)
        v_raw = np.array(datos_v) * factor_ESP_32

        if len(v_raw) == 0:
            print("No se capturaron datos.")
            return None, None

        # Filtrado de la señal capturada
        v_filt = filtfilt(b_altas, a_altas, v_raw)
        v_filt = filtfilt(b_bajas, a_bajas, v_filt)

        # Umbral: si la señal es muy pequeña, se descarta
        if np.max(np.abs(v_filt)) < UMBRAL_VOLTAJE:
            print("Señal capturada por debajo del umbral. Intente nuevamente.")
            return None, None

        t_onda = t_raw - t_raw[0]  # reiniciar tiempo a cero
        v_onda = v_filt
        print(f"Captura finalizada. Puntos: {len(t_onda)}")
        return t_onda, v_onda

    # -----------------------------------------------------------------
    # MÉTODO 1: NEWTON-RAPHSON (basado en posición, con integral)
    # -----------------------------------------------------------------
    if metodo_c == "1":
        print("\n--- Usando método de Newton-Raphson (posición) ---")

        def solver_newton_raphson_ventana(
            t_w, v_w, c_inicial, G_sub_A, factor_amp, m, k, omega, omega_n, F_0
        ):
            integral_voltaje = simpson(v_w, x=t_w)
            Electro = (-1 / (G_sub_A * factor_amp)) * integral_voltaje
            c = c_inicial
            for _ in range(80):
                c_safe = c if abs(c) > 1e-9 else 1e-9
                discriminante = np.sqrt(np.abs(c_safe**2 - 4 * m * k))
                disc_safe = discriminante if discriminante > 1e-9 else 1e-9
                zi = c_safe / (2 * np.sqrt(m * k))
                den_arctan = 1 - (omega**2 / omega_n**2)
                den_arctan = den_arctan if abs(den_arctan) > 1e-9 else 1e-9
                X_interno = (2 * zi * (omega / omega_n)) / den_arctan
                phi = np.arctan(X_interno)
                g = np.cos(phi)
                h = 1 - (discriminante / c_safe)
                p = np.exp(t_w * (-c_safe - discriminante) / (2 * m)) - np.exp(
                    t_w * (-c_safe + discriminante) / (2 * m)
                )
                dX_dc = (omega / omega_n) / (np.sqrt(m * k) * den_arctan)
                dphi_dc = (1 / (1 + X_interno**2)) * dX_dc
                g_prime = -np.sin(phi) * dphi_dc
                h_prime = (-4 * m * k) / (c_safe**2 * disc_safe)
                term1 = (-1 - c_safe / disc_safe) * np.exp(
                    t_w * (-c_safe - discriminante) / (2 * m)
                )
                term2 = (-1 + c_safe / disc_safe) * np.exp(
                    t_w * (-c_safe + discriminante) / (2 * m)
                )
                p_prime = (t_w / (2 * m)) * (term1 - term2)
                p_mean = np.mean(p)
                p_prime_mean = np.mean(p_prime)
                f_c = Electro - (F_0 / 2) * (g * h * p_mean)
                df_dc = -(F_0 / 2) * (
                    g_prime * h * p_mean
                    + g * h_prime * p_mean
                    + g * h * p_prime_mean
                )
                if abs(df_dc) < 1e-12:
                    break
                c_nuevo = c_safe - (f_c / df_dc)
                if abs(c_nuevo - c_safe) < 1e-6:
                    return c_nuevo
                c = c_nuevo
            return c

        def analizar_y_graficar(t_onda, v_onda):
            print("Procesando ventanas deslizantes (Newton-Raphson)...")
            VENTANA = 300
            PASO = 50
            t_ventana, c_evolucion = [], []
            c_semilla = 1.0
            for i in range(0, len(t_onda) - VENTANA, PASO):
                t_w = t_onda[i : i + VENTANA]
                v_w = v_onda[i : i + VENTANA]
                c_calc = solver_newton_raphson_ventana(
                    t_w,
                    v_w,
                    c_semilla,
                    G_sub_A,
                    factor_amplificacion,
                    m,
                    k,
                    omega,
                    omega_sub_n,
                    F_0,
                )
                c_evolucion.append(c_calc)
                t_ventana.append(t_w[len(t_w) // 2])
                c_semilla = c_calc
            
            c_arr = np.array(c_evolucion, dtype=float)
            c_arr[(c_arr < 0) | (c_arr > 3)] = np.nan
            valid_c = c_arr[~np.isnan(c_arr)]
            
            print("\n========== RESULTADOS (Newton-Raphson) ==========")
            if len(valid_c) > 0:
                print(f" Media c = {np.mean(valid_c):.6f} Ns/m")
                print(f" Mediana   = {np.median(valid_c):.6f}")
                print(f" Desv. est. = {np.std(valid_c):.6f}")
            else:
                print(" No hay valores válidos en el rango [0, 3].")
            print("==================================================")
            # Gráfica
            fig, ax1 = plt.subplots(figsize=(12, 7))
            plt.style.use("seaborn-v0_8-darkgrid")
            ax1.set_xlabel("Tiempo (s)")
            ax1.set_ylabel("Voltaje Centrado (V)", color="tab:blue")
            ax1.plot(
                t_onda, v_onda, color="tab:blue", alpha=0.4, label="FEM (filtrada)"
            )
            ax1.tick_params(axis="y", labelcolor="tab:blue")
            ax2 = ax1.twinx()
            ax2.set_ylabel("c(t) [Ns/m]", color="tab:red")
            ax2.plot(
                t_ventana,
                c_evolucion,
                color="tab:red",
                linewidth=2.5,
                marker=".",
                label="c estimado",
            )
            ax2.tick_params(axis="y", labelcolor="tab:red")
            fig.suptitle(
                "Método Newton-Raphson (basado en posición)",
                fontsize=14,
                fontweight="bold",
            )
            fig.tight_layout()
            plt.show()

        t_data, v_data = capturar_evento_sismico()
        if t_data is not None:
            analizar_y_graficar(t_data, v_data)

    # -----------------------------------------------------------------
    # MÉTODO 2: AJUSTE SENOIDAL (basado en velocidad)
    # -----------------------------------------------------------------
    elif metodo_c == "2":
        print("\n--- Usando método de ajuste senoidal (velocidad) ---")

        # Convertir voltaje a velocidad (ya sin offset y con ganancia)
        # v_real (FEM) ya está en voltios. La velocidad real es v_real / (G_sub_A * R_porcentaje)
        # pero la constante la podemos absorber en el ajuste, o mejor extraer la amplitud de velocidad.
        def estimar_c_desde_velocidad(
            t, v_fem, F_0, k, m, omega, G_sub_A, R_porcentaje
        ):
            # Primero obtenemos la velocidad real a partir de la FEM
            velocidad = v_fem / (G_sub_A * R_porcentaje)  # [m/s]

            # Modelo: v(t) = A * cos(omega*t - phi) + C
            def modelo(t, A, phi, C):
                return A * np.cos(omega * t - phi) + C

            # Ajuste por mínimos cuadrados
            try:
                p0 = [np.max(np.abs(velocidad)), 0.0, 0.0]
                popt, _ = curve_fit(modelo, t, velocidad, p0=p0)
                A_est, phi_est, C_est = popt
            except Exception as e:
                print(f"Error en el ajuste senoidal: {e}")
                return None, None, None, None
            # Calcular c a partir de la amplitud
            discriminante = (F_0 / A_est) ** 2 - (k - m * omega**2) ** 2
            if discriminante < 0:
                c_amp = None
            else:
                c_amp = np.sqrt(discriminante) / omega
            # Calcular c a partir de la fase (solo si el denominador no es cero)
            den_fase = k - m * omega**2
            if abs(den_fase) < 1e-9:
                c_fase = None
            else:
                c_fase = (den_fase / omega) * np.tan(phi_est)
            return c_amp, c_fase, A_est, phi_est

        def analizar_y_graficar_velocidad(t_onda, v_onda):
            print("Procesando ventanas deslizantes (ajuste senoidal)...")
            VENTANA = 300
            PASO = 50
            t_ventana, c_amp_list, c_fase_list = [], [], []
            # Se selecciona una porción estable (por ejemplo, después de los primeros 0.5 s)
            # para evitar transitorios. En ventanas pequeñas también funciona, pero mejor estabilizado.
            for i in range(0, len(t_onda) - VENTANA, PASO):
                t_w = t_onda[i : i + VENTANA]
                v_w = v_onda[i : i + VENTANA]
                c_amp, c_fase, A_est, phi_est = estimar_c_desde_velocidad(
                    t_w, v_w, F_0, k, m, omega, G_sub_A, R_porcentaje
                )
                if c_amp is not None:
                    c_amp_list.append(c_amp)
                else:
                    c_amp_list.append(np.nan)
                if c_fase is not None:
                    c_fase_list.append(c_fase)
                else:
                    c_fase_list.append(np.nan)
                t_ventana.append(t_w[len(t_w) // 2])
            # Mostrar resultados estadísticos
            c_amp_arr = np.array(c_amp_list, dtype=float)
            c_fase_arr = np.array(c_fase_list, dtype=float)
            
            c_amp_arr[(c_amp_arr < 0) | (c_amp_arr > 3)] = np.nan
            c_fase_arr[(c_fase_arr < 0) | (c_fase_arr > 3)] = np.nan
            
            valid_amp = c_amp_arr[~np.isnan(c_amp_arr)]
            valid_fase = c_fase_arr[~np.isnan(c_fase_arr)]
            
            print("\n========== RESULTADOS (Ajuste senoidal) ==========")
            if len(valid_amp) > 0:
                print(
                    f" c por AMPLITUD: media = {np.mean(valid_amp):.6f} Ns/m, mediana = {np.median(valid_amp):.6f}, desv = {np.std(valid_amp):.6f}"
                )
            if len(valid_fase) > 0:
                print(
                    f" c por FASE:     media = {np.mean(valid_fase):.6f} Ns/m, mediana = {np.median(valid_fase):.6f}, desv = {np.std(valid_fase):.6f}"
                )
            print("===================================================")
            # Gráfica comparativa
            fig, ax1 = plt.subplots(figsize=(12, 7))
            plt.style.use("seaborn-v0_8-darkgrid")
            ax1.set_xlabel("Tiempo (s)")
            ax1.set_ylabel("Voltaje Centrado (V)", color="tab:blue")
            ax1.plot(
                t_onda, v_onda, color="tab:blue", alpha=0.4, label="FEM (filtrada)"
            )
            ax1.tick_params(axis="y", labelcolor="tab:blue")
            ax2 = ax1.twinx()
            ax2.set_ylabel("c(t) [Ns/m]", color="tab:red")
            if len(c_amp_arr) > 0:
                ax2.plot(
                    t_ventana[: len(c_amp_arr)],
                    c_amp_arr,
                    "o-",
                    color="tab:red",
                    label="c por amplitud",
                )
            if len(c_fase_arr) > 0:
                ax2.plot(
                    t_ventana[: len(c_fase_arr)],
                    c_fase_arr,
                    "s-",
                    color="tab:orange",
                    label="c por fase",
                )
            ax2.tick_params(axis="y", labelcolor="tab:red")
            ax2.legend(loc="upper right")
            fig.suptitle(
                "Método de Ajuste Senoidal (basado en velocidad)",
                fontsize=14,
                fontweight="bold",
            )
            fig.tight_layout()
            plt.show()

        t_data, v_data = capturar_evento_sismico()
        if t_data is not None:
            analizar_y_graficar_velocidad(t_data, v_data)

    else:
        print("Opción no válida. Saliendo del modo 5.")


