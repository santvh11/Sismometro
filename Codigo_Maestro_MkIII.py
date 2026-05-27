import serial
from dataclasses import dataclass, field
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid, simpson
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
import collections
import csv
import time
from datetime import datetime
import logging
from Params import SismometroParamsDict, params_base

# -------------------------------------------------------------------
# 0. Selección del modelo:
# -------------------------------------------------------------------

while True:
    modelo = input(
        "Escriba (1) para mk_1 (lineal) o (2) para mk_2 (no lineal), "
        "(3) para ver voltaje en tiempo real, (4) para descargar datos crudos, "
        "o (5) para hallar el valor de 'c' experimentalmente: "
    ).strip()
    if modelo == "1":
        modelo_mk1, modelo_mk2, simular, datos_descarga, experimento_c = (
            True,
            False,
            False,
            False,
            False,
        )
        break
    elif modelo == "2":
        modelo_mk1, modelo_mk2, simular, datos_descarga, experimento_c = (
            False,
            True,
            False,
            False,
            False,
        )
        break
    elif modelo == "3":
        modelo_mk1, modelo_mk2, simular, datos_descarga, experimento_c = (
            False,
            False,
            True,
            False,
            False,  # solo simular, sin modelos teóricos
        )
        break
    elif modelo == "4":
        modelo_mk1, modelo_mk2, simular, datos_descarga, experimento_c = (
            False,
            False,
            False,
            True,
            False,
        )
        break
    elif modelo == "5":
        modelo_mk1, modelo_mk2, simular, datos_descarga, experimento_c = (
            False,
            False,
            False,
            False,
            True,
        )
        break
    else:
        print("Entrada no válida. Inténtalo de nuevo.")


# -------------------------------------------------------------------
# Factores de acoplamiento electromecánico (G_sub_L y G_sub_A)
# -------------------------------------------------------------------
def Factores_Acople(params: SismometroParamsDict) -> tuple[float, float]:
    """
    Calcula los factores de acople:
    - G_sub_L : fuerza de Laplace por unidad de corriente (N/A)
    - G_sub_A : fuerza contraelectromotriz por unidad de velocidad (V·s/m)
    """
    # Extracción de variables respetando la anidación del TypedDict
    e_sub_p = params["geometria"]["e_sub_p"]
    r_sub_p = params["geometria"]["r_sub_p"]
    mu_sub_cero = params["magnetismo"]["mu_sub_cero"]
    m_mag = params["magnetismo"]["m_mag"]
    e_sub_cs = params["geometria"]["e_sub_cs"]
    N_sub_cs_total = params["geometria"]["N_sub_c_total"]

    constantes_magneticas = (0.25 * m_mag * mu_sub_cero) / np.pi
    Radio_total = e_sub_p + r_sub_p
    Paso = e_sub_cs * 0.5 / np.pi

    Integral_linea = (1 / Radio_total**3) - (
        1 / (Radio_total**2 + (Paso * 2 * np.pi + N_sub_cs_total) ** 2) ** (3 / 2)
    )

    G_sub_A = (constantes_magneticas * 3 * Radio_total**2 / Paso) * Integral_linea
    G_sub_L = (constantes_magneticas * Radio_total) / (
        Paso * (Radio_total**2 + (4 * np.pi * Paso) ** 2) ** 3
    )

    return G_sub_L, G_sub_A


# Prueba de la función
g_l, g_a = Factores_Acople(datos_sismometro)
print(f"G_sub_L: {g_l}")
print(f"G_sub_A: {g_a}")


# -------------------------------------------------------------------
# 2. Solver de EDO´s (modelos mk1, mk2 y modos experimentales)
# -------------------------------------------------------------------
def Solver(
    modelo_mk1: bool,
    modelo_mk2: bool,
    simular: bool,
    datos_descarga: bool,
    experimento_c: bool,
    params: SismometroParamsDict,
):
    """
    Función principal que resuelve las ecuaciones del sistema según el modo elegido.
    """
    # Desempaquetado de parámetros respetando los diccionarios anidados
    m = params["mecanicos"]["m"]
    c = params["mecanicos"]["c"]
    c_sub_lambda = params["derivados"]["c_sub_lambda"]
    k = params["mecanicos"]["k"]
    L = params["electricos"]["L"]
    C = params["electricos"]["C"]
    z_0 = params["iniciales"]["z"]
    z_dot_0 = params["iniciales"]["z_dot"]
    Q_0 = params["iniciales"]["Q"]
    Q_dot_0 = params["iniciales"]["Q_dot"]
    t = params["derivados"]["t"]
    F_0 = params["derivados"]["F_0"]
    omega = params["simulacion"]["omega"]
    delta_t = params["simulacion"]["delta_t"]
    G_sub_L, G_sub_A = Factores_Acople(params)
    XC = params["derivados"]["XC"]
    R = params["electricos"]["R"]
    R_porcentaje = params["derivados"]["R_porcentaje"]
    Zeta = params["derivados"]["Zeta"]
    alpha = params["derivados"]["alpha"]
    omega_sub_0_phi_m = params["derivados"]["omega_sub_0_phi_m"]
    R_sub_e = params["geometria"]["R_sub_e"]
    r_sub_p = params["geometria"]["r_sub_p"]
    eta = params["fluidos"]["eta"]
    rho = params["derivados"]["rho"]
    m_mag = params["magnetismo"]["m_mag"]
    omega_sub_n = params["derivados"]["omega_sub_n"]
    a = params["simulacion"]["a"]
    b = params["simulacion"]["b"]
    factor_amplificacion = params["electricos"]["factor_amplificacion"]
    subida_voltaje = params["electricos"]["subida_voltaje"]

    omega_sub_n_f = omega_sub_n * 2 * np.pi
    omega_sub_0_phi_m_f = omega_sub_0_phi_m * 2 * np.pi

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

    # =================================================================
    # MODO 1: MODELO MK1 (LINEAL, SOLUCIÓN ANALÍTICA POR VALORES PROPIOS)
    # =================================================================
    if modelo_mk1:
        print("--- Entrando a Modelo MK1 (Lineal) ---")
        # Matriz de estado del sistema acoplado
        A = np.array(
            [
                [0, 1, 0, 0],
                [-k / m, -c / m, 0, -G_sub_L / m],
                [0, 0, 0, 1],
                [0, -G_sub_A / L, -1 / (L * C), -R / L],
            ]
        )
        eigenvalues, eigenvectors = np.linalg.eig(A)

        # Vector de fuerza externa
        F_t = np.array([[0], [1 / m], [0], [0]])
        I = np.eye(4)
        lado_izquierdo = 1j * omega * I - A
        lado_derecho = F_t * F_0
        Xp_complejo = np.linalg.solve(
            lado_izquierdo, lado_derecho
        )  # Solución particular fasorial

        # Solución particular en el tiempo
        X_particular = Xp_complejo * np.exp(1j * omega * t)

        # Condiciones iniciales (homogénea + particular en t=0)
        X_t_p = Xp_complejo.real
        X_t_h = np.array([[z_0, z_dot_0, Q_0, Q_dot_0]]).reshape(4, 1)
        X_t = X_t_h + X_t_p

        # Coeficientes de la solución homogénea
        coeficientes = np.linalg.solve(eigenvectors, X_t)
        X_evolucion = np.zeros((4, len(t)), dtype=complex)
        for i in range(len(eigenvalues)):
            termino_exponencial = np.exp(eigenvalues[i] * t)
            contribucion = np.outer(eigenvectors[:, i], termino_exponencial)
            X_evolucion += coeficientes[i] * contribucion

        X_total = X_evolucion + X_particular

        # Voltaje inducido por Faraday-Lenz (FEM = G_A * velocidad)
        FEM_FL = G_sub_A * R_porcentaje * X_total.real[1]
        t_filtro = t[::delta_t]

        # Voltaje por ley de Ohm fasorial (opcional)
        I_Re = X_total.real[3]
        I_Im = X_total.imag[3]
        V_Im = (R * I_Im) + (I_Re * XC)
        V_Re = (R * I_Re) - (I_Im * XC)
        V_A = np.sqrt(V_Im**2 + V_Re**2)
        V_phi = np.arctan2(V_Im, V_Re)
        FEM_Ohm = V_A * np.cos(omega * t + V_phi)

        F_Laplace = X_total.real[3] * G_sub_L  # Fuerza de retroalimentación

        # Gráficas de las variables de estado
        nombre_variables = [
            "Posición (z)",
            "Velocidad (ż)",
            "Carga (Q)",
            "Corriente (I)",
        ]
        unidades = ["[m]", "[m/s]", "[C]", "[A]"]
        paleta_colores = ["orange", "red", "blue", "green"]

        fig, axs = plt.subplots(2, 2, figsize=(7.5, 5), dpi=200)
        fig.suptitle(
            "Respuesta Temporal Completa - Detector de Gaia MK1",
            fontsize=16,
            fontweight="bold",
        )
        for i in range(4):
            ax = axs[i // 2, i % 2]
            ax.plot(
                t,
                X_total.real[i],
                color=paleta_colores[i],
                linewidth=2,
                label="Solución Total (Real)",
            )
            ax.plot(
                t,
                X_particular.real[i],
                "--",
                color="gray",
                alpha=0.4,
                label="Solo Estacionario",
            )
            ax.set_title(nombre_variables[i], fontweight="bold")
            ax.set_ylabel(f"Amplitud {unidades[i]}")
            ax.set_xlabel("Tiempo [s]")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper right", fontsize="small")
            ax.text(
                0.95,
                0.02,
                f"Pico Estac: {np.abs(Xp_complejo[i].item()):.2e}",
                transform=ax.transAxes,
                ha="right",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7),
            )
            ax.text(
                0,
                0.02,
                f"Máx: {np.max(X_total.real[i]):.2e}",
                transform=ax.transAxes,
                ha="left",
                fontsize=7,
                bbox=dict(facecolor="white", alpha=0.7),
            )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Gráfica de FEM (Ley de Ohm)
        plt.figure(figsize=(8, 6), dpi=200)
        plt.step(
            t_filtro,
            FEM_Ohm,
            where="post",
            label="Muestreo (Escalonado)",
            color="purple",
            linewidth=1,
        )
        plt.title("FEM teórica (Ley de Ohm Fasorial)")
        plt.xlabel("Tiempo [s]")
        plt.ylabel("FEM [V]")
        plt.legend()
        plt.show()

        # Gráfica de FEM (Faraday-Lenz)
        plt.figure(figsize=(8, 6), dpi=200)
        plt.step(
            t_filtro,
            FEM_FL,
            where="post",
            label="Muestreo (Escalonado)",
            color="purple",
            linewidth=1,
        )
        plt.title("FEM teórica (Faraday-Lenz)")
        plt.xlabel("Tiempo [s]")
        plt.ylabel("FEM [V]")
        plt.legend()
        plt.show()

        # Gráfica de la fuerza de Laplace
        plt.figure(figsize=(8, 6), dpi=200)
        plt.step(
            t,
            F_Laplace,
            where="post",
            label="Fuerza de repulsión electromagnética",
            color="blue",
            linewidth=1,
        )
        plt.title("Fuerza de Laplace")
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Newtons [N]")
        plt.legend()
        plt.show()

        # Estimación del número de Reynolds para verificar linealidad de c
        v_max = np.abs(np.max(X_total.real[1]))
        Re = v_max * (r_sub_p - R_sub_e) * rho / eta
        if Re < 0.1:
            print(f"Linealidad del amortiguamiento fiable. Reynolds = {Re:.3e}")
        elif 0.1 < Re < 0.9:
            print(f"Amortiguamiento ligeramente no lineal. Reynolds = {Re:.3e}")
        else:
            print(
                f"Amortiguamiento claramente no lineal (Re = {Re:.3e}) -> usar modelo MK2"
            )

    # =================================================================
    # MODO 2: MODELO MK2 (NO LINEAL, SOLUCIÓN NUMÉRICA CON SOLVE_IVP)
    # =================================================================
    elif modelo_mk2:
        print("--- Entrando a Modelo MK2 (No lineal, RK45) ---")

        # Definición de la matriz de estado con amortiguamiento cuadrático |v|*v
        def state_matrix(t, S):
            z, v, Q, I = S
            F_ext = F_0 * np.sin(omega * t)
            dz_dt = v
            dv_dt = (1 / m) * (-k * z - c * v * np.abs(v) + F_ext + I * G_sub_L)
            dQ_dt = I
            dI_dt = (1 / L) * (-R * I - (1 / C) * Q + v * G_sub_A)
            return [dz_dt, dv_dt, dQ_dt, dI_dt]

        t_span = (a, b)
        solver = solve_ivp(
            state_matrix,
            t_span,
            [z_0, z_dot_0, Q_0, Q_dot_0],
            t_eval=t,
            method="LSODA",
            rtol=1e-8,
            atol=1e-10,
        )

        if not solver.success:
            print("Error en el solver:", solver.message)
            return

        # Gráficas de las variables de estado (MK2)
        nombre_variables = [
            "Posición (z)",
            "Velocidad (ż)",
            "Carga (Q)",
            "Corriente (I)",
        ]
        unidades = ["[m]", "[m/s]", "[C]", "[A]"]
        paleta_colores = ["orange", "red", "blue", "green"]

        fig, axs = plt.subplots(2, 2, figsize=(7.5, 5), dpi=200)
        fig.suptitle(
            "Respuesta Temporal Completa - Detector de Gaia MK2 (No Lineal)",
            fontsize=16,
            fontweight="bold",
        )
        for i in range(4):
            ax = axs[i // 2, i % 2]
            ax.plot(
                solver.t,
                solver.y[i],
                color=paleta_colores[i],
                linewidth=2,
                label="Solución RK45",
            )
            ax.set_title(nombre_variables[i], fontweight="bold")
            ax.set_ylabel(f"Amplitud {unidades[i]}")
            ax.set_xlabel("Tiempo [s]")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper right", fontsize="small")
            ax.text(
                0.95,
                0.02,
                f"Pico: {np.max(np.abs(solver.y[i])):.2e}",
                transform=ax.transAxes,
                ha="right",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7),
            )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # FEM a partir de la velocidad (Faraday-Lenz)
        FEM_FL = G_sub_A * R_porcentaje * solver.y[1]
        t_filtro = solver.t[::delta_t]

        plt.figure(figsize=(8, 6), dpi=200)
        plt.step(
            t_filtro,
            FEM_FL,
            where="post",
            label="FEM (Faraday-Lenz)",
            color="purple",
            linewidth=1,
        )
        plt.title("FEM estimada (Modelo MK2)")
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Voltaje [V]")
        plt.legend()
        plt.show()

        # Fuerza de Laplace
        F_Laplace = solver.y[3] * G_sub_L
        plt.figure(figsize=(8, 6), dpi=200)
        plt.plot(
            solver.t, F_Laplace, color="blue", linewidth=1, label="Fuerza de Laplace"
        )
        plt.title("Fuerza de retroalimentación magnética")
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Newtons [N]")
        plt.legend()
        plt.show()

        # =================================================================
    # MODO 3: VISUALIZACIÓN EN TIEMPO REAL (DAQ CON ESP32)
    # =================================================================
    elif simular:
        print("\n--- Entrando a Adquisición de Datos en Tiempo Real ---")
        PUERTO = "COM3"
        BAUDIOS = 115200
        TAMANO_VENTANA = 512
        FS = 1000.0  # Frecuencia de muestreo supuesta (Hz)
        FC_PASA_ALTAS = 0.5
        FC_PASA_BAJAS = 40.0
        TIMEOUT_SEGUNDOS = 4  # Segundos sin datos para reiniciar la conexión
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
                esp = serial.Serial(puerto, baudios)
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
                "No se pudo conectar al ESP32 después de varios intentos. Saliendo del modo 3."
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

        def aplicar_filtros(tiempo, voltaje):
            """Filtro pasa-altas + pasa-bajas usando filtfilt para evitar desfase."""
            nyquist = 0.5 * FS
            b_altas, a_altas = butter(2, FC_PASA_ALTAS / nyquist, btype="high")
            b_bajas, a_bajas = butter(2, FC_PASA_BAJAS / nyquist, btype="low")
            v_filt = filtfilt(b_altas, a_altas, voltaje)
            v_filt = filtfilt(b_bajas, a_bajas, v_filt)
            return v_filt

        # ---------------------------------------------------------------
        # Función para reiniciar la conexión si se pierden los datos
        # ---------------------------------------------------------------
        def reiniciar_conexion():
            nonlocal esp32, reintentos
            print(
                "\n⚠️  Sin datos durante {} segundos. Reintentando conexión...".format(
                    TIMEOUT_SEGUNDOS
                )
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
            nombre_archivo = (
                f"log_sismometro_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
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
                        # Control de timeout también en modo guardado
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
                        # Verificar timeout
                        if (
                            time.time() - estado["ultima_actualizacion"]
                            > TIMEOUT_SEGUNDOS
                        ):
                            reiniciar_conexion()
                            if not estado["conexion_activa"]:
                                break
                            else:
                                continue
                        # Procesar cada 0.1 segundos si hay ventana llena
                        if (
                            len(buffer_t) == TAMANO_VENTANA
                            and (time.time() - ultimo_log) >= 0.1
                        ):
                            ultimo_log = time.time()
                            t_arr = np.array(buffer_t)
                            v_arr = np.array(buffer_v) - subida_voltaje
                            v_filtrado = aplicar_filtros(t_arr, v_arr)
                            v_real = v_filtrado / factor_amplificacion
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
                esp32.close()
            except SystemExit:
                pass
            finally:
                if esp32 and esp32.is_open:
                    esp32.close()
            return

        # -----------------------------------------------
        # SUB-MODO: VISUALIZACIÓN CON GRÁFICAS EN TIEMPO REAL
        # -----------------------------------------------
        else:
            # Configuración de los subplots según la vista seleccionada
            if VISTA_SELECCIONADA == 0:
                fig, axs = plt.subplots(4, 1, figsize=(12, 11))
                ax_fem, ax_vel, ax_pos, ax_fft = axs
            else:
                fig, ax_unico = plt.subplots(1, 1, figsize=(12, 7))
                if VISTA_SELECCIONADA == 1:
                    ax_fem = ax_unico
                elif VISTA_SELECCIONADA == 2:
                    ax_vel = ax_unico
                elif VISTA_SELECCIONADA == 3:
                    ax_pos = ax_unico
                elif VISTA_SELECCIONADA == 4:
                    ax_fft = ax_unico

            fig.canvas.manager.set_window_title(
                f"DAQ Sismómetro EAFIT - Vista Modo {VISTA_SELECCIONADA}"
            )

            # Inicialización de líneas según la vista
            if VISTA_SELECCIONADA in [0, 1]:
                (linea_fem_cruda,) = ax_fem.plot(
                    [], [], lw=1.5, color="purple", label="FEM Amplificada", alpha=0.7
                )
                (linea_fem_real,) = ax_fem.plot(
                    [], [], lw=1.5, color="blue", label="FEM Real"
                )
                ax_fem.set_title("Dominio del Tiempo: Voltaje Inducido (FEM)")
                ax_fem.set_ylabel("Voltaje (V)")
                ax_fem.grid(True)
                ax_fem.legend(loc="upper right")
                # Texto de timeout
                text_timeout = ax_fem.text(
                    0.5,
                    0.95,
                    "",
                    transform=ax_fem.transAxes,
                    ha="center",
                    fontsize=10,
                    color="red",
                    bbox=dict(facecolor="white", alpha=0.8),
                )
            if VISTA_SELECCIONADA in [0, 2]:
                (linea_vel,) = ax_vel.plot(
                    [], [], lw=1.5, color="orange", label="Velocidad del Imán"
                )
                ax_vel.set_title("Cinemática: Velocidad Estimada")
                ax_vel.set_ylabel("Velocidad (m/s)")
                ax_vel.grid(True)
                ax_vel.legend(loc="upper right")
                if VISTA_SELECCIONADA == 0:
                    text_timeout = ax_vel.text(
                        0.5,
                        0.95,
                        "",
                        transform=ax_vel.transAxes,
                        ha="center",
                        fontsize=10,
                        color="red",
                        bbox=dict(facecolor="white", alpha=0.8),
                    )
            if VISTA_SELECCIONADA in [0, 3]:
                (linea_pos,) = ax_pos.plot(
                    [], [], lw=1.5, color="darkgreen", label="Posición del Imán"
                )
                ax_pos.set_title("Cinemática: Posición Estimada")
                ax_pos.set_ylabel("Desplazamiento (m)")
                if VISTA_SELECCIONADA != 0:
                    ax_pos.set_xlabel("Tiempo (s)")
                ax_pos.grid(True)
                ax_pos.legend(loc="upper right")
                if VISTA_SELECCIONADA == 0:
                    text_timeout = ax_pos.text(
                        0.5,
                        0.95,
                        "",
                        transform=ax_pos.transAxes,
                        ha="center",
                        fontsize=10,
                        color="red",
                        bbox=dict(facecolor="white", alpha=0.8),
                    )
            if VISTA_SELECCIONADA in [0, 4]:
                (linea_f,) = ax_fft.plot([], [], lw=1.5, color="red")
                ax_fft.set_title("Dominio de la Frecuencia (FFT)")
                ax_fft.set_ylabel("Amplitud")
                ax_fft.set_xlabel("Frecuencia (Hz)")
                ax_fft.set_xlim(0, 120)
                ax_fft.grid(True)
                if VISTA_SELECCIONADA == 0:
                    text_timeout = ax_fft.text(
                        0.5,
                        0.95,
                        "",
                        transform=ax_fft.transAxes,
                        ha="center",
                        fontsize=10,
                        color="red",
                        bbox=dict(facecolor="white", alpha=0.8),
                    )

            plt.tight_layout()

            def actualizar(frame):
                # Leer datos del puerto mientras haya
                while esp32.in_waiting > 0:
                    try:
                        linea = (
                            esp32.readline().decode("utf-8", errors="ignore").strip()
                        )
                        if "," in linea:
                            t_micros_str, v_str = linea.split(",")
                            t_micros = int(t_micros_str)
                            v_crudo = float(v_str)
                            if estado["tiempo_inicial"] is None:
                                estado["tiempo_inicial"] = t_micros
                            t_segundos = (t_micros - estado["tiempo_inicial"]) / 1e6
                            buffer_t.append(t_segundos)
                            buffer_v.append(v_crudo)
                            estado["ultima_actualizacion"] = time.time()
                    except Exception:
                        pass

                # Verificar timeout de recepción de datos
                if time.time() - estado["ultima_actualizacion"] > TIMEOUT_SEGUNDOS:
                    reiniciar_conexion()
                    if not estado["conexion_activa"]:
                        # Cerrar figura y salir
                        plt.close(fig)
                        return ()
                    else:
                        # Mostrar mensaje de espera en la gráfica
                        if "text_timeout" in locals():
                            text_timeout.set_text("Esperando datos... Reintentando")
                        return ()

                # Limpiar mensaje de timeout si ya hay datos
                if "text_timeout" in locals():
                    text_timeout.set_text("")

                if len(buffer_t) == TAMANO_VENTANA:
                    t_arr = np.array(buffer_t)
                    v_arr = np.array(buffer_v) - subida_voltaje
                    v_filtrado = aplicar_filtros(t_arr, v_arr)
                    v_real = v_filtrado / factor_amplificacion

                    if VISTA_SELECCIONADA in [0, 2, 3]:
                        velocidad_real = -v_real / (G_sub_A * R_porcentaje)
                    if VISTA_SELECCIONADA in [0, 3]:
                        posicion_iman = cumulative_trapezoid(
                            velocidad_real, t_arr, initial=0
                        )
                    if VISTA_SELECCIONADA in [0, 4]:
                        N = TAMANO_VENTANA
                        T_muestreo = 1.0 / FS
                        yf = fft(v_real)
                        xf = fftfreq(N, T_muestreo)[: N // 2]
                        amplitud_fft = 2.0 / N * np.abs(yf[0 : N // 2])

                    elementos = []
                    if VISTA_SELECCIONADA in [0, 1]:
                        linea_fem_cruda.set_data(t_arr, v_filtrado)
                        linea_fem_real.set_data(t_arr, v_real)
                        ax_fem.set_xlim(t_arr[0], t_arr[-1])
                        margen = max(np.abs(v_filtrado)) * 1.2 + 0.01
                        ax_fem.set_ylim(-margen, margen)
                        elementos.extend([linea_fem_cruda, linea_fem_real])
                    if VISTA_SELECCIONADA in [0, 2]:
                        linea_vel.set_data(t_arr, velocidad_real)
                        ax_vel.set_xlim(t_arr[0], t_arr[-1])
                        margen_vel = max(np.abs(velocidad_real)) * 1.2 + 1e-6
                        ax_vel.set_ylim(-margen_vel, margen_vel)
                        elementos.append(linea_vel)
                    if VISTA_SELECCIONADA in [0, 3]:
                        linea_pos.set_data(t_arr, posicion_iman)
                        ax_pos.set_xlim(t_arr[0], t_arr[-1])
                        margen_pos = max(np.abs(posicion_iman)) * 1.2 + 1e-7
                        ax_pos.set_ylim(-margen_pos, margen_pos)
                        elementos.append(linea_pos)
                    if VISTA_SELECCIONADA in [0, 4]:
                        linea_f.set_data(xf, amplitud_fft)
                        ax_fft.set_ylim(0, max(amplitud_fft) * 1.2 + 1e-6)
                        elementos.append(linea_f)
                    return tuple(elementos)
                return ()

            ani = animation.FuncAnimation(fig, actualizar, interval=30, blit=False)
            plt.show()
            if esp32 and esp32.is_open:
                esp32.close()

        # =================================================================
    # MODO 5: EXPERIMENTO PARA ESTIMAR EL COEFICIENTE 'c'
    # =================================================================
    elif experimento_c:
        print("\n--- Experimento para determinar 'c' ---")
        print("Seleccione el método de estimación:")
        print("  1 - Newton-Raphson (basado en posición, requiere integración)")
        print("  2 - Ajuste senoidal (basado en velocidad, más rápido y robusto)")
        metodo_c = input("Opción (1/2): ").strip()

        PUERTO = "COM3"
        BAUDIOS = 115200
        FS = 900.0
        UMBRAL_DISPARO = 0.05
        TIEMPO_CAPTURA = 3.0

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
                        v_sin_offset = float(v_str) - subida_voltaje
                        if tiempo_inicial_micros is None:
                            tiempo_inicial_micros = t_micros
                        t_segundos = (t_micros - tiempo_inicial_micros) / 1e6
                        if estado == "IDLE":
                            historial_t.append(t_segundos)
                            historial_v.append(v_sin_offset)
                            if abs(v_sin_offset) > UMBRAL_DISPARO:
                                print("¡Trigger! Capturando onda...")
                                datos_t.extend(historial_t)
                                datos_v.extend(historial_v)
                                estado = "RECORDING"
                        elif estado == "RECORDING":
                            datos_t.append(t_segundos)
                            datos_v.append(v_sin_offset)
                            muestras_grabadas += 1
                    except Exception:
                        continue

            esp32.close()
            t_raw = np.array(datos_t)
            t_onda = t_raw - t_raw[0]
            v_onda = np.array(datos_v)
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
                c_arr = np.array(c_evolucion)
                print("\n========== RESULTADOS (Newton-Raphson) ==========")
                print(f" Media c = {np.mean(c_arr):.6f} Ns/m")
                print(f" Mediana   = {np.median(c_arr):.6f}")
                print(f" Desv. est. = {np.std(c_arr):.6f}")
                print("==================================================")
                # Gráfica
                fig, ax1 = plt.subplots(figsize=(12, 7))
                plt.style.use("seaborn-v0_8-darkgrid")
                ax1.set_xlabel("Tiempo (s)")
                ax1.set_ylabel("Voltaje Centrado (V)", color="tab:blue")
                ax1.plot(
                    t_onda, v_onda, color="tab:blue", alpha=0.4, label="FEM (ESP32)"
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
                    if c_fase is not None:
                        c_fase_list.append(c_fase)
                    t_ventana.append(t_w[len(t_w) // 2])
                # Mostrar resultados estadísticos
                c_amp_arr = np.array(c_amp_list)
                c_fase_arr = np.array(c_fase_list)
                print("\n========== RESULTADOS (Ajuste senoidal) ==========")
                if len(c_amp_arr) > 0:
                    print(
                        f" c por AMPLITUD: media = {np.mean(c_amp_arr):.6f} Ns/m, mediana = {np.median(c_amp_arr):.6f}, desv = {np.std(c_amp_arr):.6f}"
                    )
                if len(c_fase_arr) > 0:
                    print(
                        f" c por FASE:     media = {np.mean(c_fase_arr):.6f} Ns/m, mediana = {np.median(c_fase_arr):.6f}, desv = {np.std(c_fase_arr):.6f}"
                    )
                print("===================================================")
                # Gráfica comparativa
                fig, ax1 = plt.subplots(figsize=(12, 7))
                plt.style.use("seaborn-v0_8-darkgrid")
                ax1.set_xlabel("Tiempo (s)")
                ax1.set_ylabel("Voltaje Centrado (V)", color="tab:blue")
                ax1.plot(
                    t_onda, v_onda, color="tab:blue", alpha=0.4, label="FEM (ESP32)"
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


# ===================================================================
# 3. EJECUCIÓN PRINCIPAL DEL PROGRAMA
# ===================================================================
if __name__ == "__main__":
    print("\n--- INICIALIZANDO PARÁMETROS DEL SISMÓMETRO ---")
    parametros = SectionParams()

    print("\n--- INICIANDO INTEGRAL CAMPO MAGNETICO ---")
    Factores_Acople(parametros)  # Solo para precalcular (opcional)

    print("\n--- INICIANDO SOLVER Y GRÁFICAS ---")
    # ¡OJO! El orden correcto de los argumentos es:
    # modelo_mk1, modelo_mk2, simular, datos_descarga, experimento_c, params
    Solver(modelo_mk1, modelo_mk2, simular, datos_descarga, experimento_c, parametros)
