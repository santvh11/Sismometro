import serial  # Poderse conectar con el puerto USB
from dataclasses import (
    dataclass,
    field,
)  # Librería de Dataclass para almacenamiento de variables
import numpy as np  # Necesario para matrices y operaciones trigonométricas
import matplotlib.animation as animation  # Necesario para ver el voltaje en tiempo real
import matplotlib.pyplot as plt  # Necesario para graficar resultados
from scipy.integrate import solve_ivp  # Necesario para modelo Mk 2 con simulaciones
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq  # Librería de la transformada rápida de fourier
import collections
from scipy.integrate import (
    cumulative_trapezoid,
)  # Integrar numéricamente el voltaje para obtener posición

# -------------------------------------------------------------------
# 0. Selección del modelo:
# En esta sección determinamos el modelo a usar
# "modelo_mk1" corresponde a modelo con fuerza de Laplace integrada
# "modelo_mk2" corresponde a modelo sin ecuación electromagnética
# -------------------------------------------------------------------

while True:
    modelo = input(
        "Escriba (1) para mk_1 o (2) para mk_2, o (3) para ver voltaje en tiempo real: "
    ).strip()
    if modelo == "1":
        modelo_mk_1, modelo_mk_2, simular = True, False, False
        break  # Sale del bucle y sigue el programa
    elif modelo == "2":
        modelo_mk_1, modelo_mk_2, simular = False, True, False
        break
    elif modelo == "3":
        modelo_mk_1, modelo_mk_2, simular = True, False, True
        break
    else:
        print("Entrada no válida. Inténtalo de nuevo.")


# -------------------------------------------------------------------
# 1. Parámetros de sección:
# En esta sección guardamos las variables que dependen directamente del diseño (1er prden)
# Y en base a ellas, computamos las de que son combinaciones de las anteriores (2do Orden)
# Todas las unidades anteriores han de estar en el S.I
# -------------------------------------------------------------------

# Notas
# -------------------------------------------------------------------
# Tablas de viscocidad dinámica (Kg/m*s)
# Aire=1.81*(1e-6)
# Silicona 100cts=97*(1e-3)
# Silicona 150cts=1455*(1e-4)
# Silicona 1.000cts=1.1
# Silicona 10.000cts=9.7
# Silicona 12.500cts=11.64
# -------------------------------------------------------------------
# Tablas de diámetros/vueltas y longitud del resorte (Kg/m*s)
# resorte pequeño:
# vueltas:18
# d_sección_transversal:75*(1e-5) metros
# L_libre_iman=135*(1e-3) metros
# ----------------
# resorte mediano:
# vueltas: 15
# d_sección_transversal: 75*(1e-5) metros
# L_libre_iman=1125*(1e-4) metros
# ----------------
# 1.5*(1e-3) Kg (resorte pequeño)
# 3.4*(1e-3) Kg (resorte mediano)
# k= 300 (resorte pequeño)
# k= 104 (resorte mediano)


@dataclass
class SectionParams:

    # -------------------------------------------------------------------
    # 1er Orden:
    # -------------------------------------------------------------------

    # Configuraciones Reales

    # Constantes de las EDO´s
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # Todas las unidades están en el SI
    R: float = 383.1  # Resistencia Real (Ohms)
    L: float = 36.7 * (1e-3)  # Inductancia Real (H)
    C: float = 0.07 * (1e-6)  # Capacitancia Real (F)
    c = 0.43  # Coeficiente de Amortiguamiento (Kg/s)
    m: float = 1.5 * (1e-3)  # Masa del sistema masa-resorte amortiguador
    k: float = 300  # Constante Elástica del Resorte (kg/s^2)
    m_mag: float = 13.93e-03  # Escalar del momento magnético (Teslas)
    m_sis: float = 48.6 * (1e-3)  # Masa completa del sismometro
    m_mes: float = 3.148 * (1e-3)  # Masa de la parte que vibra de la mesa
    factor_amplificacion: float = 8.219  # Factor de amplifiación de electrónica

    # -------------------------------------------------------------------
    # -------------------------------------------------------------------

    # Temperatura del ambiente en Kelvin (26 grados celsiud)
    temp: float = 299.15

    # Condiciones Iniciales
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    z: float = 0  # Posición Inicial
    z_dot: float = 0  # Velocidad Inicial
    Q: float = 0  # Carga inicial
    Q_dot: float = 0  # Corriente inicial

    # -------------------------------------------------------------------
    # -------------------------------------------------------------------

    # Sección Mecánica:

    g: float = 9.77  # Constante gravitacional
    eta: float = 1.5  # Viscocidad dinámica del fluido (kg/m*s)
    R_sub_e: float = 9.5 * (1e-3)  # Radio de la Esfera
    L_cilindro: float = 10 * (1e-3)  # Longitud cilindro
    L_libre_iman: float = 135 * (1e-3)  # Longitud libre del imán a usar
    omega_Hz: float = 75  # Frecuencia de vibración de la mesa (Radianes)
    omega = omega_Hz * 2 * np.pi  # Paso a radianes (Radianes)

    e_sub_p: float = 3 * (1e-3)  # Grosor del Contenedor de PLA
    h_sub_p: float = 34 * (1e-3)  # Altura del contenedor de PLA
    r_sub_p: float = 14 * (1e-3)  # Radio del Contenedor de PLA
    h_sub_f: float = 34 * (1e-3)  # altura del fluido
    g_sub_ecs: float = 0.015 * (1e-3)  # Grosor de la capa de esmalte
    e_sub_cs: float = 0.079 * (
        1e-3
    )  # Grosor del Cable del solenoide (diámetro) #Usamos un AGW 40
    N_sub_c_total: float = 2700  # Vueltas totales de cable a través del solenoide

    # Fluidos

    rho_0: float = 1273.3  # Densidad de referencia a 0 °C [kg/m³]
    beta_rho: float = 0.6121  # Coeficiente térmico de densidad [kg/(m³·K)]
    densidad_neodimio: float = 7500  # Densidad del Neodimio N35 en Kg/m^3
    eta_0: float = 3.30e-10  # Factor preexponencial [Pa·s] de la glicerina
    b_eta: float = 6640  # Constante de Andrade   [K]

    # Sección Electromagnética/Materiales

    R_iman: float = 1.5 * (1e-3)  # Radio del imán magnético
    Br_A: float = (
        0.9  # Coeficiente para el momento magnético del Acero Inoxidable 440 C (Teslas)
    )
    Br_N32: float = (
        1.14  # Coeficiente para el momento magnético del Neodimio N32 (Teslas)
    )
    Br_N35: float = (
        1.22  # Coeficiente para el momento magnético del Neodimio N35 (Teslas)
    )

    # Sección de Resistencia

    e_sub_c: float = 1 * (1e-3)  # Grosor del Cable de conexión (diámetro)
    L_sub_c: float = 1 * (1e-2)  # Longitud del cable de conexión
    R_sub_a: float = 1 * (1e8)  # Resistencia de entrada del amplificador de señal
    R_extra: float = 1  # Resistencia añadida al circuito

    # Sección de Capacitancia

    epsilon_sub_cero: float = 8.854 * (1e-12)  # Permitividad del vacío
    epsilon_sub_e: float = 2.5  # Permitividad del esmalte del cable de cobre
    p_sub_cu: float = 1.72 * (1e-8)  # Resistividad del cobre a 20 ºC

    # Sección de Inductancia

    mu_sub_cero: float = (4) * (np.pi) * (1e-7)  # Permeabilidad del vacío
    mu_sub_PLA: float = 1  # Permeabilidad del PLA
    mu_sub_f: float = 1  # Permeabilidad del fluido

    # Sección de Control

    a: int = 0  # Límite inferior de graficación
    b: int = 0.5  # Límite superior de graficación
    puntos: int = 10000  # Densidad de la línea
    delta_t: int = 1  # tiempo en segundos que el sensor se tarda en tomar datos

    # -------------------------------------------------------------------
    # 2do Orden: (Almacenamiento)
    # -------------------------------------------------------------------

    # Sección-Masa-Resorte-Amoprtiguamiento:

    m: float = field(default=m, init=False)
    m_fluido: float = field(init=False)
    N_sub_c_capas: float = field(init=False)
    c_sub_Stokes_resorte: float = field(init=False)
    c_sub_Stokes_iman: float = field(init=False)
    c_sub_Stokes: float = field(init=False)
    lambda_c: float = field(init=False)
    c_sub_lambda: float = field(default=0, init=False)
    c: float = field(default=c, init=False)
    F_0: float = field(init=False)
    omega_sub_n: float = field(init=False)
    Zeta: float = field(init=False)
    p_error_Zeta: float = field(init=False)
    omega_sub_d: float = field(init=False)
    B: float = field(init=False)
    root_root: float = field(init=False)
    root_1: float = field(init=False)
    root_2: float = field(init=False)
    Zeta_sub_omega: float = field(init=False)
    omega_sub_s: float = field(init=False)
    phi: float = field(init=False)
    X: float = field(init=False)

    # Sección Mecánica:

    e_total: float = field(init=False)
    L_sub_s: float = field(init=False)
    A_sub_s: float = field(init=False)
    A_sub_cs: float = field(init=False)
    A_effec: float = field(init=False)
    A_sub_c: float = field(init=False)
    delta_h_sub_cs: float = field(init=False)
    h_sub_cs: float = field(init=False)
    N_sub_c: int = field(init=False)

    # Sección de fluidos

    rho: float = field(init=False)
    # Sección Electromagnética

    m_mag: float = field(default=m_mag, init=False)

    # Sección de Resistencia

    R_sub_s: float = field(init=False)
    R_sub_c: float = field(init=False)
    R_porcentaje: float = field(init=False)
    R: float = field(default=R, init=False)

    # Sección de Capacitancia

    C: float = field(default=C, init=False)
    C_inicial: float = field(init=False)
    C_N_sub_cs: float = field(init=False)
    C_N_sub_cs_capas: float = field(init=False)
    XC_capacitiva: float = field(init=False)

    # Sección de Inductancia

    L: float = field(default=L, init=False)
    XC_inductiva: float = field(init=False)

    # Frecuencias circuito RLC

    XC: float = field(init=False)
    alpha: float = field(init=False)
    omega_sub_0_phi_m: float = field(init=False)

    # Sección de Control

    t: np.ndarray = field(init=False)

    # -------------------------------------------------------------------
    # 2do Orden: (Computación)
    # -------------------------------------------------------------------

    def __post_init__(self):

        # Computaciones Base:
        # --------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------
        self.N_sub_c = round(
            self.h_sub_p / self.e_sub_cs
        )  # Número de vueltas del cable
        self.N_sub_c_capas = (
            self.N_sub_c_total / self.N_sub_c
        )  # Número de capass del cable
        self.e_total = (
            self.r_sub_p + self.e_sub_p + (0.5 * self.e_sub_cs)
        )  # Radio total del Solenoide
        self.L_sub_s = self.N_sub_c_total * (
            self.e_total * 2 * np.pi
        )  # Longitud del cable solenoide
        self.A_sub_s = self.N_sub_c * (
            self.N_sub_c_capas * (4 * (np.pi**2)) * (self.e_total * self.e_sub_cs)
        )  # Área del cable de solenoide
        self.A_effec = (
            8 * self.e_sub_cs * np.pi * self.e_total * self.N_sub_c_total
        )  # Área efectiva de contacto con campo magnético en el eje z
        self.A_sub_cs = np.pi * (
            (self.e_sub_cs * 0.5) ** 2
        )  # Área de corte de cable solenoide
        self.A_sub_c = np.pi * (
            (self.e_sub_c * 0.5) ** 2
        )  # Área de corte de cable de conexión
        self.delta_h_sub_cs = (
            self.e_sub_cs
        )  # Altura del cable de solenoide en una vuelta
        self.eta = self.eta_0 * np.exp(
            self.b_eta / self.temp
        )  # Viscosidad en base a la temperatura
        self.rho = self.rho_0 - self.beta_rho * (
            self.temp - 273.15
        )  # [kg/m³], temp en Kelvin

        # --------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------

        # Pregunta para definir parámetros directos o estimados teóricamente
        # -----------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------
        teoric_params = bool(
            input(
                "¿Desea usar los parámetros con los datos directos? "
                "(Enter) para sí y escriba algo para no"
            )
        )
        # -----------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------

        # Llamado a definiciones de cálculo teóricas:
        # -----------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------
        if teoric_params == True:
            self.masa_teorico()
            self.amortiguamiento_teorico()
            self.capacitancia_teorico()
            self.resistencia_teorico()
            self.inductancia_teorico()
            self.m_mag_teorico()

        # -----------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------

        # Cálculo de parámetros de la EDO Mecánica
        # --------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------
        self.m_fluido = np.abs(
            self.rho
            * (np.pi * (self.R_sub_e**2) * (self.h_sub_f - 0.75 * self.R_sub_e))
        )
        self.m = self.m + self.m_fluido
        self.omega_sub_n = (self.k / self.m) ** 0.5  # Frecuencia Natural (Hz)
        self.Zeta = self.c / (
            2 * ((self.k * self.m) ** 0.5)
        )  # Factor de Amortiguamiento
        self.omega_sub_d = (
            self.omega_sub_n * (1 - self.Zeta**2) ** 0.5
        )  # Frecuencia Natural Amortiguada
        # --------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------

        # Cálculo de Parámetros para el circuito RLC
        # --------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------

        self.XC_inductiva = self.L * self.omega  # Reactancia Inductiva
        self.XC_capacitiva = -1 / (self.omega * self.C)  # Reactancia Capacitiva
        self.R_porcentaje = self.R_sub_a / (
            self.R + self.R_sub_a
        )  # Resistencia para ley de faraday lenz
        self.alpha = self.R / (
            2 * self.L
        )  # Inercia Eléctrica (Velocidad de consumo de corrientes parásitas)
        self.XC = (
            self.XC_capacitiva + self.XC_inductiva
        )  # Reactancia total (parte imaginaria)
        self.omega_sub_0_phi_m = 1 / (
            (self.L * self.C) ** 0.5
        )  # Frecuencia de Resonancia (Frecuencia natural de oscilación del circuito)
        # --------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------

        # Sección de Control
        # --------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------
        self.t = np.linspace(
            self.a, self.b, self.puntos
        )  # Vector de variable continua (tiempo)
        # --------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------

        # Cálculo de la fuerza Inicial
        # --------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------
        fuerza = int(
            input("¿La fuerza es directa (1) o se calcula a partir de un MAS (2)?")
        )
        if fuerza == 1:
            self.F_0 = 5  # Fuerza en Newtons de la mesa
        elif fuerza == 2:
            amplitud = int(
                input(
                    "¿La amplitud es directa (1) o se calcula en base a la frecuencia(2)?"
                )
            )
            m_vibrante = self.m_sis + self.m_mes  # Masa vibrante
            if amplitud == 1:
                self.Y_0 = 7 * (1e-3)  # Amplitud de vibración en metros
                self.F_0 = (
                    m_vibrante * self.Y_0 * (self.omega**2)
                )  # Fuerza derivada de un MAS
            elif amplitud == 2:
                # COMPLEMENTAR

                if 26 < self.omega_Hz < 30:
                    aceleracion_G = np.abs(
                        3.5 * 2.087 / ((self.omega_Hz - 28.45) ** 2 + 2.087)
                    )
                elif 8 < self.omega_Hz < 12:
                    aceleracion_G = np.abs(
                        2.84 * 0.026 / ((self.omega_Hz - 10.33) ** 2 + 0.026)
                    )
                else:
                    aceleracion_G = np.abs(2.1 * 1e-3 * self.omega_Hz**2)

                # Definir la aceleración G

                aceleracion = aceleracion_G * 9.81
                # Función de la amplitud en base a frecuencia (Hz)
                self.F_0 = m_vibrante * aceleracion  # Fuerza derivada de un MAS
            else:
                ValueError("No se ingresó una opción válida, oprima (1) o (2)")
        else:
            ValueError("No se ingresó una opción válida, oprima (1) o (2)")

    # --------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------

    # Definiciones Teóricas
    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------
    def masa_teorico(self):
        self.m = (self.R_sub_e) ** 3 * np.pi * 1.3333 * self.densidad_neodimio

    # -----------------------------------------------------------------------------------------------
    def amortiguamiento_teorico(self):
        # Cálculo de amortiguamiento con simplificación de Navier Stokes
        self.c_sub_Stokes_iman = (
            6 * np.pi * self.eta * self.R_sub_e
        )  # Coeficiente c para una esfera
        self.c_sub_Stokes_resorte = np.abs(
            4
            * np.pi
            * self.eta
            * self.L_cilindro
            / (np.log(self.L_cilindro / 2 * self.R_sub_e) + 0.5)
        )  # Coeficiente c para un cilindro (resorte recreado)

        self.lambda_c = (
            self.R_sub_e / self.r_sub_p
        )  # Factor de lejanía (Habermann Faxen)
        if self.lambda_c > 0.6:

            # Factor de corrección de cercanía (Haberman/Sayre)

            den = 1 - (0.75857 * self.lambda_c**5)
            num = (
                1
                - (2.10444 * self.lambda_c)
                + (2.08877 * self.lambda_c**3)
                - (0.94813 * self.lambda_c**5)
            )
            self.c_sub_lambda = num / den

        elif self.lambda_c <= 0.6:

            # Factor de corrección de cercanía (Haberman/Faxen)

            self.c_sub_lambda = (
                1
                - (2.104 * (self.lambda_c))
                + (2.089 * (self.lambda_c**3))
                - 0.948 * (self.lambda_c**5)
            )

        # coeficiente de amortiguamiento final
        self.c = (
            self.c_sub_Stokes_iman + self.c_sub_Stokes_resorte
        ) / self.c_sub_lambda

    # --------------------------------------------------------------------------------------------------
    def capacitancia_teorico(self):

        # Capacitancia de 2 cilindros paralelos (cables)
        self.C_inicial = (self.L_sub_s * self.epsilon_sub_cero * self.epsilon_sub_e) / (
            np.log((self.g_sub_ecs + self.e_sub_cs) / (self.e_sub_cs))
        )

        # Añadimos número de espiras
        self.C_N_sub_cs = 2 * self.C_inicial / self.N_sub_c

        # Añadimos número de capas
        self.C = self.C_N_sub_cs * self.N_sub_c_capas

    # --------------------------------------------------------------------------------------------------
    def inductancia_teorico(self):

        # Inductancia del Solenoide
        seccion_transversal_sol = np.pi * (self.r_sub_p + self.e_sub_p) ** 2
        self.L = (
            seccion_transversal_sol
            * self.mu_sub_cero
            * (self.N_sub_c_total**2)
            / self.h_sub_p
        )

    # --------------------------------------------------------------------------------------------------
    def resistencia_teorico(self):

        # Resistencia del solenoide
        self.R_sub_s = self.p_sub_cu * self.L_sub_s / self.A_sub_cs

        # Resistencia de la conexión
        self.R_sub_c = self.p_sub_cu * self.L_sub_c / self.A_sub_c

        # Resistencia Total del sistema
        self.R = (self.R_sub_a * (self.R_sub_s + self.R_sub_c + self.R_extra)) / (
            self.R_sub_a + self.R_sub_s + self.R_sub_c + self.R_extra
        )

    # --------------------------------------------------------------------------------------------------
    def m_mag_teorico(self):

        # Preguntamos el material del imán para estimar el momento magnético
        tipo_iman = int(
            input("¿El imán es de acero 440 (1), neodimio N32(2) o Neodimio N35(3)?")
        )

        Vol = ((self.R_iman) ** 3) * 0.75 * np.pi

        if tipo_iman == 1:
            # Estimación para Acero 440C
            self.m_mag = self.Br_A * Vol / self.mu_sub_cero
        elif tipo_iman == 2:
            # Estimación para Neodimio N32
            self.m_mag = self.Br_N32 * Vol / self.mu_sub_cero
        elif tipo_iman == 3:
            # Estimación para Neodimio N35
            self.m_mag = self.Br_N35 * Vol / self.mu_sub_cero
        else:
            ValueError("Escriba (1) o (2) o (3), otra repuesta no es válida")


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
def Factores_Acople(params: SectionParams) -> float:

    # Planetamos una integral de promedios para el producto cruz de la fuerza de Laplace
    # La integral ha sido planetada de forma analítica

    # Desempaquetamos variables de section params
    e_sub_p = params.e_sub_p
    h_sub_p = params.h_sub_p
    r_sub_p = params.r_sub_p
    L_sub_s = params.L_sub_s
    mu_sub_cero = params.mu_sub_cero
    m_mag = params.m_mag
    A_effec = params.A_effec
    e_sub_cs = params.e_sub_cs
    N_sub_cs_total = params.N_sub_c_total
    e_total = params.e_total

    # Añadimos constantes físicas
    constantes_magneticas = (0.25 * m_mag * mu_sub_cero) / np.pi

    # Componente de radio y paso

    Radio_total = e_sub_p + r_sub_p
    Paso = e_sub_cs * 0.5 / np.pi

    Integral_linea = (1 / Radio_total**3) - (
        1 / (Radio_total**2 + (Paso * 2 * np.pi + N_sub_cs_total) ** 2) ** 3 / 2
    )

    # d_phi/d_t :Acople G_sub_A:

    G_sub_A = (constantes_magneticas * 3 * Radio_total**2 / Paso) * Integral_linea

    # LxB_sub_x :Acople G_sub_L:

    G_sub_L = (constantes_magneticas * Radio_total) / (
        Paso * (Radio_total**2 + (4 * np.pi * Paso) ** 2) ** 3
    )

    return G_sub_L, G_sub_A


# -------------------------------------------------------------------
# 2. Solver de EDO´s:
# El usuario puede resolverlo tanto con la parte electromagnética (mk1)
# O Con la parte mecánica solamente (mk2)
# -------------------------------------------------------------------


def Solver(modelo_mk1: bool, modelo_mk2: bool, params: SectionParams) -> float:

    # Desempaquetado de variables de Section Params
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
    puntos = params.puntos
    factor_amplificacion = params.factor_amplificacion

    # Prints de Parámetros Base:
    # --------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------
    print(f"masa estimada en :{m:.3e} Kg")  # Masa
    print(
        f"El amortiguamiento auxiliar es:{c_sub_lambda:.3e}"
    )  # Amortiguamiento Auxiliar
    print(f"La constante de amortiguamiento  es de :{c:.3e} Kg/s")  # Amortiguamiento
    print(f"La constante elástica es de :{k:.3e} Kg/s^2")  # Constante Elástica
    print(f"La resistencia es de :{R:.3e} Ohmnios")  # Resistencia
    print(f"La inductancia es de :{L:.3e} H")  # Inductancia
    print(f"La capacitancia es de :{C:.3e} F")  # Capacitancia
    print(f"Momento magnético configurado en: {m_mag:.3e}")  # Momento Magnético
    print(f"El factor  Magnético de Área es: {G_sub_A:.3e}")  # Factor Magnético de Área
    print(
        f"El factor  Magnético de Longitud es: {G_sub_L:.3e}"
    )  # Factor Magnético de Longitud
    print(
        f"El factor de amortiguamiento es de: {Zeta:.3e}"
    )  # Factor de Amortiguamiento
    print(f"La frecuencia natural es de: {omega_sub_n_f:.3e} Hz")  # Frecuencia Natural
    print(f"La inercia eléctrica es de: {alpha:.3e} ")  # Inercia Eléctrica
    print(
        f"La frecuencia eléctrica es de: {omega_sub_0_phi_m_f:.3e} Hz"
    )  # Inercia Eléctrica

    # --------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------
    if modelo_mk1 == True:

        print("--- Entrando a Modelo MK1 (Lineal) ---")

        # Procedemos a computar las EDO´s como un sistema lineal con la teoría de eigenalores y eigenvectores

        # A: Matriz de acople de constantes
        A = np.array(
            [
                [0, 1, 0, 0],
                [-k / m, -c / m, 0, -G_sub_L / m],
                [0, 0, 0, 1],
                [0, -G_sub_A / L, -1 / (L * C), -(R) / (L)],
            ]
        )

        # 3. Encuentra Eigenvalos (w) y Eigenvectores (v)
        eigenvalues, eigenvectors = np.linalg.eig(A)

        # Matriz B: La fuerza externa F(t) afecta a la aceleración (z_dot_dot)
        F_t = np.array([[0], [1 / m], [0], [0]])
        # Calculamos la solución particular como la parte imaginaria de una exponencial
        I = np.eye(4)
        lado_izquierdo = 1j * omega * I - A
        lado_derecho = F_t * F_0

        # Xp_complejo contiene la amplitud y fase de [z, z_dot, Q, Q_dot]
        Xp_complejo = np.linalg.solve(lado_izquierdo, lado_derecho)

        # --- 3. Generar la evolución de la Particular ---
        # x_p(t) = Re( Xp * exp(j * omega * t) )
        # Usamos np.outer para multiplicar el vector Xp por el vector de tiempo exponencial
        X_particular = np.zeros((4, len(t)), dtype=complex)
        X_particular = Xp_complejo * np.exp(1j * omega * t)

        # Resuelve en base al vector de condiciones iniciales/variables

        X_t_p = (
            Xp_complejo.real
        )  # Defininimos una condición inicial particular para el vector de condiciones iniciales
        X_t_h = np.array([[z_0, z_dot_0, Q_0, Q_dot_0]]).reshape(
            4, 1
        )  # Defininimos una condición inicial homogénea para el vector de condiciones iniciales
        X_t = X_t_h + X_t_p  #: X_t: Vector de variables (parte homogénea + particular)

        coeficientes = np.linalg.solve(eigenvectors, X_t)

        # Creamos una matriz para guardar evolución con respecto al tiempo

        X_evolucion = np.zeros((4, len(t)), dtype=complex)

        for i in range(len(eigenvalues)):
            # Usamos 'coeficientes' en lugar de 'c' para no confundir con amortiguamiento
            # Contribución: c_i * v_i * e^(lambda_i * t)
            termino_exponencial = np.exp(eigenvalues[i] * t)
            contribucion = np.outer(eigenvectors[:, i], termino_exponencial)
            X_evolucion += coeficientes[i] * contribucion

        # Sumamos condiciones iniciales y finales
        X_total = X_evolucion + X_particular

        # Computamos voltaje de salida en base a Faraday-Lenz
        FEM_FL = G_sub_A * R_porcentaje * X_total.real[1]

        # Hacemos el tiempo discreto para simular el envío del sensor
        t_filtro = t[::delta_t]

        # Frecuencias naturales del sistema acoplado, ya no habrá una, sino 4 por la cantidad de eignevalores
        # Los eigenvalores vienen en pares complejos conjugados
        # Tomamos la parte imaginaria y el valor absoluto para tener la frecuencia
        frecuencias_naturales = np.abs(eigenvalues.imag)

        # Como hay 4 eigenvalores, verás dos valores repetidos (los pares conjugados)
        # Eliminamos duplicados para ver las dos frecuencias principales del sistema
        frecuencias_unicas = np.unique(frecuencias_naturales) * 2 * np.pi

        print(f"Frecuencias naturales del sistema acoplado: {frecuencias_unicas} Hz ")

        # Computamos el voltaje de Salida partiendo de la ley de Ohm Fasorial (V=ZR)
        I_Re = X_total.real[3]
        I_Im = X_total.imag[3]
        V_Im = (R * I_Im) + (I_Re * XC)
        V_Re = (R * I_Re) - (I_Im * XC)
        V_A = ((V_Im**2) + (V_Re**2)) ** 0.5
        V_phi = np.arctan(V_Im / V_Re)
        FEM_Ohm = V_A * np.cos((omega * t) + V_phi)

        F_Laplace = X_total.real[3] * G_sub_L

        if simular == False:
            # Listas de colores y de nombres para las gráficas

            nombre_variables = [
                "Posición (z)",
                "Velocidad (ż)",
                "Carga (Q)",
                "Corriente (I)",
            ]
            unidades = ["[m]", "[m/s]", "[C]", "[A]"]
            paleta_colores = ["orange", "red", "blue", "green"]

            # 3. Configuración de la figura
            fig, axs = plt.subplots(2, 2, figsize=(7.5, 5), dpi=200)
            fig.suptitle(
                f"Respuesta Temporal Completa - Detector de Gaia MK2",
                fontsize=16,
                fontweight="bold",
            )

            for i in range(4):
                ax = axs[i // 2, i % 2]

                # Graficamos
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

                # Anotación del valor pico estacionario
                ax.text(
                    0.95,
                    0.02,
                    f"Pico Estac: {np.abs(Xp_complejo[i].item()):.2e}",
                    transform=ax.transAxes,
                    ha="right",
                    fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.7),
                )

                # Anotar valores máximos
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

            # Mostramos una gráfica del voltaje teórico con la corriente de la misma EDO que ha de recopilar el programa del modelo físico

            plt.figure(figsize=(8, 6), dpi=200)
            plt.step(
                t_filtro,
                FEM_Ohm,
                where="post",
                label="Muestreo (Escalonado) con base al tiempo de recolección de datos",
                color="purple",
                linewidth=1,
            )
            plt.title("FEM teórica producida por Ley de Ohm Fasorial")
            # Anotación del valor pico estacionario
            ax.text(
                0.95,
                0.02,
                f"Pico Estac: {np.abs(FEM_Ohm)}",
                transform=ax.transAxes,
                ha="right",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7),
            )
            # Anotar valores máximos
            ax.text(
                0,
                0.02,
                f"Máx: {np.max(FEM_Ohm)}",
                transform=ax.transAxes,
                ha="left",
                fontsize=7,
                bbox=dict(facecolor="white", alpha=0.7),
            )
            plt.xlabel("Tiempo")
            plt.ylabel("FEM")
            plt.legend()
            plt.show()

            # Mostramos una gráfica del voltaje teórico con el voltaje estimado con Faraday Lenz
            plt.figure(figsize=(8, 6), dpi=200)
            plt.step(
                t_filtro,
                FEM_FL,
                where="post",
                label="Muestreo (Escalonado) con base al tiempo de recolección de datos",
                color="purple",
                linewidth=1,
            )
            plt.title("FEM teórica producida por Faraday_Lenz")
            # Anotación del valor pico estacionario
            ax.text(
                0.95,
                0.02,
                f"Pico Estac: {np.abs(FEM_FL)}",
                transform=ax.transAxes,
                ha="right",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7),
            )
            # Anotar valores máximos
            ax.text(
                0,
                0.02,
                f"Máx: {np.max(FEM_FL)}",
                transform=ax.transAxes,
                ha="left",
                fontsize=7,
                bbox=dict(facecolor="white", alpha=0.7),
            )
            plt.xlabel("Tiempo")
            plt.ylabel("FEM")
            plt.legend()
            plt.show()

            # Mostramos una gráfica del voltaje teórico con el voltaje estimado con Faraday Lenz
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
            plt.xlabel("Tiempo")
            plt.ylabel("Newtons")
            plt.legend()
            plt.show()

        # ----------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------
        # Estimación del número de Reynolds
        v_max = np.abs(np.max(X_total.real[1]))
        Re = (v_max * (r_sub_p - R_sub_e) * rho) / (eta)

        if Re < 0.1:
            print(
                f"Linealidad de la constante de amortiguamiento fiable. Valor de Reynolds:{Re:.3e}"
            )
        elif 0.1 < Re < 0.9:
            print(
                f"Linealidad de la constante de amortiguamiento variable. Valor de Reynolds:{Re:.3e}"
            )
        else:
            print(
                f"La constante de amortiguamiento no es lineal, se necesita RK45. Valor de Reynolds:{Re:.3e}"
            )

    elif modelo_mk_2 == True:
        # Procedemos a computar las EDO´s como un sistema no lineal con el solver RK45

        # Definimos la matriz de estado

        def state_matrix(t, S):

            # Variables de estado
            z, v, Q, I = S

            # Fuerza externa periódica
            F_ext = F_0 * np.sin(omega * t)

            dz_dt = v
            dz_dt_2 = (1 / m) * (
                (-k * z) - (c * v * np.abs(v)) + (F_ext) + (I * G_sub_L)
            )
            dQ_dt = I
            dQ_dt_2 = (1 / L) * (-R * I - (1 / C) * Q + v * G_sub_A)

            return dz_dt, dz_dt_2, dQ_dt, dQ_dt_2

        # Vector de tiempo y puntos para gráfica
        t_span = (a, b)

        # Metemos en el solver
        solver = solve_ivp(
            state_matrix,  # Matriz de estado del Solver
            t_span,  # Intervalo de gráfica
            [z_0, z_dot_0, Q_0, Q_dot_0],  # Vector de condiciones iniciales
            t_eval=t,
            method="LSODA",
            # Tolerancias para mejor visualización (se pueden comentar o descomentar dependiendo de las preferencias)
            # rtol=1e-8,
            # atol=1e-12
        )

        # Listas de colores y de nombres para las gráficas

        nombre_variables = [
            "Posición (z)",
            "Velocidad (ż)",
            "Carga (Q)",
            "Corriente (I)",
        ]
        unidades = ["[m]", "[m/s]", "[C]", "[A]"]
        paleta_colores = ["orange", "red", "blue", "green"]

        # 3. Configuración de la figura
        fig, axs = plt.subplots(2, 2, figsize=(7.5, 5), dpi=200)
        fig.suptitle(
            f"Respuesta Temporal Completa - Detector de Gaia MK2",
            fontsize=16,
            fontweight="bold",
        )

        for i in range(4):
            ax = axs[i // 2, i % 2]

            # Graficamos
            ax.plot(
                solver.t,
                solver.y[i],
                "--",
                color=paleta_colores[i],
                alpha=0.4,
                label="Solución RK 45",
            )

            ax.set_title(nombre_variables[i], fontweight="bold")
            ax.set_ylabel(f"Amplitud {unidades[i]}")
            ax.set_xlabel("Tiempo [s]")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper right", fontsize="small")

            # Anotar valores máximos
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

        # Computamos el voltaje de Salida partiendo de la ley de Ohm Fasorial (V=ZR)
        # Los resultads de esta son netamente lineales
        I = solver.y[3]
        Z = R
        FEM_Ohm = I * Z

        # Computamos voltaje de salida en base a Faraday-Lenz
        FEM_FL = G_sub_A * R_porcentaje * solver.y[1]

        # Hacemos el tiempo discreto para simular el envío del sensor

        t_filtro = solver.t[::delta_t]

        # Mostramos una gráfica del voltaje teórico con la corriente de la misma EDO que ha de recopilar el programa del modelo físico

        plt.figure(figsize=(8, 6), dpi=200)
        plt.step(
            t_filtro,
            FEM_Ohm,
            where="post",
            label="Muestreo (Escalonado) con base al tiempo de recolección de datos",
            color="purple",
            linewidth=1,
        )
        plt.title("FEM teórica producida por Ley de Ohm Fasorial")
        plt.xlabel("Tiempo")
        plt.ylabel("FEM")
        plt.legend()
        plt.show()

        # Mostramos una gráfica del voltaje teórico con el voltaje estimado con Faraday Lenz
        plt.figure(figsize=(8, 6), dpi=200)
        plt.step(
            t_filtro,
            FEM_FL,
            where="post",
            label="Muestreo (Escalonado) con base al tiempo de recolección de datos",
            color="purple",
            linewidth=1,
        )
        plt.title("FEM teórica producida por Faraday_Lenz")
        plt.xlabel("Tiempo")
        plt.ylabel("FEM")
        plt.legend()
        plt.show()

        F_Laplace = solver.y[3] * G_sub_L

        # Mostramos una gráfica del voltaje teórico con el voltaje estimado con Faraday Lenz
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
        plt.xlabel("Tiempo")
        plt.ylabel("Newtons")
        plt.legend()
        plt.show()

        if not solver.success:
            print("Error en el solver:", solver.message)
        else:
            print("Integración exitosa del solver.")

    if modelo_mk2 == False and simular == True:

        print("\n--- Entrando a Adquisición de Datos y Gemelo Digital ---")

        # ==========================================
        # 1. FUNCIÓN MATEMÁTICA PARA INGENIERÍA INVERSA
        # ==========================================
        def modelo_gemelo_digital(t_datos, c_ajuste):
            Z_m = np.sqrt((k - m * omega**2) ** 2 + (c_ajuste * omega) ** 2)
            V_amp = (F_0 * omega) / Z_m
            phi = np.arctan2((k - m * omega**2), (c_ajuste * omega))

            velocidad_teorica = V_amp * np.cos(omega * t_datos - phi)
            FEM_FL_teorica = G_sub_A * R_porcentaje * velocidad_teorica
            return FEM_FL_teorica

        # ==========================================
        # 2. CONFIGURACIÓN DEL SISTEMA DAQ
        # ==========================================
        PUERTO = "COM3"
        BAUDIOS = 115200
        TAMANO_VENTANA = 512
        FS = 1000.0

        FC_PASA_ALTAS = 0.5
        FC_PASA_BAJAS = 40.0

        buffer_t = collections.deque(maxlen=TAMANO_VENTANA)
        buffer_v = collections.deque(maxlen=TAMANO_VENTANA)

        estado = {
            "tiempo_inicial": None,
            "c_estimado": c_sub_lambda,
            "contador_frames": 0,
        }

        try:
            esp32 = serial.Serial(PUERTO, BAUDIOS)
            print(f"Conectado exitosamente a {PUERTO}. Recibiendo datos...")
        except Exception as e:
            print(f"Error crítico al conectar con el ESP32 en el puerto {PUERTO}: {e}")
            return

        # CREACIÓN DE LOS 3 PLOTS (Ajuste de tamaño a 12x9 para dar espacio)
        fig, (ax_tiempo, ax_fft, ax_pos) = plt.subplots(3, 1, figsize=(12, 9))
        fig.canvas.manager.set_window_title("DAQ Sismómetro - Gemelo Digital EAFIT")

        # Plot 1: Voltaje
        (linea_t,) = ax_tiempo.plot(
            [], [], lw=1.5, color="purple", label="Voltaje Real (DAQ)"
        )
        (linea_teorica,) = ax_tiempo.plot(
            [], [], lw=1.5, color="red", linestyle="--", label="Voltaje Teórico"
        )
        ax_tiempo.set_title("Dominio del Tiempo (Señal Acoplada en AC)")
        ax_tiempo.set_ylabel("Voltaje (V)")
        ax_tiempo.grid(True)
        ax_tiempo.legend(loc="upper right")
        texto_metricas = ax_tiempo.text(
            0.02,
            0.75,
            "",
            transform=ax_tiempo.transAxes,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
        )

        # Plot 2: FFT
        (linea_f,) = ax_fft.plot([], [], lw=1.5, color="blue")
        ax_fft.set_title("Dominio de la Frecuencia (FFT)")
        ax_fft.set_ylabel("Amplitud")
        ax_fft.set_xlabel("Frecuencia (Hz)")
        ax_fft.set_xlim(0, 120)
        ax_fft.grid(True)

        # Plot 3: Posición calculada (Nuevo)
        (linea_pos,) = ax_pos.plot(
            [], [], lw=1.5, color="darkgreen", label="Posición del Imán ($x(t)$)"
        )
        ax_pos.set_title(
            "Posición Estimada del Imán (Integración Numérica de Faraday-Lenz)"
        )
        ax_pos.set_ylabel("Desplazamiento (m)")
        ax_pos.set_xlabel("Tiempo (s)")
        ax_pos.grid(True)
        ax_pos.legend(loc="upper right")

        plt.tight_layout()

        def aplicar_filtros(tiempo, voltaje):
            nyquist = 0.5 * FS
            b_altas, a_altas = butter(2, FC_PASA_ALTAS / nyquist, btype="high")
            b_bajas, a_bajas = butter(2, FC_PASA_BAJAS / nyquist, btype="low")
            v_filt = filtfilt(b_altas, a_altas, voltaje)
            v_filt = filtfilt(b_bajas, a_bajas, v_filt)
            return v_filt

        # ==========================================
        # 3. MOTOR DE ANIMACIÓN Y PROCESAMIENTO
        # ==========================================
        def actualizar(frame):
            while esp32.in_waiting > 0:
                try:
                    linea = esp32.readline().decode("utf-8", errors="ignore").strip()
                    if "," in linea:
                        t_micros_str, v_str = linea.split(",")
                        t_micros = int(t_micros_str)
                        v_crudo = float(v_str)

                        if estado["tiempo_inicial"] is None:
                            estado["tiempo_inicial"] = t_micros

                        t_segundos = (t_micros - estado["tiempo_inicial"]) / 1000000.0

                        buffer_t.append(t_segundos)
                        buffer_v.append(v_crudo)
                except Exception as e:
                    print(f"Aviso de lectura: {e}")

            if len(buffer_t) == TAMANO_VENTANA:
                t_arr = np.array(buffer_t)
                v_arr = np.array(buffer_v)

                v_filtrado = aplicar_filtros(t_arr, v_arr)

                # Desacoplamiento del curve_fit
                estado["contador_frames"] += 1
                if estado["contador_frames"] % 10 == 0:
                    try:
                        popt, pcov = curve_fit(
                            modelo_gemelo_digital,
                            t_arr,
                            v_filtrado,
                            p0=[c_sub_lambda],
                            bounds=(0, 10),
                        )
                        estado["c_estimado"] = popt[0]
                    except:
                        pass

                v_teorico = modelo_gemelo_digital(t_arr, estado["c_estimado"])

                # -----------------------------------------------------------
                # COMPUTACIÓN DE LA POSICIÓN (FÍSICA DE INGENIERÍA INVERSA)
                # -----------------------------------------------------------
                # 1. Despejar velocidad: v(t) = - V(t) / (G_sub_A * R_porcentaje)
                # Multiplicamos por menos el factor inverso tal como lo solicitaste
                velocidad_real = -v_filtrado / (G_sub_A * R_porcentaje)

                # 2. Integrar numéricamente con respecto al tiempo para obtener posición
                posicion_iman = cumulative_trapezoid(velocidad_real, t_arr, initial=0)
                # -----------------------------------------------------------

                # Análisis de frecuencia (FFT)
                N = TAMANO_VENTANA
                T_muestreo = 1.0 / FS
                yf = fft(v_filtrado)
                xf = fftfreq(N, T_muestreo)[: N // 2]
                amplitud_fft = 2.0 / N * np.abs(yf[0 : N // 2])

                # Renderizado de Gráfica 1: Voltaje
                linea_t.set_data(t_arr, v_filtrado)
                linea_teorica.set_data(t_arr, v_teorico)
                ax_tiempo.set_xlim(t_arr[0], t_arr[-1])
                margen_v = max(np.abs(v_filtrado)) * 1.2 + 0.01
                ax_tiempo.set_ylim(-margen_v, margen_v)

                v_pico = np.max(np.abs(v_filtrado))
                v_rms = np.sqrt(np.mean(v_filtrado**2))
                texto_metricas.set_text(
                    f"Voltaje Pico: {v_pico:.3f} V\n"
                    f"Voltaje RMS: {v_rms:.3f} V\n"
                    f"c_Teorico (Stokes): {c_sub_lambda:.4f} N·s/m\n"
                    f"c_Estimado (Real): {estado['c_estimado']:.4f} N·s/m"
                )

                # Renderizado de Gráfica 2: FFT
                linea_f.set_data(xf, amplitud_fft)
                ax_fft.set_ylim(0, max(amplitud_fft) * 1.2 + 0.01)

                # Renderizado de Gráfica 3: Posición del Imán
                linea_pos.set_data(t_arr, posicion_iman)
                ax_pos.set_xlim(t_arr[0], t_arr[-1])
                margen_p = (
                    max(np.abs(posicion_iman)) * 1.2 + 1e-7
                )  # Evita colapso si es 0
                ax_pos.set_ylim(-margen_p, margen_p)

            return linea_t, linea_teorica, linea_f, linea_pos, texto_metricas

        ani = animation.FuncAnimation(fig, actualizar, interval=30, blit=False)
        plt.show()
        esp32.close()


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------


# ===================================================================
# 3. EJECUCIÓN PRINCIPAL DEL PROGRAMA
# ===================================================================

print("\n--- INICIALIZANDO PARÁMETROS DEL SISMÓMETRO ---")
# Esto disparará el __post_init__ y las preguntas del imán
parametros = SectionParams()

print("\n--- INICIANDO INTEGRAL CAMPO MAGNETICO ---")
# Esto ejecuta tu matemática y abre las ventanas de Matplotlib
Factores_Acople(parametros)


print("\n--- INICIANDO SOLVER Y GRÁFICAS ---")
# Esto ejecuta tu matemática y abre las ventanas de Matplotlib
Solver(modelo_mk_1, modelo_mk_2, parametros)
