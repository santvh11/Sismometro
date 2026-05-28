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

# Resubir a github

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
# 1. Parámetros de sección (dataclass con todos los parámetros del sismómetro)
# -------------------------------------------------------------------

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
# Imánes:
# Ambos son de Neodimio N35
# Masa del imán de la bobina pequeña: 1
# Masa del imán de la bobina grande: 3,1
# Escalar de momento magnético del imán de la bobina pequeña: 25.23e*(1e-2)
# Escalar de momento magnético del imán de la bobina grande: 17.33e*(1e-2)
# -------------------------------------------------------------------
# Constantes Elásticas
# 1.5*(1e-3) Kg (resorte bobina grande)
# 1*(1e-3) Kg (resorte bobina pequeña)
# k= 83 (resorte bobina pequeña)
# k= 112 (resorte bobina grande)
# -------------------------------------------------------------------
# Parámetros Eléctricos:
# ----------------------------
# Bobina Pequeña:
# Inductancia: 5*10^-2 Henrios
# Capacitancia: 4*10^-5 Faradios
# Resistencia: 482 Ohmnios
# N. vueltas: 3000
# Calibre Cable / Diámetro:  AGW 38 / 0.102 * 10^-6 m^2
# ----------------------------
# Bobina Grande:
# Inductancia: 98*10^-3 Henrios
# Capacitancia: 94*10^-5 Faradios
# Resistencia: 689 Ohmnios
# N. vueltas: 3000
# Calibre Cable / Diámetro: AGW 38 / 0.102 * 10^-6 m^2
# ----------------------------
# Parámetros de Construcción:
# Bobina Pequeña:
# Masa del solenoide amrado: 44*10^-3 Kg
# Radio del solenoide (r_p): 5 *10^-3 m
# Altura del solenoide (h_p): 36 *10^-3 m
# Grosor de la capa de PLA (e_p): 3 *10^-3 m
# ----------------------------
# Bobina Grande:
# Masa del solenoide amrado: 73*10^-3 Kg
# Radio del solenoide (r_p): 7 *10^-3 m
# Altura del solenoide (h_p): 34 *10^-3 m
# Grosor de la capa de PLA (e_p): 3 *10^-3 m
# ----------------------------
# Funciones de Fuerza para la mesa de vibraciones:
# ----------------------------
# Bobina Grande:
# Modelo empírico de la aceleración de la mesa (obtenido experimentalmente)
# Ajustados por parámetros lorentzianos de picos
"""""
#Primer Pico
    Lorentz_a_1:float = 5.766*(1e-1)
    Lorentz_b_1:float = 3.150
    Lorentz_c_1:float = 2.993

    #Segundo Pico
    Lorentz_a_2:float = 2.622*(1e-1)
    Lorentz_b_2:float = 5.574*(1e-1)
    Lorentz_c_2:float = 8.964

    #Valle
    Lorentz_k:float = 6.727*(1e-6)
""" ""
# ----------------------------
# Bobina Pequeña:
# Ajustados por parámetros lorentzianos de picos
""""
#Primer Pico
    Lorentz_a_1:float = 9.368*(1e-1)
    Lorentz_b_1:float = 3.196
    Lorentz_c_1:float = 2.993

    #Segundo Pico
    Lorentz_a_2:float = 4.257*(1e-1)
    Lorentz_b_2:float = 5.581*(1e-1)
    Lorentz_c_2:float = 8.964

    #Valle
    Lorentz_k:float = 6.715*(1e-6)
"""


@dataclass
class SectionParams:
    # ---------------------------------------------------------------
    # Parámetros de primer orden (ingresados directamente por el usuario)
    # ---------------------------------------------------------------
    R: float = 482  # Resistencia total (Ohm)
    L: float = 98 * (1e-3)  # Inductancia (H)
    C: float = 94 * (1e-5)  # Capacitancia (F)
    c: float = 2  # Coeficiente de amortiguamiento (Kg/s) - valor por defecto
    m: float = 3.4 * (1e-3)  # Masa oscilante (Kg)
    k: float = 112  # Constante elástica (N/m)
    m_mag: float = 17.33 * (1e-2)  # Momento magnético (A*m^2)
    m_sis: float = 73 * (1e-3)  # Masa total del sismómetro (Kg)
    m_mes: float = 4 * (1e-3)  # Masa vibrante de la mesa (Kg)
    m_tornillo: float = 11 * (1e-3)  # Masa del tornillo de ajuste (kg)
    factor_amplificacion: float = 8.219  # Ganancia del amplificador
    subida_voltaje: float = 0  # Offset del ADC (voltios)
    temp: float = 299.15  # Temperatura ambiente (K)

    # Condiciones iniciales
    z: float = 0  # Posición inicial (m)
    z_dot: float = 0  # Velocidad inicial (m/s)
    Q: float = 0  # Carga inicial (C)
    Q_dot: float = 0  # Corriente inicial (A)

    # Parámetros mecánicos y geométricos
    g: float = 9.77  # Gravedad (m/s²)
    eta: float = 1.5  # Viscosidad dinámica (Pa·s) - se recalcula con temperatura
    R_sub_e: float = 9.5 * (1e-3)  # Radio de la esfera (m)
    L_cilindro: float = 10 * (1e-3)  # Longitud del cilindro (resorte) (m)
    L_libre_iman: float = 135 * (1e-3)  # Longitud libre del imán (m)
    omega_Hz: float = 10  # Frecuencia de excitación (Hz)
    omega = omega_Hz * 2 * np.pi  # Frecuencia angular (rad/s)
    omega_tolerancia: float = 5  # Rango de tolerancia para filtrado de frecuencias

    # Dimensiones del contenedor y solenoide
    e_sub_p: float = 3 * (1e-3)  # Espesor del contenedor de PLA (m)
    h_sub_p: float = 34 * (1e-3)  # Altura del contenedor (m)
    r_sub_p: float = 7 * (1e-3)  # Radio del contenedor (m)
    h_sub_f: float = 34 * (1e-3)  # Altura del fluido (m)
    g_sub_ecs: float = 0.015 * (1e-3)  # Grosor del esmalte (m)
    e_sub_cs: float = 0.102 * (1e-3)  # Diámetro del cable del solenoide (m)
    N_sub_c_total: float = 3000  # Número total de vueltas

    # Propiedades de fluidos (glicerina)
    rho_0: float = 1273.3  # Densidad a 0°C (kg/m³)
    beta_rho: float = 0.6121  # Coeficiente térmico de densidad (kg/(m³·K))
    densidad_neodimio: float = 7500  # Densidad del neodimio (kg/m³)
    eta_0: float = 3.30e-10  # Factor preexponencial de Andrade (Pa·s)
    b_eta: float = 6640  # Constante de Andrade (K)

    # Propiedades magnéticas
    R_iman: float = 1.5 * (1e-3)  # Radio del imán (m)
    Br_A: float = 0.9  # Remanencia acero 440C (T)
    Br_N32: float = 1.14  # Remanencia neodimio N32 (T)
    Br_N35: float = 1.22  # Remanencia neodimio N35 (T)

    # Propiedades eléctricas (cables, resistencias adicionales)
    e_sub_c: float = 1 * (1e-3)  # Diámetro cable de conexión (m)
    L_sub_c: float = 1 * (1e-2)  # Longitud cable conexión (m)
    R_sub_a: float = 1 * (1e8)  # Resistencia de entrada del amplificador (Ohm)
    R_extra: float = 1  # Resistencia añadida (Ohm)
    umbral_voltaje = 0.020  # Margen de error de voltaje del LM-358

    # Constantes físicas
    epsilon_sub_cero: float = 8.854 * (1e-12)  # Permitividad del vacío (F/m)
    epsilon_sub_e: float = 2.5  # Permitividad relativa del esmalte
    p_sub_cu: float = 1.72 * (1e-8)  # Resistividad del cobre (Ohm·m)
    mu_sub_cero: float = 4 * np.pi * (1e-7)  # Permeabilidad del vacío (H/m)
    mu_sub_PLA: float = 1  # Permeabilidad del PLA
    mu_sub_f: float = 1  # Permeabilidad del fluido

    # Rango de tiempo y discretización
    a: int = 0  # Tiempo inicial (s)
    b: float = 0.5  # Tiempo final (s)
    puntos: int = 10000  # Número de puntos para simulación
    delta_t: int = 1  # Factor de submuestreo (cada delta_t puntos)H

    # Parámetros Lorentzianos para la Fuerza de la mesa

    # Primer Pico
    Lorentz_a_1: float = 5.766 * (1e-1)
    Lorentz_b_1: float = 3.150
    Lorentz_c_1: float = 2.993

    # Segundo Pico
    Lorentz_a_2: float = 2.622 * (1e-1)
    Lorentz_b_2: float = 5.574 * (1e-1)
    Lorentz_c_2: float = 8.964

    # Valle
    Lorentz_k: float = 6.727 * (1e-6)

    # ---------------------------------------------------------------
    # Parámetros de segundo orden (se calculan automáticamente)
    # ---------------------------------------------------------------
    m_fluido: float = field(init=False)
    N_sub_c_capas: float = field(init=False)
    c_sub_Stokes_resorte: float = field(init=False)
    c_sub_Stokes_iman: float = field(init=False)
    c_sub_Stokes: float = field(init=False)
    lambda_c: float = field(init=False)
    c_sub_lambda: float = field(default=0, init=False)
    F_0: float = field(init=False)
    omega_sub_n: float = field(init=False)
    Zeta: float = field(init=False)
    p_error_Zeta: float = field(init=False)
    omega_sub_d: float = field(init=False)
    e_total: float = field(init=False)
    L_sub_s: float = field(init=False)
    A_sub_s: float = field(init=False)
    A_sub_cs: float = field(init=False)
    A_effec: float = field(init=False)
    A_sub_c: float = field(init=False)
    delta_h_sub_cs: float = field(init=False)
    N_sub_c: int = field(init=False)
    rho: float = field(init=False)
    R_sub_s: float = field(init=False)
    R_sub_c: float = field(init=False)
    R_porcentaje: float = field(init=False)
    C_inicial: float = field(init=False)
    C_N_sub_cs: float = field(init=False)
    C_N_sub_cs_capas: float = field(init=False)
    XC_capacitiva: float = field(init=False)
    XC_inductiva: float = field(init=False)
    XC: float = field(init=False)
    alpha: float = field(init=False)
    omega_sub_0_phi_m: float = field(init=False)
    t: np.ndarray = field(init=False)

    def __post_init__(self):
        """Calcula todos los parámetros derivados después de inicializar los básicos."""
        # Geometría del solenoide
        self.N_sub_c = round(self.h_sub_p / self.e_sub_cs)  # Vueltas por capa
        self.N_sub_c_capas = self.N_sub_c_total / self.N_sub_c  # Número de capas
        self.e_total = (
            self.r_sub_p + self.e_sub_p + (0.5 * self.e_sub_cs)
        )  # Radio medio del solenoide
        self.L_sub_s = self.N_sub_c_total * (
            self.e_total * 2 * np.pi
        )  # Longitud total del cable
        self.A_sub_s = self.N_sub_c * (
            self.N_sub_c_capas * (4 * np.pi**2) * (self.e_total * self.e_sub_cs)
        )
        self.A_effec = (
            8 * self.e_sub_cs * np.pi * self.e_total * self.N_sub_c_total
        )  # Área efectiva (campo magnético)
        self.A_sub_cs = np.pi * ((self.e_sub_cs * 0.5) ** 2)  # Área de corte del cable
        self.A_sub_c = np.pi * (
            (self.e_sub_c * 0.5) ** 2
        )  # Área de corte del cable de conexión
        self.delta_h_sub_cs = self.e_sub_cs  # Espaciado entre vueltas

        # Viscosidad y densidad en función de la temperatura
        self.eta = self.eta_0 * np.exp(self.b_eta / self.temp)  # Viscosidad (Pa·s)
        self.rho = self.rho_0 - self.beta_rho * (self.temp - 273.15)  # Densidad (kg/m³)

        # Preguntar si se usan valores teóricos o los directos
        teoric_params = bool(
            input(
                "¿Desea usar los parámetros con los datos directos? (Enter) para sí y escriba algo para no: "
            )
        )
        if teoric_params:
            self.masa_teorico()
            self.amortiguamiento_teorico()
            self.capacitancia_teorico()
            self.resistencia_teorico()
            self.inductancia_teorico()
            self.m_mag_teorico()

        # Masa añadida por el fluido (efecto de masa virtual)
        self.m_fluido = np.abs(
            self.rho
            * (np.pi * (self.R_sub_e**2) * (self.h_sub_f - 0.75 * self.R_sub_e))
        )
        self.m = self.m + self.m_fluido

        # Parámetros de la EDO mecánica
        self.omega_sub_n = np.sqrt(
            self.k / self.m
        )  # Frecuencia natural no amortiguada (rad/s)
        self.Zeta = self.c / (2 * np.sqrt(self.k * self.m))  # Factor de amortiguamiento
        if self.Zeta < 1:
            self.omega_sub_d = self.omega_sub_n * np.sqrt(
                1 - self.Zeta**2
            )  # Frecuencia natural amortiguada (rad/s)
        else:
            self.omega_sub_d = 0.0

        # Parámetros del circuito RLC
        self.XC_inductiva = self.L * self.omega  # Reactancia inductiva (Ohm)
        self.XC_capacitiva = -1 / (self.omega * self.C)  # Reactancia capacitiva (Ohm)
        self.R_porcentaje = self.R_sub_a / (
            self.R + self.R_sub_a
        )  # Divisor de tensión para el amplificador
        self.alpha = self.R / (2 * self.L)  # Coeficiente de amortiguamiento eléctrico
        self.XC = self.XC_capacitiva + self.XC_inductiva  # Reactancia total
        self.omega_sub_0_phi_m = 1 / np.sqrt(
            self.L * self.C
        )  # Frecuencia de resonancia del circuito (rad/s)

        # Vector de tiempo para simulaciones
        self.t = np.linspace(self.a, self.b, self.puntos)

        # Cálculo de la fuerza de excitación F_0 (a partir de la mesa vibratoria)
        fuerza = int(
            input("¿La fuerza es directa (1) o se calcula a partir de un MAS (2)?: ")
        )
        if fuerza == 1:
            self.F_0 = 5.0  # Fuerza constante (N)
        elif fuerza == 2:
            amplitud = int(
                input(
                    "¿La amplitud es directa (1) o se calcula en base a la frecuencia(2)?: "
                )
            )
            m_vibrante = (
                self.m_sis + self.m_mes + self.m_tornillo
            )  # Masa total vibrante
            if amplitud == 1:
                self.Y_0 = 7e-3  # Amplitud conocida (m)
                self.F_0 = m_vibrante * self.Y_0 * (self.omega**2)  # Fuerza = m·a
            elif amplitud == 2:
                # Modelo empírico de la aceleración de la mesa (obtenido experimentalmente)

                aceleracion_G_pico_1 = np.abs(
                    self.Lorentz_a_1
                    * self.Lorentz_b_1
                    / ((self.omega_Hz - self.Lorentz_c_1) ** 2 + self.Lorentz_b_1)
                )

                aceleracion_G_pico_2 = np.abs(
                    self.Lorentz_a_2
                    * self.Lorentz_b_2
                    / ((self.omega_Hz - self.Lorentz_c_2) ** 2 + self.Lorentz_b_2)
                )

                aceleracion_G_valle = np.abs(self.Lorentz_k * self.omega_Hz**2)

                aceleracion_G = (
                    aceleracion_G_pico_1 + aceleracion_G_pico_2 + aceleracion_G_valle
                )
                aceleracion = aceleracion_G * 9.81
                self.F_0 = m_vibrante * aceleracion
            else:
                raise ValueError("Opción no válida para amplitud")
        else:
            raise ValueError("Opción no válida para fuerza")

    # -------------------------------------------------------------------
    # Métodos para estimar parámetros teóricamente
    # -------------------------------------------------------------------
    def masa_teorico(self):
        """Estima la masa oscilante a partir del volumen de la esfera."""
        self.m = (self.R_sub_e) ** 3 * np.pi * 1.3333 * self.densidad_neodimio

    def amortiguamiento_teorico(self):
        """Calcula el coeficiente de amortiguamiento viscoso usando la ley de Stokes y corrección de Faxen."""
        self.c_sub_Stokes_iman = 6 * np.pi * self.eta * self.R_sub_e
        self.c_sub_Stokes_resorte = np.abs(
            4
            * np.pi
            * self.eta
            * self.L_cilindro
            / (np.log(self.L_cilindro / 2 * self.R_sub_e) + 0.5)
        )
        self.lambda_c = self.R_sub_e / self.r_sub_p
        if self.lambda_c > 0.6:
            # Haberman & Sayre (cercanía)
            den = 1 - (0.75857 * self.lambda_c**5)
            num = (
                1
                - 2.10444 * self.lambda_c
                + 2.08877 * self.lambda_c**3
                - 0.94813 * self.lambda_c**5
            )
            self.c_sub_lambda = num / den
        else:
            # Haberman & Faxen (lejanía)
            self.c_sub_lambda = (
                1
                - 2.104 * self.lambda_c
                + 2.089 * self.lambda_c**3
                - 0.948 * self.lambda_c**5
            )
        self.c = (
            self.c_sub_Stokes_iman + self.c_sub_Stokes_resorte
        ) / self.c_sub_lambda

    def capacitancia_teorico(self):
        """Capacitancia parásita entre espiras del solenoide (modelo de dos cilindros paralelos)."""
        self.C_inicial = (self.L_sub_s * self.epsilon_sub_cero * self.epsilon_sub_e) / (
            np.log((self.g_sub_ecs + self.e_sub_cs) / self.e_sub_cs)
        )
        self.C_N_sub_cs = 2 * self.C_inicial / self.N_sub_c
        self.C = self.C_N_sub_cs * self.N_sub_c_capas

    def inductancia_teorico(self):
        """Inductancia del solenoide (fórmula de solenoide largo)."""
        seccion_transversal_sol = np.pi * (self.r_sub_p + self.e_sub_p) ** 2
        self.L = (
            seccion_transversal_sol
            * self.mu_sub_cero
            * (self.N_sub_c_total**2)
            / self.h_sub_p
        )

    def resistencia_teorico(self):
        """Resistencia total del circuito (solenoide + cable + amplificador)."""
        self.R_sub_s = self.p_sub_cu * self.L_sub_s / self.A_sub_cs
        self.R_sub_c = self.p_sub_cu * self.L_sub_c / self.A_sub_c
        self.R = (self.R_sub_a * (self.R_sub_s + self.R_sub_c + self.R_extra)) / (
            self.R_sub_a + self.R_sub_s + self.R_sub_c + self.R_extra
        )

    def m_mag_teorico(self):
        """Momento magnético del imán a partir de su material y volumen."""
        tipo_iman = int(
            input("¿El imán es de acero 440 (1), neodimio N32(2) o Neodimio N35(3)?: ")
        )
        Vol = ((self.R_iman) ** 3) * 0.75 * np.pi
        if tipo_iman == 1:
            self.m_mag = self.Br_A * Vol / self.mu_sub_cero
        elif tipo_iman == 2:
            self.m_mag = self.Br_N32 * Vol / self.mu_sub_cero
        elif tipo_iman == 3:
            self.m_mag = self.Br_N35 * Vol / self.mu_sub_cero
        else:
            raise ValueError("Opción no válida para el tipo de imán")


# -------------------------------------------------------------------
# Factores de acoplamiento electromecánico (G_sub_L y G_sub_A)
# -------------------------------------------------------------------
def Factores_Acople(params: SectionParams) -> tuple[float, float]:
    """
    Calcula los factores de acople:
    - G_sub_L : fuerza de Laplace por unidad de corriente (N/A)
    - G_sub_A : fuerza contraelectromotriz por unidad de velocidad (V·s/m)
    """
    e_sub_p = params.e_sub_p
    r_sub_p = params.r_sub_p
    mu_sub_cero = params.mu_sub_cero
    m_mag = params.m_mag
    e_sub_cs = params.e_sub_cs
    N_sub_cs_total = params.N_sub_c_total
    h_sub_p = params.h_sub_p

    constantes_magneticas = (0.25 * m_mag * mu_sub_cero) / np.pi
    Radio_total = e_sub_p + r_sub_p
    Paso = (h_sub_p * 0.5) / (np.pi * N_sub_cs_total)
    Integral_linea = (1 / Radio_total**3) - (
        1
        / (np.sqrt(((Radio_total**2) + (Paso * 2 * np.pi * N_sub_cs_total) ** 2))) ** 3
    )
    G_sub_A = (constantes_magneticas * 3 * Radio_total**2 / Paso) * Integral_linea
    G_sub_L = (constantes_magneticas * Radio_total**2) / (
        Paso * (np.sqrt(Radio_total**2 + (2 * np.pi * Paso * N_sub_cs_total) ** 2)) ** 3
    )
    return G_sub_L, G_sub_A


# -------------------------------------------------------------------
# 2. Solver de EDO´s (modelos mk1, mk2 y modos experimentales)
# -------------------------------------------------------------------
def Solver(
    modelo_mk1: bool,
    modelo_mk2: bool,
    simular: bool,
    datos_descarga: bool,
    experimento_c: bool,
    params: SectionParams,
):
    """
    Función principal que resuelve las ecuaciones del sistema según el modo elegido.
    """
    # Desempaquetado de parámetros
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
    omega_Hz = params.omega_Hz
    omega_tolerancia = params.omega_tolerancia
    umbral_voltaje = params.umbral_voltaje

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
                [0, G_sub_A / L, -1 / (L * C), -R / L],
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
        UMBRAL_VOLTAJE = 0.02  # voltios

        TIMEOUT_SEGUNDOS = 4
        MAX_REINTENTOS = 3

        # Opción de solo guardar sin gráficos
        MODO_SOLO_GUARDAR = bool(
            input("¿Desea ver gráficas (Enter) o guardar datos (escribir algo): ")
        )

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
            print("No se pudo conectar al ESP32. Saliendo del modo 3.")
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
            """Filtro pasa-altas + pasa-bajas usando filtfilt (sin desfase)."""
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
            print(f"\n⚠️ Sin datos durante {TIMEOUT_SEGUNDOS} s. Reintentando...")
            try:
                esp32.close()
            except:
                pass
            time.sleep(1)
            reintentos += 1
            if reintentos < MAX_REINTENTOS:
                esp32 = conectar_serial(PUERTO, BAUDIOS, reintentos + 1)
                if esp32 is not None:
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
                print("Máximo de reintentos alcanzado. Cerrando modo.")
                estado["conexion_activa"] = False
                plt.close("all")
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
            print("Presione Ctrl+C para detener.")
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
                            v_real = v_filtrado / factor_amplificacion
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
        # SUB-MODO: VISUALIZACIÓN CON GRÁFICAS EN TIEMPO REAL
        # -----------------------------------------------
        else:
            # Configuración de subplots según la vista seleccionada
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

            # Inicialización de líneas y textos
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

                # Timeout y reconexión
                if time.time() - estado["ultima_actualizacion"] > TIMEOUT_SEGUNDOS:
                    reiniciar_conexion()
                    if not estado["conexion_activa"]:
                        plt.close(fig)
                        return ()
                    else:
                        if "text_timeout" in locals():
                            text_timeout.set_text("Esperando datos... Reintentando")
                        return ()
                if "text_timeout" in locals():
                    text_timeout.set_text("")

                if len(buffer_t) == TAMANO_VENTANA:
                    t_arr = np.array(buffer_t)
                    v_arr = np.array(buffer_v) - subida_voltaje
                    v_filtrado = aplicar_filtros(t_arr, v_arr)
                    v_real = v_filtrado / factor_amplificacion

                    # Umbral para no dibujar ruido
                    if np.max(np.abs(v_real)) < UMBRAL_VOLTAJE:
                        return ()

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

            ani = animation.FuncAnimation(
                fig, actualizar, interval=30, blit=False, save_count=100
            )
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

        # ---------------------------------------------------------------
        # Función de captura de datos con filtrado y umbral
        # ---------------------------------------------------------------
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
            v_raw = np.array(datos_v)

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

            def estimar_c_desde_velocidad(
                t, v_fem, F_0, k, m, omega, G_sub_A, R_porcentaje
            ):
                # Primero obtenemos la velocidad real a partir de la FEM
                velocidad = v_fem / (G_sub_A * R_porcentaje)  # [m/s]

                # Modelo: v(t) = A * cos(omega*t - phi) + C
                def modelo(t, A, phi, C):
                    return A * np.cos(omega * t - phi) + C

                try:
                    p0 = [np.max(np.abs(velocidad)), 0.0, 0.0]
                    popt, _ = curve_fit(modelo, t, velocidad, p0=p0)
                    A_est, phi_est, C_est = popt
                except Exception as e:
                    print(f"Error en el ajuste senoidal: {e}")
                    return None, None, None, None

                discriminante = (F_0 / A_est) ** 2 - (k - m * omega**2) ** 2
                if discriminante < 0:
                    c_amp = None
                else:
                    c_amp = np.sqrt(discriminante) / omega

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
