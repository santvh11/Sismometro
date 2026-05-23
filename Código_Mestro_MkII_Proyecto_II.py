import serial
from dataclasses import dataclass, field
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
import collections
import csv
import time
from datetime import datetime
import logging

# -------------------------------------------------------------------
# 0. Selección del modelo:
# -------------------------------------------------------------------

while True:
    modelo = input(
        "Escriba (1) para mk_1 o (2) para mk_2, (3) para ver voltaje en tiempo real, o (4) para descargar datos: "
    ).strip()
    if modelo == "1":
        modelo_mk1, modelo_mk2, simular, datos_descarga = True, False, False, False
        break
    elif modelo == "2":
        modelo_mk1, modelo_mk2, simular, datos_descarga = False, True, False, False
        break
    elif modelo == "3":
        modelo_mk1, modelo_mk2, simular, datos_descarga = True, False, True, False
        break
    elif modelo == "4":
        modelo_mk1, modelo_mk2, simular, datos_descarga = True, False, False, True
        break
    else:
        print("Entrada no válida. Inténtalo de nuevo.")

# -------------------------------------------------------------------
# 1. Parámetros de sección:
# -------------------------------------------------------------------


@dataclass
class SectionParams:
    # 1er Orden
    R: float = 383.1
    L: float = 36.7 * (1e-3)
    C: float = 0.07 * (1e-6)
    c: float = 1.1
    m: float = 11 * (1e-3)
    k: float = 272
    m_mag: float = 13.93e-03
    m_sis: float = 48.6 * (1e-3)
    m_mes: float = 3.148 * (1e-3)
    factor_amplificacion: float = 8.219
    temp: float = 299.15
    z: float = 0
    z_dot: float = 0
    Q: float = 0
    Q_dot: float = 0
    g: float = 9.77
    eta: float = 1.5
    R_sub_e: float = 9.5 * (1e-3)
    L_cilindro: float = 10 * (1e-3)
    L_libre_iman: float = 135 * (1e-3)
    omega_Hz: float = 10
    omega = omega_Hz * 2 * np.pi
    e_sub_p: float = 3 * (1e-3)
    h_sub_p: float = 34 * (1e-3)
    r_sub_p: float = 14 * (1e-3)
    h_sub_f: float = 34 * (1e-3)
    g_sub_ecs: float = 0.015 * (1e-3)
    e_sub_cs: float = 0.079 * (1e-3)
    N_sub_c_total: float = 3000
    rho_0: float = 1273.3
    beta_rho: float = 0.6121
    densidad_neodimio: float = 7500
    eta_0: float = 3.30e-10
    b_eta: float = 6640
    R_iman: float = 1.5 * (1e-3)
    Br_A: float = 0.9
    Br_N32: float = 1.14
    Br_N35: float = 1.22
    e_sub_c: float = 1 * (1e-3)
    L_sub_c: float = 1 * (1e-2)
    R_sub_a: float = 1 * (1e8)
    R_extra: float = 1
    epsilon_sub_cero: float = 8.854 * (1e-12)
    epsilon_sub_e: float = 2.5
    p_sub_cu: float = 1.72 * (1e-8)
    mu_sub_cero: float = (4) * (np.pi) * (1e-7)
    mu_sub_PLA: float = 1
    mu_sub_f: float = 1
    a: int = 0
    b: float = 0.5
    puntos: int = 10000
    delta_t: int = 1

    # 2do Orden
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
    B: float = field(init=False)
    root_root: float = field(init=False)
    root_1: float = field(init=False)
    root_2: float = field(init=False)
    Zeta_sub_omega: float = field(init=False)
    omega_sub_s: float = field(init=False)
    phi: float = field(init=False)
    X: float = field(init=False)
    e_total: float = field(init=False)
    L_sub_s: float = field(init=False)
    A_sub_s: float = field(init=False)
    A_sub_cs: float = field(init=False)
    A_effec: float = field(init=False)
    A_sub_c: float = field(init=False)
    delta_h_sub_cs: float = field(init=False)
    h_sub_cs: float = field(init=False)
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
        self.N_sub_c = round(self.h_sub_p / self.e_sub_cs)
        self.N_sub_c_capas = self.N_sub_c_total / self.N_sub_c
        self.e_total = self.r_sub_p + self.e_sub_p + (0.5 * self.e_sub_cs)
        self.L_sub_s = self.N_sub_c_total * (self.e_total * 2 * np.pi)
        self.A_sub_s = self.N_sub_c * (
            self.N_sub_c_capas * (4 * (np.pi**2)) * (self.e_total * self.e_sub_cs)
        )
        self.A_effec = 8 * self.e_sub_cs * np.pi * self.e_total * self.N_sub_c_total
        self.A_sub_cs = np.pi * ((self.e_sub_cs * 0.5) ** 2)
        self.A_sub_c = np.pi * ((self.e_sub_c * 0.5) ** 2)
        self.delta_h_sub_cs = self.e_sub_cs
        self.eta = self.eta_0 * np.exp(self.b_eta / self.temp)
        self.rho = self.rho_0 - self.beta_rho * (self.temp - 273.15)

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

        self.m_fluido = np.abs(
            self.rho
            * (np.pi * (self.R_sub_e**2) * (self.h_sub_f - 0.75 * self.R_sub_e))
        )
        self.m = self.m + self.m_fluido
        self.omega_sub_n = (self.k / self.m) ** 0.5
        self.Zeta = self.c / (2 * ((self.k * self.m) ** 0.5))
        self.omega_sub_d = self.omega_sub_n * (1 - self.Zeta**2) ** 0.5
        self.XC_inductiva = self.L * self.omega
        self.XC_capacitiva = -1 / (self.omega * self.C)
        self.R_porcentaje = self.R_sub_a / (self.R + self.R_sub_a)
        self.alpha = self.R / (2 * self.L)
        self.XC = self.XC_capacitiva + self.XC_inductiva
        self.omega_sub_0_phi_m = 1 / ((self.L * self.C) ** 0.5)
        self.t = np.linspace(self.a, self.b, self.puntos)

        fuerza = int(
            input("¿La fuerza es directa (1) o se calcula a partir de un MAS (2)?: ")
        )
        if fuerza == 1:
            self.F_0 = 5
        elif fuerza == 2:
            amplitud = int(
                input(
                    "¿La amplitud es directa (1) o se calcula en base a la frecuencia(2)?: "
                )
            )
            m_vibrante = self.m_sis + self.m_mes
            if amplitud == 1:
                self.Y_0 = 7 * (1e-3)
                self.F_0 = m_vibrante * self.Y_0 * (self.omega**2)
            elif amplitud == 2:
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
                aceleracion = aceleracion_G * 9.81
                self.F_0 = m_vibrante * aceleracion
            else:
                raise ValueError("No se ingresó una opción válida, oprima (1) o (2)")
        else:
            raise ValueError("No se ingresó una opción válida, oprima (1) o (2)")

    def masa_teorico(self):
        self.m = (self.R_sub_e) ** 3 * np.pi * 1.3333 * self.densidad_neodimio

    def amortiguamiento_teorico(self):
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
            den = 1 - (0.75857 * self.lambda_c**5)
            num = (
                1
                - (2.10444 * self.lambda_c)
                + (2.08877 * self.lambda_c**3)
                - (0.94813 * self.lambda_c**5)
            )
            self.c_sub_lambda = num / den
        elif self.lambda_c <= 0.6:
            self.c_sub_lambda = (
                1
                - (2.104 * (self.lambda_c))
                + (2.089 * (self.lambda_c**3))
                - 0.948 * (self.lambda_c**5)
            )
        self.c = (
            self.c_sub_Stokes_iman + self.c_sub_Stokes_resorte
        ) / self.c_sub_lambda

    def capacitancia_teorico(self):
        self.C_inicial = (self.L_sub_s * self.epsilon_sub_cero * self.epsilon_sub_e) / (
            np.log((self.g_sub_ecs + self.e_sub_cs) / (self.e_sub_cs))
        )
        self.C_N_sub_cs = 2 * self.C_inicial / self.N_sub_c
        self.C = self.C_N_sub_cs * self.N_sub_c_capas

    def inductancia_teorico(self):
        seccion_transversal_sol = np.pi * (self.r_sub_p + self.e_sub_p) ** 2
        self.L = (
            seccion_transversal_sol
            * self.mu_sub_cero
            * (self.N_sub_c_total**2)
            / self.h_sub_p
        )

    def resistencia_teorico(self):
        self.R_sub_s = self.p_sub_cu * self.L_sub_s / self.A_sub_cs
        self.R_sub_c = self.p_sub_cu * self.L_sub_c / self.A_sub_c
        self.R = (self.R_sub_a * (self.R_sub_s + self.R_sub_c + self.R_extra)) / (
            self.R_sub_a + self.R_sub_s + self.R_sub_c + self.R_extra
        )

    def m_mag_teorico(self):
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
            raise ValueError("Escriba (1) o (2) o (3), otra repuesta no es válida")


# --------------------------------------------------------------------------------------------------
def Factores_Acople(params: SectionParams) -> tuple[float, float]:
    e_sub_p = params.e_sub_p
    r_sub_p = params.r_sub_p
    mu_sub_cero = params.mu_sub_cero
    m_mag = params.m_mag
    e_sub_cs = params.e_sub_cs
    N_sub_cs_total = params.N_sub_c_total

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


# -------------------------------------------------------------------
# 2. Solver de EDO´s:
# -------------------------------------------------------------------


def Solver(
    modelo_mk1: bool,
    modelo_mk2: bool,
    simular: bool,
    datos_descarga: bool,
    params: SectionParams,
):
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

    if modelo_mk1 and not simular and not datos_descarga:
        print("--- Entrando a Modelo MK1 (Lineal) ---")
        A = np.array(
            [
                [0, 1, 0, 0],
                [-k / m, -c / m, 0, -G_sub_L / m],
                [0, 0, 0, 1],
                [0, -G_sub_A / L, -1 / (L * C), -(R) / (L)],
            ]
        )
        eigenvalues, eigenvectors = np.linalg.eig(A)
        F_t = np.array([[0], [1 / m], [0], [0]])
        I = np.eye(4)
        lado_izquierdo = 1j * omega * I - A
        lado_derecho = F_t * F_0

        Xp_complejo = np.linalg.solve(lado_izquierdo, lado_derecho)
        X_particular = Xp_complejo * np.exp(1j * omega * t)

        X_t_p = Xp_complejo.real
        X_t_h = np.array([[z_0, z_dot_0, Q_0, Q_dot_0]]).reshape(4, 1)
        X_t = X_t_h + X_t_p

        coeficientes = np.linalg.solve(eigenvectors, X_t)
        X_evolucion = np.zeros((4, len(t)), dtype=complex)

        for i in range(len(eigenvalues)):
            termino_exponencial = np.exp(eigenvalues[i] * t)
            contribucion = np.outer(eigenvectors[:, i], termino_exponencial)
            X_evolucion += coeficientes[i] * contribucion

        X_total = X_evolucion + X_particular
        FEM_FL = G_sub_A * R_porcentaje * X_total.real[1]
        t_filtro = t[::delta_t]

        frecuencias_naturales = np.abs(eigenvalues.imag)
        frecuencias_unicas = np.unique(frecuencias_naturales) * 2 * np.pi
        print(f"Frecuencias naturales del sistema acoplado: {frecuencias_unicas} Hz ")

        I_Re = X_total.real[3]
        I_Im = X_total.imag[3]
        V_Im = (R * I_Im) + (I_Re * XC)
        V_Re = (R * I_Re) - (I_Im * XC)
        V_A = ((V_Im**2) + (V_Re**2)) ** 0.5
        V_phi = np.arctan(V_Im / V_Re)
        FEM_Ohm = V_A * np.cos((omega * t) + V_phi)
        F_Laplace = X_total.real[3] * G_sub_L

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
            "Respuesta Temporal Completa - Detector de Gaia MK2",
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

        plt.figure(figsize=(8, 6), dpi=200)
        plt.step(
            t_filtro,
            FEM_Ohm,
            where="post",
            label="Muestreo (Escalonado)",
            color="purple",
            linewidth=1,
        )
        plt.title("FEM teórica producida por Ley de Ohm Fasorial")

        plt.text(
            0,
            0.02,
            f"Máx: {np.max(np.abs(FEM_Ohm)):.2e}",
            transform=plt.gca().transAxes,
            ha="left",
            fontsize=7,
            bbox=dict(facecolor="white", alpha=0.7),
        )
        plt.xlabel("Tiempo")
        plt.ylabel("FEM")
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 6), dpi=200)
        plt.step(
            t_filtro,
            FEM_FL,
            where="post",
            label="Muestreo (Escalonado)",
            color="purple",
            linewidth=1,
        )
        plt.title("FEM teórica producida por Faraday_Lenz")
        plt.text(
            0.95,
            0.02,
            f"Pico Estac: {np.max(np.abs(FEM_FL)):.2e}",
            transform=plt.gca().transAxes,
            ha="right",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7),
        )
        plt.text(
            0,
            0.02,
            f"Máx: {np.max(np.abs(FEM_FL)):.2e}",
            transform=plt.gca().transAxes,
            ha="left",
            fontsize=7,
            bbox=dict(facecolor="white", alpha=0.7),
        )
        plt.xlabel("Tiempo")
        plt.ylabel("FEM")
        plt.legend()
        plt.show()

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

    elif modelo_mk2 and not simular and not datos_descarga:

        def state_matrix(t, S):
            z, v, Q, I = S
            F_ext = F_0 * np.sin(omega * t)
            dz_dt = v
            dz_dt_2 = (1 / m) * (
                (-k * z) - (c * v * np.abs(v)) + (F_ext) + (I * G_sub_L)
            )
            dQ_dt = I
            dQ_dt_2 = (1 / L) * (-R * I - (1 / C) * Q + v * G_sub_A)
            return dz_dt, dz_dt_2, dQ_dt, dQ_dt_2

        t_span = (a, b)
        solver = solve_ivp(
            state_matrix,
            t_span,
            [z_0, z_dot_0, Q_0, Q_dot_0],
            t_eval=t,
            method="LSODA",
        )

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
            "Respuesta Temporal Completa - Detector de Gaia MK2",
            fontsize=16,
            fontweight="bold",
        )

        for i in range(4):
            ax = axs[i // 2, i % 2]
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

        I = solver.y[3]
        Z = R
        FEM_Ohm = I * Z
        FEM_FL = G_sub_A * R_porcentaje * solver.y[1]
        t_filtro = solver.t[::delta_t]

        plt.figure(figsize=(8, 6), dpi=200)
        plt.step(
            t_filtro,
            FEM_Ohm,
            where="post",
            label="Muestreo (Escalonado)",
            color="purple",
            linewidth=1,
        )
        plt.title("FEM teórica producida por Ley de Ohm Fasorial")
        plt.xlabel("Tiempo")
        plt.ylabel("FEM")
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 6), dpi=200)
        plt.step(
            t_filtro,
            FEM_FL,
            where="post",
            label="Muestreo (Escalonado)",
            color="purple",
            linewidth=1,
        )
        plt.title("FEM teórica producida por Faraday_Lenz")
        plt.xlabel("Tiempo")
        plt.ylabel("FEM")
        plt.legend()
        plt.show()

        F_Laplace = solver.y[3] * G_sub_L

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

    elif simular and not datos_descarga:
        print("\n--- Entrando a Adquisición de Datos y Gemelo Digital ---")

        def modelo_gemelo_digital(t_datos, c_ajuste, fase_extra):
            Z_m = np.sqrt((k - m * omega**2) ** 2 + (c_ajuste * omega) ** 2)
            V_amp = (F_0 * omega) / Z_m
            phi = np.arctan2((k - m * omega**2), (c_ajuste * omega))
            velocidad_teorica = V_amp * np.cos(omega * t_datos - phi + fase_extra)
            FEM_FL_teorica = (
                G_sub_A * R_porcentaje * velocidad_teorica * factor_amplificacion
            )
            return FEM_FL_teorica

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
            "c_estimado": c,
            "fase_estimada": 0.0,
            "contador_frames": 0,
        }

        try:
            esp32 = serial.Serial(PUERTO, BAUDIOS)
            print(f"Conectado exitosamente a {PUERTO}. Recibiendo datos...")
        except Exception as e:
            print(f"Error crítico al conectar con el ESP32 en el puerto {PUERTO}: {e}")
            return

        fig, (ax_tiempo, ax_fft, ax_pos) = plt.subplots(3, 1, figsize=(12, 9))
        fig.canvas.manager.set_window_title("DAQ Sismómetro - Gemelo Digital EAFIT")

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

        (linea_f,) = ax_fft.plot([], [], lw=1.5, color="blue")
        ax_fft.set_title("Dominio de la Frecuencia (FFT)")
        ax_fft.set_ylabel("Amplitud")
        ax_fft.set_xlabel("Frecuencia (Hz)")
        ax_fft.set_xlim(0, 120)
        ax_fft.grid(True)

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
                    pass

            if len(buffer_t) == TAMANO_VENTANA:
                t_arr = np.array(buffer_t)
                v_arr = np.array(buffer_v)
                v_filtrado = aplicar_filtros(t_arr, v_arr)

                UMBRAL_RUIDO = 0.05
                v_pico_actual = np.max(np.abs(v_filtrado))

                estado["contador_frames"] += 1
                if estado["contador_frames"] % 10 == 0:
                    if v_pico_actual > UMBRAL_RUIDO:
                        try:
                            popt, pcov = curve_fit(
                                modelo_gemelo_digital,
                                t_arr,
                                v_filtrado,
                                p0=[
                                    max(estado["c_estimado"], 0.01),
                                    estado["fase_estimada"],
                                ],
                                bounds=([0.0, -np.pi], [10.0, np.pi]),
                            )
                            estado["c_estimado"] = popt[0]
                            estado["fase_estimada"] = popt[1]
                        except Exception:
                            pass
                    else:
                        estado["c_estimado"] = 0.0
                        estado["fase_estimada"] = 0.0

                v_teorico = modelo_gemelo_digital(
                    t_arr, estado["c_estimado"], estado["fase_estimada"]
                )
                velocidad_real = -v_filtrado / (
                    G_sub_A * R_porcentaje * factor_amplificacion
                )
                posicion_iman = cumulative_trapezoid(velocidad_real, t_arr, initial=0)

                N = TAMANO_VENTANA
                T_muestreo = 1.0 / FS
                yf = fft(v_filtrado)
                xf = fftfreq(N, T_muestreo)[: N // 2]
                amplitud_fft = 2.0 / N * np.abs(yf[0 : N // 2])

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
                    f"c_Teorico (Stokes): {c:.4f} N·s/m\n"
                    f"c_Estimado (Real): {estado['c_estimado']:.4f} N·s/m"
                )

                linea_f.set_data(xf, amplitud_fft)
                ax_fft.set_ylim(0, max(amplitud_fft) * 1.2 + 0.01)

                linea_pos.set_data(t_arr, posicion_iman)
                ax_pos.set_xlim(t_arr[0], t_arr[-1])
                margen_p = max(np.abs(posicion_iman)) * 1.2 + 1e-7
                ax_pos.set_ylim(-margen_p, margen_p)

            return linea_t, linea_teorica, linea_f, linea_pos, texto_metricas

        ani = animation.FuncAnimation(fig, actualizar, interval=30, blit=False)
        plt.show()
        esp32.close()

    elif datos_descarga:

        def registrar_daq_headless(params, G_sub_A: float):
            logging.info("--- Iniciando DAQ Headless (Solo Registro) ---")
            PUERTO = "COM3"
            BAUDIOS = 115200
            TAMANO_VENTANA = 512
            FS = 1000.0
            FC_PASA_ALTAS = 0.5
            FC_PASA_BAJAS = 40.0
            UMBRAL_RUIDO_V = 0.05

            buffer_t = []
            buffer_v = []

            def aplicar_filtros(voltaje):
                nyquist = 0.5 * FS
                b_altas, a_altas = butter(2, FC_PASA_ALTAS / nyquist, btype="high")
                b_bajas, a_bajas = butter(2, FC_PASA_BAJAS / nyquist, btype="low")
                v_filtrado = filtfilt(b_altas, a_altas, voltaje)
                return filtfilt(b_bajas, a_bajas, v_filtrado)

            def modelo_gemelo_digital(t_datos, c_ajuste):
                Z_m = np.sqrt(
                    (params.k - params.m * params.omega**2) ** 2
                    + (c_ajuste * params.omega) ** 2
                )
                V_amp = (params.F_0 * params.omega) / Z_m
                phi = np.arctan2(
                    (params.k - params.m * params.omega**2), (c_ajuste * params.omega)
                )
                velocidad_teorica = V_amp * np.cos(params.omega * t_datos - phi)
                return (
                    G_sub_A
                    * params.R_porcentaje
                    * velocidad_teorica
                    * params.factor_amplificacion
                )

            nombre_archivo_log = f"log_sismometro_headless_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(nombre_archivo_log, mode="w", newline="") as archivo_csv:
                escritor = csv.writer(archivo_csv)
                escritor.writerow(
                    [
                        "Marca_de_Tiempo",
                        "Voltaje_Pico_(V)",
                        "Voltaje_RMS_(V)",
                        "Posicion_Z_Pico_Estimada_(m)",
                        "C_Estimado_(Kg/s)",
                        "Frec_Dominante_FFT_(Hz)",
                        "Amplitud_Max_FFT",
                    ]
                )

            print(f"Archivo de registro creado: {nombre_archivo_log}")
            print(
                f"{'HORA':<12} | {'V PICO':<10} | {'Z PICO (m)':<15} | {'C EST':<10} | {'FREQ (Hz)':<10}"
            )
            print("-" * 65)

            try:
                esp32 = serial.Serial(PUERTO, BAUDIOS)
            except Exception as e:
                logging.error(f"Error conectando a {PUERTO}: {e}")
                return

            tiempo_inicial_micros = None

            try:
                while True:
                    while esp32.in_waiting > 0:
                        try:
                            linea = esp32.readline().decode("utf-8").strip()
                            if "," in linea:
                                t_micros = int(linea.split(",")[0])
                                v_crudo = float(linea.split(",")[1])

                                if tiempo_inicial_micros is None:
                                    tiempo_inicial_micros = t_micros

                                buffer_t.append(
                                    (t_micros - tiempo_inicial_micros) / 1e6
                                )
                                buffer_v.append(v_crudo)
                        except Exception:
                            continue

                    if len(buffer_t) >= TAMANO_VENTANA:
                        t_arr = np.array(buffer_t[:TAMANO_VENTANA])
                        v_arr = np.array(buffer_v[:TAMANO_VENTANA])

                        buffer_t = buffer_t[TAMANO_VENTANA:]
                        buffer_v = buffer_v[TAMANO_VENTANA:]

                        v_filtrado = aplicar_filtros(v_arr)
                        v_pico = np.max(np.abs(v_filtrado))
                        v_rms = np.sqrt(np.mean(v_filtrado**2))

                        if v_pico > UMBRAL_RUIDO_V:
                            try:
                                popt, _ = curve_fit(
                                    modelo_gemelo_digital,
                                    t_arr,
                                    v_filtrado,
                                    p0=[0.43],
                                    bounds=(0, 100),
                                )
                                c_estimado = popt[0]
                            except Exception:
                                c_estimado = 0.0
                        else:
                            c_estimado = 0.43

                        yf = fft(v_filtrado)
                        xf = fftfreq(TAMANO_VENTANA, 1.0 / FS)[: TAMANO_VENTANA // 2]
                        amplitud_fft = (
                            2.0 / TAMANO_VENTANA * np.abs(yf[0 : TAMANO_VENTANA // 2])
                        )

                        indice_max_fft = np.argmax(amplitud_fft)
                        frecuencia_dominante = xf[indice_max_fft]
                        amplitud_maxima_fft = amplitud_fft[indice_max_fft]

                        Z_m_estimado = np.sqrt(
                            (params.k - params.m * params.omega**2) ** 2
                            + (c_estimado * params.omega) ** 2
                        )
                        velocidad_pico_estimada = (
                            params.F_0 * params.omega
                        ) / Z_m_estimado
                        posicion_pico_estimada = velocidad_pico_estimada / params.omega

                        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                        with open(
                            nombre_archivo_log, mode="a", newline=""
                        ) as archivo_csv:
                            escritor = csv.writer(archivo_csv)
                            escritor.writerow(
                                [
                                    timestamp,
                                    round(v_pico, 5),
                                    round(v_rms, 5),
                                    round(posicion_pico_estimada, 7),
                                    round(c_estimado, 5),
                                    round(frecuencia_dominante, 2),
                                    round(amplitud_maxima_fft, 5),
                                ]
                            )

                        print(
                            f"{timestamp:<12} | {v_pico:<10.4f} | {posicion_pico_estimada:<15.6e} | {c_estimado:<10.4f} | {frecuencia_dominante:<10.1f}"
                        )

            except KeyboardInterrupt:
                print("\n--- Adquisición de datos detenida por el usuario ---")
            finally:
                esp32.close()
                print("Puerto serial cerrado de manera segura.")

        registrar_daq_headless(params, G_sub_A)


# ===================================================================
# 3. EJECUCIÓN PRINCIPAL DEL PROGRAMA
# ===================================================================

print("\n--- INICIALIZANDO PARÁMETROS DEL SISMÓMETRO ---")
parametros = SectionParams()

print("\n--- INICIANDO INTEGRAL CAMPO MAGNETICO ---")
Factores_Acople(parametros)

print("\n--- INICIANDO SOLVER Y GRÁFICAS ---")
Solver(modelo_mk1, modelo_mk2, simular, datos_descarga, parametros)
