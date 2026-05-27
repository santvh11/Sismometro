import numpy as np
from typing import TypedDict


# 1. Definición de subestructuras (Agrupación lógica)
class CondicionesIniciales(TypedDict):
    z: float
    z_dot: float
    Q: float
    Q_dot: float


class ParametrosMecanicos(TypedDict):
    m: float
    k: float
    c: float
    m_sis: float
    m_mes: float
    m_tornillo: float
    g: float


class ParametrosElectricos(TypedDict):
    R: float
    L: float
    C: float
    factor_amplificacion: float
    subida_voltaje: float
    R_sub_a: float
    R_extra: float
    epsilon_sub_cero: float
    epsilon_sub_e: float
    p_sub_cu: float


class PropiedadesMagneticas(TypedDict):
    m_mag: float
    R_iman: float
    Br_A: float
    Br_N32: float
    Br_N35: float
    densidad_neodimio: float
    L_libre_iman: float
    mu_sub_cero: float
    mu_sub_PLA: float
    mu_sub_f: float


class GeometriaSolenoide(TypedDict):
    R_sub_e: float
    L_cilindro: float
    e_sub_p: float
    h_sub_p: float
    r_sub_p: float
    g_sub_ecs: float
    e_sub_cs: float
    N_sub_c_total: float
    e_sub_c: float
    L_sub_c: float


class PropiedadesFluidos(TypedDict):
    h_sub_f: float
    rho_0: float
    beta_rho: float
    eta_0: float
    b_eta: float
    eta: float


class ConfigSimulacion(TypedDict):
    temp: float
    omega_Hz: float
    omega: float
    a: int
    b: float
    puntos: int
    delta_t: int


# 'total=False' permite inicializar el diccionario principal sin estos valores,
# ya que deben calcularse en tiempo de ejecución.
class ParametrosDerivados(TypedDict, total=False):
    m_fluido: float
    N_sub_c_capas: float
    c_sub_Stokes_resorte: float
    c_sub_Stokes_iman: float
    c_sub_Stokes: float
    lambda_c: float
    c_sub_lambda: float
    F_0: float
    omega_sub_n: float
    Zeta: float
    p_error_Zeta: float
    omega_sub_d: float
    e_total: float
    L_sub_s: float
    A_sub_s: float
    A_sub_cs: float
    A_effec: float
    A_sub_c: float
    delta_h_sub_cs: float
    N_sub_c: int
    rho: float
    R_sub_s: float
    R_sub_c: float
    R_porcentaje: float
    C_inicial: float
    C_N_sub_cs: float
    C_N_sub_cs_capas: float
    XC_capacitiva: float
    XC_inductiva: float
    XC: float
    alpha: float
    omega_sub_0_phi_m: float
    t: np.ndarray


# 2. Estructura principal
class SismometroParamsDict(TypedDict):
    iniciales: CondicionesIniciales
    mecanicos: ParametrosMecanicos
    electricos: ParametrosElectricos
    magnetismo: PropiedadesMagneticas
    geometria: GeometriaSolenoide
    fluidos: PropiedadesFluidos
    simulacion: ConfigSimulacion
    derivados: ParametrosDerivados


# 3. Implementación con los valores por defecto
params: SismometroParamsDict = {
    "iniciales": {"z": 0.0, "z_dot": 0.0, "Q": 0.0, "Q_dot": 0.0},
    "mecanicos": {
        "m": 7e-3,
        "k": 272.0,
        "c": 2.0,
        "m_sis": 48.6e-3,
        "m_mes": 4e-3,
        "m_tornillo": 11e-3,
        "g": 9.77,
    },
    "electricos": {
        "R": 383.1,
        "L": 36.7e-6,
        "C": 0.07e-6,
        "factor_amplificacion": 8.219,
        "subida_voltaje": 0.0,
        "R_sub_a": 1e8,
        "R_extra": 1.0,
        "epsilon_sub_cero": 8.854e-12,
        "epsilon_sub_e": 2.5,
        "p_sub_cu": 1.72e-8,
    },
    "magnetismo": {
        "m_mag": 13.33e-03,
        "R_iman": 1.5e-3,
        "Br_A": 0.9,
        "Br_N32": 1.14,
        "Br_N35": 1.22,
        "densidad_neodimio": 7500.0,
        "L_libre_iman": 135e-3,
        "mu_sub_cero": 4 * np.pi * 1e-7,
        "mu_sub_PLA": 1.0,
        "mu_sub_f": 1.0,
    },
    "geometria": {
        "R_sub_e": 9.5e-3,
        "L_cilindro": 10e-3,
        "e_sub_p": 3e-3,
        "h_sub_p": 34e-3,
        "r_sub_p": 14e-3,
        "g_sub_ecs": 0.015e-3,
        "e_sub_cs": 0.079e-3,
        "N_sub_c_total": 2700.0,
        "e_sub_c": 1e-3,
        "L_sub_c": 1e-2,
    },
    "fluidos": {
        "h_sub_f": 34e-3,
        "rho_0": 1273.3,
        "beta_rho": 0.6121,
        "eta_0": 3.30e-10,
        "b_eta": 6640.0,
        "eta": 1.5,  # Valor por defecto
    },
    "simulacion": {
        "temp": 299.15,
        "omega_Hz": 25.0,
        "omega": 25.0 * 2 * np.pi,
        "a": 0,
        "b": 0.5,
        "puntos": 10000,
        "delta_t": 1,
    },
    "derivados": {},  # Se inicia vacío y se llena después con funciones externas
}
