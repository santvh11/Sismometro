from dataclasses import dataclass, field
import numpy as np

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
    factor_amplificacion: float = 8  # Ganancia del amplificador
    subida_voltaje: float = 1.03  # Offset del ADC (vo3ltios)
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
    omega_filtro: float = 1  # filtro para la frecuencia automático

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
    delta_t: int = 1  # Factor de submuestreo (cada delta_t puntos)
    factor_ESP_32: float = 48  # Factor de lectura del ESP 32
    ESCALA_TIEMPO: float = 5.0  # Escala para el retardo en las gráficas
    FACTOR_TIEMPO_ANIMACION: int = 100  # Retardo en ms para visualizar más lento (intervalo de animación)
    RANGO_VOLTAJE: float = 2.0  # Factor de zoom out para la gráfica de voltaje

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
