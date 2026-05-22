import serial
from dataclasses import dataclass, field
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
import collections

# ===================================================================
# 0. CONSTANTES GLOBALES
# ===================================================================
PUERTO_SERIAL = 'COM3'
BAUDIOS = 115200

# ===================================================================
# 1. PARÁMETROS DEL SISTEMA
# ===================================================================
@dataclass
class SectionParams:
    # -------------------------------------------------------------------
    # Variables de Configuración de Usuario (Inyectadas)
    # -------------------------------------------------------------------
    usar_params_teoricos: bool = False
    tipo_fuerza: int = 1
    tipo_amplitud: int = 1
    tipo_iman: int = 3

    # -------------------------------------------------------------------
    # 1er Orden: Configuraciones Reales
    # -------------------------------------------------------------------
    R: float = 383.1                                      
    L: float = 36.7 * 1e-3                                
    C: float = 0.07 * 1e-6                                
    c_base: float = 0.43                                  
    m_base: float = 1.5 * 1e-3                            
    k: float = 300                                        
    m_mag_base: float = 13.93e-03                         
    m_sis: float = 48.6 * 1e-3                            
    m_mes: float = 3.148 * 1e-3                           
    factor_amplificacion: float = 8.219                   
    temp: float = 299.15                           

    # Condiciones Iniciales
    z: float = 0                                   
    z_dot: float = 0                                
    Q: float = 0                                
    Q_dot: float = 0                            

    # Sección Mecánica
    g: float = 9.77                           
    eta_base: float = 1.5                           
    R_sub_e: float = 9.5 * 1e-3                
    L_cilindro: float = 10 * 1e-3              
    L_libre_iman: float = 135 * 1e-3           
    omega_Hz: float = 75                      
    e_sub_p: float = 3 * 1e-3                     
    h_sub_p: float = 34 * 1e-3                     
    r_sub_p: float = 14 * 1e-3                      
    h_sub_f: float = 34 * 1e-3                      
    g_sub_ecs: float = 0.015 * 1e-3                   
    e_sub_cs: float = 0.079 * 1e-3                     
    N_sub_c_total: float = 2700                            

    # Fluidos
    rho_0: float = 1273.3                           
    beta_rho: float = 0.6121                          
    densidad_neodimio: float = 7500                     
    eta_0: float = 3.30e-10                           
    b_eta: float = 6640                               

    # Electromagnética / Materiales
    R_iman: float = 1.5 * 1e-3             
    Br_A: float = 0.9                  
    Br_N32: float = 1.14                 
    Br_N35: float = 1.22                 

    # Resistencia y Capacitancia
    e_sub_c: float = 1 * 1e-3                           
    L_sub_c: float = 1 * 1e-2                          
    R_sub_a: float = 1 * 1e+8                          
    R_extra: float = 1                               
    epsilon_sub_cero: float = 8.854 * 1e-12     
    epsilon_sub_e: float = 2.5                 
    p_sub_cu: float = 1.72 * 1e-8               
    mu_sub_cero: float = 4 * np.pi * 1e-7      
    mu_sub_PLA: float = 1                         
    mu_sub_f: float = 1                            

    # Control
    a: float = 0                                      
    b: float = 0.5                                      
    puntos: int = 10000                              
    delta_t: int = 1                           

    # -------------------------------------------------------------------
    # 2do Orden: Variables Computadas (Campos Inicializados Post-Init)
    # -------------------------------------------------------------------
    omega: float = field(init=False)
    m: float = field(init=False)
    c: float = field(init=False)
    m_mag: float = field(init=False)
    m_fluido: float = field(init=False)
    N_sub_c: int = field(init=False)
    N_sub_c_capas: float = field(init=False)
    c_sub_lambda: float = field(default=0.0, init=False)
    F_0: float = field(init=False)
    omega_sub_n: float = field(init=False)
    Zeta: float = field(init=False)
    omega_sub_d: float = field(init=False)
    e_total: float = field(init=False)
    L_sub_s: float = field(init=False)
    A_sub_s: float = field(init=False) 
    A_sub_cs: float = field(init=False)
    A_effec: float = field(init=False)
    A_sub_c: float = field(init=False) 
    delta_h_sub_cs: float = field(init=False)                                                                
    rho: float = field(init=False)
    eta: float = field(init=False)
    R_sub_s: float = field(init=False) 
    R_sub_c: float = field(init=False) 
    R_porcentaje: float = field(init=False)
    C_inicial: float = field(init=False)
    C_N_sub_cs: float = field(init=False)
    XC_capacitiva: float = field(init=False)
    XC_inductiva: float = field(init=False)
    XC: float = field(init=False)
    alpha: float = field(init=False) 
    omega_sub_0_phi_m: float = field(init=False)
    t: np.ndarray = field(init=False) 

    def __post_init__(self):
        # Asignaciones base
        self.omega = self.omega_Hz * 2 * np.pi
        self.m = self.m_base
        self.c = self.c_base
        self.m_mag = self.m_mag_base

        # Computaciones geométricas y físicas
        self.N_sub_c = round(self.h_sub_p / self.e_sub_cs)
        self.N_sub_c_capas = self.N_sub_c_total / self.N_sub_c                                             
        self.e_total = (self.r_sub_p + self.e_sub_p + (0.5 * self.e_sub_cs))
        self.L_sub_s = self.N_sub_c_total * (self.e_total * 2 * np.pi)
        self.A_sub_s = (self.N_sub_c * (self.N_sub_c_capas * (4 * (np.pi**2)) * (self.e_total * self.e_sub_cs)))
        self.A_effec = (8 * self.e_sub_cs * np.pi * self.e_total * self.N_sub_c_total)
        self.A_sub_cs = np.pi * ((self.e_sub_cs * 0.5)**2)
        self.A_sub_c = np.pi * ((self.e_sub_c * 0.5)**2)
        self.delta_h_sub_cs = self.e_sub_cs
        self.eta = self.eta_0 * np.exp(self.b_eta / self.temp)
        self.rho = self.rho_0 - self.beta_rho * (self.temp - 273.15)

        # Teoría si aplica
        if self.usar_params_teoricos:
            self._calcular_parametros_teoricos()

        # EDO Mecánica
        self.m_fluido = np.abs(self.rho * (np.pi * (self.R_sub_e**2) * (self.h_sub_f - 0.75 * self.R_sub_e)))
        self.m += self.m_fluido
        self.omega_sub_n = (self.k / self.m)**0.5
        self.Zeta = self.c / (2 * ((self.k * self.m)**0.5))
        self.omega_sub_d = self.omega_sub_n * (1 - self.Zeta**2)**0.5

        # Circuito RLC
        self.XC_inductiva = self.L * self.omega
        self.XC_capacitiva = -1 / (self.omega * self.C)
        self.R_porcentaje = self.R_sub_a / (self.R + self.R_sub_a)
        self.alpha = self.R / (2 * self.L)
        self.XC = self.XC_capacitiva + self.XC_inductiva
        self.omega_sub_0_phi_m = 1 / ((self.L * self.C)**0.5)

        # Control y Tiempo
        self.t = np.linspace(self.a, self.b, self.puntos)

        # Fuerza Inicial
        if self.tipo_fuerza == 1:
            self.F_0 = 5  
        elif self.tipo_fuerza == 2:
            m_vibrante = self.m_sis + self.m_mes
            if self.tipo_amplitud == 1:   
                Y_0 = 7 * 1e-3
                self.F_0 = m_vibrante * Y_0 * (self.omega**2)
            elif self.tipo_amplitud == 2:
                f_Y = self.omega_Hz
                self.F_0 = m_vibrante * f_Y * (self.omega**2)

    def _calcular_parametros_teoricos(self):
        # Masa
        self.m = (self.R_sub_e**3) * np.pi * 1.3333 * self.densidad_neodimio
        
        # Amortiguamiento
        c_sub_Stokes_iman = 6 * np.pi * self.eta * self.R_sub_e
        c_sub_Stokes_resorte = np.abs(4 * np.pi * self.eta * self.L_cilindro / (np.log(self.L_cilindro / 2 * self.R_sub_e) + 0.5))
        lambda_c = self.R_sub_e / self.r_sub_p

        if lambda_c > 0.6:
            den = 1 - (0.75857 * lambda_c**5) 
            num = 1 - (2.10444 * lambda_c) + (2.08877 * lambda_c**3) - (0.94813 * lambda_c**5)
            self.c_sub_lambda = num / den
        else:
            self.c_sub_lambda = 1 - (2.104 * lambda_c) + (2.089 * lambda_c**3) - 0.948 * lambda_c**5
        
        self.c = (c_sub_Stokes_iman + c_sub_Stokes_resorte) / self.c_sub_lambda    

        # Capacitancia
        self.C_inicial = (self.L_sub_s * self.epsilon_sub_cero * self.epsilon_sub_e) / np.log((self.g_sub_ecs + self.e_sub_cs) / self.e_sub_cs)
        C_N_sub_cs = 2 * self.C_inicial / self.N_sub_c                                                                                       
        self.C = C_N_sub_cs * self.N_sub_c_capas  
 
        # Inductancia
        seccion_transversal_sol = np.pi * (self.r_sub_p + self.e_sub_p)**2                         
        self.L = seccion_transversal_sol * self.mu_sub_cero * (self.N_sub_c_total**2) / self.h_sub_p 

        # Resistencia
        self.R_sub_s = self.p_sub_cu * self.L_sub_s / self.A_sub_cs
        self.R_sub_c = self.p_sub_cu * self.L_sub_c / self.A_sub_c                                 
        self.R = (self.R_sub_a * (self.R_sub_s + self.R_sub_c + self.R_extra)) / (self.R_sub_a + self.R_sub_s + self.R_sub_c + self.R_extra)    
                                   
        # Momento Magnético
        Vol = ((self.R_iman)**3) * 0.75 * np.pi
        if self.tipo_iman == 1:
            self.m_mag = self.Br_A * Vol / self.mu_sub_cero
        elif self.tipo_iman == 2:
            self.m_mag = self.Br_N32 * Vol / self.mu_sub_cero
        elif self.tipo_iman == 3:
            self.m_mag = self.Br_N35 * Vol / self.mu_sub_cero           


# ===================================================================
# 2. FUNCIONES MATEMÁTICAS Y SOLVERS
# ===================================================================

def factores_acople(params: SectionParams): 
    bloque_raiz = ((params.h_sub_p)**2 + (params.e_total)**2)**0.5
    constantes_magneticas = (0.25 * params.m_mag * params.mu_sub_cero) / np.pi
    B_x = np.abs(((params.e_total - bloque_raiz) / (bloque_raiz * (params.e_total**2))) * (3/2) * constantes_magneticas)
    
    Radio_total = params.e_sub_p + params.r_sub_p
    Paso = params.e_sub_cs * 0.5 / np.pi
    Integral_linea = (1 / Radio_total**3) - (1 / (Radio_total**2 + (Paso * 2 * np.pi + params.N_sub_c_total)**2)**3/2)

    G_sub_A = (constantes_magneticas * 3 * Radio_total**2 / Paso) * Integral_linea
    G_sub_L = B_x * params.L_sub_s

    return G_sub_L, G_sub_A  

def imprimir_parametros_base(params: SectionParams, G_sub_L: float, G_sub_A: float):
    print(f"Masa estimada: {params.m:.3e} Kg")   
    print(f"Amortiguamiento auxiliar: {params.c_sub_lambda:.3e}")          
    print(f"Constante de amortiguamiento: {params.c:.3e} Kg/s")       
    print(f"Constante elástica: {params.k:.3e} Kg/s^2")                
    print(f"Resistencia: {params.R:.3e} Ohmnios")                      
    print(f"Inductancia: {params.L:.3e} H")                            
    print(f"Capacitancia: {params.C:.3e} F")                           
    print(f"Momento magnético configurado: {params.m_mag:.3e}")              
    print(f"Factor Magnético de Área (G_A): {G_sub_A:.3e}")          
    print(f"Factor Magnético de Longitud (G_L): {G_sub_L:.3e}")          
    print(f"Factor de amortiguamiento: {params.Zeta:.3e}")             
    print(f"Frecuencia natural: {params.omega_sub_n * 2 * np.pi:.3e} Hz")        
    print(f"Inercia Eléctrica: {params.alpha:.3e}")                   
    print(f"Frecuencia eléctrica: {params.omega_sub_0_phi_m * 2 * np.pi:.3e} Hz")


def graficar_resultados(t, y_matrix, y_estacionario, titulo, G_sub_A, G_sub_L, params, solver_rk=False):
    nombre_variables = ["Posición (z)", "Velocidad (ż)", "Carga (Q)", "Corriente (I)"]
    unidades = ["[m]", "[m/s]", "[C]", "[A]"]
    paleta_colores = ["orange", "red", "blue", "green"]

    fig, axs = plt.subplots(2, 2, figsize=(8, 6), dpi=150)
    fig.suptitle(titulo, fontsize=14, fontweight='bold')

    for i in range(4):
        ax = axs[i // 2, i % 2]
        if solver_rk:
            ax.plot(t, y_matrix[i], '--', color=paleta_colores[i], alpha=0.7, label="Solución RK45")
        else:
            ax.plot(t, y_matrix[i], color=paleta_colores[i], linewidth=2, label="Solución Total (Real)")
            ax.plot(t, y_estacionario[i], '--', color='gray', alpha=0.4, label="Solo Estacionario")

        ax.set_title(nombre_variables[i], fontweight='bold')
        ax.set_ylabel(f"Amplitud {unidades[i]}")
        ax.set_xlabel("Tiempo [s]")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', fontsize='small')
        
        ax.text(0.02, 0.02, f"Máx: {np.max(np.abs(y_matrix[i])):.2e}", transform=ax.transAxes, 
                ha='left', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Gráficas Adicionales
    t_filtro = t[::params.delta_t]
    
    if solver_rk:
        FEM_Ohm = y_matrix[3] * params.R
        FEM_FL = G_sub_A * params.R_porcentaje * y_matrix[1]
        F_Laplace = y_matrix[3] * G_sub_L
    else:
        I_Re = y_matrix[3]
        I_Im = y_estacionario[3] # Aproximación fasorial para gráfica
        V_Im = (params.R * I_Im) + (I_Re * params.XC)
        V_Re = (params.R * I_Re) - (I_Im * params.XC)
        V_A = ((V_Im**2) + (V_Re**2))**0.5
        V_phi = np.arctan2(V_Im, V_Re)
        FEM_Ohm = V_A * np.cos((params.omega * t_filtro) + V_phi)
        FEM_FL = G_sub_A * params.R_porcentaje * y_matrix[1][::params.delta_t]
        F_Laplace = y_matrix[3] * G_sub_L

    # FEM Ley de Ohm
    plt.figure(figsize=(7, 4), dpi=150)
    plt.step(t_filtro, FEM_Ohm[:len(t_filtro)], where='post', color="purple", linewidth=1)
    plt.title("FEM teórica (Ley de Ohm Fasorial)")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Voltaje [V]")
    plt.grid(True)
    plt.show()

    # FEM Faraday Lenz
    plt.figure(figsize=(7, 4), dpi=150)
    plt.step(t_filtro, FEM_FL, where='post', color="purple", linewidth=1)
    plt.title("FEM teórica (Faraday-Lenz)")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Voltaje [V]")
    plt.grid(True)
    plt.show()

    # Fuerza Laplace
    plt.figure(figsize=(7, 4), dpi=150)
    plt.step(t, F_Laplace, where='post', color="blue", linewidth=1)
    plt.title("Fuerza de Laplace")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Fuerza [N]")
    plt.grid(True)
    plt.show()

def simular_mk1(params: SectionParams, G_sub_L: float, G_sub_A: float):
    print("--- Ejecutando Modelo MK1 (Sistema Lineal) ---")
    A = np.array([
        [0, 1, 0, 0],
        [-params.k/params.m, -params.c/params.m, 0, -G_sub_L/params.m],
        [0, 0, 0, 1],
        [0, -G_sub_A/params.L, -1/(params.L*params.C), -params.R/params.L]
    ])

    eigenvalues, eigenvectors = np.linalg.eig(A)
    F_t = np.array([[0], [1/params.m], [0], [0]])
    I_mat = np.eye(4)
    
    lado_izquierdo = 1j * params.omega * I_mat - A
    lado_derecho = F_t * params.F_0 
    Xp_complejo = np.linalg.solve(lado_izquierdo, lado_derecho)

    X_particular = Xp_complejo * np.exp(1j * params.omega * params.t)
    X_t_p = Xp_complejo.real
    X_t_h = np.array([[params.z, params.z_dot, params.Q, params.Q_dot]]).reshape(4, 1)
    X_t = X_t_h + X_t_p 

    coeficientes = np.linalg.solve(eigenvectors, X_t)
    X_evolucion = np.zeros((4, len(params.t)), dtype=complex)

    for i in range(len(eigenvalues)):
        termino_exponencial = np.exp(eigenvalues[i] * params.t)
        contribucion = np.outer(eigenvectors[:, i], termino_exponencial)
        X_evolucion += coeficientes[i] * contribucion

    X_total = X_evolucion + X_particular

    frecuencias_unicas = np.unique(np.abs(eigenvalues.imag)) * 2 * np.pi
    print(f"Frecuencias naturales acopladas: {frecuencias_unicas} Hz")

    v_max = np.abs(np.max(X_total.real[1]))
    Re = (v_max * (params.r_sub_p - params.R_sub_e) * params.rho) / params.eta                                   
    
    if Re < 0.1:
        print(f"Amortiguamiento lineal fiable. Reynolds: {Re:.3e}")
    elif Re < 0.9:
        print(f"Amortiguamiento lineal variable. Reynolds: {Re:.3e}")
    else: 
        print(f"Amortiguamiento no lineal (requeriría RK45). Reynolds: {Re:.3e}")

    graficar_resultados(params.t, X_total.real, X_particular.real, "Detector de Gaia MK1 (Lineal)", G_sub_A, G_sub_L, params, False)

def simular_mk2(params: SectionParams, G_sub_L: float, G_sub_A: float):
    print("--- Ejecutando Modelo MK2 (No Lineal - RK45) ---")

    def state_matrix(t, S):
        z, v, Q, I = S
        F_ext = params.F_0 * np.sin(params.omega * t)
        dz_dt = v
        dv_dt = (1/params.m) * ((-params.k * z) - (params.c * v * np.abs(v)) + F_ext + (I * G_sub_L))
        dQ_dt = I
        dI_dt = (1/params.L) * (-params.R * I - (1/params.C) * Q + v * G_sub_A)
        return [dz_dt, dv_dt, dQ_dt, dI_dt]

    solver = solve_ivp(
        state_matrix,
        (params.a, params.b),
        [params.z, params.z_dot, params.Q, params.Q_dot],
        t_eval=params.t,
        method='LSODA'
    )
    
    if solver.success:
        print("Integración RK45 exitosa.")
        graficar_resultados(solver.t, solver.y, None, "Detector de Gaia MK2 (No Lineal)", G_sub_A, G_sub_L, params, True)
    else:
        print("Error en el solver:", solver.message)

def iniciar_daq_tiempo_real(params: SectionParams, G_sub_A: float):
    print("--- Iniciando DAQ en Tiempo Real ---")
    TAMANO_VENTANA = 512  
    FS = 1000.0           
    FC_PASA_ALTAS = 0.5   
    FC_PASA_BAJAS = 40.0  

    buffer_t = collections.deque(maxlen=TAMANO_VENTANA)
    buffer_v = collections.deque(maxlen=TAMANO_VENTANA)

    def aplicar_filtros(tiempo, voltaje):
        nyquist = 0.5 * FS
        b_altas, a_altas = butter(2, FC_PASA_ALTAS / nyquist, btype='high')
        b_bajas, a_bajas = butter(2, FC_PASA_BAJAS / nyquist, btype='low')
        v_filtrado = filtfilt(b_altas, a_altas, voltaje)
        return filtfilt(b_bajas, a_bajas, v_filtrado)

    def modelo_gemelo_digital(t_datos, c_ajuste):
        Z_m = np.sqrt((params.k - params.m * params.omega**2)**2 + (c_ajuste * params.omega)**2)
        V_amp = (params.F_0 * params.omega) / Z_m
        phi = np.arctan2((params.k - params.m * params.omega**2), (c_ajuste * params.omega))
        velocidad_teorica = V_amp * np.cos(params.omega * t_datos - phi)
        return G_sub_A * params.R_porcentaje * velocidad_teorica * params.factor_amplificacion

    try:
        esp32 = serial.Serial(PUERTO_SERIAL, BAUDIOS)
        print(f"Conectado a {PUERTO_SERIAL}. Esperando sincronización...")
    except Exception as e:
        print(f"Error conectando a {PUERTO_SERIAL}: {e}")
        return

    fig, (ax_tiempo, ax_fft) = plt.subplots(2, 1, figsize=(10, 7))
    fig.canvas.manager.set_window_title('DAQ Sismómetro EAFIT')

    linea_t, = ax_tiempo.plot([], [], lw=2, color='blue', label='Voltaje Filtrado')
    ax_tiempo.set_title("Dominio del Tiempo (Onda y Filtros)")
    ax_tiempo.set_ylabel("Voltaje [V]")
    ax_tiempo.set_xlabel("Tiempo [s]")
    ax_tiempo.grid(True)
    texto_metricas = ax_tiempo.text(0.02, 0.85, '', transform=ax_tiempo.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    linea_f, = ax_fft.plot([], [], lw=2, color='red')
    ax_fft.set_title("Dominio de la Frecuencia (FFT)")
    ax_fft.set_ylabel("Amplitud")
    ax_fft.set_xlabel("Frecuencia [Hz]")
    ax_fft.set_xlim(0, 100) 
    ax_fft.grid(True)

    plt.tight_layout()
    tiempo_inicial_micros = None

    def actualizar(frame):
        nonlocal tiempo_inicial_micros
        while esp32.in_waiting > 0:
            try:
                linea = esp32.readline().decode('utf-8').strip()
                if "," in linea:
                    t_micros_str, v_str = linea.split(",")
                    t_micros = int(t_micros_str)
                    
                    if tiempo_inicial_micros is None:
                        tiempo_inicial_micros = t_micros
                    
                    buffer_t.append((t_micros - tiempo_inicial_micros) / 1e6)
                    buffer_v.append(float(v_str))
            except Exception:
                pass 

        if len(buffer_t) == TAMANO_VENTANA:
            t_arr = np.array(buffer_t)
            v_arr = np.array(buffer_v)
            v_filtrado = aplicar_filtros(t_arr, v_arr)
            
            try:
                popt, _ = curve_fit(modelo_gemelo_digital, t_arr, v_filtrado, p0=[0.43], bounds=(0, 10))
                c_estimado = popt[0]
            except:
                c_estimado = 0.0 
                
            yf = fft(v_filtrado)
            xf = fftfreq(TAMANO_VENTANA, 1.0 / FS)[:TAMANO_VENTANA//2]
            amplitud_fft = 2.0 / TAMANO_VENTANA * np.abs(yf[0:TAMANO_VENTANA//2])
            
            linea_t.set_data(t_arr, v_filtrado)
            ax_tiempo.set_xlim(t_arr[0], t_arr[-1])
            margen = max(np.abs(v_filtrado)) * 1.2 + 0.01
            ax_tiempo.set_ylim(-margen, margen) 
            
            v_pico = np.max(np.abs(v_filtrado))
            v_rms = np.sqrt(np.mean(v_filtrado**2))
            texto_metricas.set_text(f"Pico: {v_pico:.3f} V\nRMS: {v_rms:.3f} V\nCoef 'c': {c_estimado:.3f}")
            
            linea_f.set_data(xf, amplitud_fft)
            ax_fft.set_ylim(0, max(amplitud_fft) * 1.2 + 0.01)

        return linea_t, linea_f, texto_metricas

    ani = animation.FuncAnimation(fig, actualizar, interval=30, blit=False)
    plt.show()
    esp32.close()

# ===================================================================
# 3. EJECUCIÓN PRINCIPAL DEL PROGRAMA
# ===================================================================

def obtener_configuracion_usuario():
    while True:
        modelo = input("Modelo a usar - (1) MK1, (2) MK2, (3) Voltaje DAQ en tiempo real: ").strip()
        if modelo in ["1", "2", "3"]:
            break
        print("Entrada no válida.")

    usar_teoricos = input("¿Usar parámetros teóricos? (Enter para Sí, otra tecla para No): ") == ""
    
    tipo_fuerza = 1
    tipo_amplitud = 1
    tipo_iman = 3

    while True:
        tipo_fuerza = input("Fuerza - (1) Directa, (2) A partir de MAS: ").strip()
        if tipo_fuerza in ["1", "2"]:
            tipo_fuerza = int(tipo_fuerza)
            break

    if tipo_fuerza == 2:
        while True:
            tipo_amplitud = input("Amplitud - (1) Directa, (2) Base a frecuencia: ").strip()
            if tipo_amplitud in ["1", "2"]:
                tipo_amplitud = int(tipo_amplitud)
                break

    if usar_teoricos:
        while True:
            tipo_iman = input("Tipo de Imán - (1) Acero 440, (2) Neodimio N32, (3) Neodimio N35: ").strip()
            if tipo_iman in ["1", "2", "3"]:
                tipo_iman = int(tipo_iman)
                break

    return int(modelo), usar_teoricos, tipo_fuerza, tipo_amplitud, tipo_iman

if __name__ == "__main__":
    modelo_seleccionado, usar_teo, fuerza, amplitud, iman = obtener_configuracion_usuario()

    print("\n--- INICIALIZANDO PARÁMETROS DEL SISMÓMETRO ---")
    parametros = SectionParams(
        usar_params_teoricos=usar_teo,
        tipo_fuerza=fuerza,
        tipo_amplitud=amplitud,
        tipo_iman=iman
    )

    print("\n--- INICIANDO INTEGRAL CAMPO MAGNETICO ---")
    G_sub_L, G_sub_A = factores_acople(parametros)
    imprimir_parametros_base(parametros, G_sub_L, G_sub_A)

    print("\n--- INICIANDO SOLVER Y GRÁFICAS ---")
    if modelo_seleccionado == 1:
        simular_mk1(parametros, G_sub_L, G_sub_A)
    elif modelo_seleccionado == 2:
        simular_mk2(parametros, G_sub_L, G_sub_A)
    elif modelo_seleccionado == 3:
        iniciar_daq_tiempo_real(parametros, G_sub_A)