import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from params import SectionParams

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



def run_modelo_mk1(params: SectionParams, G_sub_L: float, G_sub_A: float):
    print('--- Entrando a Modelo MK1 (Lineal) ---')
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
    # plt.show() moved to the end

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



def run_modelo_mk2(params: SectionParams, G_sub_L: float, G_sub_A: float):
    print('--- Entrando a Modelo MK2 (No lineal, RK45) ---')
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
    # plt.show() moved to the end

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


