# Librería de Dataclass para almacenamiento de variables
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# 0. Interfaz de Usuario y Captura de Datos
# -------------------------------------------------------------------
def obtener_parametros_usuario():
    """Maneja la lógica de entrada del usuario de forma segura."""
    opciones = {}

    while True:
        modelo = input("Escriba (1) para mk_1 o (2) para mk_2: ").strip()
        if modelo in ["1", "2"]:
            opciones["modelo_mk_1"] = modelo == "1"
            opciones["modelo_mk_2"] = modelo == "2"
            break
        print("Entrada no válida. Inténtalo de nuevo.")

    while True:
        try:
            p0 = int(
                input(
                    "¿Desea usar la masa con los datos directos? (1) para sí, (2) para no: "
                )
            )
            if p0 in [1, 2]:
                opciones["usar_masa_directa"] = p0 == 1
                break
            print("Seleccione una opción válida (1 o 2).")
        except ValueError:
            print("Entrada no válida. Por favor ingrese un número entero.")

    while True:
        try:
            iman = int(input("¿Es una esfera (1) o cilindro (2)? : "))
            if iman in [1, 2]:
                opciones["es_esfera"] = iman == 1
                break
            print("Seleccione una opción válida (1 o 2).")
        except ValueError:
            print("Entrada no válida. Por favor ingrese un número entero.")

    opciones["usar_m_mag_directo"] = False
    opciones["material_iman"] = 0

    if opciones["modelo_mk_1"]:
        while True:
            try:
                p1 = int(
                    input(
                        "¿Desea usar el momento magnético con datos directos? (1) para sí, (2) para no: "
                    )
                )
                if p1 in [1, 2]:
                    opciones["usar_m_mag_directo"] = p1 == 1
                    break
                print("Seleccione una opción válida (1 o 2).")
            except ValueError:
                print("Entrada no válida. Por favor ingrese un número entero.")

        if not opciones["usar_m_mag_directo"]:
            while True:
                try:
                    mat = int(input("¿El imán es de neodimio (1) o acero 440 (2)? : "))
                    if mat in [1, 2]:
                        opciones["material_iman"] = mat
                        break
                    print("Seleccione una opción válida (1 o 2).")
                except ValueError:
                    print("Entrada no válida. Por favor ingrese un número entero.")

    return opciones


# -------------------------------------------------------------------
# 1. Parámetros de sección (Almacenamiento y Cálculo de 2do Orden)
# -------------------------------------------------------------------
@dataclass
class SectionParams:
    # --- Entradas de Configuración de Usuario ---
    modelo_mk_1: bool
    modelo_mk_2: bool
    usar_masa_directa: bool
    es_esfera: bool
    usar_m_mag_directo: bool
    material_iman: int

    # --- 1er Orden (Diseño) ---
    k: float = 120  # Constante Elástica del Resorte (kg/s^2)
    m: float = 0.453 * (1e-3)  # Masa de la esfera
    g: float = 9.77  # Constante gravitacional
    eta: float = 1.1  # Viscosidad del fluido (kg/m*s)
    R_sub_e: float = 4 * (1e-3)  # Radio de la Esfera
    F_0: float = 7  # Fuerza en Newtons de la mesa
    omega: float = 62.8  # Frecuencia de vibración de la mesa
    A: float = 5 * (1e-3)  # elongación inicial de la mesa

    # Sección Mecánica
    e_sub_p: float = 3 * (1e-3)  # Grosor del Contenedor de PLA
    h_sub_p: float = 36 * (1e-3)  # Altura del contenedor de PLA
    r_sub_p: float = 5 * (1e-3)  # Radio del Contenedor de PLA
    h_sub_f: float = 34 * (1e-3)  # altura del fluido
    g_sub_ecs: float = 0.015 * (1e-3)  # Grosor de la capa de esmalte
    e_sub_cs: float = 0.118 * (1e-3)  # Grosor del Cable del solenoide (diámetro)
    N_sub_c_capas: int = 20  # Número de capas de vueltas de cable
    densidad_neodimio: float = 7500  # Densidad del Neodimio N35 en Kg/m^3
    L_cilindro: float = 10 * (1e-3)  # Longitud cilindro

    # Sección Electromagnética
    chi_sub_N: float = 8 * (1e8)  # Coeficiente Momento Neodimio N35
    chi_sub_A: float = 5 * (1e8)  # Coeficiente Momento Acero 440 C
    Q: float = 0  # Carga inicial
    Q_dot: float = 0  # Corriente inicial

    # Sección de Resistencia
    e_sub_c: float = 17 * (1e-3)  # Grosor Cable conexión (diámetro)
    L_sub_c: float = 5 * (1e-2)  # Longitud cable conexión
    R_sub_a: float = 1 * (1e8)  # Resistencia entrada amplificador
    R_extra: float = 1200  # Resistencia añadida

    # Sección de Capacitancia
    epsilon_sub_cero: float = 8.854 * (1e-12)
    epsilon_sub_e: float = 2.5
    p_sub_cu: float = 1.72 * (1e-8)

    # Sección de Inductancia
    mu_sub_cero: float = 4 * np.pi * (1e-7)
    mu_sub_PLA: float = 1
    mu_sub_f: float = 1

    # Sección de Control
    a: float = 0
    b: float = 5
    puntos: int = 1000
    delta_t: int = 1

    # --- 2do Orden (Calculados) ---
    c_sub_Stokes: float = field(init=False)
    lambda_c: float = field(init=False)
    c_sub_lambda: float = field(init=False)
    c: float = field(init=False)
    omega_sub_n: float = field(init=False)
    Zeta: float = field(init=False)
    p_error_Zeta: float = field(init=False)
    omega_sub_d: float = field(init=False)
    B: float = field(init=False)
    root_1: complex = field(init=False)
    root_2: complex = field(init=False)
    Zeta_sub_omega: float = field(init=False)
    omega_sub_s: float = field(init=False)
    phi: float = field(init=False)
    X: float = field(init=False)

    e_total: float = field(init=False)
    L_sub_s: float = field(init=False)
    A_sub_s: float = field(init=False)
    A_effec: float = field(init=False)
    A_sub_cs: float = field(init=False)
    A_sub_c: float = field(init=False)
    delta_h_sub_cs: float = field(init=False)
    N_sub_c: int = field(init=False)

    m_mag: float = field(init=False)

    R_sub_s: float = field(init=False)
    R_sub_c: float = field(init=False)
    R_porcentaje: float = field(init=False)
    R: float = field(init=False)

    C: float = field(init=False)
    XC_capacitiva: float = field(init=False)

    mu_prom: float = field(init=False)
    L: float = field(init=False)
    XC_inductiva: float = field(init=False)

    XC: float = field(init=False)
    alpha: float = field(init=False)
    omega_sub_0_phi_m: float = field(init=False)

    t: np.ndarray = field(init=False)

    def __post_init__(self):
        # 1. Masa
        if not self.usar_masa_directa:
            self.m = (self.R_sub_e**3) * np.pi * 1.3333 * self.densidad_neodimio
        print(f"Masa estimada en: {self.m:.6f} Kg")

        # 2. Amortiguamiento
        if self.es_esfera:
            self.c_sub_Stokes = 6 * np.pi * self.eta * self.R_sub_e
        else:
            self.c_sub_Stokes = np.abs(
                2
                * np.pi
                * self.eta
                * self.L_cilindro
                / (np.log(self.R_sub_e / self.L_cilindro) - 0.72)
            )

        self.lambda_c = self.R_sub_e / self.r_sub_p
        if self.lambda_c <= 0.6:
            print(
                "No se cumplen las condiciones de aplicación del factor Habermann Faxen"
            )
            self.c_sub_lambda = 1.0
        else:
            self.c_sub_lambda = (
                1
                - (2.104 * self.lambda_c)
                + (2.089 * (self.lambda_c**3))
                - 0.948 * (self.lambda_c**5)
            )

        print(f"Amortiguamiento Habermann Faxen: {self.c_sub_lambda:.4f}")
        self.c = self.c_sub_Stokes / self.c_sub_lambda
        print(f"Constante de amortiguamiento (c): {self.c:.6f}")

        # 3. Frecuencias y Factores
        self.omega_sub_n = (self.k / self.m) ** 0.5
        self.Zeta = self.c / (2 * np.sqrt(self.k * self.m))
        self.p_error_Zeta = self.Zeta * 0.1

        # Corrección para subamortiguado/sobreamortiguado
        term_d = 1 - self.Zeta**2 if self.Zeta < 1 else self.Zeta**2 - 1
        self.omega_sub_d = self.omega_sub_n * np.sqrt(term_d)

        self.B = (
            (self.Zeta * self.omega_sub_n * self.A / self.omega_sub_d)
            if self.omega_sub_d != 0
            else 0
        )

        root_term = np.sqrt(complex(self.c**2 - 4 * self.m * self.k))
        self.root_1 = (-self.c + root_term) / (2 * self.m)
        self.root_2 = (-self.c - root_term) / (2 * self.m)

        self.Zeta_sub_omega = (2 * self.Zeta * self.omega) / self.omega_sub_n
        self.omega_sub_s = 1 - (self.omega / self.omega_sub_n) ** 2
        self.phi = np.arctan2(self.Zeta_sub_omega, self.omega_sub_s)
        self.X = self.F_0 / (
            self.k * np.sqrt(self.Zeta_sub_omega**2 + self.omega_sub_s**2)
        )

        # 4. Mecánica del Solenoide
        self.N_sub_c = round(self.h_sub_p / self.e_sub_cs)
        self.e_total = self.r_sub_p + self.e_sub_p + (0.5 * self.e_sub_cs)
        self.L_sub_s = self.N_sub_c * self.N_sub_c_capas * (self.e_total * 2 * np.pi)
        self.A_sub_s = (
            self.N_sub_c
            * self.N_sub_c_capas
            * (4 * (np.pi**2))
            * (self.e_total * self.e_sub_cs)
        )
        self.A_effec = (
            2
            * self.e_sub_cs
            * np.pi
            * (self.r_sub_p + self.e_sub_p)
            * self.N_sub_c
            * self.N_sub_c_capas
        )
        self.A_sub_cs = np.pi * ((self.e_sub_cs * 0.5) ** 2)
        self.A_sub_c = np.pi * ((self.e_sub_c * 0.5) ** 2)
        self.delta_h_sub_cs = self.e_sub_cs

        # 5. Electromagnética
        self.m_mag = 0.0
        if self.modelo_mk_1:
            if self.usar_m_mag_directo:
                self.m_mag = 0.045
            elif self.material_iman == 2:
                self.m_mag = self.chi_sub_A * ((2 * self.R_sub_e) ** 3)
            elif self.material_iman == 1:
                self.m_mag = self.chi_sub_N * ((2 * self.R_sub_e) ** 3)
            print(f"Momento magnético configurado en: {self.m_mag:.4e}")

        # 6. Resistencia
        self.R_sub_s = self.p_sub_cu * self.L_sub_s / self.A_sub_cs
        self.R_sub_c = self.p_sub_cu * self.L_sub_c / self.A_sub_c
        self.R = (self.R_sub_a * (self.R_sub_s + self.R_sub_c + self.R_extra)) / (
            self.R_sub_a + self.R_sub_s + self.R_sub_c + self.R_extra
        )
        self.R_porcentaje = self.R_sub_a / (
            self.R_sub_a + self.R_sub_s + self.R_sub_c + self.R_extra
        )

        # 7. Capacitancia
        C_inicial = (
            self.L_sub_s * self.epsilon_sub_cero * self.epsilon_sub_e
        ) / np.log((self.g_sub_ecs + self.e_sub_cs) / self.e_sub_cs)
        self.C = (2 * C_inicial / self.N_sub_c) * self.N_sub_c_capas
        self.XC_capacitiva = (
            -1 / (self.omega * self.C) if self.omega != 0 else float("-inf")
        )

        # 8. Inductancia
        self.mu_prom = (self.mu_sub_cero / (self.r_sub_p + self.e_sub_p)) * (
            (self.mu_sub_f * self.r_sub_p) + (self.e_sub_p * self.mu_sub_PLA)
        )
        self.L = (
            self.mu_prom
            * self.A_sub_cs
            * self.N_sub_c
            * self.N_sub_c_capas
            / self.h_sub_p
        )
        self.XC_inductiva = self.L * self.omega

        # 9. Frecuencias RLC
        self.alpha = self.R / (2 * self.L)
        self.XC = self.XC_capacitiva + self.XC_inductiva
        self.omega_sub_0_phi_m = 1 / np.sqrt(self.L * self.C)

        # 10. Tiempo
        self.t = np.linspace(self.a, self.b, self.puntos)


# -------------------------------------------------------------------
# 2. Algoritmo de promedio de Fuerza de Laplace
# -------------------------------------------------------------------
def Factores_Acople(params: SectionParams):
    bloque_constante = params.r_sub_p + params.e_sub_p
    bloque_variable_lim_simetrico = params.h_sub_p * 0.5
    bloque_raiz = np.sqrt(bloque_variable_lim_simetrico**2 + bloque_constante**2)

    constantes_magneticas = (0.75 * params.m_mag * params.mu_prom) / np.pi
    B_x = (1 / (bloque_raiz * bloque_constante)) * constantes_magneticas

    dB_z = (
        (bloque_constante * params.h_sub_p) / (bloque_raiz**5)
    ) * constantes_magneticas
    dB_x = (0.5 * (bloque_constante**2) / (bloque_raiz**5)) * constantes_magneticas

    G_sub_A = params.A_effec * (dB_x + dB_z)
    G_sub_L = B_x * params.L_sub_s

    print(f"El factor G_A (promedio campo magnético) es: {G_sub_A:.4e}")
    print(f"El factor G_L (Laplace) es: {G_sub_L:.4e}")

    return G_sub_L, G_sub_A


# -------------------------------------------------------------------
# 3. Solver de EDOs
# -------------------------------------------------------------------
def Solver(params: SectionParams):
    if params.modelo_mk_1:
        print("\n--- Entrando a Modelo MK1 ---")

        G_sub_L, G_sub_A = Factores_Acople(params)

        # Matriz A (Sistema acoplado)
        A_mat = np.array(
            [
                [0, 1, 0, 0],
                [-params.k / params.m, -params.c / params.m, 0, -G_sub_L / params.m],
                [0, 0, 0, 1],
                [
                    0,
                    -G_sub_A / params.L,
                    -1 / (params.L * params.C),
                    -params.R / params.L,
                ],
            ]
        )

        eigenvalues, eigenvectors = np.linalg.eig(A_mat)

        print("Eigenvalores del sistema acoplado:")
        print(eigenvalues)

        F_t = np.array([[0], [1 / params.m], [0], [0]])
        I = np.eye(4)
        lado_izquierdo = 1j * params.omega * I - A_mat
        lado_derecho = F_t * params.F_0 * np.exp(-1j * params.phi)

        Xp_complejo = np.linalg.solve(lado_izquierdo, lado_derecho)
        X_particular = Xp_complejo * np.exp(1j * params.omega * params.t)

        X_t_p = Xp_complejo.real
        X_t_h = np.array([[params.A, params.B, params.Q, params.Q_dot]]).reshape(4, 1)
        X_t = X_t_h - X_t_p

        coeficientes = np.linalg.solve(eigenvectors, X_t)
        X_evolucion = np.zeros((4, len(params.t)), dtype=complex)

        for i in range(len(eigenvalues)):
            termino_exponencial = np.exp(eigenvalues[i] * params.t)
            contribucion = np.outer(eigenvectors[:, i], termino_exponencial)
            X_evolucion += coeficientes[i] * contribucion

        X_total = X_evolucion + X_particular

        delta_electromagnetico = params.alpha - params.omega_sub_0_phi_m
        print(f"Zeta (Mecánico): {params.Zeta:.4f}")
        print(f"Delta electromagnético (alpha - w_0): {delta_electromagnetico:.2e}")

        # Gráficas MK1
        nombre_variables = [
            "Posición (z)",
            "Velocidad (ż)",
            "Carga (Q)",
            "Corriente (I)",
        ]
        unidades = ["[m]", "[m/s]", "[C]", "[A]"]
        paleta_colores = ["orange", "red", "blue", "green"]

        fig, axs = plt.subplots(2, 2, figsize=(8, 6), dpi=100)
        fig.suptitle(
            "Respuesta Temporal - Sistema Acoplado (MK1)",
            fontsize=14,
            fontweight="bold",
        )

        for i in range(4):
            ax = axs[i // 2, i % 2]
            ax.plot(params.t, X_total.real[i], color=paleta_colores[i], label="Total")
            ax.plot(
                params.t,
                X_particular.real[i],
                "--",
                color="gray",
                alpha=0.5,
                label="Estacionario",
            )
            ax.set_title(nombre_variables[i])
            ax.set_ylabel(unidades[i])
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize="small")

        plt.tight_layout()
        plt.show()

        # Voltajes y Faraday Lenz
        I_Re = X_total.real[3]
        I_Im = X_total.imag[3]
        V_Im = (params.R * I_Im) + (I_Re * params.XC)
        V_Re = (params.R * I_Re) - (I_Im * params.XC)
        V_A = np.sqrt(V_Im**2 + V_Re**2)
        V_phi = np.arctan2(V_Im, V_Re)
        FEM_Ohm = V_A * np.cos((params.omega * params.t) + V_phi)

        FEM_FL = params.R_porcentaje * X_total.real[1]
        t_filtro = params.t[:: params.delta_t]

        plt.figure(figsize=(8, 4), dpi=100)
        plt.step(
            t_filtro,
            FEM_Ohm[:: params.delta_t],
            where="post",
            label="FEM Ley de Ohm Fasorial",
            color="purple",
        )
        plt.step(
            t_filtro,
            FEM_FL[:: params.delta_t],
            where="post",
            label="FEM Faraday Lenz",
            color="orange",
            alpha=0.7,
        )
        plt.title("Comparación de FEM")
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Voltaje [V]")
        plt.legend()
        plt.grid()
        plt.show()

        v_max = np.max(np.abs(X_total.real[1]))
        Re = (v_max * (params.r_sub_p - params.R_sub_e)) / params.eta
        print(f"Número de Reynolds estimado: {Re:.4f}")

    elif params.modelo_mk_2:
        print("\n--- Entrando a Modelo MK2 ---")

        solucion_particular = params.X * np.sin((params.omega * params.t) + params.phi)

        if np.isclose(params.Zeta, 1.0, atol=params.p_error_Zeta):
            print("Sistema críticamente amortiguado.")
            solucion_homogenea = np.exp(-params.omega_sub_n * params.t) * (
                params.A + params.B * params.t
            )
            caso = "Crítico"
        elif params.Zeta < 1.0:
            print("Sistema subamortiguado.")
            solucion_homogenea = np.exp(
                -params.Zeta * params.omega_sub_n * params.t
            ) * (
                params.A * np.cos(params.omega_sub_d * params.t)
                + params.B * np.sin(params.omega_sub_d * params.t)
            )
            caso = "Subamortiguado"
        else:
            print("Sistema sobreamortiguado.")
            solucion_homogenea = (
                np.exp(params.root_1.real * params.t) * params.A
                + np.exp(params.root_2.real * params.t) * params.B
            )
            caso = "Sobreamortiguado"

        solucion_total = solucion_homogenea + solucion_particular

        plt.figure(figsize=(10, 6), dpi=100)
        plt.plot(
            params.t,
            solucion_homogenea,
            label=f"Homogénea ({caso})",
            color="orange",
            linestyle="--",
        )
        plt.plot(
            params.t,
            solucion_particular,
            label="Estacionaria",
            color="green",
            linestyle="-.",
        )
        plt.plot(params.t, solucion_total, label="Total", color="purple", linewidth=2)

        plt.title("Respuesta Temporal (MK2)")
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Desplazamiento [m]")
        plt.legend()
        plt.grid(True, alpha=0.7)
        plt.show()


# -------------------------------------------------------------------
# 4. Ejecución Principal
# -------------------------------------------------------------------
def main():
    print("\n--- CONFIGURACIÓN INICIAL ---")
    opciones = obtener_parametros_usuario()

    print("\n--- PROCESANDO PARÁMETROS ---")
    parametros = SectionParams(**opciones)

    Solver(parametros)


if __name__ == "__main__":
    main()
