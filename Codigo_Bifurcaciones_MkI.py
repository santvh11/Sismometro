import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional

# ===================================================================
# 1. PARÁMETROS GLOBALES Y AMBIENTALES
# ===================================================================


@dataclass
class EntornoSimulacion:
    # Condiciones Ambientales Base (Valle / Montaña)
    altitud_m: float = 2150.0
    temp_ambiente_c: float = 17.0

    # Constantes Físicas
    P0: float = 101325.0  # Presión a nivel del mar [Pa]
    T0: float = 288.15  # Temp a nivel del mar [K]
    g: float = 9.793  # Gravedad local [m/s^2]
    R: float = 8.314  # Constante universal gases [J/mol·K]
    L_grad: float = 0.0065  # Gradiente térmico [K/m]
    M_aire: float = 0.028964  # Masa molar aire [kg/mol]

    # Parámetros Biológicos (Gaia / Microalgas)
    OUR_objetivo: float = 3.0  # Tasa de consumo de O2 [mmol/L·h] (Ejemplo)
    eficiencia_transferencia: float = 0.05  # 5% de eficiencia de burbujeo
    volumen_tanque_L: float = 100.0  # Volumen de trabajo por tanque

    # Variables de estado calculadas
    P_atm_local: float = field(init=False)
    densidad_O2_mmol_L: float = field(init=False)

    def __post_init__(self):
        self.P_atm_local = self.calcular_presion_atmosferica()
        self.densidad_O2_mmol_L = self.calcular_densidad_o2_vdw()

    def calcular_presion_atmosferica(self) -> float:
        """Ecuación barométrica estándar."""
        exponente = (self.g * self.M_aire) / (self.R * self.L_grad)
        base = 1.0 - ((self.L_grad * self.altitud_m) / self.T0)
        return self.P0 * (base**exponente)

    def calcular_densidad_o2_vdw(self) -> float:
        """Newton-Raphson para Van der Waals del O2."""
        T_k = self.temp_ambiente_c + 273.15
        P_o2 = 0.2095 * self.P_atm_local

        a = 0.1382  # Pa·m^6/mol^2
        b = 3.186e-5  # m^3/mol

        rho = P_o2 / (self.R * T_k)  # Estimación Gas Ideal

        for _ in range(20):  # Converge muy rápido, 20 iteraciones son suficientes
            f = (rho * self.R * T_k) / (1.0 - rho * b) - a * (rho**2) - P_o2
            df = (self.R * T_k) / ((1.0 - rho * b) ** 2) - 2 * a * rho
            rho_nueva = rho - f / df
            if abs(rho_nueva - rho) < 1e-7:
                break
            rho = rho_nueva

        return rho  # mol/m^3 es numéricamente igual a mmol/L


# ===================================================================
# 2. TOPOLOGÍA DE LA RED (POO)
# ===================================================================


class Tramo:
    def __init__(
        self,
        id_tramo: str,
        longitud: float,
        diametro: float,
        delta_z: float,
        es_hueco: bool = False,
        h_profundidad_tanque: float = 0.0,
        k_menor: float = 0.0,
        rugosidad: float = 1.5e-6,
    ):

        self.id_tramo = id_tramo
        # ¡IMPORTANTE!: Validar siempre que el input sea el diámetro completo, no el radio.
        self.D = diametro
        self.L = longitud
        self.dz = delta_z
        self.epsilon = rugosidad
        self.A = np.pi * (self.D**2) / 4.0

        # Accesorios y Fronteras
        self.K_menor = k_menor  # Para codos (ej. 0.9) o válvulas
        self.es_hueco = es_hueco  # True si es el orificio de salida final
        self.h_prof = h_profundidad_tanque  # Para la contrapresión rho*g*h

        self.hijos: List["Tramo"] = []

        # Variables de Estado Dinámico
        self.caudal: float = 0.0
        self.velocidad: float = 0.0
        self.P_entrada: float = 0.0
        self.P_salida: float = 0.0

    def agregar_bifurcacion(self, tramo_hijo: "Tramo"):
        self.hijos.append(tramo_hijo)


# ===================================================================
# 3. NÚCLEO DEL SIMULADOR
# ===================================================================


class SimuladorFlujo:
    def __init__(
        self,
        raiz: Tramo,
        entorno: EntornoSimulacion,
        rho_fluido: float = 998.0,
        mu_fluido: float = 1.0e-3,
    ):
        self.raiz = raiz
        self.env = entorno
        self.rho = rho_fluido
        self.mu = mu_fluido
        self.lista_tramos = []
        self._mapear_red(self.raiz)

    def _mapear_red(self, nodo: Tramo):
        """Recorre el árbol y aplana la red para análisis global."""
        self.lista_tramos.append(nodo)
        for hijo in nodo.hijos:
            self._mapear_red(hijo)

    def validar_red(self) -> bool:
        """Filtro de fallas lógicas (antisuicidio algorítmico)."""
        for tramo in self.lista_tramos:
            if tramo.L <= 0 and not tramo.es_hueco:
                raise ValueError(f"Tramo {tramo.id_tramo}: Longitud debe ser > 0.")
            if tramo.D <= 0:
                raise ValueError(f"Tramo {tramo.id_tramo}: Diámetro debe ser > 0.")
            if not tramo.hijos and not tramo.es_hueco:
                raise ValueError(
                    f"Tramo {tramo.id_tramo}: Nodo terminal sin hueco de salida."
                )
        print("✓ Red validada topológica y geométricamente.")
        return True

    def _calcular_friccion(self, v: float, D: float, L: float, epsilon: float) -> float:
        """Calcula la pérdida de presión estática (Moneda de Paga)."""
        if v == 0:
            return 0.0

        Re = self.rho * abs(v) * D / self.mu
        if Re < 10:
            return 0.0  # Evitar singularidades

        if Re < 2300:
            f = 64.0 / Re
        else:
            # Swamee-Jain
            f = 0.25 / (np.log10((epsilon / (3.7 * D)) + (5.74 / (Re**0.9)))) ** 2

        # Darcy-Weisbach
        return f * (L / D) * 0.5 * self.rho * (v**2)

    def resolver_flujo_directo(self, P_motor: float, Q_total_estimado: float):
        """
        Solucionador de Marcha Hacia Adelante (Forward-Marching).
        Asume un caudal inicial de prueba, lo divide por áreas en las ramas,
        calcula la EGL/HGL y verifica la presión en los tanques.
        (Para un sistema real, aquí se implementaría el bucle Newton-Raphson
         actualizando Q_total_estimado hasta que el residuo de presión sea 0).
        """
        # 1. Asignar caudales por conservación (simetría/áreas)
        self._distribuir_caudales(self.raiz, Q_total_estimado)

        # 2. Calcular pérdidas de energía (presión) nodo a nodo
        self.raiz.P_entrada = P_motor
        self._propagar_presiones(self.raiz)

    def _distribuir_caudales(self, nodo: Tramo, Q_entrada: float):
        nodo.caudal = Q_entrada
        nodo.velocidad = Q_entrada / nodo.A

        if not nodo.hijos:
            return

        # Dividir proporcional al área de los hijos (simplificación de red ramificada)
        area_total_hijos = sum(h.A for h in nodo.hijos)
        for hijo in nodo.hijos:
            Q_hijo = Q_entrada * (hijo.A / area_total_hijos)
            self._distribuir_caudales(hijo, Q_hijo)

    def _propagar_presiones(self, nodo: Tramo):
        # La fricción se paga con presión
        dp_friccion = self._calcular_friccion(
            nodo.velocidad, nodo.D, nodo.L, nodo.epsilon
        )
        dp_menor = nodo.K_menor * 0.5 * self.rho * (nodo.velocidad**2)
        dp_gravedad = self.rho * self.env.g * nodo.dz

        nodo.P_salida = nodo.P_entrada - dp_friccion - dp_menor - dp_gravedad

        for hijo in nodo.hijos:
            # Continuidad de presión estática en el nodo
            hijo.P_entrada = nodo.P_salida
            self._propagar_presiones(hijo)

    def reporte_energetico(self, P_motor: float):
        print("\n--- ANÁLISIS ENERGÉTICO Y DE EFICIENCIA ---")
        potencia_neta = self.raiz.caudal * (P_motor - self.env.P_atm_local)
        print(f"Potencia Hidráulica Entregada: {potencia_neta:.2f} W")

        perdida_friccion_total = sum(
            self._calcular_friccion(t.velocidad, t.D, t.L, t.epsilon) * t.caudal
            for t in self.lista_tramos
        )
        potencia_perdida = perdida_friccion_total

        pct_perdida = (
            (potencia_perdida / potencia_neta) * 100 if potencia_neta > 0 else 0
        )
        print(f"Energía disipada por fricción: {pct_perdida:.1f}%")

        if pct_perdida < 27.0:
            print("✓ ESTADO: Eficiencia Óptima (Pérdidas < 27%)")
        else:
            print("⚠ ALERTA: Fricción excesiva. Reevaluar diámetros o accesorios.")

        print("\n--- VALIDACIÓN METABÓLICA (OTR vs OUR) ---")
        # Asumiendo que todos los huecos terminales entregan caudal a los tanques
        caudal_tanques = sum(t.caudal for t in self.lista_tramos if t.es_hueco)
        # Convertir m3/s a L/h
        caudal_L_h = caudal_tanques * 1000 * 3600

        OTR = (
            self.env.eficiencia_transferencia * caudal_L_h * self.env.densidad_O2_mmol_L
        ) / self.env.volumen_tanque_L
        print(
            f"OTR Calculado: {OTR:.2f} mmol/L·h | OUR Requerido: {self.env.OUR_objetivo:.2f} mmol/L·h"
        )
        if OTR >= self.env.OUR_objetivo:
            print("✓ ESTADO: Oxigenación Biológica Suficiente.")
        else:
            print(
                "⚠ ALERTA: Déficit de oxígeno. Aumentar presión del motor o eficiencia del difusor."
            )

    def generar_graficas(self):
        """Genera las gráficas de EGL/HGL, Presión y Velocidad del sistema."""
        if not self.lista_tramos:
            print("No hay datos para graficar.")
            return

        nombres = [t.id_tramo for t in self.lista_tramos]
        presiones = [t.P_salida for t in self.lista_tramos]
        velocidades = [t.velocidad for t in self.lista_tramos]

        # Cálculo de HGL y EGL
        hgl = []
        egl = []
        for t in self.lista_tramos:
            # Carga de presión (m) = P / (rho * g)
            carga_presion = t.P_salida / (self.rho * self.env.g)
            # Carga de velocidad (m) = v^2 / (2g)
            carga_vel = (t.velocidad**2) / (2 * self.env.g)

            # Líneas de gradiente y energía referenciadas a su delta Z
            hgl.append(t.dz + carga_presion)
            egl.append(t.dz + carga_presion + carga_vel)

        # Crear figura con 3 subgráficas
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        fig.suptitle(
            "Análisis Hidráulico y Energético del Sistema",
            fontsize=14,
            fontweight="bold",
        )

        # 1. Plot EGL / HGL
        axs[0].plot(
            nombres,
            egl,
            label="EGL (Línea de Energía Total)",
            marker="o",
            color="#FF0000",
            linewidth=2,
        )
        axs[0].plot(
            nombres,
            hgl,
            label="HGL (Gradiente Hidráulico)",
            marker="s",
            color="#000080",
            linewidth=2,
        )
        axs[0].set_ylabel("Carga (m de fluido)")
        axs[0].set_title("Disipación de Energía a lo largo de la red")
        axs[0].legend()
        axs[0].grid(True, linestyle="--", alpha=0.6)

        # 2. Plot Presiones Estáticas
        axs[1].bar(nombres, presiones, color="#4169E1", edgecolor="black", alpha=0.8)
        axs[1].set_ylabel("Presión de salida (Pa)")
        axs[1].set_title("Caída de Presión Estática por Tramo")
        axs[1].grid(True, linestyle="--", alpha=0.4, axis="y")

        # 3. Plot Velocidades
        axs[2].plot(
            nombres, velocidades, color="#228B22", marker="D", linewidth=2, markersize=8
        )
        axs[2].set_ylabel("Velocidad (m/s)")
        axs[2].set_title("Perfil de Velocidades (Conservación de Masa)")
        axs[2].grid(True, linestyle="--", alpha=0.6)

        # Formateo del eje X para que los nombres no se superpongan
        for ax in axs:
            ax.tick_params(axis="x", rotation=15)

        plt.tight_layout(
            rect=[0, 0.03, 1, 0.96]
        )  # Ajuste para que el título global no se corte
        plt.show()


# ===================================================================
# 4. EJECUCIÓN DEL MODELO
# ===================================================================
if __name__ == "__main__":
    # 1. Entorno de simulación
    entorno = EntornoSimulacion()

    # 2. Construir Topología (Ejemplo simplificado de 1 rama, 1 bifurcación, 2 huecos)
    motor_a_T = Tramo("AB", longitud=2.0, diametro=0.04, delta_z=0.5, k_menor=0.0)

    rama_izq = Tramo(
        "BC_izq", longitud=1.5, diametro=0.02, delta_z=0.0, k_menor=0.9
    )  # Codo 90
    rama_der = Tramo("BC_der", longitud=1.5, diametro=0.02, delta_z=0.0, k_menor=0.9)

    hueco_izq = Tramo(
        "Hueco_1",
        longitud=0.01,
        diametro=0.005,
        delta_z=0.0,
        es_hueco=True,
        h_profundidad_tanque=0.6,
    )
    hueco_der = Tramo(
        "Hueco_2",
        longitud=0.01,
        diametro=0.005,
        delta_z=0.0,
        es_hueco=True,
        h_profundidad_tanque=0.6,
    )

    # Ensamblar árbol
    rama_izq.agregar_bifurcacion(hueco_izq)
    rama_der.agregar_bifurcacion(hueco_der)
    motor_a_T.agregar_bifurcacion(rama_izq)
    motor_a_T.agregar_bifurcacion(rama_der)

    # 3. Inicializar y Validar
    sim = SimuladorFlujo(motor_a_T, entorno)
    sim.validar_red()

    # 4. Resolver
    Presion_Salida_Motor = 1 * (1e5)  # Pa Absolutos
    Caudal_Estimado = 0.002  # m^3/s
    sim.resolver_flujo_directo(
        P_motor=Presion_Salida_Motor, Q_total_estimado=Caudal_Estimado
    )

    # 5. Análisis Final
    sim.reporte_energetico(P_motor=Presion_Salida_Motor)

    # 5. Análisis Final
    sim.reporte_energetico(P_motor=Presion_Salida_Motor)

    # 6. Generación de Gráficas
    sim.generar_graficas()
