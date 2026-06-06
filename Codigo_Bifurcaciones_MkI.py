import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional

# ===================================================================
# 1. PARÁMETROS GLOBALES Y AMBIENTALES
# ===================================================================


@dataclass
class EntornoSimulacion:
    altitud_m: float = 2150.0
    temp_ambiente_c: float = 17.0
    P0: float = 101325.0
    T0: float = 288.15
    g: float = 9.793
    R: float = 8.314
    L_grad: float = 0.0065
    M_aire: float = 0.028964

    OUR_objetivo: float = 3.0
    eficiencia_transferencia: float = 0.05
    volumen_tanque_L: float = 100.0

    P_atm_local: float = field(init=False)
    densidad_O2_mmol_L: float = field(init=False)

    def __post_init__(self):
        self.P_atm_local = self.calcular_presion_atmosferica()
        self.densidad_O2_mmol_L = self.calcular_densidad_o2_vdw()

    def calcular_presion_atmosferica(self) -> float:
        exponente = (self.g * self.M_aire) / (self.R * self.L_grad)
        base = 1.0 - ((self.L_grad * self.altitud_m) / self.T0)
        return self.P0 * (base**exponente)

    def calcular_densidad_o2_vdw(self) -> float:
        T_k = self.temp_ambiente_c + 273.15
        P_o2 = 0.2095 * self.P_atm_local
        a, b = 0.1382, 3.186e-5
        rho = P_o2 / (self.R * T_k)
        for _ in range(20):
            f = (rho * self.R * T_k) / (1.0 - rho * b) - a * (rho**2) - P_o2
            df = (self.R * T_k) / ((1.0 - rho * b) ** 2) - 2 * a * rho
            rho_nueva = rho - f / df
            if abs(rho_nueva - rho) < 1e-7:
                break
            rho = rho_nueva
        return rho


# ===================================================================
# 2. TOPOLOGÍA DE LA RED (MANIFOLD / MÚLTIPLE)
# ===================================================================


class Tramo:
    def __init__(
        self,
        id_tramo: str,
        longitud: float,
        diametro: float,
        delta_z: float,
        es_hueco: bool = False,
        num_tanque: Optional[int] = None,
        h_profundidad_tanque: float = 0.0,
        k_menor: float = 0.0,
        rugosidad: float = 1.5e-6,
    ):
        self.id_tramo = id_tramo
        self.D = diametro
        self.L = longitud
        self.dz = delta_z
        self.epsilon = rugosidad
        self.A = np.pi * (self.D**2) / 4.0
        self.K_menor = k_menor
        self.es_hueco = es_hueco
        self.num_tanque = num_tanque
        self.h_prof = h_profundidad_tanque
        self.hijos: List["Tramo"] = []

        self.caudal: float = 0.0
        self.velocidad: float = 0.0
        self.P_entrada: float = 0.0
        self.P_salida: float = 0.0

    def agregar_bifurcacion(self, tramo_hijo: "Tramo"):
        self.hijos.append(tramo_hijo)


# ===================================================================
# 3. NÚCLEO DEL SIMULADOR (CON BALANCEO FÍSICO ITERATIVO)
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
        self.lista_tramos.append(nodo)
        for hijo in nodo.hijos:
            self._mapear_red(hijo)

    def _calcular_friccion(self, v: float, D: float, L: float, epsilon: float) -> float:
        if v == 0:
            return 0.0
        Re = self.rho * abs(v) * D / self.mu
        if Re < 10:
            return 0.0
        if Re < 2300:
            f = 64.0 / Re
        else:
            f = 0.25 / (np.log10((epsilon / (3.7 * D)) + (5.74 / (Re**0.9)))) ** 2
        return f * (L / D) * 0.5 * self.rho * (v**2)

    def resolver_flujo_directo(
        self, P_motor_manometrica: float, Q_total_estimado: float
    ):
        """Asigna flujo iterativamente basado en la presión remanente (Torricelli modificado)"""
        # Presión Absoluta en la bomba
        self.raiz.P_entrada = self.env.P_atm_local + P_motor_manometrica

        # 1. Distribución inicial equitativa para arranque
        self._distribuir_caudales_por_area(self.raiz, Q_total_estimado)

        # 2. Bucle de relajación (Balanceo de red)
        for _ in range(15):
            self._propagar_presiones(self.raiz)

            huecos = [t for t in self.lista_tramos if t.es_hueco]
            pesos = []

            for h in huecos:
                # Contrapresión hidrostática del tanque
                P_contra = self.env.P_atm_local + (self.rho * self.env.g * h.h_prof)
                delta_p = h.P_entrada - P_contra
                # El caudal es proporcional a la raíz cuadrada de la presión estática disponible
                peso = np.sqrt(max(1e-5, delta_p)) * h.A
                pesos.append(peso)

            suma_pesos = sum(pesos)

            for i, h in enumerate(huecos):
                h.caudal = Q_total_estimado * (pesos[i] / suma_pesos)
                h.velocidad = h.caudal / h.A

            # Recalcular hacia atrás para ajustar los colectores principales
            self._actualizar_caudales_hacia_arriba(self.raiz)

    def _distribuir_caudales_por_area(self, nodo: Tramo, Q_entrada: float):
        nodo.caudal = Q_entrada
        nodo.velocidad = Q_entrada / nodo.A
        if not nodo.hijos:
            return
        area_total = sum(h.A for h in nodo.hijos)
        for hijo in nodo.hijos:
            self._distribuir_caudales_por_area(hijo, Q_entrada * (hijo.A / area_total))

    def _propagar_presiones(self, nodo: Tramo):
        dp_friccion = self._calcular_friccion(
            nodo.velocidad, nodo.D, nodo.L, nodo.epsilon
        )
        dp_menor = nodo.K_menor * 0.5 * self.rho * (nodo.velocidad**2)
        dp_gravedad = self.rho * self.env.g * nodo.dz

        nodo.P_salida = nodo.P_entrada - dp_friccion - dp_menor - dp_gravedad
        for hijo in nodo.hijos:
            hijo.P_entrada = nodo.P_salida
            self._propagar_presiones(hijo)

    def _actualizar_caudales_hacia_arriba(self, nodo: Tramo) -> float:
        if not nodo.hijos:
            return nodo.caudal
        caudal_acumulado = 0.0
        for hijo in nodo.hijos:
            caudal_acumulado += self._actualizar_caudales_hacia_arriba(hijo)
        nodo.caudal = caudal_acumulado
        nodo.velocidad = nodo.caudal / nodo.A
        return nodo.caudal

    def reporte_energetico(self, P_motor_manometrica: float):
        print("\n--- ANÁLISIS ENERGÉTICO ---")
        potencia_neta = self.raiz.caudal * P_motor_manometrica
        perdida_friccion_total = sum(
            self._calcular_friccion(t.velocidad, t.D, t.L, t.epsilon) * t.caudal
            for t in self.lista_tramos
        )
        pct_perdida = (
            (perdida_friccion_total / potencia_neta) * 100 if potencia_neta > 0 else 0
        )

        print(f"Potencia neta entregada: {potencia_neta:.2f} W")
        print(f"Energía disipada por fricción: {pct_perdida:.1f}%")
        print(
            "✓ ESTADO: Eficiencia Óptima"
            if pct_perdida < 27.0
            else "⚠ ALERTA: Fricción alta"
        )

    def generar_graficas_por_tanque(self):
        tramos_tanques = [
            t for t in self.lista_tramos if t.es_hueco and t.num_tanque is not None
        ]
        tramos_tanques.sort(key=lambda x: x.num_tanque)

        tanques_labels = [f"Grupo T{t.num_tanque}" for t in tramos_tanques]
        presiones = [
            (t.P_salida - self.env.P_atm_local) / 1000.0 for t in tramos_tanques
        ]
        velocidades = [t.velocidad for t in tramos_tanques]
        flujos_masicos = [t.caudal * self.rho for t in tramos_tanques]

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            "Condiciones Hidráulicas por Grupos de Simetría (Manifold)",
            fontsize=14,
            fontweight="bold",
        )

        axs[0].bar(
            tanques_labels, presiones, color="#3498db", edgecolor="black", alpha=0.8
        )
        axs[0].set_ylabel("Presión Manométrica Salida (kPa)")
        axs[0].set_title("Caída de Presión en Red")
        axs[0].grid(True, linestyle="--", alpha=0.5, axis="y")

        axs[1].bar(
            tanques_labels, velocidades, color="#2ecc71", edgecolor="black", alpha=0.8
        )
        axs[1].set_ylabel("Velocidad de Salida (m/s)")
        axs[1].set_title("Perfil de Velocidad por Grupo")
        axs[1].grid(True, linestyle="--", alpha=0.5, axis="y")

        axs[2].bar(
            tanques_labels,
            flujos_masicos,
            color="#e74c3c",
            edgecolor="black",
            alpha=0.8,
        )
        axs[2].set_ylabel("Flujo Másico (kg/s)")
        axs[2].set_title("Distribución de Masa Real")
        axs[2].grid(True, linestyle="--", alpha=0.5, axis="y")

        plt.tight_layout()
        plt.show()


# ===================================================================
# 4. CONFIGURACIÓN EN MANIFOLD (COLECTOR EN SERIE CON DERIVACIONES)
# ===================================================================
if __name__ == "__main__":
    entorno = EntornoSimulacion()

    # Presión Manométrica del Motor: 1 bar (100 kPa)
    Presion_Motor_Gauge = 100000.0

    # Caudal total de 0.6 L/s para evitar velocidades extremas
    Caudal_Estimado = 0.0006

    # --- TRAMO PRINCIPAL 1 ---
    colector_1 = Tramo("Colector_1", longitud=1.0, diametro=0.04, delta_z=0.0)
    derivacion_T1 = Tramo(
        "Deriv_T1", longitud=0.5, diametro=0.02, delta_z=0.0, k_menor=0.9
    )
    # Diametro 12mm representa el área equivalente de 4 huecos de 6mm (por simetría)
    hueco_T1 = Tramo(
        "Orificio_T1",
        longitud=0.02,
        diametro=0.012,
        delta_z=0.0,
        es_hueco=True,
        num_tanque=1,
        h_profundidad_tanque=0.5,
    )

    # --- TRAMO PRINCIPAL 2 ---
    colector_2 = Tramo("Colector_2", longitud=1.0, diametro=0.04, delta_z=0.0)
    derivacion_T2 = Tramo(
        "Deriv_T2", longitud=0.5, diametro=0.02, delta_z=0.0, k_menor=0.9
    )
    hueco_T2 = Tramo(
        "Orificio_T2",
        longitud=0.02,
        diametro=0.012,
        delta_z=0.0,
        es_hueco=True,
        num_tanque=2,
        h_profundidad_tanque=0.5,
    )

    # --- TRAMO PRINCIPAL 3 ---
    colector_3 = Tramo("Colector_3", longitud=1.0, diametro=0.04, delta_z=0.0)
    derivacion_T3 = Tramo(
        "Deriv_T3", longitud=0.5, diametro=0.02, delta_z=0.0, k_menor=0.9
    )
    hueco_T3 = Tramo(
        "Orificio_T3",
        longitud=0.02,
        diametro=0.012,
        delta_z=0.0,
        es_hueco=True,
        num_tanque=3,
        h_profundidad_tanque=0.5,
    )

    # ENSAMBLAJE DEL ÁRBOL EN SERIE (Flujo viaja de 1 -> 2 -> 3)
    derivacion_T1.agregar_bifurcacion(hueco_T1)
    colector_1.agregar_bifurcacion(derivacion_T1)
    colector_1.agregar_bifurcacion(colector_2)  # Sigue el flujo principal

    derivacion_T2.agregar_bifurcacion(hueco_T2)
    colector_2.agregar_bifurcacion(derivacion_T2)
    colector_2.agregar_bifurcacion(colector_3)  # Sigue el flujo principal

    derivacion_T3.agregar_bifurcacion(hueco_T3)
    colector_3.agregar_bifurcacion(derivacion_T3)  # Fin de línea

    # Simular
    sim = SimuladorFlujo(colector_1, entorno)
    sim.resolver_flujo_directo(
        P_motor_manometrica=Presion_Motor_Gauge, Q_total_estimado=Caudal_Estimado
    )

    # Resultados
    sim.reporte_energetico(P_motor_manometrica=Presion_Motor_Gauge)
    sim.generar_graficas_por_tanque()
