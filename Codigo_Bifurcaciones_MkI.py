import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# ==================================================================
# 1. PARÁMETROS REALES (basados en tus imágenes y datos)
# ==================================================================
# Soplador GF-180
Q_total_Lpm = 320.0  # L/min
Q_total = Q_total_Lpm / (1000 * 60)  # m³/s
P_soplor_manometrica = 10000.0  # 10 kPa (presión máxima de salida)

# Tubería: PVC rígido 1" (diámetro interior típico ≈ 26.6 mm)
D_mm = 26.6
D = D_mm / 1000.0  # m
A_total = np.pi * D**2 / 4.0
rugosidad_abs = 0.0015e-3  # PVC liso (1.5 µm)

# Geometría del montaje (del código MkI)
L_principal = 2.0  # m (tramo motor → bifurcación)
L_rama = 1.5  # m (bifurcación → válvula)
L_hueco = 0.01  # m (orificio final)
sumergencia_h = 0.6  # m (profundidad del difusor)

# Accesorios
K_codo90 = 0.9
K_entrada_bifurcacion = 0.5  # estimado
K_salida_tanque = 1.0  # descarga sumergida
K_accesorios_fijos = K_codo90 + K_entrada_bifurcacion + K_salida_tanque

# Fluido: aire a 20°C y 2150 m de altitud (calculado con entorno)
# Usamos los datos del EntornoSimulacion de tu código
altitud = 2150.0
temp_ambiente_c = 17.0
# Presión atmosférica local (ecuación barométrica)
P0 = 101325.0
T0 = 288.15
g = 9.793
R_univ = 8.314
M_aire = 0.028964
L_grad = 0.0065
exponente = (g * M_aire) / (R_univ * L_grad)
P_atm_local = P0 * (1 - L_grad * altitud / T0) ** exponente
# Densidad del aire (gas ideal)
T_k = temp_ambiente_c + 273.15
rho_aire = (P_atm_local * M_aire) / (R_univ * T_k)  # kg/m³
# Viscosidad dinámica del aire a 20°C
mu_aire = 1.82e-5  # Pa·s

# Parámetros del tanque (para verificar demanda biológica, no necesario para el ángulo)
OUR_objetivo = 3.0  # mmol O2 / (L·h)
V_tanque = 100.0  # L
eficiencia_O2 = 0.05
# Densidad molar del O2 en el aire a esta altitud (ya lo calculaste)
densidad_O2_molar = 8.6  # mmol/L (valor típico)


# ==================================================================
# 2. FUNCIONES HIDRÁULICAS
# ==================================================================
def factor_friccion(Re, epsilon, D):
    """Swamee-Jain (turbulento) o 64/Re (laminar)"""
    if Re < 2300:
        return 64.0 / Re
    f = 0.25 / (np.log10(epsilon / (3.7 * D) + 5.74 / Re**0.9)) ** 2
    return max(0.008, min(0.08, f))


def Kv_valvula(theta_deg):
    """theta=0 abierta, theta=90 cerrada.
    K = (sinθ/(1-sinθ))^2."""
    if theta_deg <= 0:
        return 0.0
    if theta_deg >= 90:
        return 1e6
    sinθ = np.sin(np.radians(theta_deg))
    return (sinθ / (1.0 - sinθ)) ** 2


def perdida_tramo(Q, D, L, K_local_extra, epsilon, rho, mu, es_hueco=False, h_sum=0.0):
    """Calcula la caída de presión (Pa) en un tramo de tubería.
    Si es_hueco=True, añade la contrapresión hidrostática del agua."""
    v = Q / (np.pi * D**2 / 4.0)
    Re = rho * v * D / mu
    f = factor_friccion(Re, epsilon, D)
    dp_fric = f * (L / D) * 0.5 * rho * v**2
    dp_local = K_local_extra * 0.5 * rho * v**2
    dp = dp_fric + dp_local
    if es_hueco:
        # Contrapresión del agua (rho_agua * g * h)
        dp += 1000.0 * 9.81 * h_sum
    return dp


# ==================================================================
# 3. CONFIGURACIÓN DE LA RED (dos ramas simétricas, con válvula)
# ==================================================================
# Caudal por rama (mitad del total)
Q_rama = Q_total / 2.0

# Pérdidas en el tramo principal (motor → bifurcación)
dp_principal = perdida_tramo(
    Q_total, D, L_principal, 0.0, rugosidad_abs, rho_aire, mu_aire
)

# Presión disponible a la entrada de cada rama (manométrica)
P_entrada_rama_man = P_soplor_manometrica - dp_principal


# En cada rama: tubería (L_rama) + accesorios fijos + válvula + hueco
# La pérdida total en la rama debe igualar P_entrada_rama_man (para que la presión final sea la atmosférica + hidrostática)
def ecuacion_theta(theta):
    Kv = Kv_valvula(theta)
    K_total_rama = K_accesorios_fijos + Kv
    dp_rama = perdida_tramo(
        Q_rama,
        D,
        L_rama,
        K_total_rama,
        rugosidad_abs,
        rho_aire,
        mu_aire,
        es_hueco=True,
        h_sum=sumergencia_h,
    )
    # La presión a la salida del hueco debe ser igual a la presión atmosférica (referencia manométrica = 0)
    # Por tanto, la presión disponible (P_entrada_rama_man) debe consumirse íntegramente:
    return dp_rama - P_entrada_rama_man


# Buscamos theta en [0, 90]
theta_inicial = 30.0
sol = fsolve(ecuacion_theta, theta_inicial, full_output=True)
if sol[2] == 1:
    theta_teorico = sol[0][0]
    theta_teorico = max(0.0, min(90.0, theta_teorico))
    print(f"Ángulo teórico de la válvula (θ) = {theta_teorico:.1f}°")
else:
    print("No se encontró solución. Verifique los datos.")

# Comprobación: pérdidas detalladas
Kv_calc = Kv_valvula(theta_teorico)
K_total = K_accesorios_fijos + Kv_calc
dp_rama_final = perdida_tramo(
    Q_rama,
    D,
    L_rama,
    K_total,
    rugosidad_abs,
    rho_aire,
    mu_aire,
    es_hueco=True,
    h_sum=sumergencia_h,
)
print(f"\n--- Verificación ---")
print(f"Presión manométrica del soplador: {P_soplor_manometrica/1000:.2f} kPa")
print(f"Pérdida en tubería principal: {dp_principal/1000:.2f} kPa")
print(f"Presión disponible en la rama: {P_entrada_rama_man/1000:.2f} kPa")
print(
    f"Pérdida total en la rama (con válvula a {theta_teorico:.1f}°): {dp_rama_final/1000:.2f} kPa"
)
print(f"K_v de la válvula: {Kv_calc:.2f}")

# ==================================================================
# 4. COMPARACIÓN CON LA DEMANDA BIOLÓGICA (tanques que se pueden alimentar)
# ==================================================================
# Caudal de aire necesario por tanque (L/min) según OUR
Q_necesario_Lpm = (OUR_objetivo * V_tanque) / (eficiencia_O2 * densidad_O2_molar) / 60.0
N_max_biologico = int(Q_total_Lpm / Q_necesario_Lpm)
print(f"\n--- Capacidad biológica ---")
print(f"Caudal por tanque necesario: {Q_necesario_Lpm:.2f} L/min")
print(f"Número máximo de tanques (solo biología): {N_max_biologico}")

# ==================================================================
# 5. GRÁFICA DEL COEFICIENTE K_v EN FUNCIÓN DE θ
# ==================================================================
theta_vals = np.linspace(0, 90, 200)
K_vals = [Kv_valvula(th) for th in theta_vals]
plt.figure(figsize=(8, 5))
plt.plot(theta_vals, K_vals, "b-", linewidth=2)
plt.axvline(
    theta_teorico, color="r", linestyle="--", label=f"θ teórico = {theta_teorico:.1f}°"
)
plt.yscale("log")
plt.xlabel("Ángulo de la válvula (grados)")
plt.ylabel("Coeficiente de pérdida K_v")
plt.title("Curva característica de la válvula de mariposa")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend()
plt.show()
