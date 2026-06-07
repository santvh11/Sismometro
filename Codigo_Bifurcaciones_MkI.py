import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# ============================================================
# 1. PARÁMETROS REALES (basados en tus imágenes)
# ============================================================
# Soplador GF-180
Q_total_Lpm = 320.0  # L/min
Q_total = Q_total_Lpm / (1000 * 60)  # m³/s
P_soplor_manometrica = 10000.0  # 10 kPa

# Tubería: PVC rígido 1" (diámetro interior ≈ 26.6 mm)
D_mm = 26.6
D = D_mm / 1000.0
A_total = np.pi * D**2 / 4.0
rugosidad_abs = 0.0015e-3  # PVC liso (1.5 µm)

# Geometría de la red (ajusta según tu diagrama)
L_principal = 2.0  # m (tramo motor → primera bifurcación)
L_rama = 1.5  # m (bifurcación → válvula → tanque)
L_hueco = 0.01  # m (orificio final, despreciable)
sumergencia_h = 0.6  # m (profundidad del difusor)

# Accesorios
K_codo90 = 0.9
K_entrada_bifurcacion = 0.5  # pérdida por la T o derivación
K_salida_tanque = 1.0  # descarga sumergida
K_accesorios_fijos = K_codo90 + K_entrada_bifurcacion + K_salida_tanque

# Fluido: aire a 17°C y 2150 m de altitud (calculado con entorno)
altitud = 2150.0
temp_ambiente_c = 17.0
P0 = 101325.0
T0 = 288.15
g = 9.793
R_univ = 8.314
M_aire = 0.028964
L_grad = 0.0065
exponente = (g * M_aire) / (R_univ * L_grad)
P_atm_local = P0 * (1 - L_grad * altitud / T0) ** exponente
T_k = temp_ambiente_c + 273.15
rho_aire = (P_atm_local * M_aire) / (R_univ * T_k)  # kg/m³
mu_aire = 1.82e-5  # Pa·s

# Parámetros biológicos
OUR_objetivo = 3.0  # mmol O2 / (L·h)
V_tanque = 100.0  # L
eficiencia_O2 = 0.05
densidad_O2_molar = 8.6  # mmol/L (a 2150 m)


# ============================================================
# 2. FUNCIONES HIDRÁULICAS
# ============================================================
def factor_friccion(Re, epsilon, D):
    if Re < 2300:
        return 64.0 / Re
    f = 0.25 / (np.log10(epsilon / (3.7 * D) + 5.74 / Re**0.9)) ** 2
    return max(0.008, min(0.08, f))


def Kv_valvula(theta_deg):
    """theta=0 abierta, theta=90 cerrada."""
    if theta_deg <= 0:
        return 0.0
    if theta_deg >= 90:
        return 1e6
    sinθ = np.sin(np.radians(theta_deg))
    return (sinθ / (1.0 - sinθ)) ** 2


def perdida_tramo(Q, D, L, K_local, epsilon, rho, mu, es_hueco=False, h_sum=0.0):
    v = Q / (np.pi * D**2 / 4.0)
    Re = rho * v * D / mu
    f = factor_friccion(Re, epsilon, D)
    dp_fric = f * (L / D) * 0.5 * rho * v**2
    dp_local = K_local * 0.5 * rho * v**2
    dp = dp_fric + dp_local
    if es_hueco:
        dp += 1000.0 * 9.81 * h_sum  # contrapresión del agua
    return dp, v, Re, f


# ============================================================
# 3. CONFIGURACIÓN DE LA RED (N RAMAS IDÉNTICAS)
# ============================================================
N_tanques = 12  # <-- CAMBIA AQUÍ EL NÚMERO DE TANQUES (ramas)
Q_rama = Q_total / N_tanques

# Pérdida en el tramo principal (motor → primera bifurcación)
dp_principal, v_princ, Re_princ, f_princ = perdida_tramo(
    Q_total, D, L_principal, 0.0, rugosidad_abs, rho_aire, mu_aire
)
P_entrada_rama_man = P_soplor_manometrica - dp_principal


# Función para encontrar theta que iguale la pérdida en la rama a la presión disponible
def ecuacion_theta(theta):
    Kv = Kv_valvula(theta)
    K_total_rama = K_accesorios_fijos + Kv
    dp_rama, _, _, _ = perdida_tramo(
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
    return dp_rama - P_entrada_rama_man


# Resolver
theta_inicial = 30.0
sol = fsolve(ecuacion_theta, theta_inicial, full_output=True)
if sol[2] == 1:
    theta_teorico = sol[0][0]
    theta_teorico = max(0.0, min(90.0, theta_teorico))
else:
    theta_teorico = 90.0
    print("No se encontró solución, se asume válvula completamente cerrada.")

Kv_calc = Kv_valvula(theta_teorico)
K_total_rama = K_accesorios_fijos + Kv_calc

# Calcular pérdidas detalladas en la rama (para gráficas)
dp_rama_total, v_rama, Re_rama, f_rama = perdida_tramo(
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

# Desglose de pérdidas en la rama (lineales y singulares)
v = Q_rama / (np.pi * D**2 / 4.0)
dp_fric_rama = f_rama * (L_rama / D) * 0.5 * rho_aire * v**2
dp_sing_fijos = K_accesorios_fijos * 0.5 * rho_aire * v**2
dp_sing_valvula = Kv_calc * 0.5 * rho_aire * v**2
dp_hidro = 1000.0 * 9.81 * sumergencia_h

# Pérdidas totales (lineales y singulares) de toda la red (principal + N ramas)
dp_lineal_total = dp_fric_rama * N_tanques + (
    f_princ * (L_principal / D) * 0.5 * rho_aire * (Q_total / A_total) ** 2
)
dp_singular_total = (
    K_accesorios_fijos + Kv_calc
) * 0.5 * rho_aire * v**2 * N_tanques + 0.0  # el tramo principal no tiene singulares

# ============================================================
# 4. RESULTADOS NUMÉRICOS
# ============================================================
print("=== RESULTADOS DEL MODELO ===")
print(f"Número de tanques: {N_tanques}")
print(f"Caudal total: {Q_total_Lpm:.1f} L/min")
print(f"Caudal por rama: {Q_rama * 1000 * 60:.2f} L/min")
print(f"Ángulo teórico de la válvula (θ) = {theta_teorico:.1f}°")
print(f"K_v de la válvula: {Kv_calc:.2f}")
print(f"Presión manométrica del soplador: {P_soplor_manometrica/1000:.2f} kPa")
print(f"Pérdida en tubería principal: {dp_principal/1000:.2f} kPa")
print(f"Presión disponible en la rama: {P_entrada_rama_man/1000:.2f} kPa")
print(f"Pérdida total en la rama: {dp_rama_total/1000:.2f} kPa")
print("  - Pérdida por fricción: {:.2f} kPa".format(dp_fric_rama / 1000))
print("  - Pérdida accesorios fijos: {:.2f} kPa".format(dp_sing_fijos / 1000))
print("  - Pérdida válvula: {:.2f} kPa".format(dp_sing_valvula / 1000))
print("  - Contrapresión hidrostática: {:.2f} kPa".format(dp_hidro / 1000))

# Capacidad biológica
Q_necesario_Lpm = (OUR_objetivo * V_tanque) / (eficiencia_O2 * densidad_O2_molar) / 60.0
N_max_biologico = int(Q_total_Lpm / Q_necesario_Lpm)
print(f"\n--- Capacidad biológica ---")
print(f"Caudal necesario por tanque: {Q_necesario_Lpm:.2f} L/min")
print(f"N° máximo de tanques (solo biología): {N_max_biologico}")

# ============================================================
# 5. GRÁFICAS
# ============================================================
# 5.1 Presión estática en cada tanque (salida de la rama)
presion_tanque = P_atm_local / 1000 + (dp_hidro) / 1000  # presión absoluta aprox
plt.figure(figsize=(8, 5))
tanques = [f"Tanque {i+1}" for i in range(N_tanques)]
presiones = [presion_tanque] * N_tanques
plt.bar(tanques, presiones, color="skyblue", edgecolor="black")
plt.ylabel("Presión absoluta (kPa)")
plt.title("Presión estática a la salida de cada difusor")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# 5.2 Velocidad en cada tramo (principal + ramas)
tramos = ["Principal", f"Rama (c/u)"]
velocidades = [Q_total / A_total, v_rama]
plt.figure(figsize=(6, 5))
plt.bar(tramos, velocidades, color="lightgreen", edgecolor="black")
plt.ylabel("Velocidad (m/s)")
plt.title("Velocidad del flujo en los tramos")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# 5.3 Mapa de fugas energéticas (pérdidas lineales vs singulares totales)
perdidas = [dp_lineal_total, dp_singular_total]
etiquetas = ["Pérdidas lineales", "Pérdidas singulares (codos+válvulas)"]
colores = ["#ff9999", "#66b3ff"]
plt.figure(figsize=(7, 7))
plt.pie(perdidas, labels=etiquetas, autopct="%1.1f%%", startangle=90, colors=colores)
plt.title("Distribución de pérdidas de presión en toda la red")
plt.axis("equal")
plt.show()

# 5.4 Perfil EGL y HGL a lo largo de la red (orden secuencial)
# Construimos una secuencia de puntos: entrada motor -> fin principal -> entrada rama -> salida rama (difusor)
# Para simplificar, consideramos que la red es un solo camino (la presión en todas las ramas es igual).
# Orden: [Motor, después del tramo principal, después de la válvula+accesorios, dentro del tanque]
z = [0, 0, 0, -sumergencia_h]  # alturas geodésicas (referencia motor a cota 0)
# Carga de presión (P/ρg) en cada punto, en metros de columna de aire (¡no agua!)
P_abs_motor = P_atm_local + P_soplor_manometrica
P_abs_fin_principal = P_abs_motor - dp_principal
P_abs_entrada_rama = P_abs_fin_principal(misma)
P_abs_salida_rama = P_abs_entrada_rama - dp_rama_total
# Convertir a metros de columna de aire (ρ_aire * g)
carga_presion = [
    P_abs_motor / (rho_aire * g),
    P_abs_fin_principal / (rho_aire * g),
    P_abs_entrada_rama / (rho_aire * g),
    P_abs_salida_rama / (rho_aire * g),
]
# Carga de velocidad
v_motor = Q_total / A_total
v_rama_calc = v_rama
carga_vel = [
    v_motor**2 / (2 * g),
    v_motor**2 / (2 * g),
    v_rama_calc**2 / (2 * g),
    0,
]  # en el tanque la velocidad es nula
# HGL = z + P/ρg
HGL = [z[i] + carga_presion[i] for i in range(4)]
# EGL = z + P/ρg + v²/2g
EGL = [HGL[i] + carga_vel[i] for i in range(4)]
puntos = ["Salida motor", "Fin tramo principal", "Entrada rama", "Difusor (tanque)"]

plt.figure(figsize=(10, 6))
plt.plot(
    puntos, EGL, "ro-", label="EGL (Línea de Energía Total)", linewidth=2, markersize=8
)
plt.plot(
    puntos,
    HGL,
    "bs-",
    label="HGL (Línea de Gradiente Hidráulico)",
    linewidth=2,
    markersize=8,
)
plt.xlabel("Puntos de la red")
plt.ylabel("Carga (metros de columna de aire)")
plt.title("Perfil de Energía y Gradiente Hidráulico")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(rotation=15)
plt.show()

print("\n--- FIN DEL ANÁLISIS ---")
