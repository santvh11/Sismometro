import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# ============================================================
# 1. PARÁMETROS GENERALES (soplador, fluido, tubería)
# ============================================================
Q_total_Lpm = 320.0  # L/min
Q_total = Q_total_Lpm / (1000 * 60)  # m³/s
P_soplor_man = 10000.0  # Pa manométricos
P_atm = 78000.0  # Aprox a 2150 m (ajusta si lo tienes)
# Tubería principal y ramas: mismo diámetro
D_mm = 26.6
D = D_mm / 1000.0
A = np.pi * D**2 / 4.0
rugosidad = 0.0015e-3
rho_aire = 0.96  # kg/m³ (ajustar a altitud)
mu_aire = 1.82e-5
g = 9.793
rho_agua = 1000.0


# ============================================================
# 2. FUNCIONES AUXILIARES
# ============================================================
def friccion_factor(Re, eps, D):
    if Re < 2300:
        return 64 / Re
    return 0.25 / (np.log10(eps / (3.7 * D) + 5.74 / Re**0.9)) ** 2


def Kv_valvula(theta_deg):
    if theta_deg <= 0:
        return 0.0
    if theta_deg >= 90:
        return 1e9
    sinθ = np.sin(np.radians(theta_deg))
    return (sinθ / (1.0 - sinθ)) ** 2


def perdida_tuberia(Q, L, K_local_extra, dz, h_sum=0.0, es_hueco=False):
    """Devuelve dp (Pa) para un tramo con flujo de aire.
    Si es_hueco=True, añade contrapresión hidrostática del agua."""
    v = Q / A
    Re = rho_aire * v * D / mu_aire
    f = friccion_factor(Re, rugosidad, D)
    dp_fric = f * (L / D) * 0.5 * rho_aire * v**2
    dp_local = K_local_extra * 0.5 * rho_aire * v**2
    dp_grav = rho_aire * g * dz  # cambio de altura (aire)
    dp = dp_fric + dp_local + dp_grav
    if es_hueco:
        dp += rho_agua * g * h_sum  # presión hidrostática del agua
    return dp


# ============================================================
# 3. TOPOLOGÍA DE LA RED (colector + ramas)
# ============================================================
# Supongamos 6 derivaciones a cada lado (total 12).
# El colector tiene una longitud desde el soplador hasta la primera derivación,
# luego tramos de 0.4 m entre derivaciones.
# Para simplificar, asumimos que las derivaciones izquierda y derecha están a la misma distancia.
# Definimos distancias acumuladas al punto de derivación (en metros).
distancias_derivacion = [
    1.0,
    1.4,
    1.8,
]  # ejemplo: primera a 1 m del soplador, luego cada 0.4 m
N_deriv = len(distancias_derivacion)  # =6
N_tanques = N_deriv * 4  # =12

# Características de CADA RAMA (izquierda y derecha son simétricas en este ejemplo)
# Longitud de tubería desde la derivación hasta el difusor:
L_rama_horiz = 1.7  # m (170 cm)
L_rama_subida = 0.65  # m
L_rama_bajada = 0.60  # m
L_rama_total = L_rama_horiz + L_rama_subida + L_rama_bajada  # 2.95 m
# Desnivel neto (salida menos entrada): la salida está más baja que la entrada si el tanque está por debajo
# Supongamos que la derivación está a cota 0, el tanque a cota -0.05 m (por la subida/bajada)
dz_rama = -0.05  # m (negativo, el fluido baja)
# Número de codos en la rama: 3
K_codos = 4 * 0.9
# Coeficiente de pérdida del difusor (30 orificios de 5 mm). Estimación:
# Relación de áreas: A_orificios_total = 30 * π*(0.0025)^2 = 5.89e-4 m²; A_tubo = 5.56e-4 m² -> casi igual.
# En realidad, la velocidad en los orificios será muy alta; usamos Cd=0.6.
# La fórmula para un múltiple orificio en serie: K_dif = (1/(Cd^2 * n^2 * (Ao/At)^2) - 1)
n_orificios = 30
Ao = n_orificios * np.pi * (0.0025) ** 2
At = A
K_difusor = 10
K_fijos_rama = K_codos + K_difusor
# Profundidad de los orificios dentro del agua (contrapresión hidrostática)
h_sumergencia = 0.6  # m

# ============================================================
# 4. CÁLCULO DE LA PRESIÓN DISPONIBLE EN CADA DERIVACIÓN
# ============================================================
# El colector principal: tramos entre derivaciones.
# Tramo 1: desde soplador hasta primera derivación (distancia = distancias_derivacion[0])
# Tramo i: entre derivación i-1 y derivación i (longitud = diferencia)
longitudes_entre = [distancias_derivacion[0]]
for i in range(1, N_deriv):
    longitudes_entre.append(distancias_derivacion[i] - distancias_derivacion[i - 1])

# Caudal en cada tramo del colector: va disminuyendo porque se derivan Q_rama.
Q_rama_objetivo = Q_total / N_tanques  # caudal que queremos por rama
caudales_colector = []
Q_restante = Q_total
for i in range(N_deriv):
    caudales_colector.append(Q_restante)
    Q_restante -= (
        4 * Q_rama_objetivo
    )  # porque en cada punto se derivan dos ramas (izq+der)
# El último tramo después de la última derivación ya no tiene flujo, no lo consideramos.

# Pérdida de presión en cada tramo del colector (solo fricción, sin accesorios)
presion_derivacion = [0.0] * N_deriv
P_actual = P_soplor_man  # manométrica a la salida del soplador
for i, L in enumerate(longitudes_entre):
    Q_tramo = caudales_colector[i]
    v_tramo = Q_tramo / A
    Re_tramo = rho_aire * v_tramo * D / mu_aire
    f_tramo = friccion_factor(Re_tramo, rugosidad, D)
    dp_fric_tramo = f_tramo * (L / D) * 0.5 * rho_aire * v_tramo**2
    P_actual -= dp_fric_tramo
    presion_derivacion[i] = P_actual  # presión manométrica en el punto de derivación

print("Presiones disponibles en cada derivación (kPa manométricas):")
for i, P in enumerate(presion_derivacion):
    print(
        f"Derivación {i+1} (distancia {distancias_derivacion[i]:.1f} m): {P/1000:.2f} kPa"
    )


# ============================================================
# 5. CÁLCULO DEL ÁNGULO θ PARA CADA RAMA (flujo equitativo)
# ============================================================
# Para cada derivación (que alimenta dos ramas simétricas izquierda/derecha),
# resolvemos la ecuación: P_disp = pérdida_rama(Q_rama_objetivo, theta)
def residuo_theta(theta, P_disp, Q_rama):
    # Limitar theta entre 1° y 89° para evitar divisiones por cero
    if theta < 1:
        theta = 1.0
    if theta > 89:
        theta = 89.0

    Kv = Kv_valvula(theta)
    K_total = K_fijos_rama + Kv
    dp = perdida_tuberia(
        Q_rama, L_rama_total, K_total, dz_rama, h_sumergencia, es_hueco=True
    )
    return dp - P_disp


theta_por_derivacion = []
for i, P_disp in enumerate(presion_derivacion):
    try:
        # Buscamos theta entre 0 y 90
        sol = fsolve(
            residuo_theta, 50.0, args=(P_disp, Q_rama_objetivo), full_output=True
        )
        if sol[2] == 1:
            theta = sol[0][0]
            theta = max(0, min(90, theta))
            theta_por_derivacion.append(theta)
        else:
            theta_por_derivacion.append(90.0)
    except:
        theta_por_derivacion.append(90.0)

# Mostrar resultados
print("\nÁngulos necesarios para igualar caudales (θ en grados):")
for i, theta in enumerate(theta_por_derivacion):
    print(f"Derivación {i+1} (dos tanques): θ = {theta:.1f}°")

# ============================================================
# 6. GRÁFICAS
# ============================================================
# 6.1 Presión en cada derivación
plt.figure(figsize=(10, 4))
plt.bar(range(1, N_deriv + 1), np.array(presion_derivacion) / 1000, color="lightblue")
plt.xlabel("Número de derivación (cada una alimenta 2 tanques)")
plt.ylabel("Presión manométrica (kPa)")
plt.title("Presión disponible en cada punto de bifurcación del colector")
plt.grid(axis="y", linestyle="--")
plt.show()

# 6.2 Ángulo de válvula por derivación
plt.figure(figsize=(10, 4))
plt.plot(range(1, N_deriv + 1), theta_por_derivacion, "ro-", markersize=8)
plt.xlabel("Derivación")
plt.ylabel("Ángulo de la válvula (grados)")
plt.title("Apertura necesaria de la mariposa para caudal equitativo")
plt.ylim(0, 90)
plt.grid(True)
plt.show()

# 6.3 Mapa de fugas energéticas (pérdidas lineales vs singulares en toda la red)
# Sumamos las pérdidas en el colector (lineales) y en las ramas (lineales + singulares + hidrostática)
perdida_colector_lineal = sum(
    [
        friccion_factor(rho_aire * v / D / mu_aire, rugosidad, D)
        * (L / D)
        * 0.5
        * rho_aire
        * v**2
        for L, Q in zip(longitudes_entre, caudales_colector)
        for v in [Q / A]
    ]
)
perdida_ramas_lineal = N_tanques * (
    friccion_factor(rho_aire * Q_rama_objetivo / A / D / mu_aire, rugosidad, D)
    * (L_rama_total / D)
    * 0.5
    * rho_aire
    * (Q_rama_objetivo / A) ** 2
)
perdida_ramas_singular_fijos = N_tanques * (
    K_fijos_rama * 0.5 * rho_aire * (Q_rama_objetivo / A) ** 2
)
perdida_valvulas = (
    sum(
        [
            Kv_valvula(th) * 0.5 * rho_aire * (Q_rama_objetivo / A) ** 2
            for th in theta_por_derivacion
        ]
    )
    * 2
)  # cada derivación dos válvulas
perdida_hidro_total = N_tanques * (rho_agua * g * h_sumergencia)

total_perdidas = (
    perdida_colector_lineal
    + perdida_ramas_lineal
    + perdida_ramas_singular_fijos
    + perdida_valvulas
    + perdida_hidro_total
)
labels = [
    "Colector (fricción)",
    "Ramas fricción",
    "Accesorios fijos (codos+difusor)",
    "Válvulas (mariposa)",
    "Contrapresión agua",
]
valores = [
    perdida_colector_lineal,
    perdida_ramas_lineal,
    perdida_ramas_singular_fijos,
    perdida_valvulas,
    perdida_hidro_total,
]
plt.figure(figsize=(8, 8))
plt.pie(valores, labels=labels, autopct="%1.1f%%", startangle=90)
plt.title(
    "Distribución de pérdidas de presión (total = {:.1f} kPa)".format(
        total_perdidas / 1000
    )
)
plt.show()

# 6.4 Perfil EGL/HGL a lo largo del colector + una rama representativa (la más alejada)
# Construimos puntos: salida soplador, cada derivación, y luego dentro del tanque de la última rama.
puntos_nombres = ["Soplador"]
distancias = [0]
presiones_man = [P_soplor_man]
for i, dist in enumerate(distancias_derivacion):
    puntos_nombres.append(f"Deriv.{i+1}")
    distancias.append(dist)
    presiones_man.append(presion_derivacion[i])
# Añadimos el final de la rama (dentro del tanque) para la última derivación
P_final_rama = presion_derivacion[-1] - perdida_tuberia(
    Q_rama_objetivo,
    L_rama_total,
    K_fijos_rama + Kv_valvula(theta_por_derivacion[-1]),
    dz_rama,
    h_sumergencia,
    es_hueco=True,
)
presiones_man.append(P_final_rama)
distancias.append(distancias_derivacion[-1] + 0.5)  # posición ficticia
puntos_nombres.append("Difusor (último tanque)")

# Convertir a cargas (metros de columna de aire)
z_puntos = [0] * len(
    distancias
)  # asumimos cota cero para el colector, la rama tiene dz pero ya incluido en pérdida
carga_presion = [(P_atm + P) / (rho_aire * g) for P in presiones_man]
carga_vel = []
for i, dist in enumerate(distancias):
    if i == 0:  # soplador
        v = Q_total / A
    elif i < len(distancias) - 1:  # derivaciones
        v = (
            Q_rama_objetivo * 2 * (N_deriv - (i - 1)) / A
        )  # caudal remanente en colector
    else:
        v = 0  # en el difusor la velocidad es casi nula
    carga_vel.append(v**2 / (2 * g))
HGL = [z_puntos[i] + carga_presion[i] for i in range(len(distancias))]
EGL = [HGL[i] + carga_vel[i] for i in range(len(distancias))]

plt.figure(figsize=(12, 5))
plt.plot(puntos_nombres, EGL, "ro-", label="EGL (Línea de Energía Total)")
plt.plot(puntos_nombres, HGL, "bs-", label="HGL (Gradiente Hidráulico)")
plt.xticks(rotation=45)
plt.ylabel("Carga (metros de columna de aire)")
plt.title("Perfil de energía a lo largo del colector y rama más alejada")
plt.legend()
plt.grid(True)
plt.show()
