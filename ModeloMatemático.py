#Sección 1 en adelante (estudiante)
from dataclasses import dataclass, field #Librería de Dataclass para almacenamiento de variables
#Sección 3 en adelante (Gemini)
import numpy as np #Necesario para matrices y operaciones trigonométricas
from scipy import signal #Necesario para acoplar matriz de frecuencia externa F(t)
import matplotlib.pyplot as plt  #Necesario para graficar resultados

# -------------------------------------------------------------------
# 0. Selección del modelo:
#En esta sección determinamos el modelo a usar
#"modelo_mk1" corresponde a modelo con fuerza de Laplace integrada
#"modelo_mk2" corresponde a modelo sin ecuación electromagnética
# -------------------------------------------------------------------

while True:
    modelo = input("Escriba (1) para mk_1 o (2) para mk_2: ").strip()
    if modelo == "1":
        modelo_mk_1, modelo_mk_2 = True, False
        break # Sale del bucle y sigue el programa
    elif modelo == "2":
        modelo_mk_1, modelo_mk_2 = False, True
        break
    else:
        print("Entrada no válida. Inténtalo de nuevo.")


# -------------------------------------------------------------------
# 1. Parámetros de sección:
#En esta sección guardamos las variables que dependen directamente del diseño (1er prden)
#Y en base a ellas, computamos las de que son combinaciones de las anteriores (2do Orden)
#Todas las unidades anteriores han de estar en el S.I
# -------------------------------------------------------------------

#Notas
# -------------------------------------------------------------------
#Tablas de viscocidad dinámica (Kg/m*s)
#Aire=1.81*(1e-6)
#Silicona 100cts=97*(1e-3)
#Silicona 150cts=1455*(1e-4)
#Silicona 1.000cts=1.1
#Silicona 10.000cts=9.7
#Silicona 12.500cts=11.64
# -------------------------------------------------------------------
#Tablas de diámetros/vueltas y longitud del resorte (Kg/m*s)
#resorte pequeño:
#vueltas:18
#d_sección_transversal:75*(1e-5) metros
#L_libre_iman=135*(1e-3) metros
# ----------------
#resorte mediano:
#vueltas: 15
#d_sección_transversal: 75*(1e-5) metros
#L_libre_iman=1125*(1e-4) metros
# ----------------
#1.5*(1e-3) Kg (resorte pequeño)
#3.4*(1e-3) Kg (resorte mediano)
#k= 300 (resorte pequeño)
#k= 104 (resorte mediano)
 

@dataclass
class SectionParams:

    # -------------------------------------------------------------------
    #1er Orden: 
    # -------------------------------------------------------------------
    
    #Condiciones Climáticas

    temp: float = 299.15 #Temperatura del ambiente en Kelvin (26 grados celsiud)

    #Sección-Masa-Resorte-Amoprtiguamiento:

    k: float =  300                           #Constante Elástica del Resorte (kg/s^2)
    m =  1.5*(1e-3)                                #Masa de la esfera (Kg)
    g: float =  9.77                           #Constante gravitacional
    eta: float = 1.1                          #Viscocidad del fluido (kg/m*s)
    R_sub_e: float = 9.5*(1e-3)                      #Radio de la Esfera
    L_cilindro: float = 10*(1e-3)                          #Longitud cilindro
    L_libre_iman: float = 135*(1e-3)                                  #Longitud libre del imán a usar
    omega: float = 628.3                            #Frecuencia de vibración de la mesa (Radianes)

    #Sección Mecánica:

    e_sub_p:  float =  3*(1e-3)
    e_sub_p:  float =  3*(1e-3)                     #Grosor del Contenedor de PLA
    h_sub_p:  float =  36*(1e-3)                     #Altura del contenedor de PLA 
    r_sub_p:  float =  14*(1e-3)                        #Radio del Contenedor de PLA
    h_sub_f:  float =  34*(1e-3)                         #altura del fluido
    g_sub_ecs: float = 0.015*(1e-3)                          #Grosor de la capa de esmalte
    e_sub_cs: float = 0.118*(1e-3)                     #Grosor del Cable del solenoide (diámetro) #Usamos un AGW 38 como estimación
    N_sub_c_capas= 5                                 #Número de capas de vueltas de cable
    densidad_neodimio=7500                             #Densidad del Neodimio N35 en Kg/m^3                          #
    b: float = 6750                                    #Temperatura de delta (Grados Kelvin

    #Sección Electromagnética

    chi_sub_N: float = 8*(1e+8)                 #Coeficiente para el momento magnético del Neodimio N35
    chi_sub_A: float = 5*(1e+8)                 #Coeficiente para el momento magnético del Acero Inoxidable 440 C
    Q: float = 0                                #Carga inicial
    Q_dot: float = 0                            #Corriente inicial

    #Configuraciones Reales
    R = 383.1                                      #Resistencia Real 
    L = 36.7*(1e-6)                                #Inductancia Real
    C = 0.07*(1e-6)                                #Capacitancia Real

    #Sección de Resistencia

    e_sub_c: float = 17*(1e-3)                           #Grosor del Cable de conexión (diámetro)
    L_sub_c: float = 5*(1e-2)                          #Longitud del cable de conexión
    R_sub_a: float =  1*(1e+8)                          #Resistencia de entrada del amplificador de señal
    R_extra: float = 1200                               #Resistencia añadida al circuito

    #Sección de Capacitancia

    epsilon_sub_cero: float = 8.854*(1e-12)     #Permitividad del vacío
    epsilon_sub_e: float =  2.5                 #Permitividad del esmalte del cable de cobre 
    p_sub_cu: float = 1.72*(1e-8)               #Resistividad del cobre a 20 ºC

    #Sección de Inductancia

    mu_sub_cero: float = (4)*(np.pi)*(1e-7)      #Permeabilidad del vacío
    mu_sub_PLA: float =1                         #Permeabilidad del PLA
    mu_sub_f:float=1                             #Permeabilidad del fluido

    #Sección de Control

    a:int = 0                                      #Límite inferior de graficación
    b:int = 5                                      #Límite superior de graficación
    puntos:int = 1000                              #Densidad de la línea
    delta_t: int = 1                           #tiempo en segundos que el sensor se tarda en tomar datos

    # -------------------------------------------------------------------
    #2do Orden: (Almacenamiento)
    # -------------------------------------------------------------------

    #Sección-Masa-Resorte-Amoprtiguamiento:

    m:float = field(default=m, init=False)
    c_sub_Stokes_resorte: float = field(init=False)
    c_sub_Stokes_iman: float = field(init=False)
    c_sub_Stokes: float = field(init=False)
    lambda_c: float = field(init=False)
    c_sub_lambda: float = field(init=False)
    c: float = field(init=False)
    F_O: float = field(init=False)
    Y_0_O: float = field(init=False)
    omega_sub_n: float = field(init=False)
    Zeta: float = field(init=False)
    p_error_Zeta: float = field(init=False)
    omega_sub_d: float = field(init=False)
    B: float = field(init=False)
    root_root: float = field(init=False)
    root_1: float = field(init=False)
    root_2: float = field(init=False)
    Zeta_sub_omega: float = field(init=False)
    omega_sub_s: float = field(init=False)
    phi: float = field(init=False)  
    X: float = field(init=False)
    

    #Sección Mecánica:

    e_total: float = field(init=False)
    L_sub_s: float = field(init=False)
    A_sub_s: float = field(init=False) 
    A_sub_cs: float = field(init=False) 
    A_sub_c: float = field(init=False) 
    delta_h_sub_cs: float = field(init=False)                                                                
    h_sub_cs: float = field(init=False)
    N_sub_c:  int = field(init=False)                           

    #Sección Electromagnética
    
    pregunta_1:int = field(init=False) 
    control:int = field(init=False)
    m_mag:float = field(init=False) 

    #Sección de Resistencia

    R_sub_s: float = field(init=False) 
    R_sub_c: float = field(init=False) 
    R_porcentaje: float = field(init=False)
    R: float = field(default= R, init=False)

    #Sección de Capacitancia

    C: float = field(default= C, init=False)
    C_inicial: float = field(init=False)
    C_N_sub_cs: float = field(init=False)
    C_N_sub_cs_capas: float = field(init=False)
    XC_capacitiva:float= field(init=False)
    
    #Sección de Inductancia
    
    mu_prom: float= field(init=False)
    L: float = field(default= L, init=False)
    XC_inductiva:float= field(init=False)

    #Frecuencias circuito RLC

    XC:float= field(init=False)
    alpha: float = field(init=False) 
    omega_sub_0_phi_m: float = field(init=False)
    
    #Sección de Control

    t: np.ndarray = field(init=False) 

    # -------------------------------------------------------------------
    #2do Orden: (Computación)
    # -------------------------------------------------------------------

    def __post_init__(self):

#Pregunta para definir parámetros directos o estimados teóricamente
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
        teoric_params= bool(input("¿Desea usar los parámetros con los datos teóricos? "
        "(Enter) para sí y escriba algo para no"))
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
            
#Llamado a definiciones de cálculo teóricas:
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
        if(teoric_params==True):
            self.masa_teorico()
            self.amortiguamiento_teorico()
            self.capacitancia_teorico()
            self.resistencia_teorico()
            self.inductancia_teorico()
            self.m_mag()
            
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
        
#Cálculo de la fuerza Inicial
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
        fuerza=int(input("¿La fuerza es directa (1) o se calcula a partir de un MAS (2)?"))
        if fuerza==1:
            self.F_0 = 7                              #Fuerza en Newtons de la mesa
        elif fuerza==2:
            amplitud=int(input("¿La amplitud es directa (1) o se calcula en base a la frecuencia(2)?"))
            if  amplitud==1:   
                self.Y_0= 7*(1e-3)                    #Amplitud de vibración en metros
                self.F_0 = self.m*self.Y_0*(self.omega**2) #Amplitud derivada de un MAS 
            elif amplitud==2:
                #COMPLEMENTAR
                """""
                self.F_0 =self.F_0 
                #Función de la amplitud en base a frecuencia (Hz)
                """
            else: 
                ValueError("No se ingresó una opción válida, oprima (1) o (2)")
        else:
            ValueError("No se ingresó una opción válida, oprima (1) o (2)")
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

#Cálculo de parámetros de la EDO Mecánica
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
        self.omega_sub_n = ((self.k/self.m)**0.5)                                      #Frecuencia Natural (Hz)
        self.Zeta = self.c/(2*((self.k*self.m)**0.5))                                  #Factor de Amortiguamiento
        self.p_error_Zeta = self.Zeta*0.1                                              #Factor de tolerancia para elegir la naturaleza del sistema
        self.omega_sub_d =  self.omega_sub_n*(1-self.Zeta**2)**0.5                     #Frecuencia Natural Amortiguada
        self.B = self.Zeta*self.omega_sub_n*self.A/self.omega_sub_d                    #Condición Inicial compuesta
        self.root_root = (4*self.m*self.k-(self.c**2))**0.5                            #Bloque de raíz para hallar soluciones
        self.root_1 = ((-self.c-self.root_root)*0.5)/self.m                            #Raíz uno de las soluciones
        self.root_2 = ((-self.c+self.root_root)*0.5)/self.m                            #Raíz dos de las soluciones
        self.Zeta_sub_omega = (2*self.Zeta*self.omega)/self.omega_sub_n                #Bloque de amortiguamiento
        self.omega_sub_s = 1-(self.omega/self.omega_sub_n)**2                          #Bloque de Frecuencias
        self.phi = np.arctan(self.Zeta_sub_omega/self.omega_sub_s)                     #Desfase entre la mesa y el sistema masa-resorte-amortiguador
        self.X = self.F_0/(self.k*(self.Zeta_sub_omega**2+self.omega_sub_s**2)**0.5)   #Constante para solución particular
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

#Sección Mecánica:
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
        self.N_sub_c= round(self.h_sub_p/self.e_sub_cs)                                #Número de vueltas del Cable
        self.e_total= (self.r_sub_p+self.e_sub_p+(0.5*self.e_sub_cs))                  #Radio total del Solenoide
        self.L_sub_s = self.N_sub_c*(self.N_sub_c_capas*(self.e_total*2*np.pi))        #Longitud del cable solenoide
        self.A_sub_s = (self.N_sub_c*(self.N_sub_c_capas*
        (4*(np.pi**2))*(self.e_total*self.e_sub_cs)))                                  #Área del cable de solenoide
        self.A_effec= (2*self.e_sub_cs*np.pi*(self.r_sub_p+self.e_sub_p)
        *self.N_sub_c*self.N_sub_c_capas)                                              #Área efectiva de contacto con campo magnético en el eje z
        self.A_sub_cs = np.pi*((self.e_sub_cs*0.5)**2)                                 #Área de corte de cable solenoide
        self.A_sub_c = np.pi*((self.e_sub_c*0.5)**2)                                   #Área de corte de cable de conexión
        self.delta_h_sub_cs = self.e_sub_cs                                            #Altura del cable de solenoide en una vuelta
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

#Cálculo de Parámetros para el circuito RLC
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
        self.XC_inductiva=(self.L*self.omega)                             #Reactancia Inductiva
        self.XC_capacitiva=(-1/(self.omega*self.C))                       #Reactancia Capacitiva
        self.R_porcentaje = (self.R_sub_a/(self.R))                       #Resistencia para ley de faraday lenz
        self.alpha = self.R/(2*self.L)                                    #Inercia Eléctrica (Velocidad de consumo de corrientes parásitas)
        self.XC=(self.XC_capacitiva+self.XC_inductiva)                    #Reactancia total (parte imaginaria)
        self.omega_sub_0_phi_m = (1/((self.L*self.C)**0.5))               #Frecuencia de Resonancia (Frecuencia natural de oscilación del circuito)
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

#Sección de Control
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
        self.t= np.linspace(self.a, self.b, self.puntos)                   #Vector de variable continua (tiempo)
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------



#Definiciones Teóricas
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
    def masa_teorico(self):
        self.m=(self.R_sub_e)**3*np.pi*1.3333*self.densidad_neodimio
#-----------------------------------------------------------------------------------------------
    def amortiguamiento_teorico (self):
        #Cálculo de amortiguamiento con simplificación de Navier Stokes
        self.c_sub_Stokes_iman = 6*np.pi*self.eta*self.R_sub_e                 #Coeficiente c para una esfera
        self.c_sub_Stokes_resorte= np.abs(4*np.pi*self.eta*
        self.L_cilindro/(np.log(self.L_cilindro/2*self.R_sub_e)+0.5))              #Coeficiente c para un cilindro (resorte recreado)

        self.lambda_c = (self.R_sub_e/self.r_sub_p)                                    #Factor de lejanía (Habermann Faxen)
        if (self.lambda_c > 0.6):

             #Factor de corrección de cercanía (Haberman/Sayre)

            den= 1 - (0.75857*self.lambda_c**5) 
            num= 1-(2.10444*self.lambda_c) + (2.08877 * self.lambda_c**3) - (0.94813*self.lambda_c**5)
            self.c_sub_lambda= num/den

        elif (self.lambda_c <= 0.6):

            #Factor de corrección de cercanía (Haberman/Faxen)  
               
            self.c_sub_lambda = ((1-(2.104*(self.lambda_c)) +
            (2.089*(self.lambda_c**3))-0.948*(self.lambda_c**5))) 
                      
            #coeficiente de amortiguamiento final              
            self.c= (self.c_sub_Stokes_iman+self.c_sub_Stokes_resorte)/self.c_sub_lambda    

#--------------------------------------------------------------------------------------------------
    def capacitancia_teorico(self):

        #Capacitancia de 2 cilindros paralelos (cables)
        self.C_inicial=((self.L_sub_s*self.epsilon_sub_cero*self.epsilon_sub_e)/
        (np.log((self.g_sub_ecs+self.e_sub_cs)/(self.e_sub_cs))))

        #Añadimos número de espiras                 
        self.C_N_sub_cs= 2*self.C_inicial/self.N_sub_c

        #Añadimos número de capas                                                                                       
        self.C=self.C_N_sub_cs*self.N_sub_c_capas  

#--------------------------------------------------------------------------------------------------
    def inductancia_teorico(self):
        
        #Permeatividad equivalente
        self.mu_prom=((self.mu_sub_cero/(self.r_sub_p+self.e_sub_p))*
            ((self.mu_sub_f*self.r_sub_p)+(self.e_sub_p*self.mu_sub_PLA)))
        
        #Inductancia del Solenoide                          
        self.L= self.mu_prom*self.A_sub_cs*self.N_sub_c*self.N_sub_c_capas/self.h_sub_p 

#--------------------------------------------------------------------------------------------------
    def resistencia_teorico(self):

        #Resistencia del solenoide
        self.R_sub_s = self.p_sub_cu*self.L_sub_s/self.A_sub_cs

        #Resistencia de la conexión                                 
        self.R_sub_c = self.p_sub_cu*self.L_sub_c/self.A_sub_c

        #Resistencia Total del sistema                                  
        self.R = ((self.R_sub_a*(self.R_sub_s + self.R_sub_c+self.R_extra)) /
               (self.R_sub_a + self.R_sub_s + self.R_sub_c+ self.R_extra))    
                                   
#--------------------------------------------------------------------------------------------------
    def m_mag(self):

        #Preguntamos el material del imán para estimar el momento magnético
        tipo_iman=int(input("¿El imán es de neodimio (1) o acero 440 (2)?"))   
        if (tipo_iman==2):

            #Estimación para Acero 440C                                             
            self.m_mag = self.chi_sub_A*((2*self.R_sub_e)**3)

        elif (tipo_iman==1):
            #Estimación para Neodimio N32
            self.m_mag = self.chi_sub_N*((2*self.R_sub_e)**3)                
        else:
            ValueError("Escriba (1) o (2), otra repuesta no es válida")
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------- 


def Factores_Acople (params:SectionParams)-> float: 

    #Planetamos una integral de promedios para el producto cruz de la fuerza de Laplace
    #La integral ha sido planetada de forma analítica

    #Desempaquetamos variables de section params
    e_sub_p=params.e_sub_p
    h_sub_p=params.h_sub_p
    r_sub_p=params.r_sub_p
    L_sub_s=params.L_sub_s
    A_sub_s=params.A_sub_s
    mu_prom=params.mu_prom
    m_mag=params.m_mag
    A_effec=params.A_effec

    #Precomputamos los bloques de la integral 
    bloque_constante=r_sub_p+e_sub_p
    bloque_variable_lim_simétrico=h_sub_p*0.5
    bloque_raíz=((bloque_variable_lim_simétrico)**2+(bloque_constante)**2)**0.5


    #Añadimos constantes físicas
    constantes_magneticas=(0.75*m_mag*mu_prom)/np.pi

    #Campo magnético para fuerza de Laplace
    B_x=(1/(bloque_raíz*bloque_constante))*constantes_magneticas

    #Campo Magnético para el flujo
    dB_z=((bloque_constante*h_sub_p)/(bloque_raíz**5))*constantes_magneticas
    dB_x=(0.5*(bloque_constante**2)/(bloque_raíz**5))*constantes_magneticas


    #d_phi/d_t :Acople G_sub_A:

    G_sub_A=A_effec*(dB_x+dB_z)
    
    #LxB_sub_x :Acople G_sub_L:

    G_sub_L=B_x*L_sub_s

    return G_sub_L, G_sub_A  

# -------------------------------------------------------------------
# 2. Solver de EDO´s:
#El usuario puede resolverlo tanto con la parte electromagnética (mk1)
#O Con la parte mecánica solamente (mk2)
# -------------------------------------------------------------------

def Solver (modelo_mk1:bool, modelo_mk2:bool, params:SectionParams)-> float:

    if (modelo_mk1==True) :

        print("--- Entrando a Modelo MK1 ---")

        """Hace falta definir las variables de derivción y posicionamiento"""
        """Hace falta definir las variable G, que es igual al producto cruz entre vector longitud (u_vector) y vector campo magnético"""
        #Procedemos a computar las EDO´s como un sistema lineal con la teoría de eigenalores y eigenvectores

        #Desempaquetado de variables de Section Params
        m=params.m
        c=params.c
        c_sub_lambda=params.c_sub_lambda
        k=params.k
        L=params.L
        C=params.C
        z=params.A 
        z_dot=params.B
        Q=params.Q
        Q_dot=params.Q_dot
        t=params.t
        F_0=params.F_0
        phi=params.phi
        omega=params.omega
        delta_t=params.delta_t
        G_sub_L, G_sub_A=Factores_Acople(params)
        XC=params.XC
        R=params.R
        R_porcentaje=params.R_porcentaje
        Zeta=params.Zeta
        alpha=params.alpha
        omega_sub_0_phi_m=params.omega_sub_0_phi_m
        R_sub_e=params.R_sub_e
        r_sub_p=params.r_sub_p
        eta=params.eta

# Prints de Parámetros Base:
#--------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------- 
        print(f"masa estimada en :{m} Kg")                              #Masa   
        print(f"El amortiguamiento auxiliar es:{c_sub_lambda}")         #Amortiguamiento Auxiliar
        print(f"La constante de amortiguamiento  es de :{c} Kg/s")      #Amortiguamiento
        print(f"La constante elástica es de :{k} Kg/s^2")               #Constante Elástica
        print(f"La resistencia es de :{R} Ohmnios")                     #Resistencia
        print(f"La inductancia es de :{L} H")                           #Inductancia
        print(f"La capacitancia es de :{C} F")                          #Capacitancia
        print(f"Momento magnético configurado en: {self.m_mag}")        #Momento Magnético
        print (f"El factor de Magnético de Área es: {G_sub_A}")         #Factor Magnético de Área
        print (f"El factor de Magnético de Área es: {G_sub_L}")         #Factor Magnético de Longitud
        print(f"El factor Z (amortiguamiento mecánico) es de: {Zeta}") 
#--------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------- 

        #A: Matriz de acople de constantes
        A = np.array([
        [0, 1, 0, 0],
        [-k/m, -c/m, 0, -G_sub_L/m],
        [0, 0, 0, 1],
        [0, -G_sub_A/L, -1/(L*C), -(R)/(L)]
        ])

        # 3. Encuentra Eigenvalos (w) y Eigenvectores (v)
        eigenvalues, eigenvectors = np.linalg.eig(A)

        # Matriz B: La fuerza externa F(t) afecta a la aceleración (z_dot_dot)
        F_t = np.array([[0], [1/m], [0], [0]])
        # Calculamos la solución particular como la parte imaginaria de una exponencial
        I = np.eye(4)
        lado_izquierdo = 1j * omega * I - A
        lado_derecho = F_t * F_0 

        # Xp_complejo contiene la amplitud y fase de [z, z_dot, Q, Q_dot]
        Xp_complejo = np.linalg.solve(lado_izquierdo, lado_derecho)

        # --- 3. Generar la evolución de la Particular ---
        # x_p(t) = Re( Xp * exp(j * omega * t) )
        # Usamos np.outer para multiplicar el vector Xp por el vector de tiempo exponencial
        X_particular = np.zeros((4, len(t)), dtype=complex)
        X_particular = Xp_complejo * np.exp(1j * omega * t)


       

        #Resuelve en base al vector de condiciones iniciales/variables
        
        X_t_p = Xp_complejo.real                               #Defininimos una condición inicial particular para el vector de condiciones iniciales
        X_t_h= np.array([[z, z_dot, Q, Q_dot]]).reshape(4,1)   #Defininimos una condición inicial homogénea para el vector de condiciones iniciales
        X_t= X_t_h-X_t_p                                       #: X_t: Vector de variables (parte homogénea + particular)

        coeficientes = np.linalg.solve(eigenvectors, X_t)

        #Creamos una matriz para guardar evolución con respecto al tiempo

        X_evolucion = np.zeros((4, len(t)), dtype=complex)

        for i in range(len(eigenvalues)):
            # Usamos 'coeficientes' en lugar de 'c' para no confundir con amortiguamiento
            # Contribución: c_i * v_i * e^(lambda_i * t)
            termino_exponencial = np.exp(eigenvalues[i] * t)
            contribucion = np.outer(eigenvectors[:, i], termino_exponencial)
            X_evolucion += coeficientes[i] * contribucion

        

        # Sumamos condiciones iniciales y finales
        X_total = X_evolucion + X_particular

        #Damos los parámetros de construcción

        

        delta_electromagnético=alpha-omega_sub_0_phi_m

        print(f"El delta entre amortiguamientos electromagnéticos (alpha-omega_sub_0_phi_m) de: {delta_electromagnético}")


      
        #Listas de colores y de nombres para las gráficas

        nombre_variables = ["Posición (z)", "Velocidad (ż)", "Carga (Q)", "Corriente (I)"]
        unidades = ["[m]", "[m/s]", "[C]", "[A]"]
        paleta_colores = ["orange", "red", "blue", "green"]

        # 3. Configuración de la figura
        fig, axs = plt.subplots(2, 2, figsize=(7.5, 5), dpi=200)
        fig.suptitle(f"Respuesta Temporal Completa - Detector de Gaia MK2", fontsize=16, fontweight='bold')

        for i in range(4):
            ax = axs[i // 2, i % 2]
        
        
            # Graficamos
            ax.plot(t, X_total.real[i] , color=paleta_colores[i], linewidth=2, label="Solución Total (Real)")
            ax.plot(t, X_particular.real[i] , '--', color='gray', alpha=0.4, label="Solo Estacionario")
        
            ax.set_title(nombre_variables[i], fontweight='bold')
            ax.set_ylabel(f"Amplitud {unidades[i]}")
            ax.set_xlabel("Tiempo [s]")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='upper right', fontsize='small')
        
            # Anotación del valor pico estacionario
            ax.text(0.95, 0.02, f"Pico Estac: {np.abs(Xp_complejo[i].item()):.2e}", transform=ax.transAxes, 
                    ha='right', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
            
            # Anotar valores máximos 
            ax.text(0, 0.02,f"Máx: {np.max(X_total.real[i]):.2e}",transform=ax.transAxes, 
                    ha='left', fontsize=7, bbox=dict(facecolor='white', alpha=0.7))



        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        #Líneas de Desfase

        plt.figure(figsize=(8,6), dpi=200)
        for i in range (len(X_total)):
            linea_desfase = np.full_like(t, np.angle(Xp_complejo[i]))
            plt.plot(t,linea_desfase, label= f"Desfase de {nombre_variables[i]}", color=paleta_colores[i])

        plt.title("Comparativa de Desfases")
        plt.xlabel("Tiempo")
        plt.ylabel("Desfases")
        plt.legend() 
        plt.show()


        #Frecuencias naturales del sistema acoplado, ya no habrá una, sino 4 por la cantidad de eignevalores
        # Los eigenvalores vienen en pares complejos conjugados
        # Tomamos la parte imaginaria y el valor absoluto para tener la frecuencia
        frecuencias_naturales = np.abs(eigenvalues.imag)

        # Como hay 4 eigenvalores, verás dos valores repetidos (los pares conjugados)
        # Eliminamos duplicados para ver las dos frecuencias principales del sistema
        frecuencias_unicas = np.unique(frecuencias_naturales)

        print(f"Frecuencias naturales del sistema acoplado (rad/s): {frecuencias_unicas}")

        #Computamos el voltaje de Salida partiendo de la ley de Ohm Fasorial (V=ZR)
        I_Re=X_total.real[3]
        I_Im=X_total.imag[3]
        V_Im=((R*I_Im)+(I_Re*XC))
        V_Re=((R*I_Re)-(I_Im*XC))
        V_A=((V_Im**2)+(V_Re**2))**0.5
        V_phi=np.arctan(V_Im/V_Re)
        FEM_Ohm=V_A*np.cos((omega*t)+V_phi)

        #Computamos voltaje de salida en base a Faraday-Lenz
        FEM_FL=G_sub_A*R_porcentaje*X_total.real[1]

        #Hacemos el tiempo discreto para simular el envío del sensor

        t_filtro= t[::delta_t]

        #Mostramos una gráfica del voltaje teórico con la corriente de la misma EDO que ha de recopilar el programa del modelo físico 

        plt.figure(figsize=(8,6), dpi=200)
        plt.step(t_filtro, FEM_Ohm, where='post', label='Muestreo (Escalonado) con base al tiempo de recolección de datos', color="purple", linewidth=1)
        plt.title("FEM teórica producida por Ley de Ohm Fasorial")
        plt.xlabel("Tiempo")
        plt.ylabel("FEM")
        plt.legend() 
        plt.show()

        #Mostramos una gráfica del voltaje teórico con el voltaje estimado con Faraday Lenz
        plt.figure(figsize=(8,6), dpi=200)
        plt.step(t_filtro, FEM_FL, where='post', label='Muestreo (Escalonado) con base al tiempo de recolección de datos', color="purple", linewidth=1)
        plt.title("FEM teórica producida por Faraday_Lenz")
        plt.xlabel("Tiempo")
        plt.ylabel("FEM")
        plt.legend() 
        plt.show()

        F_Laplace=X_total.real[3]*G_sub_L

        #Mostramos una gráfica del voltaje teórico con el voltaje estimado con Faraday Lenz
        plt.figure(figsize=(8,6), dpi=200)
        plt.step(t, F_Laplace, where='post', label='Fuerza de repulsión electromagnética', color="blue", linewidth=1)
        plt.title("Fuerza de Laplace")
        plt.xlabel("Tiempo")
        plt.ylabel("Newtons")
        plt.legend() 
        plt.show()



        v_max=np.abs(np.max(X_total.real[1]))
        Re = (v_max*(r_sub_p-R_sub_e))/(eta)                                   #estimación del número de Reynolds

        if Re<0.1:
            print(f"Linealidad de la constante de amortiguamiento fiable. Valor de Reynolds:{Re}")
        elif 0.1<Re<0.9:
            print(f"Linealidad de la constante de amortiguamiento variable. Valor de Reynolds:{Re}")
        else: 
            print(f"La constante de amortiguamiento no es lineal, se necesita RK45. Valor de Reynolds:{Re}")

        #Estimación de Solidworks
        V_solid=(4.333e+3)*G_sub_A*R_porcentaje
        print(f"El voltaje estimado por solidworks es de: {V_solid}")

#MOODELO MK II 

    elif (modelo_mk_2==True):
        #Procedemos a modelar la EDO como una solución de una parte homogénea y otra particular

        #Desempaquetado de variables de Section Params
        m=params.m
        c=params.c
        k=params.k
        p_error_Zeta=params.p_error_Zeta
        omega_sub_n=params.omega_sub_n
        Zeta=params.Zeta  
        A=params.A
        B=params.B
        omega_sub_d=params.omega_sub_d
        X=params.X
        phi=params.phi
        omega=params.omega
        root_1=params.root_1
        root_2=params.root_2
        t = params.t

        #Planteamos la solución particular que es universal

        solucion_particular= X*np.sin((omega*t)+phi)

        if(Zeta + p_error_Zeta < 1):
            print("La solución del sistema es subamortiguada, la solución oscila")
            print(f"Factor de amortiguamiento es de: {Zeta}")
            # Ecuación corregida (paréntesis y Zeta incluidos)
            solucion_homogenea = np.exp(-Zeta * omega_sub_n * t) * (A * np.cos(omega_sub_d * t) + B * np.sin(omega_sub_d * t))
            caso = "subamortiguado"

        elif(Zeta + p_error_Zeta > 1):
            print("La solución del sistema es sobreamortiguada, la solución decae lentamente")
            print(f"Factor de amortiguamiento es de: {Zeta}")
            solucion_homogenea = np.exp(root_1 * t) * A + np.exp(root_2 * t) * B
            caso = "sobreamortiguado"

        else:
            print("La solución del sistema es críticamente amortiguada, la solución decae")
            print(f"Factor de amortiguamiento es de: {Zeta}")
            solucion_homogenea = np.exp(-omega_sub_n * t) * (A + B * t)
            caso = "críticamente amortiguado"

        #Combinamos la solución particular y la homogénea

        solucion_total=solucion_homogenea+solucion_particular

        # Aumentamos el tamaño (12x8) y los DPI para calidad de publicación
        plt.figure(figsize=(12, 8), dpi=120)
        
        # Graficamos respetando tus colores, etiquetas y estilos de línea
        plt.plot(t, solucion_homogenea, label=f"Solución Homogénea ({caso})", color="orange", linestyle="--", linewidth=1.5, alpha=0.5)
        plt.plot(t, solucion_particular, label="Solución particular (Estacionaria)", color="green", linestyle="-.", linewidth=1.5, alpha=0.8)
        plt.plot(t, solucion_total, label="Solución total (Transitorio + Estacionaria)", color="purple", linewidth=2.5, alpha=0.5)
        
        # Títulos y Ejes con formato
        plt.title("Respuesta Temporal del Sismómetro (Modelo MK2)", fontsize=16, fontweight="bold", pad=15)
        plt.xlabel("Tiempo [s]", fontsize=12, fontweight="bold")
        plt.ylabel("Desplazamiento [m]", fontsize=12, fontweight="bold")
        
        # Leyenda y Cuadrícula
        plt.legend(loc="upper right", fontsize=11, shadow=True, fancybox=True)
        plt.grid(True, linestyle=":", alpha=0.7)
        
        # EXTRA: Cuadro de texto con el desplazamiento máximo absoluto
        amp_max = np.max(np.abs(solucion_total))
        plt.text(0.95, 0.05, f"Pico Máximo: {amp_max:.2e} m", transform=plt.gca().transAxes, 
                 ha='right', va='bottom', fontsize=11, bbox=dict(facecolor='white', alpha=0.8, edgecolor='purple'))

        # Ajuste final y visualización
        plt.tight_layout()
        plt.show()



# ===================================================================
# 3. EJECUCIÓN PRINCIPAL DEL PROGRAMA
# ===================================================================

print("\n--- INICIALIZANDO PARÁMETROS DEL SISMÓMETRO ---")
# Esto disparará el __post_init__ y las preguntas del imán
parametros = SectionParams() 

print("\n--- INICIANDO INTEGRAL CAMPO MAGNETICO ---")
# Esto ejecuta tu matemática y abre las ventanas de Matplotlib
Factores_Acople(parametros)


print("\n--- INICIANDO SOLVER Y GRÁFICAS ---")
# Esto ejecuta tu matemática y abre las ventanas de Matplotlib
Solver(modelo_mk_1, modelo_mk_2, parametros)