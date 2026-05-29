import sys
from params import SectionParams
from models import Factores_Acople, run_modelo_mk1, run_modelo_mk2
from daq import run_daq_realtime, run_experimento_c

def main():
    """
    Función principal (Punto de entrada CLI).
    
    Inicia la interfaz de línea de comandos preguntándole al usuario 
    qué modo de simulación o adquisición ejecutar (MK1, MK2, DAQ en vivo, etc.) 
    y qué configuración física de hardware se utilizará (Bobina Pequeña o Grande).
    Inyecta estos parámetros centralizados a los módulos correspondientes.
    """
    while True:
        modelo = input(
            "Escriba (1) para mk_1 (lineal) o (2) para mk_2 (no lineal), "
            "(3) para ver voltaje en tiempo real, (4) para descargar datos crudos, "
            "o (5) para hallar el valor de 'c' experimentalmente: "
        ).strip()
        
        if modelo in ["1", "2", "3", "4", "5"]:
            break
        else:
            print("Entrada no válida. Inténtalo de nuevo.")

    while True:
        opcion_bobina = input(
            "\nSeleccione la configuración del sismómetro:\n"
            "(1) Bobina pequeña\n"
            "(2) Bobina grande\n"
            "Elección: "
        ).strip()
        if opcion_bobina in ["1", "2"]:
            break
        else:
            print("Entrada no válida. Inténtalo de nuevo.")

    print("\n--- INICIALIZANDO PARÁMETROS DEL SISMÓMETRO ---")
    parametros = SectionParams(tipo_bobina=opcion_bobina)

    print("\n--- INICIANDO INTEGRAL CAMPO MAGNETICO ---")
    G_sub_L, G_sub_A = Factores_Acople(parametros)

    print("\n--- INICIANDO SOLVER Y GRÁFICAS ---")
    
    if modelo == "1":
        run_modelo_mk1(parametros, G_sub_L, G_sub_A)
    elif modelo == "2":
        run_modelo_mk2(parametros, G_sub_L, G_sub_A)
    elif modelo == "3":
        # Modo 3: ver voltaje en tiempo real
        run_daq_realtime(parametros, G_sub_L, G_sub_A)
    elif modelo == "4":
        # Modo 4: descargar datos crudos. Este modo también usa la lógica de DAQ pero
        # la selección "solo guardar" o "ver gráficas" se pregunta internamente en daq.py.
        # Originalmente modelo == "4" sólo activaba datos_descarga = True.
        # En el código original, el modo 4 no tenía un bloque explícito en el Solver, 
        # sino que probablemente caía en una de las lógicas o simplemente 
        # marcaba una bandera. Al revisar el Solver original, el bloque `elif simular:` 
        # manejaba el modo 3. Si `datos_descarga` estaba en True (modo 4), no hacía 
        # nada en el Solver porque no había un `elif datos_descarga:`! 
        # Vamos a redirigirlo a `run_daq_realtime` que internamente pregunta si se quiere solo guardar.
        run_daq_realtime(parametros, G_sub_L, G_sub_A)
    elif modelo == "5":
        run_experimento_c(parametros, G_sub_L, G_sub_A)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEjecución cancelada por el usuario.")
        sys.exit(0)
