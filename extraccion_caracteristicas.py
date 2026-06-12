import os
import pandas as pd

# Importamos las herramientas que creamos en nuestro archivo anterior
from funciones_preprocesamiento import (
    extraer_metadatos,
    cargar_y_preprocesar_audio,
    butter_bandpass_filter,
    recortar_ciclo_cardiaco,
    calcular_energia_sistole
)

# =============================================================================
# 1. CONFIGURACIONES INICIALES 
# =============================================================================
# Ruta exacta donde están todos tus archivos mezclados en Windows. Acá pongan la ruta donde tienen todos los archivos de audio, texto y anotaciones juntos.
BASE_DIR = r'C:\Users\zieli\Desktop\Facultad\TPI_AUSCULTACION_CARDIACA\TPI_SE-ALES_AUSCULTACION\training_data' 

# Creamos una lista vacía donde vamos a ir guardando el resumen de cada paciente
RESULTADOS = []

# LÓGICA: Extraer IDs de pacientes directamente de los archivos sueltos
# Lee todos los archivos que terminan en '.wav'. 
# Luego, agarra el nombre (ej: "45843_MV.wav"), lo corta por el guion bajo ('_')
# y se guarda solo la primera parte ("45843"). Usa 'set' para borrar duplicados.
archivos_wav = [f for f in os.listdir(BASE_DIR) if f.endswith('.wav')]
pacientes = list(set([f.split('_')[0] for f in archivos_wav]))

print(f"Iniciando procesamiento pesado para {len(pacientes)} pacientes encontrados...\n")

# =============================================================================
# 2. BUCLE PRINCIPAL DE EXTRACCIÓN 
# =============================================================================
# Arrancamos a iterar paciente por paciente
for paciente_id in pacientes:
    
    # Armamos la ruta del archivo de texto de este paciente específico
    ruta_txt = os.path.join(BASE_DIR, f"{paciente_id}.txt")
    
    # Si por alguna razón un paciente no tiene su archivo de texto clínico, lo salteamos
    if not os.path.exists(ruta_txt):
        continue
            
    # --- PASO A: Leer la información clínica ---
    metadatos = extraer_metadatos(ruta_txt)
    murmur = metadatos['murmur']
    timing = metadatos['timing']
    
    # --- PASO B: Aplicar la lógica de selección de válvula ---
    if murmur == 'Present' and timing == 'Holosystolic':
        # Si tiene el soplo que buscamos, usamos la válvula donde suena más fuerte
        valvula = metadatos['most_audible']
        etiqueta = 'Soplo'
    elif murmur == 'Absent':
        # Si está sano, forzamos usar la Mitral (MV) para tener una línea base estable
        valvula = 'MV'  
        etiqueta = 'Sano'
    else:
        # Ignoramos casos 'Unknown' o soplos que no sean holosistólicos (pasamos al siguiente)
        continue
        
    # --- PASO C: Armar las rutas de los archivos pesados ---
    archivo_base = f"{paciente_id}_{valvula}"
    ruta_wav = os.path.join(BASE_DIR, f"{archivo_base}.wav")
    ruta_tsv = os.path.join(BASE_DIR, f"{archivo_base}.tsv")
    
    # Verificar que los archivos de sonido y anotación realmente existan en la carpeta
    if not (os.path.exists(ruta_wav) and os.path.exists(ruta_tsv)):
        continue
        
    # --- PASO D: Secuencia de Procesamiento de Señal ---
    # Usamos try/except para que si un audio está corrupto, no se caiga todo el programa
    try:
        # 1. Cargamos el audio en la memoria
        fs, senal_cruda = cargar_y_preprocesar_audio(ruta_wav)
        
        # 2. Filtramos el ruido general (dejamos solo de 40 a 500 Hz)
        senal_filtrada = butter_bandpass_filter(senal_cruda, 40.0, 500.0, fs)
        
        # 3. Miramos el .tsv y recortamos exactamente un ciclo cardíaco
        ciclo_completo, tiempos = recortar_ciclo_cardiaco(ruta_tsv, senal_filtrada, fs)
        
        # 4. Hacemos el espectrograma, aislamos la sístole y calculamos la energía
        energia_db = calcular_energia_sistole(ciclo_completo, fs, tiempos)
        
        # 5. Guardamos todos estos datos limpios en nuestra lista de resultados
        RESULTADOS.append({
            'ID_Paciente': paciente_id,
            'Valvula_Usada': valvula,
            'Energia_Sistole_dB': energia_db,
            'Diagnostico': etiqueta
        })
        
        # Imprimimos un mensaje de éxito en la consola para saber que el programa está avanzando
        print(f"OK: {paciente_id} | {etiqueta:5s} | Válvula: {valvula} | Energía: {energia_db:.2f} dB")
        
    except Exception as e:
        # Si falló la matemática o el recorte, te avisa por qué fue, pero sigue con el resto
        print(f"Error en paciente {paciente_id}: {e}")

# =============================================================================
# 3. EXPORTACIÓN DE RESULTADOS 
# =============================================================================
# Convertimos nuestra lista de resultados en una tabla estructurada (DataFrame) usando Pandas
df_resultados = pd.DataFrame(RESULTADOS)

# Definimos cómo se va a llamar el archivo final
nombre_archivo = 'dataset_caracteristicas.csv'

# Guardamos la tabla en formato CSV (index=False es para que no agregue una columna extra de números de fila)
df_resultados.to_csv(nombre_archivo, index=False)

# Aviso de finalización
print("\n" + "="*50)
print("¡PROCESAMIENTO FINALIZADO!")
print(f"Se extrajeron características de {len(df_resultados)} pacientes válidos.")
print(f"Archivo guardado exitosamente como: '{nombre_archivo}'")
print("="*50)