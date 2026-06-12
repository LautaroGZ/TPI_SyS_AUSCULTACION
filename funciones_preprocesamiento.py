import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, spectrogram

# =============================================================================
# HERRAMIENTA 1: LECTOR DE HISTORIAS CLÍNICAS (METADATOS)
# =============================================================================
def extraer_metadatos(ruta_hea):
    """
    Abre el archivo de texto (.txt o .hea) del paciente y extrae la información clave:
    si tiene soplo, qué tipo de soplo es, y en qué válvula suena más fuerte.
    Si el paciente está sano, asigna la válvula Mitral (MV) por defecto para estandarizar.
    """
    # Forma de datos por defecto (asumimos que está sano y usamos la válvula Mitral)
    metadatos = {
        'most_audible': 'MV',  
        'murmur': 'Absent',
        'timing': 'nan'
    }
    
    # Si el archivo no existe, devolvemos la forma de datos por defecto para no frenar el programa
    if not os.path.exists(ruta_hea):
        return metadatos

    # Abrimos el texto y buscamos las palabras clave línea por línea
    with open(ruta_hea, 'r', encoding='utf-8') as f:
        for linea in f:
            linea = linea.strip() # Limpiamos espacios en blanco
            
            # Buscamos la válvula de mayor intensidad
            if linea.startswith('#Most audible location:'):
                valor = linea.split(':')[-1].strip()
                if valor != 'nan':
                    metadatos['most_audible'] = valor
                    
            # Buscamos si tiene o no tiene soplo
            elif linea.startswith('#Murmur:'):
                metadatos['murmur'] = linea.split(':')[-1].strip()
                
            # Buscamos en qué momento ocurre el soplo (ej: Holosystolic)
            elif linea.startswith('#Systolic murmur timing:'):
                metadatos['timing'] = linea.split(':')[-1].strip()
                
    return metadatos


# =============================================================================
# HERRAMIENTA 2: CARGA Y LIMPIEZA INICIAL DEL AUDIO
# =============================================================================
def cargar_y_preprocesar_audio(ruta_wav):
    """
    Carga el archivo de audio (.wav), lo pasa a mono (un solo canal) si es estéreo,
    y convierte los números a decimales largos (float64) para evitar errores de memoria.
    """
    # Leemos el archivo. fs = Frecuencia de muestreo (ej: 4000 muestras por segundo)
    fs, senal_cruda = wavfile.read(ruta_wav)
    
    # Si la señal tiene más de una dimensión (estéreo), nos quedamos solo con la primera
    if len(senal_cruda.shape) > 1:
        senal_cruda = senal_cruda[:, 0]
        
    # Convertimos los datos a float64 para que la computadora no se sature al hacer cálculos
    senal_cruda = senal_cruda.astype(np.float64)
    
    return fs, senal_cruda


# =============================================================================
# HERRAMIENTA 3: FILTRO DIGITAL PASABANDA 
# =============================================================================
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Filtro pasabanda que elimina ruidos muy graves (movimiento) y muy agudos (interferencia).
    Usa 'filtfilt' para filtrar hacia adelante y hacia atrás, evitando que la onda se desplace en el tiempo.
    """
    # Frecuencia de Nyquist (es el límite matemático, siempre es la mitad de fs)
    nyq = 0.5 * fs
    
    # Normalizamos las frecuencias de corte para que SciPy las entienda
    low = lowcut / nyq
    high = highcut / nyq
    
    # Diseñamos el filtro Butterworth de orden 4
    b, a = butter(order, [low, high], btype='band')
    
    # Aplicamos el filtro de fase cero (filtfilt) a nuestros datos de audio
    return filtfilt(b, a, data)


# =============================================================================
# HERRAMIENTA 4: RECORTADOR DEL CICLO CARDÍACO 
# =============================================================================
def recortar_ciclo_cardiaco(ruta_tsv, senal_filtrada, fs):
    """
    Lee el archivo .tsv donde están anotados los tiempos de S1 y S2.
    Recorta y devuelve exactamente ese pedacito de la señal de audio (el ciclo cardíaco),
    junto con los instantes exactos en los que ocurre cada evento.
    """
    # Usamos Pandas para leer la tabla de tiempos del paciente
    anotaciones = pd.read_csv(
        ruta_tsv,
        sep='\t',
        header=None,
        names=['inicio', 'fin', 'estado']
    )
    
    # Definimos los bordes del recorte (Inicio de S1 hasta Fin de S2, más un pequeño margen de 0.05s)
    t_inicio_ciclo = anotaciones.iloc[2]['inicio'] - 0.05
    t_fin_ciclo = anotaciones.iloc[4]['fin'] + 0.05

    # Convertimos esos segundos en "índices" (posiciones dentro de la lista de números del audio)
    idx_inicio_ciclo = int(t_inicio_ciclo * fs)
    idx_fin_ciclo = int(t_fin_ciclo * fs)

    # Hacemos el recorte en la señal original
    ciclo_completo = senal_filtrada[idx_inicio_ciclo:idx_fin_ciclo]

    # Guardamos los instantes clave en un diccionario para usarlos en el espectrograma
    tiempos_eventos = {
        't_inicio_ciclo': t_inicio_ciclo,
        't_sistole_inicio': anotaciones.iloc[3]['inicio'],
        't_s2_inicio': anotaciones.iloc[4]['inicio']
    }
    
    return ciclo_completo, tiempos_eventos


# =============================================================================
# HERRAMIENTA 5: EXTRACCIÓN DE ENERGÍA EN SÍSTOLE 
# =============================================================================
def calcular_energia_sistole(ciclo_completo, fs, tiempos_eventos, f_min=150.0, f_max=400.0):
    """
    Convierte el audio en un espectrograma. Luego recorta un "cuadrado" específico:
    solo el tiempo de la sístole y solo las frecuencias medias-altas (150-400 Hz) donde
    resalta el soplo holosistólico. Finalmente, suma toda esa energía y la devuelve en dB.
    """
    # 1. Generamos el espectrograma (imagen 3D de Potencia vs. Tiempo y Frecuencia)
    f_stft, t_stft_rel, Sxx = spectrogram(
        ciclo_completo,
        fs,
        nperseg=128,
        noverlap=115
    )
    
    # Ajustamos el tiempo para que coincida con los segundos reales del paciente
    t_stft_abs = t_stft_rel + tiempos_eventos['t_inicio_ciclo']
    
    # 2. MÁSCARA HORIZONTAL: Filtramos para quedarnos solo con frecuencias entre 150 y 400 Hz
    # Cortamos acá abajo para evitar que el latido normal (que es de baja frecuencia) nos tape el soplo
    idx_frecuencias = (f_stft >= f_min) & (f_stft <= f_max)
    
    # 3. MÁSCARA VERTICAL: Filtramos para quedarnos estrictamente en el período de la sístole
    # (Desde que arranca la sístole hasta que arranca el segundo ruido S2)
    idx_tiempo = (t_stft_abs >= tiempos_eventos['t_sistole_inicio']) & (t_stft_abs <= tiempos_eventos['t_s2_inicio'])
    
    # Recortamos la matriz del espectrograma usando nuestras dos máscaras
    Sxx_sistole = Sxx[idx_frecuencias, :][:, idx_tiempo]
    
    # Sumamos toda la potencia espectral que quedó dentro de nuestro "cuadrado" de interés
    energia_total = np.sum(Sxx_sistole)
    
    # Pasamos el resultado final a una escala logarítmica (Decibelios)
    # Le sumamos 1e-12 para evitar que intente hacer el logaritmo de 0 si el audio está mudo
    energia_db = 10 * np.log10(energia_total + 1e-12)
    
    return energia_db