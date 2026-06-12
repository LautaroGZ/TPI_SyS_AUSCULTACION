import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. CARGA Y BALANCEO DE DATOS 
# =============================================================================
print("Cargando base de datos...")
# Leemos el archivo que fabricó el Script 1
df = pd.read_csv('dataset_caracteristicas.csv')

# Filtramos la tabla gigante en dos listas separadas
sanos = df[df['Diagnostico'] == 'Sano']
soplos = df[df['Diagnostico'] == 'Soplo']

# Buscamos cuál es el grupo más chico (seguramente los soplos) para nivelar
cantidad_minima = min(len(sanos), len(soplos))
print(f"Balanceando: Usaremos {cantidad_minima} Sanos y {cantidad_minima} Soplos.")

# .sample elige pacientes al azar. random_state=42 asegura que siempre elija los mismos si lo volvés a correr
sanos_balanceados = sanos.sample(n=cantidad_minima, random_state=42)
soplos_balanceados = soplos.sample(n=cantidad_minima, random_state=42)

# =============================================================================
# 2. DIVISIÓN 80/20 CON PANDAS 
# =============================================================================
# Calculamos exactamente cuánto es el 80% matemático de nuestra muestra balanceada
limite_80_porciento = int(cantidad_minima * 0.80)

# .iloc rebana la lista. [:limite] agarra desde el principio hasta el 80%
entrenamiento_sanos = sanos_balanceados.iloc[:limite_80_porciento]
# [limite:] agarra desde el 80% hasta el final de la lista (el 20% restante)
pruebas_sanos = sanos_balanceados.iloc[limite_80_porciento:]

# Hacemos el mismo corte para los pacientes enfermos
entrenamiento_soplos = soplos_balanceados.iloc[:limite_80_porciento]
pruebas_soplos = soplos_balanceados.iloc[limite_80_porciento:]

# Unimos las mitades correspondientes (80 con 80, y 20 con 20)
df_entrenamiento = pd.concat([entrenamiento_sanos, entrenamiento_soplos])
df_pruebas = pd.concat([pruebas_sanos, pruebas_soplos])

# Mezclamos las filas aleatoriamente (shuffle) para que no queden agrupadas por diagnóstico
df_entrenamiento = df_entrenamiento.sample(frac=1, random_state=42).reset_index(drop=True)
df_pruebas = df_pruebas.sample(frac=1, random_state=42).reset_index(drop=True)

# GUARDAMOS EL 20% (No lo tocamos hasta la validación final del Script 3)
df_pruebas.to_csv('dataset_prueba.csv', index=False)
print(f"\nSe guardaron {len(df_pruebas)} pacientes en 'dataset_prueba.csv' para la validación final.")
print(f"Iniciando calibración de umbral con los {len(df_entrenamiento)} pacientes de entrenamiento...")

# =============================================================================
# 3. BÚSQUEDA DEL UMBRAL ÓPTIMO 
# =============================================================================
# Nos fijamos cuáles son las energías extremas en nuestros datos
energia_min = df_entrenamiento['Energia_Sistole_dB'].min()
energia_max = df_entrenamiento['Energia_Sistole_dB'].max()

# Creamos 600 posibles umbrales equiespaciados entre la energía mínima y máxima
umbrales_a_probar = np.linspace(energia_min, energia_max, 600)

# Lista para guardar el rendimiento de cada uno de los 600 intentos
resultados = []

# Empezamos el barrido de fuerza bruta
for umbral in umbrales_a_probar:
    vp = 0  # Verdaderos Positivos (Acierto Soplo)
    vn = 0  # Verdaderos Negativos (Acierto Sano)
    fp = 0  # Falsos Positivos (Falsa Alarma)
    fn = 0  # Falsos Negativos (Soplo que se escapó)
    
    # Evaluamos a cada paciente de entrenamiento con el umbral actual
    for index, paciente in df_entrenamiento.iterrows():
        energia = paciente['Energia_Sistole_dB']
        diagnostico_real = paciente['Diagnostico']
        
        # Nuestra Regla determinista (el corazón del algoritmo)
        prediccion = 'Soplo' if energia > umbral else 'Sano'
        
        # Clasificamos el resultado en la Matriz de Confusión
        if diagnostico_real == 'Soplo' and prediccion == 'Soplo':
            vp += 1
        elif diagnostico_real == 'Sano' and prediccion == 'Sano':
            vn += 1
        elif diagnostico_real == 'Sano' and prediccion == 'Soplo':
            fp += 1
        elif diagnostico_real == 'Soplo' and prediccion == 'Sano':
            fn += 1
            
    # Calculamos la métrica de Exactitud (Accuracy) para este umbral
    accuracy = (vp + vn) / len(df_entrenamiento) * 100
    
    # Anotamos los resultados
    resultados.append({
        'Umbral': umbral,
        'Accuracy': accuracy,
        'Falsos_Negativos': fn,
        'Falsos_Positivos': fp
    })

# Convertimos los 600 resultados en una tabla
df_resultados = pd.DataFrame(resultados)

# =============================================================================
# 4. VISUALIZACIÓN PARA TOMAR LA DECISIÓN (EL GRÁFICO)
# =============================================================================
plt.figure(figsize=(12, 6))

# Dibujamos las tres curvas clave
plt.plot(df_resultados['Umbral'], df_resultados['Accuracy'], label='Accuracy (%)', color='green', linewidth=2)
plt.plot(df_resultados['Umbral'], df_resultados['Falsos_Negativos'], label='Falsos Negativos (Cant.)', color='red', linestyle='--', linewidth=2)
plt.plot(df_resultados['Umbral'], df_resultados['Falsos_Positivos'], label='Falsos Positivos (Cant.)', color='orange', linestyle=':', linewidth=2)

# Estética del gráfico
plt.title('Calibración del Umbral de Energía en Sístole', fontweight='bold')
plt.xlabel('Umbral de Energía [dB]')
plt.ylabel('Porcentaje (%) / Cantidad de Pacientes')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Buscamos automáticamente cuál fue el umbral que dio el pico más alto de Accuracy
mejor_accuracy = df_resultados.loc[df_resultados['Accuracy'].idxmax()]

# Imprimimos el resultado como guía
print("\n" + "="*50)
print("=== SUGERENCIA ESTADÍSTICA AUTOMÁTICA ===")
print("="*50)
print(f"El umbral matemático que maximiza el Accuracy es: {mejor_accuracy['Umbral']:.2f} dB")
print(f"Rendimiento esperado: Accuracy {mejor_accuracy['Accuracy']:.1f}% | Falsos Negativos: {mejor_accuracy['Falsos_Negativos']}")
print("\nNOTA: La decisión final del umbral no es solo matemática, también debe considerar el contexto clínico y la tolerancia a errores. Nos tenemos que fijar en el gráfico y elegir un punto que tenga un buen equilibrio entre Accuracy, Falsos Negativos y Falsos Positivos según lo que consideremos más importante para nuestro diagnóstico.")