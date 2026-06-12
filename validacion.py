import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURACIÓN DEL UMBRAL ELEGIDO
# =============================================================================
# Este es el valor en dB que elegimos mirando el gráfico del Script 2.
# A partir de este número, el algoritmo decide si el paciente es patológico o sano.
UMBRAL_ELEGIDO = 50.0  

# =============================================================================
# 1. CARGA DE DATOS DEL DATASET DE PRUEBA
# =============================================================================
print("=" * 60)
print("             EVALUACIÓN DEL RENDIMIENTO FINAL")
print("=" * 60)
print(f"Cargando el 20% de datos aislados para la prueba ...")

# Usamos try/except por si nos olvidamos de correr el Script 2 primero
try:
    # Cargamos a los pacientes que el algoritmo NUNCA vio durante el entrenamiento
    df_test = pd.read_csv('dataset_prueba.csv')
except FileNotFoundError:
    print("Error: No se encontró 'dataset_prueba.csv'.")
    print("Por favor, corré el calibracion_umbral.py primero para generar este archivo.")
    exit()

print(f"Se encontraron {len(df_test)} pacientes para evaluar.")
print(f"Umbral fijo configurado: {UMBRAL_ELEGIDO:.2f} dB\n")

# =============================================================================
# 2. PROCESAMIENTO Y CLASIFICACIÓN (EL CONSULTORIO)
# =============================================================================
# Contadores para armar nuestra Matriz de Confusión
vp = 0  # Verdaderos Positivos (Enfermo detectado correctamente)
vn = 0  # Verdaderos Negativos (Sano detectado correctamente)
fp = 0  # Falsos Positivos (Sano diagnosticado como enfermo por error)
fn = 0  # Falsos Negativos (Enfermo diagnosticado como sano por error)

print("Evaluando pacientes uno por uno...")
print("-" * 60)

# Hacemos pasar al 20% de los pacientes uno por uno
for index, paciente in df_test.iterrows():
    paciente_id = paciente['ID_Paciente']
    energia = paciente['Energia_Sistole_dB']
    diagnostico_real = paciente['Diagnostico']  # Lo que dijo el médico de la base de datos
    
    # Aplicación de la regla matemática fija (La predicción de tu algoritmo)
    prediccion = 'Soplo' if energia > UMBRAL_ELEGIDO else 'Sano'
    
    # Clasificación en la Matriz de Confusión comparando la predicción vs la realidad
    if diagnostico_real == 'Soplo' and prediccion == 'Soplo':
        vp += 1
        resultado = "Verdadero Positivo (Soplo Detectado)"
    elif diagnostico_real == 'Sano' and prediccion == 'Sano':
        vn += 1
        resultado = "Verdadero Negativo (Sano Confirmado)"
    elif diagnostico_real == 'Sano' and prediccion == 'Soplo':
        fp += 1
        resultado = "Falso Positivo (Falsa Alarma)"
    elif diagnostico_real == 'Soplo' and prediccion == 'Sano':
        fn += 1
        resultado = "Falso Negativo (Soplo No Detectado - ¡Alerta!)"

    # Imprimimos el detalle de cada paciente en la consola para ver cómo le fue
    print(f"ID: {paciente_id} | Real: {diagnostico_real:5s} | Pred: {prediccion:5s} -> {resultado}")

print("-" * 60)

# =============================================================================
# 3. CÁLCULO DE MÉTRICAS DIAGNÓSTICAS 
# =============================================================================
total_pacientes = len(df_test)

# Exactitud General: Porcentaje total de aciertos
accuracy = (vp + vn) / total_pacientes * 100

# Sensibilidad: Capacidad de encontrar la enfermedad. 
# Le agregamos el 'if' para evitar que intente dividir por cero matemáticamente.
sensibilidad = (vp / (vp + fn) * 100) if (vp + fn) > 0 else 0.0

# Especificidad: Capacidad de reconocer a los sanos.
especificidad = (vn / (vn + fp) * 100) if (vn + fp) > 0 else 0.0

# =============================================================================
# 4. REPORTE FINAL DE RESULTADOS 
# =============================================================================
# Este bloque solo imprime un recuadro de texto ordenado y fácil de leer
print("\n" + "=" * 60)
print("                      INFORME DE RENDIMIENTO")
print("" + "=" * 60)
print(f"Muestra total de validación: {total_pacientes} pacientes.")
print(f"Umbral de corte utilizado:   {UMBRAL_ELEGIDO:.2f} dB")
print("-" * 60)
print(f"  [VP] Verdaderos Positivos: {vp:2d} | [FP] Falsos Positivos: {fp:2d}")
print(f"  [FN] Falsos Negativos:     {fn:2d} | [VN] Verdaderos Negativos: {vn:2d}")
print("-" * 60)
print(f"ACCURACY (Exactitud General):  {accuracy:.1f} %")
print(f"SENSIBILIDAD:                  {sensibilidad:.1f} %  (Capacidad de detectar soplos)")
print(f"ESPECIFICIDAD:                 {especificidad:.1f} %  (Capacidad de reconocer sanos)")
print("=" * 60)

# Un mensaje final de interpretación clínica para entender el impacto de los errores
if fn == 0:
    print("El algoritmo logró un 0% de Falsos Negativos en la prueba ciega.")
else:
    print(f"Atención: Se escaparon {fn} casos patológicos. Considerá bajar un poco el umbral.")
print("=" * 60)