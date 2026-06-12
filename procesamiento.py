import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, spectrogram

# =============================================================================
# 1. CARGA DE DATOS Y PREPROCESAMIENTO
# =============================================================================

archivo_wav = 'TPI_SE-ALES_AUSCULTACION\\45843\\45843_MV.wav'
archivo_tsv = 'TPI_SE-ALES_AUSCULTACION\\45843\\45843_MV.tsv'

# Lectura del archivo de audio
fs, senal_cruda = wavfile.read(archivo_wav)

# Si el audio fuera estéreo, se selecciona un único canal
if len(senal_cruda.shape) > 1:
    senal_cruda = senal_cruda[:, 0]

# Conversión a float
senal_cruda = senal_cruda.astype(np.float64)

# Lectura de las anotaciones clínicas
anotaciones = pd.read_csv(
    archivo_tsv,
    sep='\t',
    header=None,
    names=['inicio', 'fin', 'estado']
)

# =============================================================================
# 2. FILTRADO PASABANDA (40 - 500 Hz)
# =============================================================================

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):

    nyq = 0.5 * fs

    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')

    return filtfilt(b, a, data)

senal_filtrada = butter_bandpass_filter(
    senal_cruda,
    40.0,
    500.0,
    fs
)

# =============================================================================
# 3. SELECCIÓN DE UN CICLO CARDÍACO REPRESENTATIVO
# =============================================================================

t_inicio_ciclo = anotaciones.iloc[2]['inicio'] - 0.05
t_fin_ciclo = anotaciones.iloc[4]['fin'] + 0.05

idx_inicio_ciclo = int(t_inicio_ciclo * fs)
idx_fin_ciclo = int(t_fin_ciclo * fs)

ciclo_completo = senal_filtrada[idx_inicio_ciclo:idx_fin_ciclo]

# Instantes característicos del ciclo
t_s1_inicio = anotaciones.iloc[2]['inicio']
t_sistole_inicio = anotaciones.iloc[3]['inicio']
t_s2_inicio = anotaciones.iloc[4]['inicio']
t_s2_fin = anotaciones.iloc[4]['fin']

# =============================================================================
# ESPECTROGRAMA (STFT)
# =============================================================================

plt.figure(figsize=(14, 6))

f_stft, t_stft_rel, Sxx = spectrogram(
    ciclo_completo,
    fs,
    nperseg=128,
    noverlap=115
)

t_stft_abs = t_stft_rel + t_inicio_ciclo

plt.pcolormesh(
    t_stft_abs,
    f_stft,
    10 * np.log10(Sxx + 1e-12),
    shading='gouraud',
    cmap='jet',
    vmin=-10,
    vmax=45
)

# Marcadores de eventos cardíacos

plt.axvline(
    x=t_s1_inicio,
    color='white',
    linestyle='--',
    alpha=0.8,
    label='Inicio S1'
)

plt.axvline(
    x=t_sistole_inicio,
    color='cyan',
    linestyle='-',
    linewidth=2,
    label='Inicio Sístole'
)

plt.axvline(
    x=t_s2_inicio,
    color='magenta',
    linestyle='-',
    linewidth=2,
    label='Inicio S2'
)

plt.axvline(
    x=t_s2_fin,
    color='white',
    linestyle='--',
    alpha=0.8,
    label='Fin S2'
)

plt.title(
    'Espectrograma del Ciclo Cardíaco',
    fontweight='bold',
    pad=15
)

plt.xlabel('Tiempo [s]')
plt.ylabel('Frecuencia [Hz]')
plt.ylim(0, 500)

plt.legend(loc='upper right')
plt.colorbar(label='Potencia [dB]')

plt.tight_layout()
plt.show()