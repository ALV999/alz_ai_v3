import mne
import numpy as np
import pandas as pd
import os
import logging
from scipy.stats import entropy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

STANDARD_CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']

BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

def load_eeg_data(file_path):
    try:
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.edf':
            raw = mne.io.read_raw_edf(file_path, preload=True)
        elif ext == '.set':
            raw = mne.io.read_raw_eeglab(file_path, preload=True)
        elif ext == '.csv':
            df = pd.read_csv(file_path)
            info = mne.create_info(ch_names=list(df.columns), sfreq=250, ch_types='eeg')  
            raw = mne.io.RawArray(df.T.values, info)
        else:
            raise ValueError(f"Formato no soportado: {ext}. Usa .edf, .set o .csv.")
        
        # Log canales cargados
        logging.info(f"Archivo EEG cargado: {file_path}. Canales presentes: {raw.ch_names}")
        
        # Manejo de canales missing: Agregar dummies con zero-filled data
        missing_channels = set(STANDARD_CHANNELS) - set(raw.ch_names)
        if missing_channels:
            logging.warning(f"Canales faltantes: {missing_channels}. Imputando con datos zero-filled.")
            for ch in missing_channels:
                info = mne.create_info(ch_names=[ch], sfreq=raw.info['sfreq'], ch_types='eeg')
                dummy = mne.io.RawArray(np.zeros((1, len(raw.times))), info)
                raw.add_channels([dummy], force_update_info=True)
        
        # Filtro (movido de app.py para centralizar)
        raw.filter(l_freq=0.5, h_freq=45, picks='eeg', fir_design='firwin')
        
        logging.info("Preprocesamiento completado. Canales finales: {raw.ch_names}")
        return raw

    except Exception as e:
        logging.error(f"Error cargando/preprocesando EEG: {e}")
        raise

def extract_spectral_features(raw):
    try:
        # Primero, calcular RMS por canal (agregado para resolver missing 'rms_O2', etc.)
        features = {}
        for ch_idx, ch_name in enumerate(raw.ch_names):
            signal = raw.get_data(picks=[ch_name])[0]  # Obtener signal del canal
            if len(signal) == 0:
                rms = 0.0
            else:
                rms = np.sqrt(np.mean(signal**2))  # Cálculo estándar de RMS
            features[f'rms_{ch_name}'] = [rms]
        
        # Cálculo espectral existente
        spectrum = raw.compute_psd(method='welch', fmin=0.5, fmax=45, n_fft=512, n_overlap=256, picks='eeg')
        psds = spectrum.get_data()
        freqs = spectrum.freqs

        total_power = np.sum(psds, axis=1)  
        for band, (fmin, fmax) in BANDS.items():
            band_idx = np.where((freqs >= fmin) & (freqs < fmax))[0]
            if len(band_idx) == 0:
                logging.warning(f"Banda {band} sin índices válidos. Saltando.")
                continue

            abs_power = np.sum(psds[:, band_idx], axis=1)
            for ch_idx, ch_name in enumerate(raw.ch_names):
                features[f'abs_power_{band}_{ch_name}'] = [abs_power[ch_idx]]

            rel_power = abs_power / (total_power + 1e-10)  # Evitar div/0
            for ch_idx, ch_name in enumerate(raw.ch_names):
                features[f'rel_power_{band}_{ch_name}'] = [rel_power[ch_idx]]

            psd_norm = psds[:, band_idx] / (np.sum(psds[:, band_idx], axis=1, keepdims=True) + 1e-10)  # Evitar div/0
            band_entropy = entropy(psd_norm, axis=1)
            for ch_idx, ch_name in enumerate(raw.ch_names):
                features[f'entropy_{band}_{ch_name}'] = [band_entropy[ch_idx]]

        # Ratios con safeguards para div/0
        for ch_idx, ch_name in enumerate(raw.ch_names):
            abs_theta = features.get(f'abs_power_theta_{ch_name}', [0])[0]
            abs_alpha = features.get(f'abs_power_alpha_{ch_name}', [1])[0] or 1  # Evitar div/0
            abs_beta = features.get(f'abs_power_beta_{ch_name}', [0])[0]
            theta_alpha = abs_theta / abs_alpha
            beta_alpha = abs_beta / abs_alpha
            features[f'ratio_theta_alpha_{ch_name}'] = [theta_alpha]
            features[f'ratio_beta_alpha_{ch_name}'] = [beta_alpha]

        features_df = pd.DataFrame(features)
        
        # Log para depuración
        logging.info(f"Features espectrales extraídas: {list(features_df.columns)}")
        logging.info(f"Número de features: {len(features_df.columns)}")
        
        return features_df

    except Exception as e:
        logging.error(f"Error extrayendo features: {e}")
        raise
