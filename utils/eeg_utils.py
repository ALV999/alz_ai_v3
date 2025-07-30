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

def load_and_preprocess_eeg(file_path, file_type='set'):
    try:
        if file_type == 'set':
            raw = mne.io.read_raw_eeglab(file_path, preload=True)
            logging.info(f"Archivo EEG .set cargado: {file_path}")
        elif file_type == 'csv':
            df = pd.read_csv(file_path)
            info = mne.create_info(ch_names=list(df.columns), sfreq=250, ch_types='eeg')  
            raw = mne.io.RawArray(df.T.values, info)
            logging.info(f"Archivo CSV cargado: {file_path}")
        else:
            raise ValueError("Formato no soportado. Usa .set o .csv.")

        missing_channels = set(STANDARD_CHANNELS) - set(raw.ch_names)
        if missing_channels:
            raise ValueError(f"Canales faltantes: {missing_channels}. Se esperan {STANDARD_CHANNELS}.")

        raw.filter(l_freq=0.5, h_freq=45, picks='eeg')

        logging.info("Preprocesamiento completado.")
        return raw

    except Exception as e:
        logging.error(f"Error cargando/preprocesando EEG: {e}")
        raise

def extract_spectral_features(raw):
    try:
        psds, freqs = mne.time_frequency.psd_welch(raw, fmin=0.5, fmax=45, n_fft=512, n_overlap=256, picks='eeg')

        features = {}
        total_power = np.sum(psds, axis=1)  
        for band, (fmin, fmax) in BANDS.items():
            band_idx = np.where((freqs >= fmin) & (freqs < fmax))[0]
            if len(band_idx) == 0:
                continue

            abs_power = np.sum(psds[:, band_idx], axis=1)
            for ch_idx, ch_name in enumerate(raw.ch_names):
                features[f'abs_power_{band}_{ch_name}'] = [abs_power[ch_idx]]

            rel_power = abs_power / total_power
            for ch_idx, ch_name in enumerate(raw.ch_names):
                features[f'rel_power_{band}_{ch_name}'] = [rel_power[ch_idx]]

            psd_norm = psds[:, band_idx] / np.sum(psds[:, band_idx], axis=1, keepdims=True)
            band_entropy = entropy(psd_norm, axis=1)
            for ch_idx, ch_name in enumerate(raw.ch_names):
                features[f'entropy_{band}_{ch_name}'] = [band_entropy[ch_idx]]

        for ch_idx, ch_name in enumerate(raw.ch_names):
            theta_alpha = features.get(f'abs_power_theta_{ch_name}', [0])[0] / features.get(f'abs_power_alpha_{ch_name}', [1])[0]
            beta_alpha = features.get(f'abs_power_beta_{ch_name}', [0])[0] / features.get(f'abs_power_alpha_{ch_name}', [1])[0]
            features[f'ratio_theta_alpha_{ch_name}'] = [theta_alpha]
            features[f'ratio_beta_alpha_{ch_name}'] = [beta_alpha]

        features_df = pd.DataFrame(features)
        logging.info("Features espectrales extraídas exitosamente.")
        return features_df

    except Exception as e:
        logging.error(f"Error extrayendo features: {e}")
        raise
