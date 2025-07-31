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
        
        # Filtro (movido aquí para centralizar)
        raw.filter(l_freq=0.5, h_freq=45, picks='eeg')
        
        logging.info(f"Archivo EEG cargado: {file_path}. Canales: {raw.ch_names}")
        return raw
    
    except Exception as e:
        logging.error(f"Error cargando EEG: {e}")
        raise

def extract_spectral_features(raw, expected_features=None):
    """
    Extrae features espectrales, optimizando si se proporcionan expected_features (solo calcula lo necesario).
    Si no, calcula todo y deja filtrar en app.py.
    """
    try:
        # Manejo de canales: Imputar missing con 0 en features (no raise)
        channels = raw.ch_names
        missing_channels = set(STANDARD_CHANNELS) - set(channels)
        if missing_channels:
            logging.warning(f"Canales faltantes: {missing_channels}. Imputando features=0 para ellos.")
        
        # Unir canales reales + missing (para features consistentes)
        all_channels = list(set(channels) | missing_channels)
        
        # Computar PSD
        spectrum = raw.compute_psd(method='welch', fmin=0.5, fmax=45, n_fft=512, n_overlap=256, picks='eeg')
        psds = spectrum.get_data()
        freqs = spectrum.freqs
        
        # Ajustar psds para missing channels (agregar ceros)
        if missing_channels:
            missing_psds = np.zeros((len(missing_channels), len(freqs)))
            psds = np.vstack((psds, missing_psds))
            channels = list(channels) + list(missing_channels)  # Actualizar lista
        
        features = {}
        
        # RMS: Calcular siempre (está en expected, e.g., 'rms_O2')
        data = raw.get_data(picks='eeg')
        if missing_channels:
            missing_data = np.zeros((len(missing_channels), data.shape[1]))
            data = np.vstack((data, missing_data))
        
        for ch_idx, ch_name in enumerate(all_channels):
            if expected_features and not any(f'rms_{ch_name}' in feat for feat in expected_features):
                continue  # Optimizar: skip si no requerido
            rms = np.sqrt(np.mean(data[ch_idx] ** 2)) if ch_name in raw.ch_names else 0.0
            features[f'rms_{ch_name}'] = [rms]
        
        # Potencia total (para rel_power)
        total_power = np.sum(psds, axis=1)
        
        # Features por banda
        for band, (fmin, fmax) in BANDS.items():
            if expected_features and not any(band in feat for feat in expected_features):
                continue  # Skip bandas no requeridas
            
            band_idx = np.where((freqs >= fmin) & (freqs < fmax))[0]
            if len(band_idx) == 0:
                logging.warning(f"No índices para banda {band}. Skipping.")
                continue
            
            abs_power = np.sum(psds[:, band_idx], axis=1)
            for ch_idx, ch_name in enumerate(all_channels):
                if expected_features and not any(f'abs_power_{band}_{ch_name}' in feat for feat in expected_features):
                    continue
                features[f'abs_power_{band}_{ch_name}'] = [abs_power[ch_idx] if ch_name in raw.ch_names else 0.0]
            
            rel_power = abs_power / (total_power + 1e-10)  # Evitar div/0
            for ch_idx, ch_name in enumerate(all_channels):
                if expected_features and not any(f'rel_power_{band}_{ch_name}' in feat for feat in expected_features):
                    continue
                features[f'rel_power_{band}_{ch_name}'] = [rel_power[ch_idx] if ch_name in raw.ch_names else 0.0]
            
            # Entropy: Solo si requerido (en tu config, NO lo está; coméntalo si no)
            # psd_norm = psds[:, band_idx] / np.sum(psds[:, band_idx], axis=1, keepdims=True) + 1e-10
            # band_entropy = entropy(psd_norm, axis=1)
            # for ch_idx, ch_name in enumerate(all_channels):
            #     if expected_features and not any(f'entropy_{band}_{ch_name}' in feat for feat in expected_features):
            #         continue
            #     features[f'entropy_{band}_{ch_name}'] = [band_entropy[ch_idx] if ch_name in raw.ch_names else 0.0]
        
        # Ratios: Solo si requeridos (en config, NO; pero algunos abs/rel implican)
        for ch_idx, ch_name in enumerate(all_channels):
            if expected_features and not any(f'ratio_' in feat and ch_name in feat for feat in expected_features):
                continue
            abs_theta = features.get(f'abs_power_theta_{ch_name}', [0])[0]
            abs_alpha = features.get(f'abs_power_alpha_{ch_name}', [1])[0] or 1.0  # Evitar div/0
            abs_beta = features.get(f'abs_power_beta_{ch_name}', [0])[0]
            features[f'ratio_theta_alpha_{ch_name}'] = [abs_theta / abs_alpha if ch_name in raw.ch_names else 0.0]
            features[f'ratio_beta_alpha_{ch_name}'] = [abs_beta / abs_alpha if ch_name in raw.ch_names else 0.0]
        
        features_df = pd.DataFrame(features)
        
        logging.info(f"Features espectrales extraídas: {list(features_df.columns)}")
        logging.info(f"Número de features: {len(features_df.columns)}")
        
        return features_df

    except Exception as e:
        logging.error(f"Error extrayendo features: {e}")
        raise
