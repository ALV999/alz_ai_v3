import os
import logging
import pandas as pd
import numpy as np
import mne  # Para cargar y procesar EEG
from sklearn.utils import resample  # Para bootstrap (intervalos de confianza)
from utils.eeg_utils import extract_spectral_features  # Asumiendo en utils/
from utils.pdf_utils import generate_pdf  # Asumiendo en utils/
from utils.predictor import DementiaPredictor, predict_dementia  # Integración de predictor.py en utils/

# Configuración de logging (consistente con otros archivos)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(eeg_file_path, subject_name, age, gender, model_dir='assets', output_dir='docs', use_bootstrap=True, n_bootstraps=100):
    """
    Flujo principal: Carga EEG, extrae features, predice demencia (con bootstrap opcional), genera PDF.
    
    Args:
    - eeg_file_path: str, ruta al archivo EEG (e.g., 'assets/test.edf').
    - subject_name: str, nombre del sujeto.
    - age: float/int, edad.
    - gender: str, género (e.g., 'male'/'female' — será encoded por predictor si aplica).
    - model_dir: str, directorio con modelos (default: 'assets').
    - output_dir: str, directorio para outputs (default: 'docs').
    - use_bootstrap: bool, si True, calcula intervalos de confianza para MMSE (default: True).
    - n_bootstraps: int, número de iteraciones de bootstrap (default: 100).
    
    Returns:
    - str: Ruta al PDF generado.
    """
    try:
        # Paso 1: Cargar y preprocesar datos EEG
        logging.info(f"Cargando datos EEG de {eeg_file_path}")
        if not os.path.exists(eeg_file_path):
            raise FileNotFoundError(f"Archivo EEG no encontrado: {eeg_file_path}")
        
        raw = mne.io.read_raw_edf(eeg_file_path, preload=True)  # Ajusta para otros formatos si es necesario
        raw.filter(1, 45, fir_design='firwin')  # Filtro bandpass básico (ajusta según necesidades)

        # Paso 2: Extraer features espectrales
        logging.info("Extrayendo features espectrales...")
        features_df = extract_spectral_features(raw)  # Debe devolver pd.DataFrame con columnas como 'abs_power_delta_Fp1', etc.
        
        # Agregar inputs demográficos (asumiendo que predictor espera 'age' y 'gender' como columnas)
        features_df['age'] = age
        features_df['gender'] = gender  # predictor.py lo encodeará si usa additional_encoders

        # Validar que features_df coincida con feature_names de predictor
        predictor = DementiaPredictor(model_dir)
        if not predictor.load_model():
            raise RuntimeError("Fallo al cargar el modelo en predictor.py")
        
        missing_features = set(predictor.feature_names) - set(features_df.columns)
        if missing_features:
            raise ValueError(f"Faltan features en features_df: {missing_features}")

        # Paso 3: Predicción usando predictor.py
        logging.info("Realizando predicción principal...")
        prediction = predict_dementia(features_df, model_dir=model_dir)
        if prediction is None:
            raise ValueError("Predicción fallida en predictor.py")

        # Mapear outputs de predictor a formato esperado por pdf_utils.py
        results = {
            'group': prediction['classification'],
            'mmse': prediction['mmse_score'],
            'mmse_ic': 'N/A',  # Se actualizará si use_bootstrap=True
            'metrics': {
                'Probability': prediction['probability'],
                'Class Probabilities': prediction['class_probabilities']
            }
        }

        # Opcional: Bootstrap para intervalos de confianza en MMSE
        if use_bootstrap:
            logging.info(f"Calculando bootstrap con {n_bootstraps} iteraciones...")
            bootstrapped_mmse = []
            for _ in range(n_bootstraps):
                boot_df = resample(features_df)  # Resamplea el DataFrame
                boot_pred = predict_dementia(boot_df, model_dir=model_dir)
                if boot_pred:
                    bootstrapped_mmse.append(boot_pred['mmse_score'])
            
            if bootstrapped_mmse:
                lower = np.percentile(bootstrapped_mmse, 2.5)
                upper = np.percentile(bootstrapped_mmse, 97.5)
                results['mmse_ic'] = f"[{lower:.2f}, {upper:.2f}]"
                results['metrics']['Bootstrap Mean MMSE'] = np.mean(bootstrapped_mmse)
            else:
                logging.warning("Bootstrap falló; no se generaron muestras válidas.")

        # Inputs usados para PDF
        inputs_used = {'age': age, 'gender': gender}

        # Paso 4: Generar PDF
        logging.info("Generando PDF...")
        pdf_path = generate_pdf(results, subject_name, features_df, inputs_used, output_dir=output_dir)
        logging.info(f"Proceso completado. PDF en: {pdf_path}")
        
        return pdf_path

    except Exception as e:
        logging.error(f"Error en el flujo principal: {e}")
        raise

# Ejemplo de uso para testing
if __name__ == "__main__":
    # Reemplaza con valores reales (asegúrate de que 'assets/test.edf' exista)
    eeg_file = 'assets/test.edf'
    subject = 'Paciente001'
    age = 65
    gender = 'male'  # Debe coincidir con el encoding en le_gender.pkl si aplica
    model_dir = 'assets'

    main(eeg_file, subject, age, gender, model_dir=model_dir, use_bootstrap=True)
