import os
import logging
import pandas as pd
import numpy as np
import mne
from sklearn.utils import resample
from utils.eeg_utils import extract_spectral_features, load_eeg_data  # Agregado load_eeg_data
from utils.pdf_utils import generate_pdf
from utils.predictor import DementiaPredictor, predict_dementia

import streamlit as st
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(eeg_file_path, subject_name, age, gender, model_dir='assets', output_dir='docs', use_bootstrap=True, n_bootstraps=100):
    try:
        logging.info(f"Usando model_dir: {model_dir}")  # Cambio mínimo: Log para depurar el path pasado (nuevo)
        logging.info(f"Cargando datos EEG de {eeg_file_path}")
        if not os.path.exists(eeg_file_path):
            raise FileNotFoundError(f"Archivo EEG no encontrado: {eeg_file_path}")
        
        raw = load_eeg_data(eeg_file_path)  # Usa la nueva función para .edf o .set
        raw.filter(1, 45, fir_design='firwin')

        logging.info("Extrayendo features espectrales...")
        features_df = extract_spectral_features(raw)
        
        features_df['age'] = age
        features_df['gender'] = gender

        predictor = DementiaPredictor(model_dir)
        if not predictor.load_model():
            raise RuntimeError("Fallo al cargar el modelo en predictor.py")
        
        missing_features = set(predictor.feature_names) - set(features_df.columns)
        if missing_features:
            raise ValueError(f"Faltan features en features_df: {missing_features}")

        logging.info("Realizando predicción principal...")
        prediction = predict_dementia(features_df, model_dir=model_dir)
        if prediction is None:
            raise ValueError("Predicción fallida en predictor.py")

        results = {
            'group': prediction['classification'],
            'mmse': prediction['mmse_score'],
            'mmse_ic': 'N/A',
            'metrics': {
                'Probability': prediction['probability'],
                'Class Probabilities': prediction['class_probabilities']
            }
        }

        if use_bootstrap:
            logging.info(f"Calculando bootstrap con {n_bootstraps} iteraciones...")
            bootstrapped_mmse = []
            for _ in range(n_bootstraps):
                boot_df = resample(features_df)
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

        inputs_used = {'age': age, 'gender': gender}

        logging.info("Generando PDF...")
        pdf_path = generate_pdf(results, subject_name, features_df, inputs_used, output_dir=output_dir)
        logging.info(f"Proceso completado. PDF en: {pdf_path}")
        
        return {
            'raw': raw,
            'features_df': features_df,
            'results': results,
            'pdf_path': pdf_path
        }

    except Exception as e:
        logging.error(f"Error en el flujo principal: {e}")
        raise

def streamlit_app():
    st.title("Predictor de Demencia con EEG")
    st.write("Usa el sidebar izquierdo para ingresar datos y procesar. Las visualizaciones se mostrarán aquí.")

    with st.sidebar:
        st.header("Inputs del Paciente")
        uploaded_file = st.file_uploader("Sube archivo EEG (.edf o .set)", type=["edf", "set"])
        subject_name = st.text_input("Nombre/ID del Sujeto", value="Paciente001")
        age = st.number_input("Edad", min_value=0, max_value=120, value=65)
        gender = st.selectbox("Género", ["male", "female", "other"], index=0)
        
        process_button = st.button("Procesar Predicción")

    if process_button:
        if uploaded_file is not None:
            try:
                temp_dir = "temp"
                os.makedirs(temp_dir, exist_ok=True)
                eeg_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(eeg_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                output = main(eeg_file_path, subject_name, age, gender, model_dir='assets', output_dir='docs', use_bootstrap=True)
                
                st.subheader("Resultados de Predicción")
                st.write(f"**Clasificación:** {output['results']['group']}")
                st.write(f"**MMSE Score:** {output['results']['mmse']}")
                st.write(f"**Intervalo de Confianza MMSE:** {output['results']['mmse_ic']}")
                st.write(f"**Probabilidad:** {output['results']['metrics']['Probability']}")
                
                with open(output['pdf_path'], "rb") as pdf_file:
                    st.download_button("Descargar PDF", pdf_file, file_name=f"{subject_name}_report.pdf")
                
                st.subheader("Visualizaciones de EEG")
                
                st.write("**Señales Raw EEG (Primeros 5 canales, 10 segundos):**")
                fig_raw, ax = plt.subplots()
                output['raw'].plot(duration=10, n_channels=5, show=False, ax=ax)
                st.pyplot(fig_raw)
                
                st.write("**Power Spectral Density (PSD):**")
                fig_psd = output['raw'].compute_psd().plot(show=False)
                st.pyplot(fig_psd)
                
                st.write("**Features Espectrales Extraídas (Promedios por Banda):**")
                bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
                band_means = {}
                for band in bands:
                    band_cols = [col for col in output['features_df'].columns if band in col and 'abs_power' in col]
                    if band_cols:
                        band_means[band] = output['features_df'][band_cols].mean().mean()
                
                if band_means:
                    fig_features, ax = plt.subplots()
                    ax.bar(band_means.keys(), band_means.values())
                    ax.set_xlabel("Banda de Frecuencia")
                    ax.set_ylabel("Potencia Absoluta Promedio")
                    ax.set_title("Features Espectrales por Banda")
                    st.pyplot(fig_features)
                else:
                    st.warning("No se encontraron features para graficar por banda.")
                
                st.success("Procesamiento completado.")

            except Exception as e:
                st.error(f"Error durante el procesamiento: {str(e)}")
        else:
            st.error("Por favor, sube un archivo EEG.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 or 'streamlit' not in sys.modules:
        eeg_file = 'assets/test.edf'
        subject = 'Paciente001'
        age = 65
        gender = 'male'
        model_dir = 'assets'

        main(eeg_file, subject, age, gender, model_dir=model_dir, use_bootstrap=True)
    else:
        streamlit_app()
