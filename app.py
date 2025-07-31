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
        logging.info(f"Cargando datos EEG de {eeg_file_path}")
        if not os.path.exists(eeg_file_path):
            raise FileNotFoundError(f"Archivo EEG no encontrado: {eeg_file_path}")
        
        raw = load_eeg_data(eeg_file_path)  # Filtro ya aplicado internamente

        predictor = DementiaPredictor(model_dir)
        if not predictor.load_model():
            raise RuntimeError("Fallo al cargar el modelo en predictor.py")

        expected_features = predictor.feature_names
        logging.info(f"Extrayendo features espectrales con expected_features: {expected_features}")
        features_df = extract_spectral_features(raw, expected_features=expected_features)
        
        # Agregar features clínicas (con encoding para Gender)
        features_df['Age'] = age
        gender_map = {'male': 1, 'female': 0}  # Ajusta basado en tu training; asume 1=male, 0=female
        features_df['Gender'] = gender_map.get(gender.lower(), 0)  # Default 0 si inválido

        # Validación automática de calidad de datos
        features_df = features_df.fillna(0)
        outlier_cols = []
        for col in features_df.columns:
            col_data = features_df[col]
            if np.any(np.abs(col_data - col_data.mean()) > 5 * col_data.std()):
                outlier_cols.append(col)
        if outlier_cols:
            logging.warning(f"Outliers detectados en las columnas: {outlier_cols}")
        if not (0 <= features_df['Age'].iloc[0] <= 120):
            logging.warning(f"Edad fuera de rango: {features_df['Age'].iloc[0]}")
        if features_df['Gender'].iloc[0] not in [0, 1]:
            logging.warning(f"Valor de género inesperado: {features_df['Gender'].iloc[0]}")
    try:
        logging.info(f"Cargando datos EEG de {eeg_file_path}")
        if not os.path.exists(eeg_file_path):
            raise FileNotFoundError(f"Archivo EEG no encontrado: {eeg_file_path}")
        
        raw = load_eeg_data(eeg_file_path)  # Filtro ya aplicado internamente

        predictor = DementiaPredictor(model_dir)
        if not predictor.load_model():
            raise RuntimeError("Fallo al cargar el modelo en predictor.py")

        expected_features = predictor.feature_names
        logging.info(f"Extrayendo features espectrales con expected_features: {expected_features}")
        features_df = extract_spectral_features(raw, expected_features=expected_features)

        # Agregar features clínicas (con encoding para Gender)
        features_df['Age'] = age
        gender_map = {'male': 1, 'female': 0}
        features_df['Gender'] = gender_map.get(gender.lower(), 0)

        # Validación automática de calidad de datos
        features_df = features_df.fillna(0)
        outlier_cols = []
        for col in features_df.columns:
            col_data = features_df[col]
            if np.any(np.abs(col_data - col_data.mean()) > 5 * col_data.std()):
                outlier_cols.append(col)
        if outlier_cols:
            logging.warning(f"Outliers detectados en las columnas: {outlier_cols}")
        if not (0 <= features_df['Age'].iloc[0] <= 120):
            logging.warning(f"Edad fuera de rango: {features_df['Age'].iloc[0]}")
        if features_df['Gender'].iloc[0] not in [0, 1]:
            logging.warning(f"Valor de género inesperado: {features_df['Gender'].iloc[0]}")

        # FILTRAR A SOLO LAS FEATURES ESPERADAS (fix principal para mismatch)
        expected_features = predictor.feature_names
        logging.info(f"Features esperadas por el modelo: {expected_features}")

        missing_features = set(expected_features) - set(features_df.columns)
        if missing_features:
            logging.warning(f"Faltan features: {missing_features}. Imputando con 0.")
            for feat in missing_features:
                features_df[feat] = 0.0

        extra_features = set(features_df.columns) - set(expected_features)
        if extra_features:
            logging.info(f"Eliminando {len(extra_features)} features extras no requeridas.")
            features_df = features_df.drop(columns=list(extra_features))

        # Seleccionar solo las esperadas, en orden exacto
        features_df = features_df[expected_features]

        # Validación final
        if features_df.shape[1] != len(expected_features):
            raise ValueError(f"Features finales no match: {features_df.shape[1]} vs {len(expected_features)} esperadas.")
        logging.info(f"Features finales para predicción: {list(features_df.columns)} (Total: {features_df.shape[1]})")

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
            
            # Identificar columnas EEG (todas excepto Age y Gender)
            eeg_columns = [col for col in features_df.columns if col not in ['Age', 'Gender']]
            
            for _ in range(n_bootstraps):
                # Resample solo features EEG
                boot_eeg_df = resample(features_df[eeg_columns])
                
                # Agregar Age y Gender fijos (no resamplear)
                boot_df = boot_eeg_df.copy()
                boot_df['Age'] = age
                boot_df['Gender'] = features_df['Gender'].iloc[0]  # Encoded value
                
                # Asegurar orden de expected_features
                boot_df = boot_df[expected_features]
                
                boot_pred = predict_dementia(boot_df, model_dir=model_dir)
                if boot_pred:
                    bootstrapped_mmse.append(boot_pred['mmse_score'])
            
            if bootstrapped_mmse:
                mmse_mean = np.mean(bootstrapped_mmse)
                mmse_std = np.std(bootstrapped_mmse)
                results['mmse_ic'] = f"{mmse_mean:.2f} ± {mmse_std:.2f}"
            else:
                logging.warning("Bootstrap falló en todas las iteraciones.")
        
        # Generar PDF
        inputs_used = {'age': age, 'gender': gender}
        pdf_path = generate_pdf(results, subject_name, features_df, inputs_used, output_dir=output_dir)

        return {
            'results': results,
            'pdf_path': pdf_path,
            'raw': raw,
            'features_df': features_df
        }

    except Exception as e:
        logging.error(f"Error en main: {str(e)}")
        raise

def streamlit_app():
    st.title("Alzheimer AI Detector")

    with st.sidebar:
        st.header("Inputs")
        uploaded_file = st.file_uploader("Sube un archivo EEG (.edf o .set)", type=['edf', 'set'])
        subject_name = st.text_input("Nombre del Sujeto", "Paciente001")
        age = st.number_input("Edad", min_value=0, max_value=120, value=65)
        gender = st.selectbox("Género", ["male", "female"])
        procesar = st.button("Procesar")

    if procesar:
        if uploaded_file is not None:
            eeg_file_path = os.path.join("temp", uploaded_file.name)
            os.makedirs("temp", exist_ok=True)
            with open(eeg_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            try:
                output = main(eeg_file_path, subject_name, age, gender, use_bootstrap=True)
                
                st.subheader("Resultados")
                st.write(f"**Grupo Predicho:** {output['results']['group']}")
                st.write(f"**Puntaje MMSE Estimado:** {output['results']['mmse']}")
                st.write(f"**Intervalo de Confianza MMSE:** {output['results']['mmse_ic']}")
                st.write(f"**Probabilidad:** {output['results']['metrics']['Probability']}")
                
                with open(output['pdf_path'], "rb") as pdf_file:
                    st.download_button("Descargar PDF", pdf_file, file_name=f"{subject_name}_report.pdf")
                
                st.subheader("Visualizaciones de EEG")
                
                st.write("**Señales Raw EEG (Primeros 5 canales, 10 segundos):**")
                fig_raw = output['raw'].plot(duration=10, n_channels=5, show=False)
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

                # Histograma de todas las features
                st.write("**Distribución de Features (Histograma):**")
                fig_hist, ax = plt.subplots(figsize=(10, 4))
                output['features_df'].hist(ax=ax, bins=20, color='skyblue', edgecolor='black')
                plt.tight_layout()
                st.pyplot(fig_hist)

                # Matriz de correlación
                st.write("**Matriz de Correlación de Features:**")
                corr = output['features_df'].corr()
                fig_corr, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(corr, cmap='coolwarm', aspect='auto')
                ax.set_title('Correlación entre Features')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig_corr)

                # Boxplot de bandas
                st.write("**Boxplot de Potencias por Banda:**")
                fig_box, ax = plt.subplots()
                band_data = [output['features_df'].filter(like=f'abs_power_{band}_').values.flatten() for band in bands]
                ax.boxplot(band_data, labels=bands)
                ax.set_xlabel("Banda de Frecuencia")
                ax.set_ylabel("Potencia Absoluta")
                ax.set_title("Boxplot de Potencias por Banda")
                st.pyplot(fig_box)

                # Gráfico de barras de probabilidades de clase
                st.write("**Probabilidades por Clase Predicha:**")
                class_probs = output['results']['metrics']['Class Probabilities']
                fig_probs, ax = plt.subplots()
                ax.bar(class_probs.keys(), class_probs.values(), color='orange')
                ax.set_xlabel("Clase")
                ax.set_ylabel("Probabilidad")
                ax.set_title("Probabilidades de Clasificación")
                st.pyplot(fig_probs)
                
                st.success("Procesamiento completado.")

            except Exception as e:
                st.error(f"Error durante el procesamiento: {str(e)}")
                logging.error(f"Error detallado en Streamlit: {e}")
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
