import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import logging
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import joblib
import json
import shutil
import matplotlib.pyplot as plt
import mne  # Asumiendo que utils/eeg_utils.py lo usa

# Importar funciones de utils (el próximo archivo en la lista)
from utils.eeg_utils import load_and_preprocess_eeg, extract_spectral_features

# Configuración de logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'app_log.txt'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Caching para rapidez
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('assets/dementia_model.h5')
        logging.info("Modelo cargado exitosamente.")
        return model
    except Exception as e:
        logging.error(f"Error cargando modelo: {e}")
        st.error("Error cargando el modelo. Verifica assets/dementia_model.h5.")
        return None

@st.cache_resource
def load_assets():
    try:
        scaler = joblib.load('assets/scaler.pkl')
        with open('assets/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        label_encoder = joblib.load('assets/label_encoder.pkl')
        le_gender = joblib.load('assets/le_gender.pkl')
        with open('assets/bootstrap_metrics.json', 'r') as f:
            bootstrap_metrics = json.load(f)
        logging.info("Assets cargados exitosamente.")
        return scaler, feature_names, label_encoder, le_gender, bootstrap_metrics
    except Exception as e:
        logging.error(f"Error cargando assets: {e}")
        st.error("Error cargando assets. Verifica la carpeta assets.")
        return None, None, None, None, None

# Función para generar visualizaciones estáticas (ejemplo simple; expande en utils si necesitas)
def generate_visualizations(features_df):
    # Ejemplo: Gráfico de barras para powers promedio
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    avg_powers = {band: features_df.filter(like=f'abs_power_{band}').mean().mean() for band in bands}
    fig, ax = plt.subplots()
    ax.bar(avg_powers.keys(), avg_powers.values())
    ax.set_title('Average Spectral Powers')
    ax.set_ylabel('Power')
    return fig

# Función para generar PDF
def generate_pdf(results, subject_name, features_df, inputs_used):
    pdf_path = f'temp/{subject_name}_results.pdf' if subject_name else 'temp/results.pdf'
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.drawString(100, 750, f"Resultados para Sujeto: {subject_name}")
    c.drawString(100, 730, f"Inputs Usados: Edad={inputs_used['age']}, Género={inputs_used['gender']}")
    c.drawString(100, 700, f"Grupo Predicho: {results['group']}")
    c.drawString(100, 680, f"MMSE Predicho: {results['mmse']} (IC: {results['mmse_ic']})")
    c.drawString(100, 650, "Métricas: " + str(results['metrics']))
    
    # Agregar visualización
    fig = generate_visualizations(features_df)
    fig.savefig('temp/temp_plot.png')
    c.drawImage('temp/temp_plot.png', 100, 400, width=400, height=200)
    
    c.save()
    return pdf_path

# Main app
def main():
    st.title("App de Clasificación y Predicción de Alzheimer")
    st.write("Sube un EEG (.set) o CSV con features para predecir grupo y MMSE.")

    # Cargar assets
    model = load_model()
    scaler, feature_names, label_encoder, le_gender, bootstrap_metrics = load_assets()
    if model is None or scaler is None:
        return

    # Sidebar inputs
    st.sidebar.header("Datos Demográficos")
    subject_name = st.sidebar.text_input("Nombre del Sujeto", "")
    age = st.sidebar.number_input("Edad", min_value=0, max_value=120, value=70)
    gender = st.sidebar.selectbox("Género", ['F', 'M'], index=0)
    
    # Warnings para defaults
    if age == 70 and gender == 'F':
        st.sidebar.warning("Usando valores por default (Edad=70, Género=F). Ingresa valores para mayor precisión.")

    # Upload
    uploaded_file = st.file_uploader("Sube EEG (.set) o CSV", type=['set', 'csv'])
    
    if uploaded_file:
        temp_dir = 'temp'
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        try:
            if uploaded_file.name.endswith('.set'):
                # Procesar EEG
                raw = load_and_preprocess_eeg(file_path)  # De utils
                features = extract_spectral_features(raw)  # De utils
                
                # Validación de canales (ejemplo: asumir 19 canales estándar)
                expected_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']  # Ajusta a tus estándares
                if set(raw.ch_names) != set(expected_channels):
                    raise ValueError("Canales no coinciden con estándar 10-20.")
                
                features_df = pd.DataFrame([features])
            elif uploaded_file.name.endswith('.csv'):
                features_df = pd.read_csv(file_path)
                # Validar columns
                if set(features_df.columns) != set(feature_names):
                    raise ValueError("Columnas del CSV no coinciden con features esperadas.")
            else:
                raise ValueError("Formato no soportado.")
            
            # Agregar demográficos
            gender_encoded = le_gender.transform([gender])[0]  # Asumiendo le_gender es LabelEncoder para 'F'/'M'
            features_df['Age'] = age
            features_df['Gender_M'] = 1 if gender == 'M' else 0  # O usa le_gender si es más complejo
            
            # Ordenar y escalar features
            features_df = features_df[feature_names]
            scaled_features = scaler.transform(features_df)
            
            # Predicciones
            predictions = model.predict(scaled_features)
            group_pred = label_encoder.inverse_transform(np.argmax(predictions[0], axis=1))  # Asumiendo predictions[0] es para clasificación
            mmse_pred = predictions[1][0][0]  # Asumiendo predictions[1] es para MMSE (ajusta a tu modelo multi-task)
            
            # IC de bootstrap (ejemplo)
            mmse_ic = bootstrap_metrics.get('mmse_ic', 'N/A')
            metrics = bootstrap_metrics  # O selecciona relevantes
            
            results = {
                'group': group_pred[0],
                'mmse': mmse_pred,
                'mmse_ic': mmse_ic,
                'metrics': metrics
            }
            
            st.success("Predicciones completadas!")
            st.write(results)
            
            # Generar y descargar PDF
            inputs_used = {'age': age, 'gender': gender}
            pdf_path = generate_pdf(results, subject_name, features_df, inputs_used)
            with open(pdf_path, 'rb') as f:
                st.download_button("Descargar PDF", f, file_name=os.path.basename(pdf_path))
            
            logging.info(f"Procesamiento exitoso para {uploaded_file.name}")
        
        except ValueError as ve:
            st.error(f"Archivo EEG/CSV no válido: {ve}. Se esperan canales estándar 10-20.")
            logging.error(f"Error de validación: {ve}")
        except Exception as e:
            st.error(f"Error procesando archivo: {e}")
            logging.error(f"Error general: {e}")
        finally:
            # Limpiar temp
            shutil.rmtree(temp_dir)
    
    # Nota informativa sobre formatos (como pediste)
    st.info("Archivos soportados: EEG en formato .set (EEGLAB, con canales estándar 10-20 como Fp1, Cz, etc.) o CSV con features pre-extraídas (columnas coincidentes con el modelo). Asegúrate de que incluya al menos 19 canales para compatibilidad.")

if __name__ == "__main__":
    main()
