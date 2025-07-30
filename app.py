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

# Importar funciones de utils
from utils.eeg_utils import load_and_preprocess_eeg, extract_spectral_features

# Suprimir warnings de TensorFlow (basado en logs previos)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Silencia depreciaciones

# Configuración de logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'app_log.txt'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Directorio de assets (usamos paths relativos para portabilidad local/cloud)
ASSETS_DIR = 'assets'

# Caching para rapidez (EDUCACIÓN: @st.cache_resource evita recargar assets en cada interacción de Streamlit, mejorando performance)
@st.cache_resource
def load_model():
    model_path = os.path.join(ASSETS_DIR, 'dementia_model.keras')  # Actualizado a .keras (más moderno y compatible, como en nuestra conversación)
    if not os.path.exists(model_path):
        error_msg = f"Modelo no encontrado en: {model_path}. Verifica la carpeta assets y sube a Git si es cloud."
        logging.error(error_msg)
        st.error(error_msg)
        return None
    
    try:
        # Carga directa para .keras (no necesita compile=False; asume que se guardó con info de compilación)
        model = tf.keras.models.load_model(model_path)
        logging.info("Modelo .keras cargado exitosamente.")
        return model
    except Exception as e:
        logging.error(f"Error cargando modelo: {e}")
        st.error(f"Error cargando el modelo: {str(e)}. Verifica compatibilidad de TF/Keras (e.g., versión usada en entrenamiento).")
        return None

@st.cache_resource
def load_assets():
    scaler_path = os.path.join(ASSETS_DIR, 'scaler.pkl')  # Scaler moderno y compatible (como generamos previamente)
    feature_names_path = os.path.join(ASSETS_DIR, 'feature_names.json')
    label_encoder_path = os.path.join(ASSETS_DIR, 'label_encoder.pkl')
    le_gender_path = os.path.join(ASSETS_DIR, 'le_gender.pkl')
    bootstrap_metrics_path = os.path.join(ASSETS_DIR, 'bootstrap_metrics.json')
    
    # Checks de existencia (EDUCACIÓN: Esto resuelve issues de paths "no reconocidos" de tests previos; imprime cwd para depuración)
    for path in [scaler_path, feature_names_path, label_encoder_path, le_gender_path, bootstrap_metrics_path]:
        if not os.path.exists(path):
            error_msg = f"Archivo faltante: {path}. Verifica assets/ y sube a Git. Directorio actual: {os.getcwd()}"
            logging.error(error_msg)
            st.error(error_msg)
            return None, None, None, None, None
    
    try:
        scaler = joblib.load(scaler_path)
        with open(feature_names_path, 'r') as f:
            feature_names = json.load(f)
        label_encoder = joblib.load(label_encoder_path)
        le_gender = joblib.load(le_gender_path)
        with open(bootstrap_metrics_path, 'r') as f:
            bootstrap_metrics = json.load(f)
        logging.info("Assets cargados exitosamente.")
        return scaler, feature_names, label_encoder, le_gender, bootstrap_metrics
    except Exception as e:
        logging.error(f"Error cargando assets: {e}")
        st.error(f"Error cargando assets: {str(e)}. Posible incompatibilidad (e.g., versiones de sklearn/joblib).")
        return None, None, None, None, None

# Función para generar visualizaciones estáticas (sin cambios, pero agregué comentario)
def generate_visualizations(features_df):
    """
    EDUCACIÓN: Genera un gráfico simple de potencias espectrales. Puedes expandir con más visualizaciones si necesitas.
    """
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    avg_powers = {band: features_df.filter(like=f'abs_power_{band}').mean().mean() for band in bands}
    fig, ax = plt.subplots()
    ax.bar(avg_powers.keys(), avg_powers.values())
    ax.set_title('Average Spectral Powers')
    ax.set_ylabel('Power')
    return fig

# Función para generar PDF (sin cambios mayores, pero aseguramos temp/ exista)
def generate_pdf(results, subject_name, features_df, inputs_used):
    temp_dir = 'temp'
    os.makedirs(temp_dir, exist_ok=True)  # Asegura que temp/ exista
    pdf_path = os.path.join(temp_dir, f'{subject_name}_results.pdf' if subject_name else 'results.pdf')
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.drawString(100, 750, f"Resultados para Sujeto: {subject_name}")
    c.drawString(100, 730, f"Inputs Usados: Edad={inputs_used['age']}, Género={inputs_used['gender']}")
    c.drawString(100, 700, f"Grupo Predicho: {results['group']}")
    c.drawString(100, 680, f"MMSE Predicho: {results['mmse']} (IC: {results['mmse_ic']})")
    c.drawString(100, 650, "Métricas: " + str(results['metrics']))
    
    # Agregar visualización
    fig = generate_visualizations(features_df)
    plot_path = os.path.join(temp_dir, 'temp_plot.png')
    fig.savefig(plot_path)
    c.drawImage(plot_path, 100, 400, width=400, height=200)
    
    c.save()
    return pdf_path

# Main app
def main():
    st.title("App de Clasificación y Predicción de Alzheimer")
    st.write("Sube un EEG (.set) o CSV con features para predecir grupo y MMSE.")

    # Cargar assets (EDUCACIÓN: Cargamos al inicio para eficiencia; si falla, detiene la app como en fixes previos)
    model = load_model()
    scaler, feature_names, label_encoder, le_gender, bootstrap_metrics = load_assets()
    if model is None or scaler is None:
        return

    # Sidebar inputs (sin cambios)
    st.sidebar.header("Datos Demográficos")
    subject_name = st.sidebar.text_input("Nombre del Sujeto", "")
    age = st.sidebar.number_input("Edad", min_value=0, max_value=120, value=70)
    gender = st.sidebar.selectbox("Género", ['F', 'M'], index=0)
    
    # Warnings para defaults (sin cambios)
    if age == 70 and gender == 'F':
        st.sidebar.warning("Usando valores por default (Edad=70, Género=F). Ingresa valores para mayor precisión.")
    
    # Upload (sin cambios mayores, pero usamos paths relativos)
    uploaded_file = st.file_uploader("Sube EEG (.set) o CSV", type=['set', 'csv'])
    
    if uploaded_file:
        temp_dir = 'temp'
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        try:
            if uploaded_file.name.endswith('.set'):
                # Procesar EEG (sin cambios)
                raw = load_and_preprocess_eeg(file_path)  # De utils
                features = extract_spectral_features(raw)  # De utils
                
                # Validación de canales (sin cambios)
                expected_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
                if set(raw.ch_names) != set(expected_channels):
                    raise ValueError(f"Canales no coinciden con estándar 10-20. Encontrados: {raw.ch_names}")
                
                features_df = pd.DataFrame([features])
            elif uploaded_file.name.endswith('.csv'):
                features_df = pd.read_csv(file_path)
                # Validar columns (sin cambios)
                if set(features_df.columns) != set(feature_names):
                    raise ValueError(f"Columnas del CSV no coinciden. Esperadas: {feature_names}")
            else:
                raise ValueError("Formato no soportado.")
            
            # Agregar demográficos (sin cambios, asumiendo le_gender transforma a numérico)
            gender_encoded = le_gender.transform([gender])[0]
            features_df['Age'] = age
            features_df['Gender'] = gender_encoded
            
            # Ordenar y escalar features (sin cambios)
            features_df = features_df[feature_names]
            scaled_features = scaler.transform(features_df)
            
            # Predicciones (ajustado para robustez: maneja shapes; asume output [clasif_probs, mmse])
            predictions = model.predict(scaled_features)
            # EDUCACIÓN: Ajusta estos índices si el output del modelo .keras es diferente (e.g., verifica con model.summary() en un notebook)
            group_pred = label_encoder.inverse_transform(np.argmax(predictions[0], axis=1))[0]  # Clasificación
            mmse_pred = predictions[1][0][0]  # MMSE (ajusta si no es predictions[1])
            
            # IC de bootstrap (sin cambios)
            mmse_ic = bootstrap_metrics.get('mmse_ic', 'N/A')
            metrics = bootstrap_metrics
            
            results = {
                'group': group_pred,
                'mmse': mmse_pred,
                'mmse_ic': mmse_ic,
                'metrics': metrics
            }
            
            st.success("Predicciones completadas!")
            st.write(results)
            
            # Generar y descargar PDF (sin cambios)
            inputs_used = {'age': age, 'gender': gender}
            pdf_path = generate_pdf(results, subject_name, features_df, inputs_used)
            with open(pdf_path, 'rb') as f:
                st.download_button("Descargar PDF", f, file_name=os.path.basename(pdf_path))
            
            logging.info(f"Procesamiento exitoso para {uploaded_file.name}")
        
        except ValueError as ve:
            st.error(f"Archivo EEG/CSV no válido: {ve}")
            logging.error(f"Error de validación: {ve}")
        except Exception as e:
            st.error(f"Error procesando archivo: {e}")
            logging.error(f"Error general: {e}")
        finally:
            # Limpiar temp de manera segura (sin cambios)
            try:
                shutil.rmtree(temp_dir)
            except Exception as cleanup_err:
                logging.warning(f"Error limpiando temp: {cleanup_err}")

    # Nota informativa (sin cambios)
    st.info("Archivos soportados: EEG en formato .set (EEGLAB, con canales estándar 10-20 como Fp1, Cz, etc.) o CSV con features pre-extraídas (columnas coincidentes con el modelo). Asegúrate de que incluya al menos 19 canales para compatibilidad.")

if __name__ == "__main__":
    main()
