# AlzIA: Aplicación Web para Predicción de Demencia Usando EEG

## Descripción
AlzIA es una aplicación web desarrollada con Streamlit que utiliza un modelo de deep learning multi-task (basado en TensorFlow/Keras) para predecir el grupo de demencia (e.g., Control, Alzheimer, etc.) y el puntaje MMSE a partir de datos EEG. Soporta carga de archivos EEG en formato .set (EEGLAB) o CSV con features pre-extraídas, preprocesamiento automático, extracción de features espectrales (powers absolutos/relativos, entropy, ratios de bandas), predicciones con intervalos de confianza via bootstrap, y generación de reportes en PDF.

Esta app está diseñada para ser un prototipo preciso y eficiente, con énfasis en modularidad, manejo de errores y logging para depuración. Basada en un script original (`alz_ia_v3.py`), se enfoca en precisión médica/ingenieril, pero no es un sustituto de diagnósticos profesionales.

## Objetivos
- Proporcionar una interfaz intuitiva para cargar y analizar datos EEG/CSV.
- Realizar predicciones multi-task: clasificación de grupo de demencia y regresión de puntaje MMSE.
- Incorporar métricas de confianza (e.g., intervalos via bootstrap resampling).
- Generar reportes descargables en PDF con resultados, inputs y visualizaciones.
- Asegurar compatibilidad con canales EEG estándar (sistema 10-20, mínimo 19 canales).
- Mantener código modular y extensible para futuras expansiones (e.g., más features o modelos).

## Funcionalidades Principales
- **Carga de Archivos**: Soporte para .set (EEGLAB) o .csv (features pre-extraídas). Validación automática de canales y formatos.
- **Inputs Adicionales**: Edad, género, nombre del sujeto (codificado con LabelEncoder).
- **Preprocesamiento y Features**: Filtrado bandpass, extracción de PSD (Welch), powers por banda (delta, theta, alpha, beta, gamma), entropy, ratios (e.g., theta/alpha).
- **Predicción**: Uso de modelo .h5 cargado desde assets/. Predicciones con bootstrap para robustez.
- **Resultados**: Visualización en app (grupo, MMSE, IC, métricas) y descarga de PDF con tablas y gráficos.
- **Seguridad y Limpieza**: Manejo temporal de archivos, eliminación automática post-procesamiento.
- **Logging**: Registros detallados para depuración.

## Especificaciones Técnicas
- **Framework Principal**: Streamlit para la UI web.
- **Modelo**: Multi-task DNN (clasificación + regresión), cargado desde `dementia_model.h5`.
- **Dependencias**: Ver `requirements.txt` (incluye TensorFlow, MNE, Pandas, NumPy, Matplotlib, Scikit-learn, Joblib, ReportLab).
- **Formatos Soportados**: EEG .set con canales 10-20 (e.g., Fp1, Cz); CSV con columnas matching `feature_names.json`.
- **Entorno**: Python 3.8+; recomendado usar venv/conda. No soporta GPU por default (ajusta TensorFlow si es necesario).
- **Limitaciones**: Asume sfreq=250 Hz para EEG; no maneja artifacts avanzados (e.g., ICA); para uso investigativo, no clínico.

## Instalación
1. Clona el repositorio (si aplica) o crea la estructura de directorios:
