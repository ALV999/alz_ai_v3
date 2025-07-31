import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DementiaPredictor:
    def __init__(self, model_dir='assets'):  # Default a 'assets' (carpeta separada); cambia a 'utils' si los archivos están allí. Asegúrate de que coincida con app.py
        # Cambio mínimo: Subir un nivel al root del repo (desde utils/), luego join con model_dir (e.g., 'assets')
        # Si los archivos están dentro de utils/, revierte a: os.path.join(os.path.dirname(__file__), model_dir)
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_dir)
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.config = None
        self.additional_encoders = {}

    def load_model(self):
        """Carga el modelo y todos los componentes necesarios"""
        logging.info("Cargando modelo...")
        logging.info(f"Intentando cargar desde path absoluto: {os.path.abspath(self.model_dir)}")  # Log para depuración en cloud
        # Cargar configuración
        try:
            config_path = os.path.join(self.model_dir, 'model_config.json')
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logging.info("Configuración cargada.")
        except FileNotFoundError:
            logging.error(f"Error: model_config.json no encontrado en {self.model_dir}")
            return False

        # Cargar modelo (formato .keras)
        model_path_keras = os.path.join(self.model_dir, 'dementia_model.keras')
        if os.path.exists(model_path_keras):
            try:
                self.model = tf.keras.models.load_model(model_path_keras)
                logging.info("Modelo .keras cargado exitosamente.")
            except Exception as e:
                logging.error(f"Error cargando modelo .keras: {e}")
                return False
        else:
            # Fallback to .h5 if .keras is not found (for backward compatibility)
            model_path_h5 = os.path.join(self.model_dir, 'dementia_model.h5')
            if os.path.exists(model_path_h5):
                try:
                    self.model = tf.keras.models.load_model(model_path_h5)
                    logging.info("Modelo .h5 cargado exitosamente como fallback.")
                except Exception as e:
                    logging.error(f"Error cargando modelo .h5: {e}")
                    return False
            else:
                logging.error(f"Error: Archivo de modelo no encontrado en {self.model_dir} (ni .keras ni .h5)")
                return False

        # Cargar preprocessors
        try:
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            self.scaler = joblib.load(scaler_path)
            logging.info("Scaler cargado.")
        except FileNotFoundError:
            logging.error(f"Error: scaler.pkl no encontrado en {self.model_dir}")
            return False

        try:
            # Load le_group as the primary label encoder
            le_group_path = os.path.join(self.model_dir, 'le_group.pkl')
            self.label_encoder = joblib.load(le_group_path)
            logging.info("Label encoder (Group) cargado.")
        except FileNotFoundError:
            logging.error(f"Error: le_group.pkl no encontrado en {self.model_dir}")
            # Fallback to older name if exists
            try:
                label_encoder_path = os.path.join(self.model_dir, 'label_encoder.pkl')
                self.label_encoder = joblib.load(label_encoder_path)
                logging.info("Label encoder (label_encoder.pkl) cargado como fallback.")
            except FileNotFoundError:
                logging.error(f"Error: label_encoder.pkl no encontrado en {self.model_dir}")
                return False

        # Cargar feature names
        try:
            feature_names_path = os.path.join(self.model_dir, 'feature_names.json')
            with open(feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
            logging.info("Nombres de features cargados.")
            logging.info(f"Features esperadas: {len(self.feature_names)}")
        except FileNotFoundError:
            logging.error(f"Error: feature_names.json no encontrado en {self.model_dir}")
            return False

        # Cargar encoders adicionales (e.g., le_gender)
        if 'additional_encoders' in self.config:
            for encoder_name, encoder_file in self.config['additional_encoders'].items():
                encoder_path = os.path.join(self.model_dir, encoder_file)
                if os.path.exists(encoder_path):
                    try:
                        self.additional_encoders[encoder_name] = joblib.load(encoder_path)
                        logging.info(f"Encoder adicional '{encoder_name}' cargado desde {encoder_file}.")
                    except Exception as e:
                        logging.warning(f"Advertencia: Error cargando encoder adicional '{encoder_name}' desde {encoder_file}: {e}")
                else:
                    logging.warning(f"Advertencia: Encoder adicional '{encoder_name}' no encontrado en {self.model_dir}.")

        return True

    def predict(self, data):
        """Realiza predicción sobre nuevos datos"""
        logging.info(f"Columnas en features_df (antes de validación): {list(data.columns)}")
        if self.model is None or self.scaler is None or self.label_encoder is None or self.feature_names is None:
            logging.error("Error: El modelo o los preprocesadores no se han cargado correctamente.")
            return None

        # Convertir a DataFrame si es necesario y asegurar nombres de columnas correctos
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, np.ndarray):
            if data.shape[1] != len(self.feature_names):
                raise ValueError(f"Número incorrecto de features. Esperado: {len(self.feature_names)}, Recibido: {data.shape[1]}")
            data = pd.DataFrame(data, columns=self.feature_names)
        elif not isinstance(data, pd.DataFrame):
            raise TypeError("El dato de entrada debe ser un diccionario, un array de NumPy o un DataFrame de Pandas.")

        # Asegurar que el DataFrame tenga las columnas esperadas en el orden correcto
        try:
            data = data[self.feature_names]
        except KeyError as e:
            logging.error(f"Error: Faltan features en los datos de entrada: {e}")
            return None

        # Escalar datos
        try:
            data_scaled = self.scaler.transform(data)
        except Exception as e:
            logging.error(f"Error durante el escalado de features: {e}")
            return None

        # Predicción
        try:
            # Predict returns a list of outputs for multi-output models
            predictions = self.model.predict(data_scaled, verbose=0)
            class_pred = predictions[0]
            reg_pred = predictions[1]
        except Exception as e:
            logging.error(f"Error durante la predicción del modelo: {e}")
            return None

        # Procesar resultados
        results = []
        # Ensure reg_pred is flattened for consistent processing
        reg_pred_flat = reg_pred.flatten()

        for i in range(len(data)):
            predicted_class_idx = np.argmax(class_pred[i])
            # inverse_transform expects a list-like input
            predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            class_probability = np.max(class_pred[i])
            predicted_mmse = reg_pred_flat[i] # Use flattened prediction

            result = {
                'classification': predicted_class,
                'probability': float(class_probability),
                'mmse_score': float(predicted_mmse),
                'class_probabilities': {
                    cls_name: float(prob) for cls_name, prob in zip(self.label_encoder.classes_, class_pred[i])
                }
            }
            results.append(result)

        return results[0] if len(results) == 1 else results

# Función de conveniencia para uso rápido
def predict_dementia(data, model_dir='assets'):  # Default a 'assets'; cambia a 'utils' si los archivos están allí
    """Función simple para predicción rápida"""
    predictor = DementiaPredictor(model_dir)
    if predictor.load_model():
        return predictor.predict(data)
    else:
        logging.error("No se pudo cargar el modelo para predicción.")
        return None

# Ejemplo de uso:
if __name__ == "__main__":
    # Directorio donde se exportó el modelo
    # Cambia esto si tu directorio de exportación es diferente (e.g., 'assets' o 'utils')
    model_export_directory = 'assets'  # Default a 'assets'; ajusta según tu estructura de repo

    predictor = DementiaPredictor(model_export_directory)

    # Intentar cargar el modelo y preprocesadores
    if predictor.load_model():
        logging.info("\nModelo cargado. Listo para predecir.")

        # Crear datos de ejemplo para la predicción
        # DEBES REEMPLAZAR ESTO CON DATOS REALES O GENERADOS CORRECTAMENTE
        # Asegúrate de que las features coincidan EXACTAMENTE con predictor.feature_names
        logging.info(f"Generando datos de ejemplo con {len(predictor.feature_names)} features...")
        # Create a dictionary with dummy data for prediction
        example_data_dict = {feature: 0.1 * (i + 1) for i, feature in enumerate(predictor.feature_names)}
        # If Gender is a feature and was label encoded, we need to handle it
        # Assuming 'Gender' is one of the features and was encoded by le_gender
        if 'Gender' in example_data_dict and 'gender' in predictor.additional_encoders:
            # Assuming 'M' maps to 1 and 'F' maps to 0 based on previous encoding
            # You might need to adjust this based on how le_gender was fitted
            example_data_dict['Gender'] = 1 # Example: Encode 'M' as 1

        try:
            # Realizar la predicción
            # Pass the dictionary directly to the predict method
            result = predictor.predict(example_data_dict)

            # Imprimir el resultado
            logging.info("\nResultado de la predicción:")
            logging.info(f"  Clasificación: {result['classification']}")
            logging.info(f"  Probabilidad (Clase Predicha): {result['probability']:.3f}")
            logging.info(f"  Score MMSE: {result['mmse_score']:.2f}")
            logging.info(f"  Probabilidades por clase: {result['class_probabilities']}")

        except ValueError as e:
            logging.error(f"Error al predecir: {e}")
        except TypeError as e:
            logging.error(f"Error de tipo al predecir: {e}")
        except Exception as e:
            logging.error(f"Ocurrió un error inesperado durante la predicción: {e}")

    else:
        logging.error("\nNo se pudo cargar el modelo. No se realizará la predicción de ejemplo.")
