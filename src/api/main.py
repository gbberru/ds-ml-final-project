"""
API Básica usando FastAPI para servir el modelo entrenado.
"""
from fastapi import FastAPI
import joblib
import pandas as pd

# Crear app
app = FastAPI(title="Modelo de Predicción de Viviendas")

# Cargar modelo al iniciar
MODEL_PATH = "models/best_lgbm_model.joblib"
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
feature_names = artifact["features"]

# Endpoint raíz
@app.get("/")
def home():
    return {"mensaje": "API de predicción de precios de viviendas funcionando"}

# Endpoint de predicción
@app.post("/predict")
def predict(data: dict):
    """
    Recibe un JSON con las variables del modelo
    y devuelve el precio estimado
    """

    # Convertir input a DataFrame
    df = pd.DataFrame([data])

    # Alinear columnas con el modelo
    df = df.reindex(columns=feature_names, fill_value=0)

    # Predicción
    prediction = model.predict(df)[0]

    return {
        "predicted_price": round(float(prediction), 2)
    }