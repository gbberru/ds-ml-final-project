


from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score


TARGET = "median_house_value"


def train_best_model(processed_train_data_path: str, model_save_path: str):
    """
    Entrena el mejor modelo encontrado en la fase de experimentación
    y lo guarda en disco junto con la lista de columnas usadas.
    """
    # Cargar datos procesados de entrenamiento
    df_train = pd.read_csv(processed_train_data_path)

    # Separar predictoras y target
    X_train = df_train.drop(columns=[TARGET])
    y_train = df_train[TARGET]

    # Instanciar mejor modelo con los mejores hiperparámetros encontrados
    model = LGBMRegressor(
        colsample_bytree=0.8,
        learning_rate=0.05,
        max_depth=-1,
        min_child_samples=15,
        n_estimators=500,
        num_leaves=50,
        reg_alpha=0.1,
        reg_lambda=0.0,
        subsample=0.8,
        random_state=42,
        n_jobs=-1
    )

    # Entrenar
    model.fit(X_train, y_train)

    # Guardar modelo + columnas
    model_artifact = {
        "model": model,
        "features": X_train.columns.tolist(),
        "target": TARGET
    }

    model_save_path = Path(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_artifact, model_save_path)

    print("Entrenamiento finalizado.")
    print(f"Train original: {df_train.shape}")
    print(f"Número de variables predictoras: {X_train.shape[1]}")
    print(f"Modelo guardado en: {model_save_path}")


def evaluate_model(model_path: str, processed_test_data_path: str):
    """
    Carga el modelo guardado y lo evalúa sobre el set de prueba.
    """
    # Cargar modelo
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_names = artifact["features"]
    target = artifact["target"]

    # Cargar test
    df_test = pd.read_csv(processed_test_data_path)

    X_test = df_test.drop(columns=[target])
    y_test = df_test[target]

    # Alinear columnas de test con train
    X_test = X_test.reindex(columns=feature_names, fill_value=0)

    # Predicción
    y_pred = model.predict(X_test)

    # Métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n=== Resultados en TEST ===")
    print(f"LGBM final -> RMSE: {rmse:,.2f} | R2: {r2:.4f}")


if __name__ == "__main__":
    PROCESSED_TRAIN_PATH = "data/processed/train_features_sin_log.csv"
    PROCESSED_TEST_PATH = "data/processed/test_features_sin_log.csv"
    MODEL_OUTPUT_PATH = "models/best_lgbm_model.joblib"

    train_best_model(PROCESSED_TRAIN_PATH, MODEL_OUTPUT_PATH)
    evaluate_model(MODEL_OUTPUT_PATH, PROCESSED_TEST_PATH)