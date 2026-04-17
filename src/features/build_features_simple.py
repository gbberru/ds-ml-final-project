"""
Módulo para limpieza y enriquecimiento (Feature Engineering) usando funciones simples.
"""
from pathlib import Path
import pandas as pd
import numpy as np

from pathlib import Path
import pandas as pd
import numpy as np


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa valores faltantes de total_bedrooms usando el ratio mediano:
    total_bedrooms / total_rooms.
    """
    df = df.copy()


    if "total_bedrooms" in df.columns:
        df = df.dropna(subset=["total_bedrooms"])

    return df


def remove_invalid_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina registros inválidos según reglas de coherencia lógica.
    """
    df = df.copy()

    valid_mask = (
        (df["total_rooms"] > 0) &
        (df["total_bedrooms"] > 0) &
        (df["population"] > 0) &
        (df["households"] > 0) &
        (df["median_income"] > 0) &
        (df["total_bedrooms"] <= df["total_rooms"]) &
        (df["households"] <= df["population"])
    )

    return df.loc[valid_mask].copy()



def encode_ocean_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte ocean_proximity a variables dummy.
    """
    df = df.copy()

    if "ocean_proximity" in df.columns:
        df = pd.get_dummies(df, columns=["ocean_proximity"], prefix="ocean", drop_first=True)

    return df


def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo de limpieza y enriquecimiento.
    """
    df = df.copy()

    # 1. Imputar faltantes
    df = impute_missing_values(df)

    # 2. Eliminar registros inválidos
    df = remove_invalid_records(df)

    # 5. Codificar variable categórica
    df = encode_ocean_proximity(df)

    return df


if __name__ == "__main__":
    train_input_path = Path("data/interim/train_set.csv")
    test_input_path = Path("data/interim/test_set.csv")

    train_output_path = Path("data/processed/train_features_simple.csv")
    test_output_path = Path("data/processed/test_features_simple.csv")

    # Crear carpeta de salida si no existe
    train_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Procesar train
    df_train = pd.read_csv(train_input_path)
    df_train_processed = preprocess_pipeline(df_train)
    df_train_processed.to_csv(train_output_path, index=False)

    # Procesar test
    df_test = pd.read_csv(test_input_path)
    df_test_processed = preprocess_pipeline(df_test)
    df_test_processed.to_csv(test_output_path, index=False)

    print("Proceso finalizado.")
    print(f"Train original: {df_train.shape}")
    print(f"Train procesado: {df_train_processed.shape}")
    print(f"Archivo guardado en: {train_output_path}")

    print(f"Test original: {df_test.shape}")
    print(f"Test procesado: {df_test_processed.shape}")
    print(f"Archivo guardado en: {test_output_path}")