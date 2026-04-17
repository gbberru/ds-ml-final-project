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


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables derivadas útiles para modelado.
    """
    df = df.copy()

    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    df["bedrooms_per_household"] = df["total_bedrooms"] / df["households"]
    df["rooms_per_person"] = df["total_rooms"] / df["population"]

    return df


def clip_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Controla extremos en variables derivadas usando percentiles 1 y 99.
    """
    df = df.copy()

    derived_cols = [
        "rooms_per_household",
        "bedrooms_per_room",
        "population_per_household",
        "bedrooms_per_household",
        "rooms_per_person",
    ]

    for col in derived_cols:
        if col in df.columns:
            p1 = df[col].quantile(0.01)
            p99 = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=p1, upper=p99)

    return df


def encode_ocean_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte ocean_proximity a variables dummy.
    """
    df = df.copy()

    if "ocean_proximity" in df.columns:
        df = pd.get_dummies(df, columns=["ocean_proximity"], prefix="ocean", drop_first=True)

    return df


def create_log_features(
    df: pd.DataFrame,
    target: str = "median_house_value",
    skew_threshold: float = 1.0,
    drop_original: bool = False,
) -> pd.DataFrame:
    """
    Crea variables logarítmicas para columnas numéricas con alta asimetría positiva.
    Mantiene las columnas originales cuando drop_original=False.
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    predictor_cols = [col for col in numeric_cols if col != target]

    skewness = df[predictor_cols].skew()
    skewed_cols = skewness[skewness > skew_threshold].index.tolist()

    # Solo columnas no negativas
    skewed_cols = [col for col in skewed_cols if (df[col] >= 0).all()]

    for col in skewed_cols:
        df[f"log_{col}"] = np.log1p(df[col])

    if drop_original:
        df.drop(columns=skewed_cols, inplace=True)

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

    # 3. Crear variables derivadas
    df = create_features(df)

    # 4. Controlar extremos en variables derivadas
    df = clip_derived_features(df)

    # 5. Codificar variable categórica
    df = encode_ocean_proximity(df)

    # 6. Crear variables logarítmicas sin eliminar las originales
    """
    df = create_log_features(
        df,
        target="median_house_value",
        skew_threshold=1.0,
        drop_original=True
    )
    """
    return df


if __name__ == "__main__":
    train_input_path = Path("data/interim/train_set.csv")
    test_input_path = Path("data/interim/test_set.csv")

    train_output_path = Path("data/processed/train_features_delnull.csv")
    test_output_path = Path("data/processed/test_features_delnull.csv")

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