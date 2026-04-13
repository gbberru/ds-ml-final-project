"""
Script para dividir los datos en conjunto de entrenamiento y conjunto de prueba.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_and_save_data(raw_data_path: str, interim_data_path: str):
    """
    INSTRUCCIONES:
    1. Lee el archivo CSV descargado previamente en `raw_data_path` usando pandas.
    2. Separa los datos con `train_test_split()`. Te recomendamos un test_size=0.2 y random_state=42.
    3. (Opcional pero recomendado) Puedes usar `StratifiedShuffleSplit` basado en la variable
       del ingreso medio (median_income) para que la muestra sea representativa.
    4. Guarda los archivos resultantes (ej. train_set.csv y test_set.csv) en la carpeta `interim_data_path`.
    """
    # Crear carpeta de salida si no existe
    Path(interim_data_path).mkdir(parents=True, exist_ok=True)

    # Leer datos
    housing = pd.read_csv(raw_data_path)

    # Crear categoría de ingreso para estratificar
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, float("inf")],
        labels=[1, 2, 3, 4, 5]
    )

    # División train/test estratificada
    train_set, test_set = train_test_split(
        housing,
        test_size=0.2,
        random_state=42,
        stratify=housing["income_cat"]
    )

    # Quitar la columna auxiliar para no arrastrarla a los datos finales
    train_set = train_set.drop("income_cat", axis=1)
    test_set = test_set.drop("income_cat", axis=1)

    # Guardar resultados
    train_set.to_csv(Path(interim_data_path) / "train_set.csv", index=False)
    test_set.to_csv(Path(interim_data_path) / "test_set.csv", index=False)

if __name__ == "__main__":
    RAW_PATH = "data/raw/housing/housing.csv"
    INTERIM_PATH = "data/interim/"
    split_and_save_data(RAW_PATH, INTERIM_PATH)
    print("Script para dividir datos completado.")
