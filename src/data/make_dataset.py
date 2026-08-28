from pathlib import Path

import pandas as pd

from src.utils.logger import logger


def load_raw_data(base_path: Path = Path("data/raw")):
    train_path = base_path / "train.csv"
    test_path = base_path / "test.csv"

    logger.info(f"Cargando dataset de entrenamiento desde: {train_path}")
    df_train = pd.read_csv(train_path)

    logger.info(f"Cargando dataset de prueba desde: {test_path}")
    df_test = pd.read_csv(test_path)

    logger.success(f"Datos cargados exitosamente -> Train: {df_train.shape}, Test: {df_test.shape}")
    return df_train, df_test


if __name__ == "__main__":
    load_raw_data()
