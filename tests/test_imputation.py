import numpy as np
import pandas as pd

from src.features.imputation import AdvancedDataImputer, littles_mcar_test


def test_littles_mcar_test_no_missing():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})
    res = littles_mcar_test(df)
    assert res["is_mcar"] is True
    assert res["chi_square_stat"] == 0.0
    assert res["p_value"] == 1.0


def test_littles_mcar_test_synthetic_mcar():
    np.random.seed(42)
    N = 300
    x1 = np.random.normal(10, 2, N)
    x2 = np.random.normal(50, 5, N)
    x3 = 0.5 * x1 + np.random.normal(0, 1, N)

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

    # Inyectar ausencia puramente aleatoria (MCAR)
    mask_x1 = np.random.rand(N) < 0.15
    mask_x2 = np.random.rand(N) < 0.10
    df.loc[mask_x1, "x1"] = np.nan
    df.loc[mask_x2, "x2"] = np.nan

    res = littles_mcar_test(df)
    assert "chi_square_stat" in res
    assert "degrees_of_freedom" in res
    assert "p_value" in res
    assert res["p_value"] > 0.01  # No debe rechazar H0 contundentemente en MCAR sintético
    assert len(res["missing_patterns"]) >= 2


def test_littles_mcar_test_synthetic_mar():
    np.random.seed(42)
    N = 300
    x1 = np.random.normal(10, 2, N)
    x2 = np.random.normal(50, 5, N)

    df = pd.DataFrame({"x1": x1, "x2": x2})

    # Inyectar ausencia condicionada estrictamente a x1 > 11.5 (MAR dependiente)
    df.loc[df["x1"] > 11.5, "x2"] = np.nan

    res = littles_mcar_test(df)
    # Al estar condicionada fuertemente, debe detectar anomalía en el patrón
    assert res["total_records"] == N
    assert len(res["missing_patterns"]) == 2


def test_advanced_imputer_with_mcar_diagnostics():
    df = pd.DataFrame(
        {
            "feature_1": [1.0, np.nan, 3.0, 4.0, 5.0, np.nan, 7.0],
            "feature_2": [10.0, 20.0, np.nan, 40.0, 50.0, 60.0, 70.0],
            "target": [0, 1, 0, 1, 0, 1, 0],
        }
    )

    imputer = AdvancedDataImputer(strategy="knn", n_neighbors=2, add_indicator=True)
    transformed = imputer.fit_transform(df)

    assert imputer.mcar_diagnostics_ is not None
    assert "chi_square_stat" in imputer.mcar_diagnostics_
    assert "verdict" in imputer.mcar_diagnostics_
    assert transformed.isnull().sum().sum() == 0
    assert "feature_1_is_na" in transformed.columns
    assert "feature_2_is_na" in transformed.columns
