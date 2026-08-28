import numpy as np
import pandas as pd

from src.features.target_encoder import BayesianTargetEncoder


def test_bayesian_target_encoder_basic():
    df = pd.DataFrame({"Category": ["A", "A", "A", "B", "B", "C"], "Target": [1, 1, 0, 0, 0, 1]})

    encoder = BayesianTargetEncoder(cols=["Category"], m=2.0, cv=3)
    encoded_oof = encoder.fit_transform(df[["Category"]], df["Target"])

    assert "Category_TE" in encoded_oof.columns
    assert len(encoded_oof) == len(df)
    assert not encoded_oof["Category_TE"].isnull().any()

    # Test transform with seen and unseen categories
    df_new = pd.DataFrame({"Category": ["A", "B", "UNKNOWN_CATEGORY"]})
    encoded_new = encoder.transform(df_new)

    assert len(encoded_new) == 3
    assert not encoded_new["Category_TE"].isnull().any()
    # UNKNOWN_CATEGORY should receive the global mean
    np.testing.assert_almost_equal(encoded_new.iloc[2]["Category_TE"], encoder.global_mean_, decimal=4)


def test_bayesian_target_encoder_smoothing_effect():
    # Category with 1 sample vs Category with 100 samples
    df = pd.DataFrame(
        {
            "Category": ["RARE"] * 1 + ["FREQUENT"] * 100,
            "Target": [1] + [1] * 80 + [0] * 20,  # Target mean: RARE=1.0, FREQUENT=0.8
        }
    )

    encoder = BayesianTargetEncoder(cols=["Category"], m=10.0, cv=2)
    encoder.fit(df[["Category"]], df["Target"])

    encoded = encoder.transform(df[["Category"]])
    global_mean = df["Target"].mean()

    rare_encoded = encoded[df["Category"] == "RARE"]["Category_TE"].iloc[0]
    freq_encoded = encoded[df["Category"] == "FREQUENT"]["Category_TE"].iloc[0]

    # RARE should be strongly pulled towards global_mean
    assert abs(rare_encoded - global_mean) < abs(1.0 - global_mean)
    # FREQUENT should be close to 0.8
    assert abs(freq_encoded - 0.8) < 0.05
