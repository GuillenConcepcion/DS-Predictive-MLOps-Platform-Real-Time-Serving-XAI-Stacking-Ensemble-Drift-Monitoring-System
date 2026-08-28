import pandas as pd

from src.features.build_features import TitanicFeaturePipeline


def test_feature_pipeline_no_leakage():
    sample_train = pd.DataFrame(
        {
            "PassengerId": [1, 2],
            "Survived": [0, 1],
            "Pclass": [3, 1],
            "Name": ["Braund, Mr. Owen Harris", "Cumings, Mrs. John Bradley (Florence Briggs Thayer)"],
            "Sex": ["male", "female"],
            "Age": [22.0, None],
            "SibSp": [1, 1],
            "Parch": [0, 0],
            "Ticket": ["A/5 21171", "PC 17599"],
            "Fare": [7.25, 71.28],
            "Cabin": [None, "C85"],
            "Embarked": ["S", "C"],
        }
    )

    sample_test = pd.DataFrame(
        {
            "PassengerId": [892],
            "Pclass": [3],
            "Name": ["Kelly, Mr. James"],
            "Sex": ["male"],
            "Age": [34.5],
            "SibSp": [0],
            "Parch": [0],
            "Ticket": ["330911"],
            "Fare": [7.8292],
            "Cabin": [None],
            "Embarked": ["Q"],
        }
    )

    pipeline = TitanicFeaturePipeline()
    X_train, y_train = pipeline.fit_transform(sample_train)
    X_test = pipeline.transform(sample_test)

    assert len(X_train) == 2
    assert len(y_train) == 2
    assert len(X_test) == 1
    assert list(X_train.columns) == list(X_test.columns)
    assert not X_train.isnull().values.any()
    assert not X_test.isnull().values.any()
