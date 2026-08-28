import re
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.features.imputation import AdvancedDataImputer
from src.features.target_encoder import BayesianTargetEncoder


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Transformador de Scikit-Learn para seleccionar determinísticamente
    un subconjunto de columnas (ej. las variables seleccionadas por RFECV).
    """

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            cols_to_use = [c for c in self.columns if c in X.columns]
            return X[cols_to_use]
        return X


class TitanicFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Pipeline de Feature Engineering e Imputación Avanzada para Titanic MLOps.
    Hereda de BaseEstimator y TransformerMixin para integración nativa con sklearn.pipeline.Pipeline.
    Garantiza cero Data Leakage y consistencia determinística para inferencias unitarias.
    """

    AGE_BINS = [0, 12, 18, 35, 60, 100]
    AGE_LABELS = ["Child", "Teen", "Adult", "Middle", "Senior"]
    FAMILY_BINS = [0, 1, 4, 20]
    FAMILY_LABELS = ["Solo", "Small", "Large"]

    def __init__(self):
        self.age_imputer = AdvancedDataImputer(strategy="knn", n_neighbors=5, add_indicator=True)
        self.target_encoder = BayesianTargetEncoder(
            cols=["TicketPrefix", "CabinDeck", "Title", "Embarked"],
            m=10.0,
            cv=5,
            random_state=42,
        )
        self.fare_median_by_pclass: dict[int, float] = {}
        self.embarked_mode = "S"
        self.fitted_columns: list[str] | None = None
        self.categorical_categories: dict[str, list[str]] = {}
        self.ticket_frequency_map_: dict[str, int] = {}
        self.fare_quartiles_: list[float] = []

    @staticmethod
    def _extract_title(name: str) -> str:
        match = re.search(r",\s*([^\.]+)\.", name)
        if match:
            title = match.group(1).strip()
            if title in ["Mr", "Miss", "Mrs", "Master"]:
                return title
            elif title in ["Mlle", "Ms"]:
                return "Miss"
            elif title == "Mme":
                return "Mrs"
            elif title in ["Dr", "Prof"]:
                return "Rare_Prof"
            elif title in ["Rev"]:
                return "Rare_Clergy"
            elif title in ["Col", "Major", "Capt"]:
                return "Rare_Military"
            else:
                return "Rare"
        return "Rare"

    @staticmethod
    def _extract_cabin_deck(cabin: Any) -> str:
        if pd.isnull(cabin) or not str(cabin).strip():
            return "U"
        return str(cabin).strip()[0].upper()

    @staticmethod
    def _extract_ticket_prefix(ticket: Any) -> str:
        if pd.isnull(ticket):
            return "NUMERIC"
        t = str(ticket).replace(".", "").replace("/", "").strip()
        parts = t.split()
        if len(parts) > 1:
            return parts[0].upper()
        elif not parts[0].isdigit():
            return parts[0].upper()
        return "NUMERIC"

    def _build_ticket_frequency(self, df: pd.DataFrame) -> dict[str, int]:
        """Calcula la frecuencia de aparicion de cada numero de ticket (proxy de grupo)."""
        return df["Ticket"].value_counts().to_dict()

    def _add_new_features(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """
        Añade las features avanzadas de forma deterministica.
        En train calcula y almacena los parametros; en test los aplica.
        """
        # 1. FarePerPerson
        df["FarePerPerson"] = (df["Fare"] / df["FamilySize"]).round(4)

        # 2. TicketFrequency
        if is_train:
            self.ticket_frequency_map_ = self._build_ticket_frequency(df)
        df["TicketFrequency"] = df["Ticket"].map(self.ticket_frequency_map_).fillna(1).astype(int)

        # 3. AgeBin
        df["AgeBin"] = (
            pd.cut(
                df["Age"],
                bins=self.AGE_BINS,
                labels=self.AGE_LABELS,
                right=False,
                include_lowest=True,
            )
            .astype(str)
            .replace("nan", "Adult")
        )

        # 4. FareBin
        if is_train:
            self.fare_quartiles_ = df["Fare"].quantile([0.25, 0.50, 0.75]).tolist()
        fare_bins = [0.0] + self.fare_quartiles_ + [float("inf")]
        df["FareBin"] = (
            pd.cut(
                df["Fare"],
                bins=fare_bins,
                labels=["Q1", "Q2", "Q3", "Q4"],
                right=True,
                include_lowest=True,
            )
            .astype(str)
            .replace("nan", "Q1")
        )

        # 5. Pclass_Sex_Interaction
        df["Pclass_Sex_Interaction"] = df["Pclass"].astype(int) * (df["Sex"] == "female").astype(int)

        # 6. FamilyBin
        df["FamilyBin"] = (
            pd.cut(
                df["FamilySize"],
                bins=self.FAMILY_BINS,
                labels=self.FAMILY_LABELS,
                right=True,
                include_lowest=True,
            )
            .astype(str)
            .replace("nan", "Solo")
        )

        return df

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        """Ajusta todos los componentes del pipeline sobre el conjunto de entrenamiento."""
        df = X.copy()
        if y is None and "Survived" in df.columns:
            y = df["Survived"]
            df = df.drop(columns=["Survived"])

        df["Title"] = df["Name"].apply(self._extract_title)
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        df["HasCabin"] = df["Cabin"].notnull().astype(int)
        df["CabinDeck"] = df["Cabin"].apply(self._extract_cabin_deck)
        df["TicketPrefix"] = df["Ticket"].apply(self._extract_ticket_prefix)

        self.embarked_mode = df["Embarked"].mode()[0] if not df["Embarked"].empty else "S"
        df["Embarked"] = df["Embarked"].fillna(self.embarked_mode)
        self.fare_median_by_pclass = {int(k): float(v) for k, v in df.groupby("Pclass")["Fare"].median().items()}

        age_features = ["Age", "Pclass", "SibSp", "Parch", "FamilySize", "Fare"]
        imputed_age_df = self.age_imputer.fit_transform(df[age_features])
        df["Age"] = imputed_age_df["Age"]
        if "Age_is_na" in imputed_age_df.columns:
            df["Age_is_na"] = imputed_age_df["Age_is_na"].astype(int)

        df = self._add_new_features(df, is_train=True)

        if y is not None:
            self.target_encoder.fit(df, y)
            df_te = self.target_encoder.transform(df)
            for c in df_te.columns:
                df[c] = df_te[c]
        else:
            for col in self.target_encoder.cols:
                df[f"{col}_TE"] = 0.3838

        categorical_cols = ["Sex", "Embarked", "Title", "CabinDeck", "Pclass", "AgeBin", "FareBin", "FamilyBin"]
        for col in categorical_cols:
            self.categorical_categories[col] = sorted(list(df[col].astype(str).unique()))
            df[col] = pd.Categorical(df[col].astype(str), categories=self.categorical_categories[col])

        numerical_features = [
            "Age",
            "Fare",
            "FamilySize",
            "IsAlone",
            "HasCabin",
            "FarePerPerson",
            "TicketFrequency",
            "Pclass_Sex_Interaction",
            "TicketPrefix_TE",
            "CabinDeck_TE",
            "Title_TE",
            "Embarked_TE",
        ]
        if "Age_is_na" in df.columns:
            numerical_features.append("Age_is_na")

        features_to_encode = numerical_features + categorical_cols
        X_train_encoded = pd.get_dummies(df[features_to_encode], columns=categorical_cols, drop_first=True)
        self.fitted_columns = list(X_train_encoded.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforma un conjunto de datos (crudo) de forma determinística."""
        df = X.copy()
        if "Survived" in df.columns:
            df = df.drop(columns=["Survived"])

        df["Title"] = df["Name"].apply(self._extract_title)
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        df["HasCabin"] = df["Cabin"].notnull().astype(int)
        df["CabinDeck"] = df["Cabin"].apply(self._extract_cabin_deck)
        df["TicketPrefix"] = df["Ticket"].apply(self._extract_ticket_prefix)

        df["Embarked"] = df["Embarked"].fillna(self.embarked_mode)

        for pclass, median_fare in self.fare_median_by_pclass.items():
            mask = (df["Pclass"] == pclass) & (df["Fare"].isnull())
            df.loc[mask, "Fare"] = median_fare

        age_features = ["Age", "Pclass", "SibSp", "Parch", "FamilySize", "Fare"]
        imputed_age_df = self.age_imputer.transform(df[age_features])
        df["Age"] = imputed_age_df["Age"]
        if "Age_is_na" in imputed_age_df.columns:
            df["Age_is_na"] = imputed_age_df["Age_is_na"].astype(int)

        df = self._add_new_features(df, is_train=False)

        df_te = self.target_encoder.transform(df)
        for c in df_te.columns:
            df[c] = df_te[c]

        categorical_cols = ["Sex", "Embarked", "Title", "CabinDeck", "Pclass", "AgeBin", "FareBin", "FamilyBin"]
        for col in categorical_cols:
            df[col] = pd.Categorical(df[col].astype(str), categories=self.categorical_categories.get(col, []))

        numerical_features = [
            "Age",
            "Fare",
            "FamilySize",
            "IsAlone",
            "HasCabin",
            "FarePerPerson",
            "TicketFrequency",
            "Pclass_Sex_Interaction",
            "TicketPrefix_TE",
            "CabinDeck_TE",
            "Title_TE",
            "Embarked_TE",
        ]
        if "Age_is_na" in df.columns:
            numerical_features.append("Age_is_na")

        features_to_encode = numerical_features + categorical_cols
        df_encoded = pd.get_dummies(df[features_to_encode], columns=categorical_cols, drop_first=True)

        if self.fitted_columns is not None:
            for col in self.fitted_columns:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            df_encoded = df_encoded[self.fitted_columns]

        return df_encoded

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None):
        """
        Ajusta y transforma simultáneamente.
        Mantiene compatibilidad tanto para uso directo como para Pipelines de Scikit-Learn.
        """
        df = X.copy()
        if y is None and "Survived" in df.columns:
            y_series = df["Survived"]
            X_df = df.drop(columns=["Survived"])
            self.fit(X_df, y_series)
            X_encoded = self.transform(X_df)
            return X_encoded, y_series

        self.fit(df, y)
        X_encoded = self.transform(df)
        if y is not None:
            return X_encoded, y
        return X_encoded
