from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class PassengerInput(BaseModel):
    PassengerId: int | None = Field(default=1, description="Identificador único del pasajero")
    Pclass: int = Field(..., ge=1, le=3, description="Clase del ticket (1 = 1ª, 2 = 2ª, 3 = 3ª)", examples=[1])
    Name: str = Field(
        ...,
        min_length=2,
        max_length=200,
        description="Nombre completo del pasajero con título de cortesía",
        examples=["Cumings, Mrs. John Bradley (Florence Briggs Thayer)"],
    )
    Sex: str = Field(..., description="Género ('male' o 'female')", examples=["female"])
    Age: float | None = Field(
        None, ge=0.0, le=120.0, description="Edad en años (opcional, imputada si es null)", examples=[38.0]
    )
    SibSp: int = Field(default=0, ge=0, le=20, description="Número de hermanos / cónyuges a bordo", examples=[1])
    Parch: int = Field(default=0, ge=0, le=20, description="Número de padres / hijos a bordo", examples=[0])
    Ticket: str | None = Field(default="PC 17599", description="Número de boleto")
    Fare: float | None = Field(
        None, ge=0.0, le=1000.0, description="Tarifa del boleto (opcional, imputada si es null)", examples=[71.2833]
    )
    Cabin: str | None = Field(None, max_length=50, description="Número o letra de cabina/camarote", examples=["C85"])
    Embarked: str | None = Field(
        "S", description="Puerto de embarque ('C' = Cherbourg, 'Q' = Queenstown, 'S' = Southampton)", examples=["C"]
    )

    @field_validator("Sex")
    @classmethod
    def validate_and_normalize_sex(cls, v: str) -> str:
        v_clean = str(v).strip().lower()
        if v_clean not in ["male", "female"]:
            raise ValueError(f"Género inválido: '{v}'. Debe ser estrictamente 'male' o 'female'.")
        return v_clean

    @field_validator("Embarked")
    @classmethod
    def validate_and_normalize_embarked(cls, v: str | None) -> str:
        if v is None:
            return "S"
        v_clean = str(v).strip().upper()
        if v_clean not in ["S", "C", "Q"]:
            raise ValueError(f"Puerto de embarque inválido: '{v}'. Debe ser 'S', 'C', o 'Q'.")
        return v_clean

    @field_validator("Pclass")
    @classmethod
    def validate_pclass(cls, v: int) -> int:
        if v not in [1, 2, 3]:
            raise ValueError(f"Clase inválida: '{v}'. Debe ser 1, 2 o 3.")
        return v

    @model_validator(mode="after")
    def validate_domain_consistency(self) -> "PassengerInput":
        if self.Age is not None and self.Age < 0.0:
            raise ValueError("La edad no puede ser negativa.")
        if self.Fare is not None and self.Fare < 0.0:
            raise ValueError("La tarifa no puede ser negativa.")
        return self


class BatchPredictionInput(BaseModel):
    passengers: list[PassengerInput]


class PredictionOutput(BaseModel):
    passenger_id: int | None
    prediction: int = Field(..., description="0 = No sobrevivió, 1 = Sobrevivió")
    survival_probability: float = Field(..., description="Probabilidad estimada de supervivencia [0.0 - 1.0]")
    status: str = Field(..., description="Etiqueta descriptiva ('Survived' o 'Did Not Survive')")
    risk_level: str = Field(..., description="Nivel de probabilidad ('High', 'Moderate', 'Low')")


class BatchPredictionOutput(BaseModel):
    total_samples: int
    survival_rate: float
    predictions: list[PredictionOutput]


class ModelMetadataResponse(BaseModel):
    model_name: str
    framework: str
    algorithm: str
    cv_accuracy: float
    cv_roc_auc: float
    features_count: int
    top_shap_features: list[str]


class PredictionDriftResponse(BaseModel):
    sample_size: int
    baseline_samples: int
    probability_drift: dict[str, Any]
    decision_drift: dict[str, Any]
    overall_status: str
    is_drift_detected: bool
    summary: str


class InferenceMetricsResponse(BaseModel):
    total_inferences: int
    current_buffer_size: int
    average_survival_probability: float
    positive_prediction_rate: float
    optimal_threshold: float
    buffer_status: str
