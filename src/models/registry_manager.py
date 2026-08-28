"""
registry_manager.py — Gestor MLOps de Model Registry con Patrón Champion / Challenger.

Administra el ciclo de vida, versionado y promoción de modelos en MLflow Model Registry
utilizando aliases formales (@champion, @challenger, @previous_champion).
"""

import os
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.logger import logger

DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://172.17.212.149:5000")
REGISTERED_MODEL_NAME = "Titanic_Survival_Production_Pipeline"


class ModelRegistryManager:
    """Administrador del ciclo de vida y gobierno de modelos en MLflow Model Registry."""

    def __init__(self, tracking_uri: str | None = None):
        self.tracking_uri = tracking_uri or DEFAULT_TRACKING_URI
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def register_and_promote(
        self,
        model_name: str = REGISTERED_MODEL_NAME,
        run_id: str = "",
        artifact_path: str = "pipeline",
        candidate_auc: float = 0.0,
        baseline_threshold: float = 0.880,
    ) -> dict[str, Any]:
        """
        Registra una nueva versión de modelo y evalúa su promoción automática a @champion.

        Criterio:
            - Si candidate_auc > champion_auc y candidate_auc >= baseline_threshold:
                -> Nueva versión = @champion
                -> Versión anterior = @previous_champion / @challenger
            - De lo contrario:
                -> Nueva versión = @challenger
        """
        logger.info(
            f"Evaluando promoción en Model Registry para modelo '{model_name}' (Run ID: {run_id}, AUC: {candidate_auc:.4f})"
        )

        try:
            # 1. Crear el modelo registrado si no existe
            try:
                self.client.create_registered_model(
                    model_name,
                    description="Pipeline Atómico Productivo de Supervivencia Titanic (Feature Pipeline + RFECV + Calibrated Stacking)",
                )
                logger.info(f"Modelo registrado '{model_name}' creado en MLflow Registry.")
            except Exception:
                pass  # Ya existe

            # 2. Crear nueva versión vinculada al run_id
            model_uri = f"runs:/{run_id}/{artifact_path}"
            model_version = self.client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run_id,
                description=f"Versión candidata con CV ROC-AUC: {candidate_auc:.4f}",
            )
            v_number = model_version.version
            logger.success(f"Versión {v_number} creada para '{model_name}'.")

            # 3. Consultar si existe un @champion actual
            current_champion_version = None
            current_champion_auc = 0.0

            try:
                model_meta = self.client.get_registered_model(model_name)
                aliases = model_meta.aliases or {}
                if "champion" in aliases:
                    current_champion_version = aliases["champion"]
                    # Obtener métricas del run del champion actual
                    champ_ver_details = self.client.get_model_version(model_name, current_champion_version)
                    if champ_ver_details.run_id:
                        champ_run = self.client.get_run(champ_ver_details.run_id)
                        current_champion_auc = float(champ_run.data.metrics.get("cv_roc_auc", 0.0))
            except Exception as e:
                logger.warning(f"No se pudo consultar el @champion previo: {e}")

            # 4. Decisión de Promoción
            is_new_champion = False

            if current_champion_version is None:
                if candidate_auc >= baseline_threshold:
                    is_new_champion = True
                    logger.info(
                        f"Sin @champion previo. Candidato supera baseline ({baseline_threshold:.4f}) -> Nuevo @champion."
                    )
            else:
                logger.info(
                    f"Comparando Challenger v{v_number} (AUC: {candidate_auc:.4f}) vs Champion v{current_champion_version} (AUC: {current_champion_auc:.4f})"
                )
                if candidate_auc > current_champion_auc and candidate_auc >= baseline_threshold:
                    is_new_champion = True
                    logger.success(f"Challenger v{v_number} supera al Champion actual v{current_champion_version}!")

            # 5. Aplicar Aliases y Tags
            if is_new_champion:
                if current_champion_version:
                    self.client.set_registered_model_alias(model_name, "previous_champion", current_champion_version)
                    self.client.set_model_version_tag(
                        model_name, current_champion_version, "deployment_status", "demoted_champion"
                    )

                self.client.set_registered_model_alias(model_name, "champion", v_number)
                self.client.set_model_version_tag(model_name, v_number, "deployment_status", "champion")
                status = "promoted_to_champion"
                logger.success(f"🎉 Modelo v{v_number} promovido a @champion exitosamente.")
            else:
                self.client.set_registered_model_alias(model_name, "challenger", v_number)
                self.client.set_model_version_tag(model_name, v_number, "deployment_status", "challenger")
                status = "registered_as_challenger"
                logger.info(f"Modelo v{v_number} etiquetado como @challenger.")

            return {
                "status": status,
                "model_name": model_name,
                "version": v_number,
                "is_champion": is_new_champion,
                "candidate_auc": candidate_auc,
                "champion_auc": candidate_auc if is_new_champion else current_champion_auc,
            }

        except Exception as e:
            logger.error(f"Error gestionando promoción en Model Registry: {e}")
            return {
                "status": "error",
                "error": str(e),
                "model_name": model_name,
            }
