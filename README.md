# 🚢 Predictive MLOps Platform: Real-Time Serving, XAI, Stacking Ensemble & Drift Monitoring System
### *Sistema de Grado de Producción para Modelado Predictivo, Inferencia Atómica Contenerizada y Gobernanza MLOps*

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Serving%20API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Atomic%20Pipeline-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Champion%20Registry-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Podman](https://img.shields.io/badge/Podman-Rootless%20Container-892CA0?logo=podman&logoColor=white)](https://podman.io/)
[![Pytest](https://img.shields.io/badge/Pytest-27%20Passed%20(100%25)-brightgreen?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Ruff](https://img.shields.io/badge/Code%20Style-Ruff%200%20Errors-black)](https://astral.sh/ruff)

**Lead Architect:** [Guillén Concepción](https://www.linkedin.com/in/guillen-concepcion-25266b127) *(Senior Data Scientist & MLOps Engineer)*  
**Contacto:** [LinkedIn](https://www.linkedin.com/in/guillen-concepcion-25266b127) • [GitHub](https://github.com/GuillenConcepcion) • [Email](mailto:guillenconcepcion@gmail.com)

---

## 💡 Motivación y Propósito: De Cuadernos Experimentales a Arquitecturas MLOps Enterprise

> *"En la industria del Machine Learning, más del 85% de los modelos predictivos nunca logran desplegarse en producción. Aquellos que lo logran frecuentemente colapsan ante problemas invisibles: Data Leakage, fallas silenciosas en la imputación de datos, falta de explicabilidad para el negocio, degradación por drift y servicios de inferencia frágiles acoplados manualmente."*

Este proyecto nace con la misión de ser un **Framework Canónico y Arquitectura de Referencia MLOps**, tomando el icónico problema de clasificación tabular del *Titanic* no como un ejercicio académico elemental, sino como un **banco de pruebas de grado Senior** para demostrar cómo transformar un problema clásico en un sistema de producción robusto, auditable, escalable y matemáticamente riguroso.

### 🌟 Los 5 Pilares Fundamentales de este Proyecto:
1. **🔬 Rigor Estadístico y Contraste de Hipótesis:** Reemplazo de imputaciones intuitivas por el contraste formal de **Little's MCAR Test (1988)** ($p=2.84\times 10^{-7} \implies \text{Ausencia MAR}$), guiando el uso fundamentado de `KNNImputer` y `MissingIndicator`.
2. **🛡️ Ingeniería de Características sin Fugas (*Zero Data Leakage*):** Implementación de **Bayesian Target Encoding OOF con $m$-estimate** ($m=10.0$) y selección recursiva con **RFECV (14 variables élite)**, garantizando representaciones óptimas sin sesgo de optimismo.
3. **🧠 Ensambles Calibrados de 2 Niveles (Stacking Generalization):** Combinación de algoritmos basados en árboles (`GBM`, `LightGBM`, `XGBoost`) coordinados por un Meta-Learner con regularización Ridge ($L_2$) y calibración monótona con **Isotonic Calibration (PAVA)**, alcanzando un **ROC-AUC récord de 89.32%**.
4. **⚡ Arquitectura de Inferencia Atómica & Observabilidad en Tiempo Real:** Unificación de todo el flujo en un único objeto serializado `sklearn.pipeline.Pipeline`, servido mediante un microservicio **FastAPI con contratos defensivos Pydantic V2** y monitoreo continuo de **Data & Prediction Drift** (PSI, Kolmogorov-Smirnov, Wasserstein, TVD).
5. **🐳 Infraestructura Cloud-Native & Gobernanza Automatizada:** Empaquetado **Multi-Stage Rootless en Podman / Docker** (UID 10001), ciclo de vida **Champion / Challenger en MLflow Registry** y una puerta de calidad infranqueable (**Performance & Quality Regression Gate**) en CI/CD que protege el SLA en producción.

---

## 📊 1. Resumen Ejecutivo de Métricas — Champion Stacking Pipeline v3

### 🏆 Modelo Champion: `Titanic_Survival_Production_Pipeline` (Promovido a `@champion` en MLflow)

| Métrica MLOps | CV 5-Fold (OOF) | Variación Acumulada |
| :--- | :---: | :---: |
| **ROC-AUC Score** | **0.8932 (89.32%)** | **+0.80%** 🚀 *(Nuevo Récord Histórico)* |
| **Accuracy (Exactitud)** | **84.06%** | Balance óptimo con regularización $L_2$ |
| **F1-Score (Macro)** | **83.08%** | **+0.08%** ⬆️ |
| **Threshold Óptimo** | **0.390** (OOF-F1 Max) | Calibrado isotónicamente |
| **Arquitectura de Ensamble** | Stacking 2-Level (`GBM + LGBM + XGB` $\to$ `LogisticRegression L2`) | Meta-Learner: $w_{xgb}=1.41$, $w_{gbm}=1.39$, $w_{lgbm}=1.17$ |
| **Selección de Variables** | RFECV (14 variables élite) | Reducción de 38 a 14 features sin pérdida de señal |
| **Encoding Categórico** | Bayesian Target Encoding OOF ($m=10.0$) | Sin Data Leakage en CV (`TicketPrefix`, `CabinDeck`, `Title`, `Embarked`) |
| **Empaquetado de Inferencia** | Pipeline Atómico Unificado | Inferencia directa `pipeline.predict_proba(df)` en 1 línea |

### Evolución Comparativa de Modelos

| Iteración / Modelo | CV ROC-AUC | CV Accuracy | CV F1-Macro | Estrategia Clave |
| :--- | :---: | :---: | :---: | :--- |
| **v1: Baseline XGBoost** | 88.23% | 84.40% | 83.16% | Feature Engineering v1 (23 variables), One-Hot |
| **v2: Optuna GBM (100 trials)** | 89.19% | 84.18% | 83.00% | Feature Engineering v2 (38 variables), Bins continuos |
| **v2.1: Voting Ensemble** | 88.94% | 83.95% | 82.65% | Soft blending ($1/3$ GBM, $1/3$ LGBM, $1/3$ XGB) |
| **v3: Stacking Champion Pipeline** | **89.32%** | **84.06%** | **83.08%** | **Bayesian TE + RFECV 14 + Meta-Learner $L_2$ + Calibración** |

---

## 🏛 2. Arquitectura del Pipeline Atómico de Producción

Todo el ciclo de transformación, filtrado de variables e inferencia calibrada está encapsulado en un único artefacto serializado: [`models/titanic_production_pipeline.pkl`](models/titanic_production_pipeline.pkl):

```
+---------------------------------------------------------------------------------------------------+
|                            TITANIC ATOMIC PRODUCTION PIPELINE                                     |
+---------------------------------------------------------------------------------------------------+
|  [Raw Payload] (Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked)              |
|                                         |                                                         |
|                                         v                                                         |
|  Step 1: 'features' -> TitanicFeaturePipeline (TransformerMixin)                                  |
|          - Imputación KNN + Indicadores de Ausencia (Little's MCAR compliance)                    |
|          - Extracción de Títulos, Prefijos de Ticket y Decks de Cabina                            |
|          - Target Encoding Bayesiano OOF (m=10.0)                                                 |
|          - Discretización AgeBin & FareBin + One-Hot Encoding                                     |
|                                         |                                                         |
|                                         v (38 variables generadas)                                |
|  Step 2: 'selector' -> ColumnSelector (TransformerMixin)                                          |
|          - Selección determinística de las 14 características élite filtradas por RFECV           |
|                                         |                                                         |
|                                         v (14 variables de alta señal)                            |
|  Step 3: 'model' -> CalibratedClassifierCV(StackingClassifier)                                    |
|          - Base: GradientBoosting + LightGBM + XGBoost                                            |
|          - Meta-Learner: LogisticRegression(C=0.1, L2)                                            |
|          - Calibración: Isotonic Calibration (cv=5)                                               |
|                                         |                                                         |
|                                         v                                                         |
|  [Inference Output] pipeline.predict_proba(df)[:, 1] -> Probabilidades Calibradas                 |
+---------------------------------------------------------------------------------------------------+
```

---

## 🎯 3. Innovaciones Técnicas & Gobierno MLOps

### 3.1. Target Encoding Bayesiano Out-of-Fold (OOF) con $m$-estimate
Para capturar la señal predictiva de variables de media/alta cardinalidad (`TicketPrefix`, `CabinDeck`, `Title`, `Embarked`) sin inducir sobreajuste ni *Target Leakage*:
$$\hat{S}_i = \frac{n_i \cdot \bar{y}_i + m \cdot \mu_{\text{global}}}{n_i + m}$$
Donde $m=10.0$ actúa como parámetro de regularización empírico que contrae las categorías poco frecuentes hacia la tasa global de supervivencia.

### 3.2. Selección de Características con RFECV
Mediante Eliminación Recursiva de Características con Validación Cruzada (`RFECV`), se eliminó el ruido colineal, reduciendo el espacio dimensional de 38 a **14 variables de élite**:
* `Sex_female`, `Pclass_3`, `Title_Mr`, `TicketPrefix_TE`, `FarePerPerson`, `FareBin_Q4`, `IsAlone`, `Age`, `Fare`, `FamilySize`, `CabinDeck_TE`, `Title_TE`, `Embarked_TE`, `Age_Is_Missing`.

### 3.3. MLflow Model Registry Champion / Challenger Pattern
El módulo [`src/models/registry_manager.py`](src/models/registry_manager.py) automatiza la promoción en el Model Registry:
* Evalúa la métrica ROC-AUC del candidato frente al modelo actual en producción.
* Asigna dinámicamente el alias **`@champion`** al superar el baseline ($0.8800$), permitiendo despliegues Canary y rollback inmediato.

### 3.4. Contratos de Datos Estrictos con Pydantic V2
En [`src/serving/schemas.py`](src/serving/schemas.py), la clase `PassengerInput` implementa:
* `@field_validator`: Normalización de texto (`Sex`, `Embarked`), validación de clases (`Pclass` $\in \{1,2,3\}$) y plausibilidad física de rangos (`Age` $\in [0, 120]$, `Fare` $\in [0, 1000]$).
* `@model_validator(mode='after')`: Validación de consistencia cruzada de dominio.

### 3.5. Observabilidad Integral: Data & Prediction Drift
* **Data Drift:** KS 2-Sample Test, Population Stability Index (PSI) y Chi-Square en variables de entrada.
* **Prediction Drift ($\hat{p}$ y $\hat{y}$):** Monitoreo continuo del desplazamiento en la confianza del modelo y la tasa de supervivencia predicha, con alertas `STABLE`, `MODERATE_DRIFT` y `CRITICAL_DRIFT`.
* **Endpoints:** `GET /monitoring/prediction-drift`, `GET /monitoring/inference-metrics`, `GET /monitoring/drift/dashboard`.

### 3.6. Performance & Quality Regression Gate en CI/CD
El script [`src/models/performance_gate.py`](src/models/performance_gate.py) actúa como puerta de control de calidad en el pipeline de CI/CD ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)), bloqueando la integración si:
* $ROC\text{-}AUC_{\text{OOF}} < 0.880$ ✅ (`0.8932`)
* $Accuracy_{\text{OOF}} < 0.830$ ✅ (`0.8406`)
* $F_{1}\text{-}Macro_{\text{OOF}} < 0.820$ ✅ (`0.8308`)
* Latencia SLA $P_{95} > 250\text{ ms}$ por inferencia individual ✅ (`175.57 ms`)

---

## 🔬 4. Diagnóstico de Ausencia: Little's MCAR Test (1988)

- **Estadístico Chi-Cuadrado ($d^2$):** `36.036` ($df=4$)
- **P-Valor:** `2.8447e-07` ($p \le 0.05$)
- **Veredicto:** **Rechaza $H_0$ (Ausencia MAR - Missing at Random)**. La ausencia de edad depende de la clase social y la estructura familiar, justificando formalmente el uso de `KNNImputer` y `MissingIndicator` frente a imputaciones univariadas.

---

## 🧠 5. Explicabilidad XAI y Top Características (SHAP)

| Ranking | Feature | Mean \|SHAP Value\| | Interpretación de Negocio |
| :---: | :--- | :---: | :--- |
| **1** | `Title_Mr` | **0.9259** | Mayor penalización predictiva (prioridad de evacuación a mujeres y niños). |
| **2** | `Pclass_3` | **0.4018** | Fuerte impacto negativo asociado a la 3ª clase y ubicación en cubiertas inferiores. |
| **3** | `Sex_male` | **0.3808** | Factor clave de supervivencia histórica en el naufragio. |
| **4** | `Fare` | **0.3710** | Tarifas más altas correlacionan positivamente con cabinas premium y acceso a botes. |
| **5** | `Age` | **0.2885** | Menores de edad obtienen mayor probabilidad de supervivencia condicionada por clase. |
| **6** | `FamilySize` | **0.2233** | Familias pequeñas (2-4 miembros) maximizan supervivencia frente a pasajeros solitarios o familias numerosas. |
| **7** | `HasCabin` | **0.1812** | Registro de camarote asignado actúa como indicador de supervivencia. |

---

## 📁 6. Catálogo de Artefactos Generados

| Artefacto | Ruta | Tipo / Formato | Descripción |
| :--- | :--- | :--- | :--- |
| **Pipeline Champion v3** | [`models/titanic_production_pipeline.pkl`](models/titanic_production_pipeline.pkl) | Scikit-Learn Pipeline | Pipeline unificado (`Features -> Bayesian TE -> RFECV -> Calibrated Stacking`). |
| **Metadata del Modelo** | [`models/stacking_metadata.json`](models/stacking_metadata.json) | JSON | Pesos del meta-learner, umbral óptimo (0.390) y métricas CV. |
| **Variables RFECV** | [`models/rfecv_selected_features.json`](models/rfecv_selected_features.json) | JSON | Lista de las 14 variables de alta relevancia seleccionadas. |
| **Reporte Performance Gate** | [`reports/performance_gate_report.json`](reports/performance_gate_report.json) | JSON | Resultados de la auditoría de calidad y latencia SLA en CI/CD. |
| **Importancia SHAP** | [`reports/shap_feature_importance.csv`](reports/shap_feature_importance.csv) | CSV tabular | Ranking SHAP de interpretabilidad global con TreeExplainer. |
| **Dashboard EDA & BI** | [`reports/eda_bi_dashboard.html`](reports/eda_bi_dashboard.html) | Dashboard HTML5 | Visualización ejecutiva interactiva con KPIs demográficos y BI. |
| **Reporte de Data Drift** | [`reports/live_drift_report.html`](reports/live_drift_report.html) | Dashboard HTML5 | Auditoría continua con Kolmogorov-Smirnov y PSI. |
| **Predicciones Kaggle** | [`data/processed/titanic_stacking_submission.csv`](data/processed/titanic_stacking_submission.csv) | CSV oficial | Predicciones generadas por el pipeline atómico champion. |

---

## 🐳 7. Contenerización Cloud-Native Rootless (Podman / Docker)

Construcción optimizada mediante **Multi-Stage Build con `uv`** y usuario sin privilegios (**UID 10001**):

```bash
# 1. Construir imagen con Podman
podman build -t odysseus-titanic-api:latest -f Containerfile .

# 2. Ejecutar contenedor en modo Rootless
podman run -d -p 8001:8001 --name titanic-serving odysseus-titanic-api:latest

# 3. Comprobar salud del microservicio
curl -f http://localhost:8001/health
```

---

## 🧪 8. Suite de Pruebas Automatizadas (27/27 Tests Aprobados)

```bash
# Ejecutar suite completa con cobertura
uv run pytest --cov=src --cov-report=term-missing tests/
```

* **`tests/test_benchmark.py`**: Inicialización y compatibilidad de estimadores.
* **`tests/test_imputation.py`**: Validación de Little's MCAR Test y KNN Imputer.
* **`tests/test_monitoring.py`**: Algoritmos de PSI, KS-Test, Wasserstein y TVD.
* **`tests/test_pipeline.py`**: Integridad del pipeline sin data leakage.
* **`tests/test_target_encoder.py`**: Contracción Bayesiana $m$-estimate OOF.
* **`tests/test_prediction_drift.py`**: Validación Pydantic V2 y shift de probabilidades.
* **`tests/test_serving.py`**: Endpoints FastAPI unitarios, por lotes y dashboards.
* **`tests/test_performance_gate.py`**: Verificación de SLAs de calidad y latencia.

---

## 📐 9. Fundamentos Matemáticos y Formulaciones Algorítmicas

### 9.1. Little's MCAR Test (1988) — Diagnóstico de Ausencia Multivariada
Evalúa el contraste de hipótesis $H_0: \text{Datos MCAR}$ vs $H_1: \text{Datos MAR/MNAR}$ mediante estimación de máxima verosimilitud vía Expectation-Maximization (EM):
$$d^2 = \sum_{s=1}^S N_s \left( \mathbf{\bar{y}}_{\text{obs}, s} - \mathbf{\hat{\mu}}_s \right)^T \mathbf{\hat{\Sigma}}_s^{-1} \left( \mathbf{\bar{y}}_{\text{obs}, s} - \mathbf{\hat{\mu}}_s \right)$$
Donde $S$ es el número de patrones únicos de ausencia, $N_s$ es la frecuencia del patrón $s$, $\mathbf{\bar{y}}_{\text{obs}, s}$ es el vector de medias muestrales observadas, y $\mathbf{\hat{\mu}}_s, \mathbf{\hat{\Sigma}}_s$ son los subvectores y submatrices estimados por EM. Grados de libertad:
$$df = \left( \sum_{s=1}^S P_s \right) - P$$
* **Resultado Obtenido:** $d^2 = 36.036$, $df = 4$, $p = 2.84\times 10^{-7} < 0.05 \implies$ **Rechazo estricto de $H_0$** (justifica `KNNImputer` y `MissingIndicator`).

### 9.2. Bayesian Target Encoding con $m$-estimate (Micci-Barreca 2001)
Suavizado bayesiano empírico ajustado estrictamente Out-of-Fold (OOF) para evitar *Target Leakage*:
$$\hat{S}_i = \lambda(n_i) \cdot \bar{y}_i + (1 - \lambda(n_i)) \cdot \mu_{\text{global}} = \frac{n_i \cdot \bar{y}_i + m \cdot \mu_{\text{global}}}{n_i + m}$$
Donde $n_i$ es el conteo muestral de la categoría $i$, $\bar{y}_i$ es la media observada del target, $\mu_{\text{global}}$ es la tasa global de supervivencia, y $m=10.0$ es el hiperparámetro de regularización que contrae categorías de baja frecuencia hacia el prior global.

### 9.3. Stacking Classifier con Meta-Learner Regularizado $L_2$ (Wolpert 1992)
La probabilidad final de supervivencia se modela mediante regresión logística sobre las probabilidades OOF de los modelos base ($M=3$: GBM, LGBM, XGB):
$$P(\text{Survived}=1 \mid \mathbf{x}) = \sigma\left( \beta_0 + \sum_{m=1}^M \beta_m \hat{p}_m(\mathbf{x}) \right) = \frac{1}{1 + \exp\left( -\left(\beta_0 + \sum_{m=1}^M \beta_m \hat{p}_m(\mathbf{x})\right) \right)}$$
Sujeto a penalización Ridge ($L_2$ con $C=0.1$):
$$\min_{\boldsymbol{\beta}} \left\{ -\sum_{i=1}^N \left[ y_i \ln \hat{y}_i + (1 - y_i) \ln (1 - \hat{y}_i) \right] + \frac{1}{2C} \|\boldsymbol{\beta}\|_2^2 \right\}$$
* **Pesos Aprendidos en Producción:** $\beta_0 = -2.1346$, $\beta_{\text{XGB}} = +1.4118$, $\beta_{\text{GBM}} = +1.3891$, $\beta_{\text{LGBM}} = +1.1699$.

### 9.4. Calibración Isotónica de Probabilidades (PAVA Algorithm)
Ajuste no paramétrico monótono para transformar scores en probabilidades reales calibradas:
$$\min_{\hat{m}} \sum_{i=1}^N (y_i - \hat{m}(f_i))^2 \quad \text{sujeto a } \hat{m}(f_i) \le \hat{m}(f_j) \text{ siempre que } f_i \le f_j$$
Resuelto mediante el algoritmo *Pool Adjacent Violators* (PAVA) en validación cruzada estratificada de 5 folds.

### 9.5. Population Stability Index (PSI) & Jeffreys Divergence
Cuantifica la divergencia simétrica de Kullback-Leibler entre la distribución base ($Expected$) y la observada en producción ($Actual$):
$$\text{PSI} = \sum_{b=1}^B \left( \%Actual_b - \%Expected_b \right) \times \ln\left( \frac{\%Actual_b + \epsilon}{\%Expected_b + \epsilon} \right)$$
Binning adaptativo: $B = \min\left(10, \max\left(3, \left\lfloor \frac{N}{25} \right\rfloor \right)\right)$.
* $\text{PSI} < 0.10$: `STABLE` | $0.10 \le \text{PSI} < 0.25$: `MODERATE_DRIFT` | $\text{PSI} \ge 0.25$: `CRITICAL_DRIFT`.

### 9.6. Test de Kolmogorov-Smirnov (KS 2-Sample)
Evalúa el cambio de forma en distribuciones continuas mediante la distancia supremum entre funciones empíricas de distribución acumulada (ECDF):
$$D = \sup_x |F_{\text{ref}}(x) - F_{\text{curr}}(x)|$$

### 9.7. Distancia de Wasserstein ($W_1$ - Earth Mover's Distance)
Mide el costo mínimo de transporte de masa de probabilidad:
$$W_1(u, v) = \int_{-\infty}^{\infty} |U(x) - V(x)| dx$$

### 9.8. Explicabilidad Aditiva con SHAP (Shapley Values)
Descomposición unívoca y aditiva basada en teoría de juegos cooperativos:
$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]$$
Garantiza propiedades de **Eficiencia**, **Simetría**, **Dummy** y **Aditividad**.

---

## 📚 10. Bibliografía Científica y Referencias Canónicas

1. **Little, Roderick J. A. (1988).** *A Test of Missing Completely at Random for Multivariate Data with Missing Values.* **Journal of the American Statistical Association (JASA)**, 83(404), 1198–1202. [DOI: 10.1080/01621459.1988.10478722](https://doi.org/10.1080/01621459.1988.10478722).
2. **Rubin, Donald B. (1976).** *Inference and Missing Data.* **Biometrika**, 63(3), 581–592. [DOI: 10.1093/biomet/63.3.581](https://doi.org/10.1093/biomet/63.3.581).
3. **Micci-Barreca, Daniele (2001).** *A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems.* **ACM SIGKDD Explorations**, 3(1), 27–32. [DOI: 10.1145/507533.507538](https://doi.org/10.1145/507533.507538).
4. **Wolpert, David H. (1992).** *Stacked Generalization.* **Neural Networks**, 5(2), 241–259. [DOI: 10.1016/S0893-6080(05)80023-1](https://doi.org/10.1016/S0893-6080(05)80023-1).
5. **Zadrozny, Bianca, & Elkan, Charles (2002).** *Transforming Classifier Scores into Accurate Multiclass Probability Estimates.* **Proceedings of the 8th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '02)**, 694–699.
6. **Lundberg, Scott M., & Lee, Su-In (2017).** *A Unified Approach to Interpreting Model Predictions.* **Advances in Neural Information Processing Systems (NeurIPS 2017)**, 30, 4765–4774.
7. **Chen, Tianqi, & Guestrin, Carlos (2016).** *XGBoost: A Scalable Tree Boosting System.* **Proceedings of the 22nd ACM SIGKDD (KDD '16)**, 785–794. [DOI: 10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785).
8. **Akiba, Takuya, et al. (2019).** *Optuna: A Next-generation Hyperparameter Optimization Framework.* **KDD '19**, 2623–2631. [DOI: 10.1145/3292500.3330701](https://doi.org/10.1145/3292500.3330701).
9. **Jha, Pratibha Kumari (2024).** *Data Science Best Practices: End-to-End Enterprise Standards.* [LinkedIn Pulse Publication](https://www.linkedin.com/pulse/data-science-best-practices-pratibha-kumari-jha).
10. **Saini, Alankrit (2024).** *Complete Guide to Data Imputation Techniques: From Basic to Advanced.* [LinkedIn Pulse Publication](https://www.linkedin.com/pulse/complete-guide-data-imputation-techniques-from-basic-advanced-saini--uqhdc/).
11. **Sehgal, Manav (2017).** *Titanic Data Science Solutions.* [Kaggle Notebook](https://www.kaggle.com/code/startupsci/titanic-data-science-solutions).
12. **Freeman, LD (2018).** *A Data Science Framework: To Achieve 99% Accuracy.* [Kaggle Notebook](https://www.kaggle.com/code/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy).

---

## 👨‍💻 Autor y Contacto

<p align="left">
  <img src="images/guillen_logo.png" alt="Guillén Concepción" width="110" style="border-radius: 50%;" />
</p>

**Guillén Concepción**  
*Senior Data Scientist & MLOps Engineer*  

Especialista en diseño, desarrollo y despliegue de soluciones integrales de Inteligencia Artificial. Enfoque pragmático orientado a valor de negocio, abarcando desde investigación (CRISP-DM) hasta arquitecturas Cloud-Native resilientes y sistemas de producción auditables.

- **LinkedIn:** [https://www.linkedin.com/in/guillen-concepcion-25266b127](https://www.linkedin.com/in/guillen-concepcion-25266b127)  
- **GitHub:** [https://github.com/GuillenConcepcion](https://github.com/GuillenConcepcion)  
- **Email:** [guillenconcepcion@gmail.com](mailto:guillenconcepcion@gmail.com)
