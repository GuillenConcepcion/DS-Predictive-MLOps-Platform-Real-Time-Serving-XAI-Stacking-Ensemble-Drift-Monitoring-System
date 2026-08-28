# 📘 Manual Canónico de Ciencia de Datos & Clasificación: Técnicas, Modelos, Métricas y MLOps
### *Playbook de Autoaprendizaje, Rigor Estadístico y Registro Histórico de Portafolio para Data Science Senior*

**Lead Architect & Author:** [Guillén Concepción](https://www.linkedin.com/in/guillen-concepcion-25266b127) *(Senior Data Scientist & MLOps Engineer)*  
**Repositorio Oficial:** [DS-Predictive-MLOps-Platform](https://github.com/GuillenConcepcion/DS-Predictive-MLOps-Platform-Real-Time-Serving-XAI-Stacking-Ensemble-Drift-Monitoring-System)  
**Última Actualización:** Agosto 2026 • **Versión:** 3.0 (Champion Stacking Architecture)  
**Contacto:** [LinkedIn](https://www.linkedin.com/in/guillen-concepcion-25266b127) • [GitHub](https://github.com/GuillenConcepcion) • [Email](mailto:guillenconcepcion@gmail.com)

---

## 🧭 Índice General del Manual

1. [Visión y Filosofía: El Salto de "Toy Notebooks" a Ciencia de Datos Enterprise](#1-visión-y-filosofía-el-salto-de-toy-notebooks-a-ciencia-de-datos-enterprise)
2. [Marco de Trabajo de 14 Pilares de Ciencia de Datos](#2-marco-de-trabajo-de-14-pilares-de-ciencia-de-datos)
3. [Diagnóstico de Datos y Mecanismos de Ausencia (Missing Data Theory)](#3-diagnóstico-de-datos-y-mecanismos-de-ausencia-missing-data-theory)
4. [Ingeniería de Características y Prevención de Data Leakage](#4-ingeniería-de-características-y-prevención-de-data-leakage)
5. [Taxonomía de Modelos y Estrategias de Ensamble](#5-taxonomía-de-modelos-y-estrategias-de-ensamble)
6. [Métricas de Evaluación, Calibración de Probabilidades y Umbrales](#6-métricas-de-evaluación-calibración-de-probabilidades-y-umbrales)
7. [Explicabilidad e Interpretabilidad (XAI con SHAP)](#7-explicabilidad-e-interpretabilidad-xai-con-shap)
8. [Observabilidad, Monitoreo Continuo y Detección de Drift](#8-observabilidad-monitoreo-continuo-y-detección-de-drift)
9. [Registro Histórico Evolutivo del Proyecto (Portfolio Case Study)](#9-registro-histórico-evolutivo-del-proyecto-portfolio-case-study)
10. [Guía de Defensa Técnica en Entrevistas de Nivel Senior / Lead](#10-guía-de-defensa-técnica-en-entrevistas-de-nivel-senior--lead)
11. [Bibliografía Canónica y Artículos Científicos](#11-bibliografía-canónica-y-artículos-científicos)

---

## 1. Visión y Filosofía: El Salto de "Toy Notebooks" a Ciencia de Datos Enterprise

En la industria tecnológica contemporánea, más del **85% de los proyectos de Machine Learning fracasan antes de llegar a producción**. Aquellos pocos que superan la barrera del despliegue suelen degradarse rápidamente debido a fallas invisibles:

```
+-----------------------------------------------------------------------------------------+
|                  DISCREPANCIA ENTRE DESARROLLO ACADÉMICO Y ENTERPRISE                  |
+-----------------------------------------------------------------------------------------+
|  Enfoque "Toy / Kaggle Notebook"             |  Enfoque "Enterprise MLOps & Production" |
+----------------------------------------------+------------------------------------------+
|  Imputación ingenua (media/moda global)      |  Contraste formal Little's MCAR (1988)   |
|  One-Hot Encoding ciego sin regularización   |  Bayesian Target Encoding OOF (m-est)    |
|  Ajuste sobre todo el dataset (Data Leakage) |  Aislamiento estricto dentro de CV folds |
|  Maximización ciega de Accuracy              |  ROC-AUC, Brier Score y Calibración PAVA |
|  Umbral fijo por defecto en 0.50             |  Optimización OOF de Umbral (Youden / F1)|
|  Scripts monolíticos y pickle dispersos      |  Pipeline Serializado Atómico (Sklearn)  |
|  Payloads sin validación de tipos            |  Contratos de Datos Pydantic V2          |
|  Despliegue "ciego" sin telemetría           |  Monitoreo en tiempo real de Drift (PSI) |
|  Pruebas manuales                            |  CI/CD Performance Gate automatizado     |
+-----------------------------------------------------------------------------------------+
```

Este manual formaliza la metodología utilizada en el proyecto **Odysseus / Predictive MLOps Framework**, estructurando un marco de trabajo replicable para cualquier problema de clasificación supervisada en el mundo real (Fintech, Churn, Detección de Fraude, Riesgo Crediticio, Propensión de Compra, etc.).

---

## 2. Marco de Trabajo de 14 Pilares de Ciencia de Datos

*(Referencia canónica: Pratibha Kumari Jha, 2024)*

```
[1. Business Goal] ---> [2. Data Audit] ---> [3. Robust EDA] ---> [4. Feature Eng.]
         |                                                                |
[8. Continuous MLOps] <-- [7. XAI (SHAP)] <-- [6. Calibrated CV] <-- [5. Model Stacking]
         |
[9. Modular Code] ----> [10. Data Governance] -> [11. Container (Podman)] -> [12. Optuna TPE]
                                                                                   |
                                    [14. Lifelong Learning] <--- [13. BI Dashboards]
```

1. **Definición Clara del Problema:** Formulación de la función de pérdida alineada al impacto financiero (costo de Falsos Positivos vs Falsos Negativos).
2. **Auditoría e Inmutabilidad de Datos Crudos:** Los datos en `data/raw/` son de solo lectura; todo procesamiento se ejecuta mediante transformadores reproducibles.
3. **Análisis Exploratorio de Datos (EDA) con Trazabilidad:** Identificación de interacciones multivariadas (género × clase, tarifas por puerto).
4. **Ingeniería de Características sin Fugas:** Todo encoder, imputador y selector se ajusta estrictamente sobre el subconjunto de entrenamiento.
5. **Selección de Modelos y Stacking de 2 Niveles:** Diversidad algorítmica combinando árboles con regularización lineal ($L_2$).
6. **Validación Cruzada Estratificada y Calibración:** Estimación fuera de pliegue (OOF) con calibración monótona de probabilidades.
7. **Interpretabilidad y Explicabilidad (XAI):** Cuantificación de aportes marginales con valores de Shapley (`TreeExplainer`).
8. **Monitoreo Continuo de Data & Prediction Drift:** Auditoría continua con Kolmogorov-Smirnov, PSI y Wasserstein.
9. **Código Modular y Tipado Estricto:** Tipado estático con `mypy`, linting PEP 8 con `ruff` y suite de pruebas unitarias (`pytest`).
10. **Gobernanza y Versionado:** Registro de experimentos y promoción de artefactos con **MLflow Model Registry** (`@champion`).
11. **Reproducibilidad y Contenedores Rootless:** Gestión determinística de dependencias con `uv` y ejecución sin privilegios en Podman (UID 10001).
12. **Optimización Bayesiana de Hiperparámetros:** Muestreo inteligente del espacio de búsqueda mediante Optuna TPE.
13. **Comunicación Ejecutiva y Dashboards BI:** Dashboards HTML5 interactivos autónomos para stakeholders de negocio.
14. **Autoaprendizaje y Escalabilidad:** Documentación de decisiones arquitectónicas y formulaciones matemáticas.

---

## 3. Diagnóstico de Datos y Mecanismos de Ausencia (Missing Data Theory)

### 3.1. Taxonomía de Rubin (1976)

```
                              Mecanismos de Ausencia
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
      MCAR                             MAR                            MNAR
(Missing Completely              (Missing at Random)             (Missing Not at
    at Random)                   P(M|Yobs, Ymis)=P(M|Yobs)           Random)
P(M|Yobs, Ymis)=P(M)                    │                       P(M|Yobs, Ymis)
        │                               ▼                               │
        ▼                      Imputación Multivariada                  ▼
Válido Imputación Media/       (KNN / MICE / MissForest)       Modelado Explícito del
Moda o Eliminación de Filas   + Missing Indicators (Obligatorio)   Mecanismo de Ausencia
```

### 3.2. Formulación Matemática de Little's MCAR Test (1988)

El test evalúa si las diferencias entre las medias observadas en los distintos patrones de ausencia son atribuibles al azar:

$$\begin{cases} H_0: \text{El mecanismo de ausencia es MCAR} \\ H_1: \text{El mecanismo de ausencia es MAR o MNAR} \end{cases}$$

#### Estadístico Chi-Cuadrado de Little ($d^2$):
$$d^2 = \sum_{s=1}^S N_s \left( \mathbf{\bar{y}}_{\text{obs}, s} - \mathbf{\hat{\mu}}_s \right)^T \mathbf{\hat{\Sigma}}_s^{-1} \left( \mathbf{\bar{y}}_{\text{obs}, s} - \mathbf{\hat{\mu}}_s \right)$$

* $S$: Número de patrones únicos de valores ausentes.
* $N_s$: Número de observaciones con el patrón $s$.
* $\mathbf{\bar{y}}_{\text{obs}, s}$: Vector de medias muestrales de las variables presentes en el patrón $s$.
* $\mathbf{\hat{\mu}}_s, \mathbf{\hat{\Sigma}}_s$: Vector de medias y matriz de covarianza estimados por **Expectation-Maximization (EM)**.
* Grados de libertad: $df = \left(\sum_{s=1}^S P_s\right) - P$.

#### Regla de Decisión Operativa:
* Si $p\text{-valor} \le 0.05 \implies$ **Rechazar $H_0$**. Prohibido usar imputaciones univariadas simples (media/mediana). Se debe usar `KNNImputer` o `IterativeImputer` (MICE) acompañado de banderas binarias `MissingIndicator`.

---

## 4. Ingeniería de Características y Prevención de Data Leakage

### 4.1. Bayesian Target Encoding con $m$-estimate (Micci-Barreca 2001)

Cuando se manejan variables categóricas con alta cardinalidad (`TicketPrefix`, `CabinDeck`), el One-Hot Encoding genera matrices dispersas que inducen sobreajuste. El **Target Encoding Bayesiano** sustituye la categoría por una combinación convexa entre la media local de la categoría y el prior global:

$$\hat{S}_i = \lambda(n_i) \cdot \bar{y}_i + (1 - \lambda(n_i)) \cdot \mu_{\text{global}} = \frac{n_i \cdot \bar{y}_i + m \cdot \mu_{\text{global}}}{n_i + m}$$

* $n_i$: Frecuencia muestral de la categoría $i$.
* $\bar{y}_i$: Tasa observada del target en la categoría $i$.
* $\mu_{\text{global}}$: Media global del target en el conjunto de entrenamiento.
* $m$: Parámetro de contracción bayesiana ($m=10.0$). Categorías raras ($n_i \to 0$) convergen hacia $\mu_{\text{global}}$, mientras que categorías frecuentes ($n_i \gg m$) conservan su media empírica.

#### Protocolo Anti-Leakage (Out-of-Fold Computation):
```python
# Durante el entrenamiento, el valor de target encoding para la muestra 'j' en el Fold 'k'
# se calcula EXCLUSIVAMENTE utilizando los datos de los Folds restantes (\k):
S_ik = (n_i_minus_k * y_mean_i_minus_k + m * global_mean_minus_k) / (n_i_minus_k + m)
```

### 4.2. Selección Recursiva de Características con Validación Cruzada (RFECV)

Para evitar la "maldición de la dimensionalidad" ($p \gg N$), se ejecuta **RFECV** con un estimador regularizado y scoring en `roc_auc`:
1. Entrenar el estimador en $K$ folds sobre el conjunto total de variables ($p=38$).
2. Calcular la importancia de cada característica ($|w_i|$ o Gini Importance).
3. Eliminar la característica de menor aporte y reevaluar el CV ROC-AUC.
4. Identificar el subespacio óptimo que maximiza estrictamente la métrica sin sobreajustar ($p^* = 14$ variables).

---

## 5. Taxonomía de Modelos y Estrategias de Ensamble

```
                               Jerarquía de Ensambles
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        ▼                                                                 ▼
Voting Classifier (Nivel 1)                                   Stacking Classifier (Nivel 2)
  Promedio aritmético/ponderado                                 Meta-modelo entrenado con
  p_final = (1/M) * sum(p_m)                                     probabilidades OOF
        │                                                                 │
        ▼                                                                 ▼
Sensible a modelos mal calibrados                             Aprende confianza óptima
(Un modelo sobreconfiado domina el ensamble)                   y penaliza colinealidad (L2)
```

### 5.1. Stacking Classifier con Meta-Learner Regularizado $L_2$ (Wolpert 1992)

El ensamble final de producción combina 3 familias distintas de árboles de gradiente (`GradientBoostingClassifier`, `LightGBM`, `XGBoost`):

$$P(y=1 \mid \mathbf{x}) = \sigma\left( \beta_0 + \sum_{m=1}^M \beta_m \hat{p}_m^{\text{OOF}}(\mathbf{x}) \right) = \frac{1}{1 + \exp\left( -\left(\beta_0 + \sum_{m=1}^M \beta_m \hat{p}_m^{\text{OOF}}(\mathbf{x})\right) \right)}$$

Sujeto a la minimización de la entropía cruzada binaria con penalización Ridge:

$$\min_{\boldsymbol{\beta}} \left\{ -\sum_{i=1}^N \left[ y_i \ln \hat{y}_i + (1 - y_i) \ln (1 - \hat{y}_i) \right] + \frac{1}{2C} \|\boldsymbol{\beta}\|_2^2 \right\}$$

* **Pesos del Meta-Estimador en Producción ($C=0.1$):**
  * $\beta_0 = -2.1346$ (Intercepto)
  * $\beta_{\text{XGB}} = +1.4118$ (Mayor peso asignado por mejor calibración en colas)
  * $\beta_{\text{GBM}} = +1.3891$
  * $\beta_{\text{LGBM}} = +1.1699$

### 5.2. Calibración Isotónica de Probabilidades (PAVA Algorithm)

Los clasificadores de boosting tienden a producir probabilidades distorsionadas cerca de 0 y 1. Para garantizar que $P(\text{predicción} = p) \approx p$, se aplica **Calibración Isotónica**:

$$\min_{\hat{m}} \sum_{i=1}^N (y_i - \hat{m}(f_i))^2 \quad \text{sujeto a } \hat{m}(f_i) \le \hat{m}(f_j) \text{ siempre que } f_i \le f_j$$

Resuelto de forma monótona no paramétrica mediante el algoritmo *Pool Adjacent Violators* (PAVA) en 5 folds estratificados.

---

## 6. Métricas de Evaluación, Calibración de Probabilidades y Umbrales

### 6.1. Matriz de Métricas para Clasificación Binaria

| Métrica | Definición Matemática | Cuándo Utilizar | Vulnerabilidad |
| :--- | :---: | :--- | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Clases perfectamente balanceadas ($50/50$). | Inútil con desbalance de clases. |
| **ROC-AUC** | $\int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) dt$ | Capacidad de ordenamiento global e invariancia de escala. | Puede ser optimista si los negativos dominan. |
| **PR-AUC** | $\int_0^1 \text{Precision}(\text{Recall}) dr$ | Detección de eventos raros (Fraude $< 1\%$). | Depende de la prevalencia de la clase. |
| **F1-Macro** | $\frac{1}{K} \sum_{k=1}^K \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$ | Balance armónico entre clases minoritarias y mayoritarias. | Requiere fijar un umbral de decisión. |
| **Brier Score** | $\frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$ | Evaluación estricta de la calibración de probabilidad. | Penaliza fuertemente predicciones overconfident. |

### 6.2. Optimización del Umbral de Decisión ($\tau^*$)

El umbral estándar $\tau = 0.50$ es subóptimo en problemas reales. Se utiliza una búsqueda sobre las predicciones OOF para maximizar el F1-Score:

$$\tau^* = \arg\max_{\tau \in (0, 1)} F_1\left(y_{\text{true}}, \mathbb{I}(\hat{p}_{\text{OOF}} \ge \tau)\right)$$

* **Resultado Óptimo en el Ensamble Champion:** $\tau^* = 0.390 \implies$ Maximiza la captura de supervivientes reduciendo los Falsos Negativos sin degradar la precisión global.

---

## 7. Explicabilidad e Interpretabilidad (XAI con SHAP)

Basado en la teoría de juegos cooperativos (Lloyd Shapley, 1953), **TreeSHAP** (Lundberg & Lee, 2017) calcula la contribución marginal de cada variable:

$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]$$

### Propiedades Matemáticas Garantizadas:
1. **Eficiencia:** $\sum_{i=1}^M \phi_i(x) = f(x) - \mathbb{E}[f(x)]$. La suma de las atribuciones equivale exactamente a la desviación frente a la predicción base.
2. **Simetría:** Si dos variables contribuyen idénticamente a todas las coaliciones, $\phi_i = \phi_j$.
3. **Dummy:** Si una variable no altera ninguna predicción, $\phi_i = 0$.
4. **Aditividad:** Para ensambles, $\phi_i^{\text{ensamble}} = \sum w_m \phi_i^{(m)}$.

---

## 8. Observabilidad, Monitoreo Continuo y Detección de Drift

### 8.1. Population Stability Index (PSI) & Divergencia de Jeffreys

$$\text{PSI} = \sum_{b=1}^B \left( \%Actual_b - \%Expected_b \right) \times \ln\left( \frac{\%Actual_b + \epsilon}{\%Expected_b + \epsilon} \right)$$

* Binning adaptativo: $B = \min\left(10, \max\left(3, \lfloor N/25 \rfloor\right)\right)$ para evitar varianza espuria en muestras pequeñas.
* **Umbrales Operativos:**
  * $\text{PSI} < 0.10 \implies$ `STABLE` (Operación nominal).
  * $0.10 \le \text{PSI} < 0.25 \implies$ `MODERATE_DRIFT` (Advertencia y aumento de telemetría).
  * $\text{PSI} \ge 0.25 \implies$ `CRITICAL_DRIFT` (Disparo de reentrenamiento y alerta MLOps).

### 8.2. Kolmogorov-Smirnov (KS 2-Sample) y Wasserstein Distance ($W_1$)

* **KS 2-Sample:** $D = \sup_x |F_{\text{ref}}(x) - F_{\text{curr}}(x)|$ (Evalúa cambios en la forma acumulada ECDF).
* **Wasserstein ($W_1$):** $W_1(u, v) = \int_{-\infty}^\infty |U(x) - V(x)| dx$ (Esfuerzo de transporte de masa de probabilidad).

---

## 9. Registro Histórico Evolutivo del Proyecto (Portfolio Case Study)

```
[Iteración v1] ----> [Iteración v2] ----> [Iteración v2.1] ----> [Iteración v3 CHAMPION]
Baseline XGBoost     Optuna GBM (100)      Voting Ensemble        Stacking Calibrado L2
ROC-AUC: 88.23%      ROC-AUC: 89.19%       ROC-AUC: 88.94%        ROC-AUC: 89.32% (RÉCORD)
Accuracy: 84.40%     Accuracy: 84.18%      Accuracy: 83.95%       Accuracy: 84.06%
F1: 83.16%           F1: 83.00%            F1: 82.65%             F1: 83.08%
23 Features          38 Features           38 Features            14 Features Élite (RFECV)
One-Hot Encoding     Discretización Bins   Soft Average           Bayesian Target Encoding
```

### Resumen de la Toma de Decisiones Técnicas:
1. **¿Por qué se descartó el Voting Classifier en v2.1?**  
   El promedio aritmético simple ($1/3, 1/3, 1/3$) asignaba el mismo peso a estimadores con diferentes niveles de calibración OOF.
2. **¿Por qué triunfó el Stacking Classifier en v3?**  
   El meta-estimador con penalización Ridge ($L_2$) aprendió a ponderar a XGBoost ($1.41$) y GBM ($1.39$) por encima de LightGBM ($1.17$), mientras que la selección de 14 variables con RFECV eliminó el 63% del ruido colineal.
3. **¿Por qué se serializó en un único Pipeline de Scikit-Learn?**  
   Garantiza que la API de inferencia ejecute `pipeline.predict_proba(df)` en una sola llamada, eliminando inconsistencias de esquema entre entrenamiento y producción.

---

## 10. Guía de Defensa Técnica en Entrevistas de Nivel Senior / Lead

### P1: ¿Cómo garantizaste que no hubiera Data Leakage en el Target Encoding y la Imputación?
> **Respuesta Modelo:**  
> *"La regla de oro aplicada fue el encapsulamiento estricto dentro de los pliegues de validación cruzada. El `BayesianTargetEncoder` y el `KNNImputer` fueron programados como clases compatibles con `TransformerMixin` de Scikit-Learn. Durante el entrenamiento, los priors y estadísticos se calcularon exclusivamente con las muestras de los folds de entrenamiento ($K-1$). En inferencia, el objeto congelado aplica los mapeos aprendidos sin recalcular ningún estadístico sobre las muestras entrantes."*

### P2: ¿Por qué utilizaste Little's MCAR Test en lugar de imputar directamente con la mediana?
> **Respuesta Modelo:**  
> *"La imputación univariada por media o mediana asume implícitamente que los datos son Missing Completely at Random (MCAR). Mediante el test de Little (1988), obtuvimos un estadístico $d^2=36.036$ con un $p\text{-valor}=2.84\times 10^{-7}$, lo que rechazó formalmente $H_0$ a favor de un mecanismo MAR. Esto demostró que la ausencia de edad dependía de la clase socioeconómica y la estructura familiar, haciendo mandatorio el uso de imputación multivariada no lineal (KNN) y variables indicadoras de ausencia (`MissingIndicator`) para evitar sesgos en el gradiente."*

### P3: ¿Cómo proteges el sistema contra regresiones de rendimiento en CI/CD?
> **Respuesta Modelo:**  
> *"Diseñé un script autónomo de control de calidad (`src/models/performance_gate.py`) integrado en el pipeline de GitHub Actions / GitLab CI. El gate bloquea cualquier merge request si el modelo candidato no supera simultáneamente 4 umbrales estrictos: $ROC\text{-}AUC \ge 0.880$, $Accuracy \ge 0.830$, $F_1 \ge 0.820$ y un SLA de latencia $P_{95} \le 250\text{ ms}$ por inferencia unitaria."*

---

## 11. Bibliografía Canónica y Artículos Científicos

1. **Little, Roderick J. A. (1988).** *A Test of Missing Completely at Random for Multivariate Data with Missing Values.* **JASA**, 83(404), 1198–1202. [DOI: 10.1080/01621459.1988.10478722](https://doi.org/10.1080/01621459.1988.10478722).
2. **Rubin, Donald B. (1976).** *Inference and Missing Data.* **Biometrika**, 63(3), 581–592. [DOI: 10.1093/biomet/63.3.581](https://doi.org/10.1093/biomet/63.3.581).
3. **Micci-Barreca, Daniele (2001).** *A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems.* **ACM SIGKDD Explorations**, 3(1), 27–32.
4. **Wolpert, David H. (1992).** *Stacked Generalization.* **Neural Networks**, 5(2), 241–259.
5. **Zadrozny, Bianca, & Elkan, Charles (2002).** *Transforming Classifier Scores into Accurate Multiclass Probability Estimates.* **ACM KDD '02**, 694–699.
6. **Lundberg, Scott M., & Lee, Su-In (2017).** *A Unified Approach to Interpreting Model Predictions.* **NeurIPS 2017**, 30, 4765–4774.
7. **Chen, Tianqi, & Guestrin, Carlos (2016).** *XGBoost: A Scalable Tree Boosting System.* **ACM KDD '16**, 785–794.
8. **Akiba, Takuya, et al. (2019).** *Optuna: A Next-generation Hyperparameter Optimization Framework.* **KDD '19**, 2623–2631.
9. **Jha, Pratibha Kumari (2024).** *Data Science Best Practices: End-to-End Enterprise Standards.* **LinkedIn Pulse**.
10. **Saini, Alankrit (2024).** *Complete Guide to Data Imputation Techniques: From Basic to Advanced.* **LinkedIn Pulse**.
11. **Sehgal, Manav (2017) & Freeman, LD (2018).** *Titanic Data Science Solutions & A Data Science Framework to Achieve 99% Accuracy.* **Kaggle Competitions**.
