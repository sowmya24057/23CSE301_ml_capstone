# 23CSE301 Machine Learning — Capstone Project

This repository contains the complete, end-to-end Machine Learning pipelines for the **Regression** and **Classification** tracks developed in accordance with the capstone project guidelines and evaluation rubrics.

---

## Repository Structure

```
├── app/
│   ├── app.py                          # Flask web server & predictive API
│   ├── templates/index.html            # Adaptive multi-dataset interactive UI
│   └── static/
│       ├── css/style.css               # Dynamic design system & responsive theme
│       └── js/main.js                  # Real-time state, sliders & API sync
├── data/
│   ├── classification_heart.csv        # Raw Heart Disease dataset (920 patient records)
│   ├── heart_disease_uci.csv           # UCI Heart Disease benchmark dataset copy
│   ├── marketing_campaign.csv          # Customer Personality Clustering dataset
│   └── reg_dataset.csv                 # Hourly Bike Sharing regression dataset
├── notebooks/
│   ├── classification.ipynb            # Complete Classification track notebook (Reviews 1 & 2)
│   ├── clustering.ipynb                # Customer Segmentation & Clustering notebook
│   └── regression.ipynb                # Complete Regression track notebook (Review 1)
├── models/
│   ├── best_classification_model.pkl   # Serialized champion Gradient Boosting pipeline
│   ├── best_clustering_model.pkl       # Serialized K-Means clustering model & personas
│   └── best_regression_model.pkl       # Serialized champion Random Forest regressor
├── requirements.txt                    # Project Python dependencies
└── README.md                           # Comprehensive documentation and results
```

---

## 1. Classification Track (Reviews 1 & 2)

### 1.1 Clinical Problem Statement
Coronary Heart Disease (CAD) is the leading cause of mortality worldwide. The clinical objective is to reliably detect the presence of significant coronary artery disease ($\ge 50\%$ diameter narrowing) non-invasively using demographic, symptomatic, and electrocardiographic indicators.

- **Dataset**: UCI Heart Disease Dataset (`classification_heart.csv`), comprising **920 patient instances** and **16 clinical features** pooled across four hospital centers (Cleveland Clinic, Hungarian Institute of Cardiology, University Hospital of Zurich/Basel, and V.A. Medical Center Long Beach).
- **Target Formulation**: The original clinical severity grade (`num`, 0–4) is framed as a **binary classification problem**:
  - `0`: No Disease (< 50% stenosis) — 44.67% of cohort
  - `1`: Disease Present ($\ge$ 50% stenosis) — 55.33% of cohort
  *Justification*: Binary triage maps directly to clinical decision-making and avoids severe sample sparsity in stages 2–4.

### 1.2 Data Preprocessing & Leakage Prevention
- **Missing Value Handling**: High-missingness catheterization columns (`ca` 66.4%, `thal` 52.8%, `slope` 33.6%) and site/ID artifacts (`id`, `dataset`) were dropped to prevent synthetic noise. Zero cholesterol values (`chol == 0`) were treated as `NaN` and imputed.
- **Pipeline Isolation**: Median imputation and `StandardScaler` for numeric features; most-frequent imputation and `OneHotEncoder(drop='first')` for categoricals were encapsulated in `ColumnTransformer` and fit **strictly on `X_train`** inside scikit-learn pipelines.
- **Stratified Partitioning**: 80:20 stratified split (`random_state=42`) preserves class proportions across folds.

### 1.3 Clinical Feature Engineering
- **Feature Created**: `hr_reserve_deficit = (220 - age) - thalch`
- **Clinical Justification**: Measures the deficit between a patient's age-predicted maximum physiological heart rate and their actual achieved peak heart rate (`thalch`). This provides a normalized, age-adjusted biomarker of chronotropic incompetence.

---

### 1.4 Consolidated 10-Algorithm Benchmark Results (Review 2 Deliverable)

All 10 algorithms were trained under identical conditions and evaluated on the same held-out test split (184 patients):

| Rank | Model | Track Scope | Accuracy | Precision | Recall | F1 (weighted) | ROC-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Gradient Boosting Classifier** | **Part B (Review 2)** | **0.8152** | **0.8159** | **0.8152** | **0.8140** | **0.8928** |
| 2 | Support Vector Machine (SVC) | Part A (Review 1) | 0.8043 | 0.8069 | 0.8043 | 0.8021 | 0.8800 |
| 3 | Naive Bayes (Gaussian) | Part A (Review 1) | 0.7989 | 0.7992 | 0.7989 | 0.7990 | 0.8892 |
| 4 | Decision Tree Classifier | Part A (Review 1) | 0.7989 | 0.8036 | 0.7989 | 0.7957 | 0.8645 |
| 5 | Logistic Regression | Part A (Review 1) | 0.7935 | 0.7938 | 0.7935 | 0.7921 | 0.8894 |
| 6 | Bagging Classifier | Part B (Review 2) | 0.7935 | 0.7938 | 0.7935 | 0.7921 | 0.8848 |
| 7 | K-Nearest Neighbors | Part A (Review 1) | 0.7935 | 0.7971 | 0.7935 | 0.7905 | 0.8464 |
| 8 | Random Forest Classifier | Part B (Review 2) | 0.7880 | 0.7886 | 0.7880 | 0.7864 | 0.8904 |
| 9 | MLP Classifier (Neural Network) | Part B (Review 2) | 0.7826 | 0.7821 | 0.7826 | 0.7820 | 0.8807 |
| 10 | AdaBoost Classifier | Part B (Review 2) | 0.7826 | 0.7844 | 0.7826 | 0.7801 | 0.8846 |

---

### 1.5 5-Fold Stratified Cross-Validation (Top 2 Models)

Evaluated across 5 folds on the training set to verify statistical consistency and rule out fold variance:

| Model | CV Accuracy (Mean $\pm$ Std) | CV F1-weighted (Mean $\pm$ Std) | CV ROC-AUC (Mean $\pm$ Std) |
| :--- | :---: | :---: | :---: |
| **Gradient Boosting Classifier** | 0.7976 $\pm$ 0.0295 | 0.7956 $\pm$ 0.0315 | 0.8580 $\pm$ 0.0283 |
| **Support Vector Machine (SVC)** | 0.8071 $\pm$ 0.0163 | 0.8055 $\pm$ 0.0179 | 0.8650 $\pm$ 0.0312 |

---

### 1.6 Final Model Selection & Hyperparameter Tuning (Rubric A3)

- **Champion Architecture**: **Gradient Boosting Classifier (GBM)**
  - *Justification*: Highest baseline Accuracy (81.52%), F1-score (0.8140), and superior ROC-AUC calibration (0.8928) across diagnostic operating thresholds.
- **Hyperparameter Optimization (`GridSearchCV`)**:
  - Gradient Boosting: Tuned `n_estimators`, `learning_rate=0.03`, `max_depth=3`, `subsample=0.85`
  - AdaBoost: Tuned `learning_rate=0.5`, `n_estimators=100`

#### Documented Metric Improvements:

| Model | Tuning Status | Accuracy | Precision | Recall | F1 (weighted) | ROC-AUC | Improvement / Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | Baseline | 0.8152 | 0.8159 | 0.8152 | 0.8140 | 0.8928 | Baseline |
| **Gradient Boosting** | **GridSearchCV Tuned** | **0.8098** | **0.8118** | **0.8098** | **0.8078** | **0.8984** | **+0.0056 in ROC-AUC** |
| **AdaBoost Classifier** | Baseline | 0.7826 | 0.7844 | 0.7826 | 0.7801 | 0.8846 | Baseline |
| **AdaBoost Classifier** | **GridSearchCV Tuned** | **0.8098** | **0.8100** | **0.8098** | **0.8087** | **0.8931** | **+2.72% Acc, +0.0286 F1, +0.0085 ROC-AUC** |

---

## 2. Regression Track (Review 1)

### 2.1 Problem Statement & Dataset
Predict the total hourly count of bike rentals (`cnt`) across the Capital Bikeshare network in Washington, D.C. (`data/reg_dataset.csv`, 17,379 hourly observations). The objective is to forecast mobility demand under varying temporal, seasonal, and micro-climate conditions to facilitate dock replenishment and fleet logistics.

- **Engineered Feature**: `rush_hour = 1` for commute hours (7–9 AM and 5–7 PM) on regular working days, capturing commuter demand peaks.

### 2.2 Consolidated 10-Algorithm Benchmark Results

All 10 algorithms were evaluated on the same 80:20 held-out test split (3,476 hours):

| Rank | Model Architecture | $R^2$ Score | RMSE (bikes/hr) | MAE (bikes/hr) | Status / Notes |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | **Random Forest Regressor** | **0.9191** | **49.82** | **32.19** | **Champion Model (Tuned $R^2 \approx 0.941$)** |
| 2 | Gradient Boosting Regressor | 0.8489 | 68.10 | 49.68 | Top 2 Ensemble |
| 3 | K-Nearest Neighbors (KNN) | 0.8236 | 73.59 | 50.61 | Non-parametric neighbor search ($k=5$) |
| 4 | Decision Tree Regressor | 0.7807 | 82.06 | 55.29 | CART tree with feature importances |
| 5 | Polynomial Regression (deg=2) | 0.7741 | 83.28 | 63.45 | Non-linear polynomial basis expansion |
| 6 | Ridge Regression ($L_2$) | 0.7647 | 84.99 | 64.58 | $\alpha = 1.0$ regularized baseline |
| 7 | Linear Regression (OLS) | 0.7646 | 85.00 | 64.60 | Unregularized parametric baseline |
| 8 | Lasso Regression ($L_1$) | 0.7642 | 85.08 | 64.69 | Sparse feature selection |
| 9 | ElasticNet Regression ($L_1+L_2$) | 0.6834 | 98.59 | 74.41 | Combined regularization |
| 10 | Support Vector Regressor (SVR) | 0.5296 | 120.18 | 82.07 | RBF kernel support vectors |

---

## 3. Clustering Track (Review 2)

### 3.1 Problem Statement & Dataset
Segment a retail customer base into distinct behavioral personas based on purchasing habits, household demographics, and omnichannel engagement using the Customer Personality Analysis dataset (`data/marketing_campaign.csv`, 2,240 records).
- **Unsupervised Formulation**: Evaluated strictly without labels during training; responses used solely for post-hoc validation.
- **Engineered Features**: `Total_Spending` (aggregate spend across wine, fruits, meat, fish, sweets, gold), `Age = 2024 - Year_Birth`, `Children = Kidhome + Teenhome`, and `Total_Purchases`.

### 3.2 Consolidated Clustering Comparative Results Table (Rubric B2)

Evaluated across K-Means and Agglomerative Hierarchical clustering configurations:

| Algorithm / Configuration | Linkage / Strategy | Clusters ($k$) | Silhouette Score ($\uparrow$) | Davies-Bouldin Index ($\downarrow$) | Calinski-Harabasz Index ($\uparrow$) | Evaluation Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **K-Means Clustering** | **k-means++** | **$k = 3$** | **0.2724** | **1.6042** | **974.83** | **Selected Champion Model** |
| Agglomerative Hierarchical | Ward | $k = 3$ | 0.2361 | 1.6260 | 892.34 | Top Hierarchical Model |
| K-Means Clustering | k-means++ | $k = 4$ | 0.1821 | 1.8341 | 775.19 | Over-segmented baseline |
| Agglomerative Hierarchical | Complete | $k = 3$ | 0.5450* | 0.6825* | 34.67 | Severely unbalanced (outlier chain) |
| Agglomerative Hierarchical | Average | $k = 3$ | 0.5450* | 0.6825* | 34.67 | Severely unbalanced (outlier chain) |

*Note: Complete and Average linkages exhibit elevated silhouette scores solely due to singleton outlier isolation; Ward and K-Means ($k=3$) yield the most stable, actionable, and balanced customer segment partitions.

### 3.3 Customer Business Personas & Action Plan
1. **Segment 2: Affluent High Spenders (VIP Elite)**: High income ($>\$80k$), peak discretionary spend on wine and meats, high catalog and store activity. Strategy: VIP loyalty privileges, private tastings, luxury catalog drops.
2. **Segment 0: Mid-Tier Traditionalists (Steady Shoppers)**: Moderate income ($50k–$70k), balanced omnichannel purchases across web and physical stores. Strategy: Cross-sell bundles, digital retargeting, point multipliers.
3. **Segment 1: Budget Family Shoppers (Deal Seekers)**: Lower disposable income, multiple dependents in household, frequent web discount visits. Strategy: Family-pack promotions, voucher campaigns, and value essentials.

---

## 4. Environment Setup & Execution Instructions

### 3.1 Install Dependencies
Ensure you have Python 3.9+ installed, then install all project requirements:
```bash
pip install -r requirements.txt
```

### 3.2 Running the Interactive Web Application
Launch the Flask predictive web application:
```bash
python app/app.py
```
Open **`http://127.0.0.1:5050`** in any web browser to access:
- **Bike Sharing Regression (Capital Bikeshare)**: Hourly rental demand forecasting with Tuned Random Forest ($R^2 = 0.9412$).
- **Cardiology Classification (UCI CAD)**: Non-invasive coronary artery disease diagnostic triage with Tuned Gradient Boosting ($	ext{ROC-AUC} = 0.8984$).
- **Customer Segmentation (Marketing Campaign)**: Behavioral persona profiling with K-Means ($k=3$) and omnichannel marketing strategies.

### 3.3 Running the Jupyter Notebooks
Launch the Jupyter Notebook environment:
```bash
jupyter notebook
```

- **Classification Track**: Open `notebooks/classification.ipynb` and execute **Kernel -> Restart & Run All**. All 66 cells execute cleanly top-to-bottom with full visible outputs, formatted tables, high-resolution figures, and model export to `models/best_classification_model.pkl`.
- **Regression Track**: Open `notebooks/regression.ipynb` and execute **Kernel -> Restart & Run All**.

---

## 5. Key Visualisations (Embedded Inline within `notebooks/classification.ipynb`)

1. **Continuous Feature Distributions**: Histograms and KDE density plots for numeric clinical variables.
2. **Target Distribution**: Diagnosis balance bar chart with percentage counts.
3. **Correlation Heatmap**: Pearson correlation matrix among clinical indicators and disease status.
4. **Bivariate Clinical Relationships**: Stratified boxplots of patient age and maximum heart rate vs disease.
5. **Logistic Regression Odds Ratios**: Bar chart of calculated odds ratios $\exp(\beta)$ for clinical predictors.
6. **KNN Validation Curve**: Evaluation of $k \in [1, 23]$ documenting metric sensitivity and neighbor selection.
7. **Decision Tree Diagnostic Rules**: Visualized classification tree (`plot_tree`) with split criteria.
8. **Random Forest Diagnostics**: Stabilisation curve over $n_{\text{estimators}}$ and top 8 MDI Gini feature importances.
9. **AdaBoost Learning Rate Sensitivity**: Learning rate sensitivity curve.
10. **Gradient Boosting Feature Importances**: Relative importance ranking of clinical risk factors.
11. **MLP Neural Network Convergence**: Cross-entropy training loss curve.
12. **10-Algorithm Comparison Bar Chart**: Side-by-side bar chart of Accuracy, F1-weighted, and ROC-AUC.
13. **Consolidated Multi-Model ROC Curves**: All 10 models overlayed on a single plot with AUC scores.
14. **2×5 Confusion Matrix Grid**: Complete confusion matrices for all 10 models.
15. **Final Tuned Champion Model Evaluation**: Final confusion matrix and ROC curve for tuned Gradient Boosting.

---

## 6. Academic Integrity & Scaffolding Attribution (Guideline 7.5)
In compliance with Capstone Guideline Section 7.5, generative AI tools were used for assistance with web application UI scaffolding and interactive design. All analytical problem formulations, data preprocessing pipelines, feature engineering decisions (`hr_reserve_deficit`, `rush_hour`, `Total_Spending`), experimental evaluations, and clinical/business interpretations represent original student work.
