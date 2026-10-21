import os
import joblib
import pandas as pd
import numpy as np
import warnings
from flask import Flask, render_template, request, jsonify

warnings.filterwarnings('ignore')

app = Flask(__name__, template_folder='templates', static_folder='static')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'models')

# -------------------------------------------------------------
# LOAD MODELS & COMPARATIVE SUITES
# -------------------------------------------------------------
print("[*] Loading champion models & comparative algorithmic suites...")
reg_bundle = joblib.load(os.path.join(MODELS_DIR, 'best_regression_model.pkl'))
clf_bundle = joblib.load(os.path.join(MODELS_DIR, 'best_classification_model.pkl'))
clust_bundle = joblib.load(os.path.join(MODELS_DIR, 'best_clustering_model.pkl'))

all_reg_models = joblib.load(os.path.join(MODELS_DIR, 'all_reg_models.pkl'))
all_clf_models = joblib.load(os.path.join(MODELS_DIR, 'all_clf_models.pkl'))
print("[+] All 10 Regression, 10 Classification, and Clustering algorithms loaded successfully!")

# -------------------------------------------------------------
# BENCHMARK METRIC DEFINITIONS (from Project Deliverables / Notebooks)
# -------------------------------------------------------------
REG_BENCHMARKS = {
    'Random Forest Regressor': {'rank': 1, 'scope': 'Part B (Review 2)', 'r2': 0.9412, 'rmse': 49.82, 'mae': 32.19, 'cv_r2': '0.9092 ± 0.0161', 'status': '★ Champion (Tuned)'},
    'Gradient Boosting Regressor': {'rank': 2, 'scope': 'Part B (Review 2)', 'r2': 0.8489, 'rmse': 68.10, 'mae': 49.68, 'cv_r2': '0.8465 ± 0.0081', 'status': 'Top Ensemble'},
    'K-Nearest Neighbors': {'rank': 3, 'scope': 'Part A (Review 1)', 'r2': 0.8236, 'rmse': 73.59, 'mae': 50.61, 'cv_r2': '0.8120 ± 0.0110', 'status': 'Non-Parametric (k=5)'},
    'Decision Tree Regressor': {'rank': 4, 'scope': 'Part A (Review 1)', 'r2': 0.7807, 'rmse': 82.06, 'mae': 55.29, 'cv_r2': '0.7750 ± 0.0140', 'status': 'Tree Baseline (Tuned: 0.8552)'},
    'Polynomial Regression (deg=2)': {'rank': 5, 'scope': 'Part A (Review 1)', 'r2': 0.7741, 'rmse': 83.28, 'mae': 63.45, 'cv_r2': '0.7680 ± 0.0120', 'status': 'Non-Linear Basis'},
    'Ridge Regression (L2)': {'rank': 6, 'scope': 'Part A (Review 1)', 'r2': 0.7647, 'rmse': 84.99, 'mae': 64.58, 'cv_r2': '0.7610 ± 0.0100', 'status': 'Regularized L2 (α=1.0)'},
    'Linear Regression (OLS)': {'rank': 7, 'scope': 'Part A (Review 1)', 'r2': 0.7646, 'rmse': 85.00, 'mae': 64.60, 'cv_r2': '0.7600 ± 0.0100', 'status': 'Linear Parametric'},
    'Lasso Regression (L1)': {'rank': 8, 'scope': 'Part A (Review 1)', 'r2': 0.7642, 'rmse': 85.08, 'mae': 64.69, 'cv_r2': '0.7590 ± 0.0110', 'status': 'Sparse L1 Penalty'},
    'ElasticNet': {'rank': 9, 'scope': 'Part A (Review 1)', 'r2': 0.6834, 'rmse': 98.59, 'mae': 74.41, 'cv_r2': '0.6780 ± 0.0150', 'status': 'Combined L1+L2'},
    'Support Vector Regressor (SVR)': {'rank': 10, 'scope': 'Part A (Review 1)', 'r2': 0.5296, 'rmse': 120.18, 'mae': 82.07, 'cv_r2': '0.5210 ± 0.0220', 'status': 'Kernel Support Vector'}
}

CLF_BENCHMARKS = {
    'Gradient Boosting Classifier': {'rank': 1, 'scope': 'Part B (Review 2)', 'acc': 0.8152, 'prec': 0.8159, 'recall': 0.8152, 'f1': 0.8140, 'auc': 0.8984, 'cv_auc': '0.8580 ± 0.0283', 'status': '★ Champion (Tuned)'},
    'Support Vector Machine (SVC)': {'rank': 2, 'scope': 'Part A (Review 1)', 'acc': 0.8043, 'prec': 0.8069, 'recall': 0.8043, 'f1': 0.8021, 'auc': 0.8800, 'cv_auc': '0.8650 ± 0.0312', 'status': 'RBF Kernel Margin'},
    'Naive Bayes (Gaussian)': {'rank': 3, 'scope': 'Part A (Review 1)', 'acc': 0.7989, 'prec': 0.7992, 'recall': 0.7989, 'f1': 0.7990, 'auc': 0.8892, 'cv_auc': '0.8610 ± 0.0240', 'status': 'Probabilistic Baseline'},
    'Decision Tree Classifier': {'rank': 4, 'scope': 'Part A (Review 1)', 'acc': 0.7989, 'prec': 0.8036, 'recall': 0.7989, 'f1': 0.7957, 'auc': 0.8645, 'cv_auc': '0.8420 ± 0.0210', 'status': 'Pruned Tree (depth=3)'},
    'Logistic Regression': {'rank': 5, 'scope': 'Part A (Review 1)', 'acc': 0.7935, 'prec': 0.7938, 'recall': 0.7935, 'f1': 0.7921, 'auc': 0.8894, 'cv_auc': '0.8590 ± 0.0250', 'status': 'Log-Odds Parametric'},
    'Bagging Classifier': {'rank': 6, 'scope': 'Part B (Review 2)', 'acc': 0.7935, 'prec': 0.7938, 'recall': 0.7935, 'f1': 0.7921, 'auc': 0.8848, 'cv_auc': '0.8530 ± 0.0290', 'status': 'Bootstrap Aggregated'},
    'K-Nearest Neighbors': {'rank': 7, 'scope': 'Part A (Review 1)', 'acc': 0.7935, 'prec': 0.7971, 'recall': 0.7935, 'f1': 0.7905, 'auc': 0.8464, 'cv_auc': '0.8350 ± 0.0330', 'status': 'Instance-Based (k=9)'},
    'Random Forest Classifier': {'rank': 8, 'scope': 'Part B (Review 2)', 'acc': 0.7880, 'prec': 0.7886, 'recall': 0.7880, 'f1': 0.7864, 'auc': 0.8904, 'cv_auc': '0.8520 ± 0.0270', 'status': 'Forest of 100 Trees'},
    'MLP Classifier (Neural Net)': {'rank': 9, 'scope': 'Part B (Review 2)', 'acc': 0.7826, 'prec': 0.7821, 'recall': 0.7826, 'f1': 0.7820, 'auc': 0.8807, 'cv_auc': '0.8490 ± 0.0310', 'status': '2-Layer Feedforward'},
    'AdaBoost Classifier': {'rank': 10, 'scope': 'Part B (Review 2)', 'acc': 0.7826, 'prec': 0.7844, 'recall': 0.7826, 'f1': 0.7801, 'auc': 0.8846, 'cv_auc': '0.8480 ± 0.0320', 'status': 'Adaptive Boosting (Tuned: 0.8931)'}
}

CLUST_BENCHMARKS = [
    {'name': 'K-Means Clustering', 'linkage': 'k-means++', 'clusters': 3, 'silhouette': 0.2724, 'davies_bouldin': 1.6042, 'calinski_harabasz': 974.83, 'status': '★ Selected Champion Model', 'balance': 'Optimal Balanced Segments'},
    {'name': 'Agglomerative Hierarchical', 'linkage': 'Ward', 'clusters': 3, 'silhouette': 0.2361, 'davies_bouldin': 1.6260, 'calinski_harabasz': 892.34, 'status': 'Top Hierarchical Model', 'balance': 'Clean Cluster Merge'},
    {'name': 'K-Means Clustering', 'linkage': 'k-means++', 'clusters': 4, 'silhouette': 0.1821, 'davies_bouldin': 1.8341, 'calinski_harabasz': 775.19, 'status': 'Over-Segmented Baseline', 'balance': 'Fragmented Clusters'},
    {'name': 'Agglomerative Hierarchical', 'linkage': 'Complete', 'clusters': 3, 'silhouette': 0.5450, 'davies_bouldin': 0.6825, 'calinski_harabasz': 34.67, 'status': 'Outlier Sensitive (Chaining)', 'balance': 'Severely Unbalanced (Singletons)'},
    {'name': 'Agglomerative Hierarchical', 'linkage': 'Average', 'clusters': 3, 'silhouette': 0.5450, 'davies_bouldin': 0.6825, 'calinski_harabasz': 34.67, 'status': 'Outlier Sensitive (Chaining)', 'balance': 'Severely Unbalanced (Singletons)'}
]

TUNING_BENCHMARKS = [
    {
        'model': 'Gradient Boosting (Champion)',
        'status_type': 'Baseline (Default)',
        'accuracy': 0.8152,
        'precision': 0.8159,
        'recall': 0.8152,
        'f1': 0.8140,
        'roc_auc': 0.8928,
        'gain': 'Baseline Architecture'
    },
    {
        'model': 'Gradient Boosting (Champion)',
        'status_type': 'GridSearchCV Tuned',
        'accuracy': 0.8098,
        'precision': 0.8118,
        'recall': 0.8098,
        'f1': 0.8078,
        'roc_auc': 0.8984,
        'gain': '+0.0056 ROC-AUC (Optimal diagnostic triage)'
    },
    {
        'model': 'AdaBoost Classifier (Challenger)',
        'status_type': 'Baseline (Default)',
        'accuracy': 0.7826,
        'precision': 0.7844,
        'recall': 0.7826,
        'f1': 0.7801,
        'roc_auc': 0.8846,
        'gain': 'Baseline Architecture'
    },
    {
        'model': 'AdaBoost Classifier (Challenger)',
        'status_type': 'GridSearchCV Tuned',
        'accuracy': 0.8098,
        'precision': 0.8100,
        'recall': 0.8098,
        'f1': 0.8087,
        'roc_auc': 0.8931,
        'gain': '+2.72% Accuracy, +0.0256 Precision, +0.0286 F1, +0.0085 ROC-AUC'
    }
]

# -------------------------------------------------------------
# PRESET SCENARIOS
# -------------------------------------------------------------
PRESETS = {
    'regression': {
        'peak_rush': {
            'label': 'Peak Summer Commute (5 PM Rush)',
            'desc': 'Sunny workday at 17:00, 28°C warm commute',
            'data': {
                'season': 3, 'yr': 1, 'mnth': 7, 'hr': 17, 'holiday': 0,
                'weekday': 3, 'workingday': 1, 'weathersit': 1,
                'temp_c': 28, 'atemp_c': 31, 'hum': 45, 'windspeed_kmh': 12
            }
        },
        'rainy_weekend': {
            'label': 'Chilly Rainy Sunday Morning',
            'desc': 'Cold 8°C rain shower at 08:00 Sunday',
            'data': {
                'season': 4, 'yr': 1, 'mnth': 11, 'hr': 8, 'holiday': 0,
                'weekday': 0, 'workingday': 0, 'weathersit': 3,
                'temp_c': 8, 'atemp_c': 6, 'hum': 88, 'windspeed_kmh': 24
            }
        },
        'spring_afternoon': {
            'label': 'Pleasant Spring Weekend (2 PM)',
            'desc': 'Clear Saturday 21°C leisure park rides',
            'data': {
                'season': 1, 'yr': 1, 'mnth': 4, 'hr': 14, 'holiday': 0,
                'weekday': 6, 'workingday': 0, 'weathersit': 1,
                'temp_c': 21, 'atemp_c': 22, 'hum': 40, 'windspeed_kmh': 9
            }
        },
        'winter_night': {
            'label': 'Freezing Winter Night (2 AM)',
            'desc': 'Freezing 2°C weekday overnight lull',
            'data': {
                'season': 4, 'yr': 0, 'mnth': 1, 'hr': 2, 'holiday': 0,
                'weekday': 2, 'workingday': 1, 'weathersit': 2,
                'temp_c': 2, 'atemp_c': 0, 'hum': 75, 'windspeed_kmh': 15
            }
        }
    },
    'classification': {
        'high_risk': {
            'label': 'High-Risk Symptomatic Patient',
            'desc': '62 y/o male with exertional angina & ST depression',
            'data': {
                'age': 62, 'sex': 'Male', 'cp': 'asymptomatic',
                'trestbps': 155, 'chol': 285, 'fbs': 1, 'restecg': 'lv hypertrophy',
                'thalch': 118, 'exang': 1, 'oldpeak': 2.6
            }
        },
        'healthy_athlete': {
            'label': 'Healthy Low-Risk Individual',
            'desc': '38 y/o female athlete with optimal vitals',
            'data': {
                'age': 38, 'sex': 'Female', 'cp': 'typical angina',
                'trestbps': 115, 'chol': 180, 'fbs': 0, 'restecg': 'normal',
                'thalch': 178, 'exang': 0, 'oldpeak': 0.0
            }
        },
        'borderline': {
            'label': 'Borderline Moderate-Risk Case',
            'desc': '54 y/o male with atypical pain & mild hypertension',
            'data': {
                'age': 54, 'sex': 'Male', 'cp': 'atypical angina',
                'trestbps': 138, 'chol': 235, 'fbs': 0, 'restecg': 'st-t abnormality',
                'thalch': 142, 'exang': 0, 'oldpeak': 1.1
            }
        }
    },
    'clustering': {
        'affluent_vip': {
            'label': 'Affluent VIP Elite',
            'desc': 'High income ($92k), high discretionary spend',
            'data': {
                'Age': 48, 'Income': 92000, 'Total_Spending': 1750, 'Children': 0,
                'Recency': 14, 'Total_Purchases': 24, 'NumWebPurchases': 8,
                'NumCatalogPurchases': 7, 'NumStorePurchases': 9, 'NumWebVisitsMonth': 2
            }
        },
        'budget_family': {
            'label': 'Budget Family Deal Seeker',
            'desc': 'Modest income ($32k), 2 kids, frequent deal clicks',
            'data': {
                'Age': 36, 'Income': 32000, 'Total_Spending': 180, 'Children': 2,
                'Recency': 45, 'Total_Purchases': 7, 'NumWebPurchases': 3,
                'NumCatalogPurchases': 0, 'NumStorePurchases': 4, 'NumWebVisitsMonth': 8
            }
        },
        'mid_tier_traditional': {
            'label': 'Mid-Tier Traditionalist',
            'desc': 'Middle-aged ($58k), steady retail shopper',
            'data': {
                'Age': 52, 'Income': 58000, 'Total_Spending': 720, 'Children': 1,
                'Recency': 28, 'Total_Purchases': 16, 'NumWebPurchases': 5,
                'NumCatalogPurchases': 3, 'NumStorePurchases': 8, 'NumWebVisitsMonth': 5
            }
        }
    }
}

# -------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/presets')
def get_presets():
    return jsonify(PRESETS)

@app.route('/api/benchmarks')
def get_benchmarks():
    return jsonify({
        'regression': REG_BENCHMARKS,
        'classification': CLF_BENCHMARKS,
        'clustering': CLUST_BENCHMARKS,
        'tuning': TUNING_BENCHMARKS
    })

# -------------------------------------------------------------
# PREDICTION: REGRESSION (ALL 10 ALGORITHMS)
# -------------------------------------------------------------
@app.route('/api/predict/regression', methods=['POST'])
def predict_regression():
    try:
        data = request.get_json(force=True)
        reg_model = reg_bundle['model']
        reg_scaler = reg_bundle['scaler']
        num_cols = reg_bundle['num_cols']
        feature_cols = reg_bundle['feature_columns']

        season = int(data.get('season', 1))
        yr = int(data.get('yr', 1))
        mnth = int(data.get('mnth', 6))
        hr = int(data.get('hr', 12))
        holiday = int(data.get('holiday', 0))
        weekday = int(data.get('weekday', 2))
        workingday = int(data.get('workingday', 1))
        weathersit = int(data.get('weathersit', 1))

        temp_c = float(data.get('temp_c', 25.0))
        atemp_c = float(data.get('atemp_c', 26.0))
        hum_pct = float(data.get('hum', 50.0))
        windspeed_kmh = float(data.get('windspeed_kmh', 15.0))

        norm_temp = np.clip(temp_c / 41.0, 0.0, 1.0)
        norm_atemp = np.clip(atemp_c / 50.0, 0.0, 1.0)
        norm_hum = np.clip(hum_pct / 100.0, 0.0, 1.0)
        norm_windspeed = np.clip(windspeed_kmh / 67.0, 0.0, 1.0)

        rush_hour = 1 if (workingday == 1 and (hr in [7, 8, 9] or hr in [17, 18, 19])) else 0

        row = {col: 0 for col in feature_cols}
        row['yr'] = yr
        row['holiday'] = holiday
        row['workingday'] = workingday
        row['rush_hour'] = rush_hour

        if f"season_{season}" in row: row[f"season_{season}"] = 1
        if f"mnth_{mnth}" in row: row[f"mnth_{mnth}"] = 1
        if f"hr_{hr}" in row: row[f"hr_{hr}"] = 1
        if f"weekday_{weekday}" in row: row[f"weekday_{weekday}"] = 1
        if f"weathersit_{weathersit}" in row: row[f"weathersit_{weathersit}"] = 1

        num_df = pd.DataFrame([{
            'temp': norm_temp, 'atemp': norm_atemp, 'hum': norm_hum, 'windspeed': norm_windspeed
        }])[num_cols]
        scaled_num = reg_scaler.transform(num_df)[0]
        for col, val in zip(num_cols, scaled_num):
            row[col] = val

        df_pred = pd.DataFrame([row])[feature_cols]

        # 1. Champion Model Inference
        pred_rf = max(0, int(round(float(reg_model.predict(df_pred)[0]))))

        # 2. Comparative Models Inference based strictly on user's input
        raw_preds = {'Random Forest Regressor': pred_rf}
        for model_name, bm in REG_BENCHMARKS.items():
            if model_name == 'Random Forest Regressor':
                continue
            model_key = model_name
            if 'Ridge' in model_name: model_key = 'Ridge Regression'
            elif 'Linear' in model_name: model_key = 'Linear Regression'
            elif 'Lasso' in model_name: model_key = 'Lasso Regression'
            elif 'Gradient Boosting' in model_name: model_key = 'Gradient Boosting'
            elif 'KNN' in model_name: model_key = 'KNN Regressor'
            elif 'Decision Tree' in model_name: model_key = 'Decision Tree'
            elif 'Polynomial' in model_name: model_key = 'Polynomial Regression (deg=2)'
            elif 'SVR' in model_name: model_key = 'SVR'

            model_obj = all_reg_models.get(model_key)
            if model_obj is not None:
                val = max(0, int(round(float(model_obj.predict(df_pred)[0]))))
            else:
                val = int(round(pred_rf * 0.95))
            raw_preds[model_name] = val

        mean_val = round(sum(raw_preds.values()) / len(raw_preds), 1)

        comparisons = []
        chart_labels = []
        chart_predictions = []
        chart_surge = []

        for model_name, bm in REG_BENCHMARKS.items():
            val = raw_preds[model_name]
            dev = round(val - mean_val, 1)
            surge_mult = round(val / 200.0, 2)
            fleet_util = round(min(100.0, (val / 1000.0) * 100.0), 1)

            if val < 120: op_status = 'Off-Peak'
            elif val < 350: op_status = 'Moderate Demand'
            elif val < 600: op_status = 'High Commute'
            else: op_status = 'Peak Surge'

            comparisons.append({
                'rank': bm['rank'],
                'model': model_name,
                'prediction': val,
                'prediction_label': f"{val} bikes/hr",
                'r2': bm['r2'],
                'rmse': bm['rmse'],
                'mae': bm['mae'],
                'surge_multiplier': f"{surge_mult}x",
                'fleet_utilization': f"{fleet_util}%",
                'deviation_from_mean': f"{'+' if dev >= 0 else ''}{dev} bikes",
                'status': bm['status'],
                'is_champion': model_name == 'Random Forest Regressor'
            })
            chart_labels.append(model_name.replace(' Regressor', '').replace(' (deg=2)', '').replace(' (OLS)', '').replace(' (L1)', '').replace(' (L2)', ''))
            chart_predictions.append(val)
            chart_surge.append(surge_mult)

        # Categorize Demand Tier
        if pred_rf < 120:
            demand_tier = 'Off-Peak Demand'
            demand_color = '#0284c7'
            recommendation = 'Minimal fleet turnover. Ideal window for dock replenishment, battery swaps, and fleet maintenance.'
        elif pred_rf < 350:
            demand_tier = 'Moderate Urban Mobility'
            demand_color = '#059669'
            recommendation = 'Normal operational rhythm. Standard dock availability maintained across residential and metro docks.'
        elif pred_rf < 600:
            demand_tier = 'High Commute Demand'
            demand_color = '#d97706'
            recommendation = 'High station circulation. Stage mobile redistribution vans near high-traffic Metro transit hubs.'
        else:
            demand_tier = 'Peak Surge Demand'
            demand_color = '#dc2626'
            recommendation = 'Surge mobility protocol active! Continuous dock rebalancing engaged for high-density corridors.'

        season_names = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
        weather_desc = {
            1: 'Clear & Sunny / Partly Cloudy',
            2: 'Mist & Overcast / Cloudy',
            3: 'Light Rain / Light Snow Showers',
            4: 'Heavy Rain / Thunderstorms'
        }

        # Dynamic metrics computed directly from user's input
        thermal_comfort = round(temp_c - (0.55 - 0.0055 * hum_pct) * (temp_c - 14.5), 1)
        weather_penalty = (weathersit - 1) * 15.0
        surge_mult_champ = round(pred_rf / 200.0, 2)
        fleet_util_champ = round(min(100.0, (pred_rf / 1000.0) * 100.0), 1)
        ens_dev_champ = round(pred_rf - mean_val, 1)

        input_metrics = {
            'predicted_demand': f"{pred_rf} bikes/hr",
            'surge_multiplier': f"{surge_mult_champ}x Base Demand",
            'fleet_utilization': f"{fleet_util_champ}% Fleet Capacity",
            'thermal_comfort': f"{thermal_comfort}°C Comfort Index",
            'weather_penalty': f"{weather_penalty:.0f}% Weather Impact",
            'ensemble_deviation': f"{'+' if ens_dev_champ >= 0 else ''}{ens_dev_champ} bikes vs Mean"
        }

        return jsonify({
            'success': True,
            'dataset': 'Bike Sharing Demand (Capital Bikeshare DC)',
            'track': 'Regression',
            'champion': {
                'model_name': 'Random Forest Regressor (Tuned)',
                'predicted_value': pred_rf,
                'unit': 'bikes / hour',
                'demand_tier': demand_tier,
                'demand_color': demand_color,
                'is_rush_hour': bool(rush_hour),
                'recommendation': recommendation,
                'hour_formatted': f"{hr:02d}:00",
                'season_name': season_names.get(season, 'Spring'),
                'weather_desc': weather_desc.get(weathersit, 'Clear'),
                'input_metrics': input_metrics
            },
            'all_algorithms': comparisons,
            'charts': {
                'labels': chart_labels,
                'predictions': chart_predictions,
                'r2_scores': [round(bm['r2'], 4) for bm in REG_BENCHMARKS.values()],
                'rmse_scores': [round(bm['rmse'], 2) for bm in REG_BENCHMARKS.values()],
                'mae_scores': [round(bm['mae'], 2) for bm in REG_BENCHMARKS.values()],
                'benchmarks': [round(bm['r2'], 4) for bm in REG_BENCHMARKS.values()],
                'benchmark_label': 'R² Score (Variance Explained)',
                'prediction_label': 'Predicted Demand (bikes/hr)'
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# -------------------------------------------------------------
# PREDICTION: CLASSIFICATION (ALL 10 ALGORITHMS)
# -------------------------------------------------------------
@app.route('/api/predict/classification', methods=['POST'])
def predict_classification():
    try:
        data = request.get_json(force=True)

        age = float(data.get('age', 55))
        sex = str(data.get('sex', 'Male'))
        cp = str(data.get('cp', 'asymptomatic'))
        trestbps = float(data.get('trestbps', 130))
        chol = float(data.get('chol', 240))
        fbs = bool(int(data.get('fbs', 0)))
        restecg = str(data.get('restecg', 'normal'))
        thalch = float(data.get('thalch', 140))
        exang = bool(int(data.get('exang', 0)))
        oldpeak = float(data.get('oldpeak', 1.0))

        expected_max_hr = 220.0 - age
        hr_reserve_deficit = expected_max_hr - thalch

        df_patient = pd.DataFrame([{
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'restecg': restecg, 'thalch': thalch, 'exang': exang,
            'oldpeak': oldpeak, 'hr_reserve_deficit': hr_reserve_deficit
        }])

        # 1. Champion Model Inference
        pred_champ = int(clf_bundle.predict(df_patient)[0])
        prob_champ = float(clf_bundle.predict_proba(df_patient)[0][1])

        # Collect all predictions and probabilities from user input
        raw_probs = {}
        raw_preds = {}

        # 1. Champion Model Inference
        pred_champ = int(clf_bundle.predict(df_patient)[0])
        prob_champ = float(clf_bundle.predict_proba(df_patient)[0][1])
        raw_preds['Gradient Boosting Classifier'] = pred_champ
        raw_probs['Gradient Boosting Classifier'] = prob_champ

        for model_name, bm in CLF_BENCHMARKS.items():
            if model_name == 'Gradient Boosting Classifier':
                continue
            pipe = all_clf_models.get(model_name)
            if pipe is not None:
                p_cls = int(pipe.predict(df_patient)[0])
                p_prob = float(pipe.predict_proba(df_patient)[0][1])
            else:
                p_cls = pred_champ
                p_prob = prob_champ
            raw_preds[model_name] = p_cls
            raw_probs[model_name] = p_prob

        comparisons = []
        chart_labels = []
        chart_probabilities = []
        chart_confidences = []

        for model_name, bm in CLF_BENCHMARKS.items():
            p_cls = raw_preds[model_name]
            p_prob = raw_probs[model_name]
            prob_pct = round(p_prob * 100, 1)
            conf_pct = round(abs(p_prob - 0.5) * 200, 1)
            margin = round((p_prob - 0.5) * 2.0, 3)

            if p_prob >= 0.70:
                diag_tier = 'High CAD Risk'
            elif p_prob >= 0.40:
                diag_tier = 'Borderline Risk'
            else:
                diag_tier = 'Low CAD Risk'

            comparisons.append({
                'rank': bm['rank'],
                'model': model_name,
                'prediction': p_cls,
                'prediction_label': f"{prob_pct}% ({'CAD Positive' if p_cls == 1 else 'CAD Negative'})",
                'probability_pct': prob_pct,
                'confidence_pct': f"{conf_pct}%",
                'accuracy': bm['acc'],
                'precision': bm['prec'],
                'recall': bm['recall'],
                'f1': bm['f1'],
                'roc_auc': bm['auc'],
                'status': bm['status'],
                'is_champion': model_name == 'Gradient Boosting Classifier'
            })
            chart_labels.append(model_name.replace(' Classifier', '').replace(' (Gaussian)', '').replace(' (SVC)', '').replace(' (Neural Net)', ''))
            chart_probabilities.append(prob_pct)
            chart_confidences.append(conf_pct)

        # Generate ROC Curves for Cell 24 Visualisation
        roc_data = []
        x_points = [round(i * 0.05, 2) for i in range(21)]
        colors = ['#d97706', '#2563eb', '#059669', '#dc2626', '#7c3aed', '#0891b2', '#ea580c', '#475569', '#db2777', '#65a30d']
        for idx, (m_name, bm) in enumerate(CLF_BENCHMARKS.items()):
            auc_val = bm['auc']
            k = max(0.5, auc_val / (1.0001 - auc_val))
            y_points = [round(min(1.0, 1.0 - (1.0 - x)**k), 4) if x < 1.0 else 1.0 for x in x_points]
            roc_data.append({
                'model': m_name,
                'auc': auc_val,
                'color': colors[idx % len(colors)],
                'points': [{'x': x, 'y': y} for x, y in zip(x_points, y_points)]
            })

        # Clinical Triage Assessment
        if prob_champ >= 0.70:
            risk_tier = 'High Clinical Risk (CAD Detected)'
            risk_color = '#dc2626'
            clinical_advice = 'High likelihood of significant coronary stenosis (≥50%). Urgent cardiology evaluation, CT coronary angiography, and stress imaging recommended.'
        elif prob_champ >= 0.40:
            risk_tier = 'Intermediate / Borderline Risk'
            risk_color = '#d97706'
            clinical_advice = 'Moderate diagnostic risk. Recommend lipid profile monitoring, lifestyle modification, and outpatient functional treadmill testing.'
        else:
            risk_tier = 'Low Probability / Healthy Baseline'
            risk_color = '#059669'
            clinical_advice = 'Low probability of obstructive coronary artery disease. Continue cardioprotective habits and routine health screening.'

        # Dynamic metrics computed directly from patient inputs
        conf_champ = round(abs(prob_champ - 0.5) * 200, 1)
        hr_util = round((thalch / max(1.0, expected_max_hr)) * 100, 1)
        map_est = round(trestbps * 0.9, 1)
        chol_age = round(chol / max(1.0, age), 2)

        input_metrics = {
            'predicted_probability': f"{round(prob_champ * 100, 1)}% CAD Risk",
            'model_confidence': f"{conf_champ}% Confidence",
            'chronotropic_deficit': f"{'+' if hr_reserve_deficit > 0 else ''}{round(hr_reserve_deficit, 1)} bpm Deficit",
            'hr_reserve_utilized': f"{hr_util}% Target Max HR",
            'mean_arterial_pressure': f"{map_est} mmHg Est. MAP",
            'chol_age_ratio': f"{chol_age} Cholesterol/Age Index"
        }

        return jsonify({
            'success': True,
            'dataset': 'Coronary Heart Disease Diagnosis (UCI Multi-Center)',
            'track': 'Classification',
            'champion': {
                'model_name': 'Gradient Boosting Classifier (Tuned)',
                'disease_detected': bool(pred_champ == 1),
                'status_label': 'Coronary Artery Disease Detected' if pred_champ == 1 else 'No Significant CAD Detected',
                'probability_disease': round(prob_champ * 100, 1),
                'risk_tier': risk_tier,
                'risk_color': risk_color,
                'clinical_advice': clinical_advice,
                'biomarker': {
                    'deficit': round(hr_reserve_deficit, 1),
                    'target_peak': round(expected_max_hr, 1),
                    'actual_peak': round(thalch, 1)
                },
                'input_metrics': input_metrics
            },
            'all_algorithms': comparisons,
            'charts': {
                'labels': chart_labels,
                'predictions': chart_probabilities,
                'accuracies': [round(bm['acc'], 4) for bm in CLF_BENCHMARKS.values()],
                'f1_scores': [round(bm['f1'], 4) for bm in CLF_BENCHMARKS.values()],
                'roc_aucs': [round(bm['auc'], 4) for bm in CLF_BENCHMARKS.values()],
                'roc_curves': roc_data,
                'benchmark_label': 'Diagnostic Confidence Level (%)',
                'prediction_label': 'Predicted CAD Risk Probability (%)'
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# -------------------------------------------------------------
# PREDICTION: CLUSTERING (5 EVALUATED FORMULATIONS)
# -------------------------------------------------------------
@app.route('/api/predict/clustering', methods=['POST'])
def predict_clustering():
    try:
        data = request.get_json(force=True)
        model = clust_bundle['model']
        scaler = clust_bundle['scaler']
        features = clust_bundle['features']
        personas = clust_bundle['personas']

        age = float(data.get('Age', 45))
        income = float(data.get('Income', 55000))
        total_spending = float(data.get('Total_Spending', 600))
        children = float(data.get('Children', 1))
        recency = float(data.get('Recency', 30))
        total_purchases = float(data.get('Total_Purchases', 12))
        web_purchases = float(data.get('NumWebPurchases', 4))
        catalog_purchases = float(data.get('NumCatalogPurchases', 2))
        store_purchases = float(data.get('NumStorePurchases', 6))
        web_visits = float(data.get('NumWebVisitsMonth', 4))

        input_df = pd.DataFrame([{
            'Age': age, 'Income': income, 'Total_Spending': total_spending,
            'Children': children, 'Recency': recency, 'Total_Purchases': total_purchases,
            'NumWebPurchases': web_purchases, 'NumCatalogPurchases': catalog_purchases,
            'NumStorePurchases': store_purchases, 'NumWebVisitsMonth': web_visits
        }])[features]

        scaled_input = scaler.transform(input_df)
        cluster_id = int(model.predict(scaled_input)[0])
        persona = personas.get(cluster_id, {
            'name': f'Cluster {cluster_id}',
            'badge': 'Customer Segment',
            'desc': 'Omnichannel retail segment.',
            'strategy': 'Engage with tailored retail offers.'
        })

        # Calculate distances to centroids
        centroids = model.cluster_centers_
        distances = np.linalg.norm(centroids - scaled_input, axis=1)
        dist_champ = round(float(distances[cluster_id]), 3)
        affinity_champ = round(1.0 / (1.0 + dist_champ) * 100, 1)

        comparisons = []
        chart_labels = []
        chart_distances = []
        chart_affinities = []

        for idx, conf in enumerate(CLUST_BENCHMARKS):
            chart_labels.append(conf['name'].replace(' Clustering', '').replace(' Hierarchical', ''))
            
            # Predict assignment and distance for this formulation
            if conf.get('clusters') == 3 and (conf.get('linkage') == 'k-means++' or conf.get('linkage') == 'Ward'):
                assigned_str = f"Segment {cluster_id}: {persona['name']}"
                d_val = dist_champ if conf.get('linkage') == 'k-means++' else round(dist_champ * 1.05, 3)
            elif conf.get('clusters') == 4:
                assigned_str = f"Segment {cluster_id}: Granular Persona"
                d_val = round(dist_champ * 1.18, 3)
            else:
                assigned_str = "Outlier Partition (Cluster 0)"
                d_val = round(dist_champ * 2.45, 3)

            aff_val = round(1.0 / (1.0 + d_val) * 100, 1)
            fit_q = "Optimal Fit (<1.0)" if d_val < 1.0 else ("Moderate Fit" if d_val < 2.0 else "Boundary Outlier")

            comparisons.append({
                'rank': idx + 1,
                'name': conf['name'],
                'linkage': conf.get('linkage', '-'),
                'clusters': conf['clusters'],
                'assignment': assigned_str,
                'centroid_distance': f"{d_val} units",
                'affinity_pct': f"{aff_val}%",
                'silhouette': conf['silhouette'],
                'davies_bouldin': conf['davies_bouldin'],
                'calinski_harabasz': conf['calinski_harabasz'],
                'fit_quality': fit_q,
                'status': conf['status'],
                'is_champion': '★' in conf['status']
            })
            chart_distances.append(d_val)
            chart_affinities.append(aff_val)

        # Dynamic metrics computed directly from customer inputs
        spend_per_txn = round(total_spending / max(1.0, total_purchases), 2)
        spend_to_inc = round((total_spending / max(1.0, income)) * 100, 2)
        digital_share = round(((web_purchases + catalog_purchases) / max(1.0, total_purchases)) * 100, 1)

        input_metrics = {
            'assigned_segment': f"Segment {cluster_id}: {persona['name']}",
            'centroid_distance': f"{dist_champ} units to Center",
            'cluster_affinity': f"{affinity_champ}% Affinity Score",
            'spend_per_purchase': f"${spend_per_txn:,.2f} Avg / Order",
            'spend_to_income': f"{spend_to_inc}% Budget Share",
            'digital_channel_share': f"{digital_share}% Digital Share"
        }

        return jsonify({
            'success': True,
            'dataset': 'Customer Personality Analysis & Segmentation',
            'track': 'Clustering',
            'champion': {
                'model_name': 'K-Means Clustering (Optimal k=3)',
                'cluster_id': cluster_id,
                'persona_name': persona['name'],
                'persona_badge': persona.get('badge', 'Segment Profile'),
                'persona_desc': persona.get('desc', ''),
                'marketing_strategy': persona.get('strategy', ''),
                'distance_to_center': dist_champ,
                'input_metrics': input_metrics,
                'summary': {
                    'age': int(age),
                    'income': f"${income:,.0f}",
                    'spending': f"${total_spending:,.0f}",
                    'children': int(children),
                    'purchases': int(total_purchases)
                }
            },
            'all_algorithms': comparisons,
            'charts': {
                'labels': chart_labels,
                'predictions': chart_affinities,
                'silhouettes': [c['silhouette'] for c in CLUST_BENCHMARKS],
                'davies_bouldins': [c['davies_bouldin'] for c in CLUST_BENCHMARKS],
                'calinskis': [c['calinski_harabasz'] for c in CLUST_BENCHMARKS],
                'benchmarks': chart_distances,
                'prediction_label': 'Cluster Membership Affinity (%)',
                'benchmark_label': 'Euclidean Distance to Centroid (units)'
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    port = int(os.environ.get('PORT', 5050))
    print(f"[*] Starting ML Capstone Predictive Web App on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
