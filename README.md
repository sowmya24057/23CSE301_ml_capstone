# 23CSE301 — Machine Learning Capstone Project

B.Tech CSE, AY 2026-27. End-to-end ML pipelines across Regression, Classification,
and Clustering tracks.

## Team
- *<Name 1, Roll No.>*
- *<Name 2, Roll No.>*
- *<Name 3, Roll No.>*

## Datasets

| Track | Dataset | Target | File(s) |
|---|---|---|---|
| Regression | UCI Bike Sharing Dataset | `cnt` (daily rental count) | `data/day.csv`, `data/hour.csv` |
| Classification | UCI Heart Disease Dataset | `num` → binarized to disease presence | `data/heart_disease_uci.csv` |
| Clustering | *TBD (Review 2)* | — | — |

## Repository Structure

```
├── README.md
├── requirements.txt
├── data/           raw dataset files
├── notebooks/      regression.ipynb, classification.ipynb, clustering.ipynb
├── models/         saved model files (.pkl), if any
└── app/            GUI / deployment code (bonus), if attempted
```

## Setup & How to Run

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Open the notebook for the track you want under `notebooks/` and run all cells
top-to-bottom (`Cell → Run All`).

## Progress

- [x] Review 1 — Regression track (10 algorithms) + Classification Part A (5 algorithms)
- [ ] Review 2 — Classification Part B + Clustering track

## Results Summary


**Regression (best model: Random Forest Regressor)** — R² ≈ 0.88 on held-out test set.

**Classification Part A (best model: SVM by weighted F1)** — Accuracy ≈ 0.80, ROC-AUC ≈ 0.88.
