# 🎓 Student Placement Prediction

A machine learning project that predicts whether a student will be **placed** or **not placed** based on academic performance, skills, and extracurricular profile.

## 📌 Overview

This project uses a dataset of 10,000 student records to build and compare multiple classification models for predicting campus placement outcomes. It covers the full ML workflow: data cleaning, exploratory data analysis (EDA), preprocessing, model training, cross-validation, hyperparameter tuning, and model serialization for deployment.

## 📂 Dataset

The dataset (`placementdata.csv`) contains **10,000 student records** with the following features:

| Column | Description | Type |
|---|---|---|
| `StudentID` | Unique student identifier (dropped before modeling) | int |
| `CGPA` | Cumulative Grade Point Average | float |
| `Internships` | Number of internships completed | int |
| `Projects` | Number of projects completed | int |
| `Workshops/Certifications` | Number of workshops/certifications | int |
| `AptitudeTestScore` | Aptitude test score | int |
| `SoftSkillsRating` | Soft skills rating | float |
| `ExtracurricularActivities` | Participation in extracurriculars (Yes/No) | object |
| `PlacementTraining` | Attended placement training (Yes/No) | object |
| `SSC_Marks` | Secondary school marks | int |
| `HSC_Marks` | Higher secondary marks | int |
| `PlacementStatus` | Target: `Placed` / `NotPlaced` | object |

**Class distribution:** NotPlaced — 5,803 · Placed — 4,197 (mild class imbalance, ~58/42 split). No missing values or duplicates in the dataset.

## 🔍 Project Workflow

1. **Data Cleaning** – checked for nulls/duplicates, dropped the non-predictive `StudentID` column.
2. **Exploratory Data Analysis** – visualized placement status distribution and its relationship with CGPA, internships, projects, aptitude scores, and SSC/HSC marks using box plots and count plots.
3. **Preprocessing**
   - Label-encoded the target (`PlacementStatus`)
   - One-hot encoded categorical features (`ExtracurricularActivities`, `PlacementTraining`)
   - 80/20 stratified train-test split
   - Feature scaling with `StandardScaler` (for scale-sensitive models)
4. **Model Training** – trained and evaluated six classifiers:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
   - Naive Bayes
   - Decision Tree
   - Random Forest
5. **Cross-Validation** – 5-fold stratified CV to assess model stability.
6. **Hyperparameter Tuning** – `GridSearchCV` for all base models, plus ensemble methods (Random Forest, AdaBoost, Gradient Boosting).
7. **Model Selection & Export** – best-performing model saved with `joblib` for reuse/deployment.

## 📊 Results

### Baseline model accuracy (test set)

| Model | Accuracy |
|---|---|
| **Logistic Regression** | **0.8085** |
| Naive Bayes | 0.8005 |
| SVM | 0.7990 |
| Random Forest | 0.7960 |
| KNN | 0.7780 |
| Decision Tree | 0.7275 |

### After hyperparameter tuning (test accuracy)

| Model | CV Accuracy | Test Accuracy |
|---|---|---|
| **Logistic Regression** | 0.7968 | **0.8085** |
| AdaBoost | 0.7979 | 0.8035 |
| SVM | 0.7979 | 0.8030 |
| KNN | 0.7949 | 0.8010 |
| Naive Bayes | 0.7945 | 0.8005 |
| Gradient Boosting | 0.7980 | 0.7995 |
| Random Forest | 0.7970 | 0.7955 |
| Decision Tree | 0.7823 | 0.7815 |

**Final selected model: Logistic Regression** (best test accuracy at ~80.85%, with stable cross-validation performance).

## 🗂️ Saved Artifacts

Running the notebook produces the following files (used for inference without retraining):

- `logistic_regression_model.pkl` — trained final model
- `scaler.pkl` — fitted `StandardScaler` for input features
- `feature_names.pkl` — ordered list of feature names expected by the model

## 🛠️ Tech Stack

- **Language:** Python
- **Data Handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly
- **Machine Learning:** scikit-learn
- **Model Persistence:** joblib

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn joblib
```

### Running the Notebook

1. Place `placementdata.csv` in the same directory as the notebook.
2. Open and run `Placement_prediction.ipynb` in Jupyter Notebook / JupyterLab.
3. The notebook will train all models, run tuning, and export the final model artifacts.

### Using the Saved Model for Prediction

```python
import joblib

model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# Prepare new input data as a DataFrame with the same columns as `feature_names`,
# then scale and predict:
X_new_scaled = scaler.transform(new_data[feature_names])
prediction = model.predict(X_new_scaled)
```

## 📈 Possible Improvements

- Address class imbalance (e.g., SMOTE, class weights)
- Try additional ensemble/stacking techniques
- Feature engineering (interaction terms, polynomial features)
- Deploy as a REST API or simple web app for real-time predictions

## 📄 License

This project is for educational purposes. Add a license of your choice if distributing.
