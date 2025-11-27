# 🏥 Obesity-Level-Predictor-with-Streamlit

> Predict obesity levels using XGBoost with a Streamlit frontend and FastAPI backend.

A machine learning project that classifies individuals into obesity categories based on lifestyle and health factors.
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-latest-red.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-green.svg)](https://fastapi.tiangolo.com/)

---

## 📋 Features

- Real-time predictions via **Streamlit UI**
- RESTful API with **FastAPI**
- **XGBoost classifier** with preprocessing pipeline
- Supports **7 obesity categories**
- Easy integration and local processing (data privacy)

---

## 📊 Dataset

**17 features** including demographics, genetics, diet, activity, and habits.  
**Target**: 7-class obesity classification.

| Class                   | Description              |
|-------------------------|-------------------------|
| `Insufficient_Weight`   | Underweight             |
| `Normal_Weight`         | Healthy range           |
| `Overweight_Level_I`    | Mildly overweight       |
| `Overweight_Level_II`   | Moderately overweight   |
| `Obesity_Type_I`        | Class I obesity         |
| `Obesity_Type_II`       | Class II obesity        |
| `Obesity_Type_III`      | Severe obesity          |

---
## 📈 Model Performance

| Metric    | Score    |
|-----------|---------|
| Accuracy  | 96.21%  |
| Precision | 96%     |
| Recall    | 96%     |
| F1-Score  | 96.08%  |


## 🛠️ Tech Stack

- **ML**: XGBoost, Scikit-learn, Pandas, NumPy  
- **Backend**: FastAPI, Uvicorn  
- **Frontend**: Streamlit  
- **Development**: Python 3.8+, Jupyter Notebook


