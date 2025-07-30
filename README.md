# 🔬 Shell.ai Hackathon 2025 – Sustainable Fuel Blend Property Prediction

## 🏁 Overview

This repository contains the full machine learning pipeline developed by **Team Alpha Warriors** (Ayush, Arjun, Animesh, and Pranay) for the **Shell.ai Hackathon 2025**.  
The goal of the challenge was to develop predictive models that estimate the **physical and chemical properties of sustainable aviation fuel blends**, given:

- The **proportions** of each component in the blend
- The **batch-specific chemical properties** of each component

We built a hybrid ensemble pipeline that leverages **feature engineering**, **deep learning**, **gradient boosting**, and **transformer-based tabular modeling** to achieve robust, high-performing predictions.

---

## 📦 Dataset Description

The competition provided 3 key files:

- `train.csv`: 65 columns – 5 blend composition features, 50 component properties, 10 target blend properties
- `test.csv`: Same structure as train (excluding targets)
- `sample_submission.csv`: Format required for submission

Each row represents a unique fuel blend and its corresponding chemical profile.

| Feature Group          | Columns                             |
|------------------------|--------------------------------------|
| Blend Composition      | Component1_fraction ... Component5_fraction |
| Component Properties   | Component1_Property1 ... Component5_Property10 |
| Blend Targets (labels) | BlendProperty1 ... BlendProperty10  |

---

## 🎯 Objective

Develop models that accurately predict the final **10 blend properties** from the input features. These predictions guide real-world decisions in **fuel formulation**, enabling safer, more efficient, and sustainable fuel development.

---

## 🛠️ ML Pipeline Summary

### 🔧 Feature Engineering Highlights

- **Weighted Property Averages**: Based on blend proportions
- **Variance and Squared Stats**: Per target property
- **Min/Max per Property**: Across all components
- **Deviation Terms**: Difference from weighted mean per component
- **Polynomial Features**: 2nd-degree interactions
- **Ratio & Interaction Terms**: Between blend fractions
- **Per-target Feature Selection**: Based on correlation thresholds

> Final dataset after engineering: ~150 enriched features

---

### 🧠 Modeling Strategy

Our model ensemble was trained per target (10 total):

| Model Type          | Framework               |
|---------------------|--------------------------|
| FT-Transformer      | PyTorch Tabular          |
| LightGBM            | LightGBM                 |
| CatBoost            | CatBoost                 |
| XGBoost             | XGBoost                  |
| ANN                 | PyTorch                  |

Each model was trained using:

- 5-Fold Cross-Validation
- Per-target normalization + feature selection
- Early stopping and model checkpoints (where applicable)

### 🤖 Meta-Ensembling

We used **Bayesian Ridge Regression** as a final layer to blend model outputs (`OOF predictions`) into a single prediction per target.

---

## 🧪 Training Details

- Total submissions tested:  
  - 🔹 Pranay: **420+**  
  - 🔹 Ayush: **220+**
- Frameworks: `scikit-learn`, `PyTorch`, `LightGBM`, `CatBoost`, `XGBoost`, `pytorch-tabular`
- Scoring Metric: **Mean Absolute Percentage Error (MAPE)**
- Public Leaderboard Score: `94.201`
- Final Output: `submission_hybrid.csv` (500 × 10)

---

## 🚀 Results

- ✅ Achieved **Top 25 Global Rank**
- 🧠 Designed and tested **100s of variations**
- 🏗️ Built modular, reproducible code for reuse in real-world scenarios

---

## 💡 Key Takeaways

- ML + chemistry requires respecting the **physics** and **constraints** of the domain  
- Deep tabular models (like FT-Transformer) work well when combined with **feature-rich inputs**  
- Ensembling different model types improves robustness  
- Real-world modeling challenges involve **data cleaning, domain understanding, and iteration** – not just model tuning

---

## 📂 How to Reproduce

1. Install required libraries:
```bash
pip install -r requirements.txt


