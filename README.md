# Comparative Analysis Using Supervised and Deep Learning Algorithms for Binary Classification on a Superhero Dataset
## CS634: Data mining

## Project Overview
This project performs a **comparative analysis** of three machine learning models — **k-Nearest Neighbors (kNN)**, **Random Forest**, and **Long Short-Term Memory (LSTM)** — to determine which performs best for a **binary classification task** on a custom superhero dataset.  
The goal is to predict whether a superhero is **“good” or “evil”** based on physical attributes, training habits, and special powers.

---

## Algorithms Used
1. **k-Nearest Neighbors (kNN)** – Classifies based on the majority vote among the k nearest data points.  
2. **Random Forest** – An ensemble method combining multiple decision trees for better accuracy and generalization.  
3. **Long Short-Term Memory (LSTM)** – A deep learning model adapted for tabular classification; requires tuning to avoid overfitting.

---

## Dataset Description
**Dataset:** `superhero dataset.csv`

Each row represents a superhero with these features:

| Feature | Description |
|----------|-------------|
| `height_cm` | Height in centimeters |
| `weight_kg` | Weight in kilograms |
| `age` | Age of the superhero |
| `years_active` | Total years active |
| `training_hours_per_week` | Average training hours per week |
| `civilian_casualties_past_year` | Civilian casualties in the past year |
| `power_level` | Overall power rating |
| `public_approval_rating` | Public approval rating (0–100%) |
| `super_strength`, `flight`, `telepathy`, `healing_factor`, `shape_shifting`, `invisibility`, `telekinesis` | Binary power indicators (1 = Yes, 0 = No) |
| `is_good` | Target variable (1 = good, 0 = evil) |

---

## Environment Setup

### Clone the Repository
```bash
git clone https://github.com/KenilAvaiyaa/Avaiya_Kenil_finaltermproj.git
cd Avaiya_Kenil_finaltermproj
```

### Create a Virtual Environment
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```
**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies
Install all required libraries:
```bash
pip install -r requirements.txt
```
Or manually:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
```

---

## How to Run
### Option 1 – Run Jupyter Notebook
Open `main.ipynb` and run all cells.

### Option 2 – Run Python Script
```bash
python main.py   # or python3 main.py on macOS/Linux
```
Make sure `superhero dataset.csv` is in the same folder.

---

## Project Workflow
1. Load dataset using pandas  
2. Handle missing/duplicate values  
3. Perform EDA with Seaborn and Matplotlib  
4. Standardize features using `StandardScaler`  
5. Split data into 80% training and 20% testing  
6. Train models: kNN, Random Forest, LSTM  
7. Perform 10-fold cross-validation  
8. Evaluate with metrics: Accuracy, Precision, Recall, F1, Specificity, AUC, Brier Score, TSS, HSS, BSS, etc.  

---

## Results Summary

| Model | Accuracy | Precision | Recall | Specificity | AUC |
|--------|-----------|-----------|---------|--------------|------|
| **Random Forest** | ~65% | **0.67** | **0.92** | Best Balance | **Highest** |
| **kNN** | Moderate | Balanced | Fair | Moderate | Average |
| **LSTM** | High Recall | Low Specificity | 1.0 | 0.0 | Unstable |

**Best Performing Model:** Random Forest  
It achieved the most balanced and reliable performance across all metrics.

---

## Tools & Libraries
- Python 3.x  
- Pandas  
- NumPy  
- Scikit-learn  
- TensorFlow / Keras  
- Matplotlib  
- Seaborn  
