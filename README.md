# Stock Price Regression — Data Mining Project

Comparative regression analysis on daily stock prices of 10 aerospace companies (1988–1991), predicting the value of one company based on the other nine.

**Best result: Random Forest — MAE 0.568, RMSE 0.863, MAPE 1.2%**

---

## Dataset

- **Source:** [OpenML — Stock dataset (id=223)](https://openml.org/search?type=data&id=223), originally from the StatLib repository
- **Instances:** 950 daily records
- **Features:** 10 continuous variables (daily stock prices, float64), no missing values
- **Target:** `company10` — predicted from the remaining 9 companies

---

## Pipeline

**Preprocessing**
- Exploratory analysis: descriptive statistics and boxplots (outliers found in `company4` and `company9`, kept due to low count)
- 80/20 train/test split with `random_state=42`
- StandardScaler fitted on training set only — applied separately to features and target to prevent data leakage

**Models** (all scikit-learn, default parameters)
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

**Evaluation metrics:** MAE, RMSE, MAPE, SMAPE (custom implementation)

**Dimensionality reduction:** PCA (2 components) applied post-standardization, experiments repeated for comparison

---

## Results

### Full Feature Space

| Model | MAE | RMSE | MAPE | SMAPE |
|---|---|---|---|---|
| **Random Forest** | **0.568** | **0.863** | **1.2%** | **0.6%** |
| Decision Tree | 0.670 | 1.000 | 1.4% | 0.7% |
| Gradient Boosting | 0.753 | 1.012 | 1.6% | 0.8% |
| Linear Regression | 1.797 | 2.359 | 3.9% | 1.9% |

### After PCA (2 components)

All models degraded significantly after reducing to 2 PCA components. Cumulative variance analysis revealed that 2 components capture insufficient variance for this dataset — 3 or 4 components would be more appropriate.

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Random Forest | 1.720 | 2.829 | 3.6% |
| Decision Tree | 1.991 | 3.601 | 4.2% |
| Gradient Boosting | 2.009 | 2.831 | 4.2% |
| Linear Regression | 4.518 | 5.290 | 9.5% |

---

## Key Findings

- **Random Forest** is the best model for this problem, with no significant overfitting or underfitting
- **`company1`** is the most influential predictor across all tree-based models (71% correlation with `company10`)
- **PCA with 2 components** causes a dramatic performance drop — feature selection on `company1` and `company9` would likely be a better dimensionality reduction strategy
- **Linear Regression** underperforms likely due to non-linear relationships in financial data and sensitivity to outliers

---

## Stack

`Python` · `scikit-learn` · `pandas` · `numpy` · `matplotlib` · `seaborn`

---

## How to Run

```bash
git clone https://github.com/Matti02co/stock-price-regression
cd stock-price-regression
pip install -r requirements.txt
jupyter notebook NotebookProgetto.ipynb
```

The dataset is fetched automatically via `sklearn.datasets.fetch_openml`.

---

*Data Mining project — University of Cagliari, Computer Science*  
*Cocco Mattia · Lepuri Tomas*
