# 🏠 House Price Prediction - Ames Housing Dataset

Predicting house sale prices using the [Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniquestechniques) from Kaggle. Built as part of a data science challenge focused on numerical features, interpretable modelling, and justified decision-making.

---

## 📁 Repository Contents

| File | Description |
|------|-------------|
| `notebook.ipynb` | Full project notebook - EDA, cleaning, engineering, and modelling |
| `house_price_report.docx` | Written report covering all five parts of the brief |
| `submission.csv` | Kaggle submission file with predicted SalePrice for test set |

---

## 📌 Project Overview

A junior data scientist scenario: given only the measurable, numerical characteristics of residential properties in Ames, Iowa, build a model that predicts sale price and justify every decision made along the way.

**Constraint:** Only numerical features (`int64` / `float64`) were used. No categorical encoding - 36 features after filtering.

---

## 🔍 Methodology

### Part 1 - Exploratory Data Analysis
- Filtered dataset to 36 numerical features
- Identified top 5 features by correlation with SalePrice: `OverallQual`, `GrLivArea`, `GarageCars`, `GarageArea`, `TotalBsmtSF`
- Analysed scatter plots for linearity and variance patterns
- Diagnosed right-skewed SalePrice (skewness: 1.88) and justified log transformation
- Identified and dropped 2 confirmed outliers (large homes, implausibly low prices)

### Part 2 - Data Cleaning
- **LotFrontage** (259 missing): filled with median - robust to skewed lot sizes
- **MasVnrArea** (8 missing): filled with 0 - missing means no masonry veneer
- **GarageYrBlt** (81 missing): filled with 0 - cross-validated against GarageType = NA, confirming no garage exists
- Detected multicollinearity pairs (|r| > 0.8): dropped `GarageCars` and `TotRmsAbvGrd`
- Applied `StandardScaler` for fair regularisation in Ridge regression

### Part 3 - Feature Engineering
Four new features created from existing columns:

| Feature | Formula | Correlation w/ SalePrice |
|---------|---------|--------------------------|
| `TotalSF` | 1stFlrSF + 2ndFlrSF + TotalBsmtSF | 0.782 |
| `HouseAge` | YrSold − YearBuilt | −0.523 |
| `TotalBathrooms` | FullBath + 0.5×HalfBath + BsmtFullBath + 0.5×BsmtHalfBath | 0.632 |
| `YearsSinceRemodel` | YrSold − YearRemodAdd | −0.507 |

`log(SalePrice)` used as the model target to normalise the distribution (skewness: 1.88 → 0.12).

### Part 4 - Modelling & Validation

| Model | CV RMSE (log SalePrice) | Std Dev |
|-------|------------------------|---------|
| Linear Regression (OLS) | 0.1289 | 0.0059 |
| Ridge Regression (α=50) | 0.1289 | 0.0057 |

- 5-fold cross-validation used throughout - no single train/test split
- Both models pass the target threshold of **RMSE < 0.16**
- Ridge marginally reduces variance across folds; performance is near-identical at this feature-to-row ratio

---

## 📊 Results

> **Final CV RMSE: 0.1289** on log(SalePrice) - below the 0.16 competition threshold.

Predictions were generated using the Ridge model (α=50), back-transformed with `exp()` for the Kaggle submission.

---

## 🛠️ Tech Stack

- Python 3
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

---

## 📂 Data Source

Kaggle - [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

Download `train.csv` and `test.csv` and place them in the root directory before running the notebook.
