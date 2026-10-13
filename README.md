# Credit Card Default Prediction – EDA, Visualization & Deep Learning

**Course:** U21ADP05 – Exploratory Data Analysis and Visualization (Assignment II – Case Study)
**Institution:** KPR Institute of Engineering and Technology, Dept. of Artificial Intelligence and Data Science
**Author:** AKHILASH-SK (GitHub) – *name / roll no. to be filled in*

## Problem
A bank wants to know **which credit-card customers will miss next month's payment**, so it can set limits, trigger early-warning calls and reduce losses. This project explores a real-world dataset end to end: cleaning, statistical EDA, 19 visualizations, a deep-learning model (PyTorch MLP) and a full evaluation.

## Dataset
*Default of Credit Card Clients* – UCI Machine Learning Repository (Yeh & Lien, 2009).
Link: <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>
30,000 customers (Taiwan, Apr–Sep 2005), **23 input features** (credit limit, demographics, 6 months of repayment status, bills and payments) and a binary target: *default payment next month* (22.1 % positive).
A CSV copy is stored in [`data/credit_default.csv`](data/credit_default.csv).

## Repository structure
```
├── README.md
├── requirements.txt
├── data/credit_default.csv            # dataset (CSV copy of the UCI file)
├── notebooks/eda_dl_case_study.ipynb  # full, commented, executed notebook (code + outputs + interpretations)
├── figures/                           # every chart as PNG (+ one interactive Plotly HTML)
└── report/                            # test_metrics.csv, summary_numbers.json (and the project report)
```

## What the notebook does
| Step | Content |
|---|---|
| 1-2 | Load data, data dictionary, summary statistics, missing values, duplicates (35 removed), invalid category codes, IQR outliers, feature engineering (6 new features) |
| 3 | 13 visualizations: histograms, boxplots, bar charts, distribution fitting (Normal / Log-normal / Gamma, Binomial / Poisson / Bernoulli), default-rate charts, chi-square & Mann-Whitney tests, scatterplots, correlation heatmap, pairplot, multivariate heatmap, interactive Plotly chart |
| 4 | Signed-log transform, one-hot encoding, standardization, stratified 70/15/15 train/validation/test split (no leakage) |
| 5 | PyTorch **MLP** with batch-norm, dropout, early stopping; grid search over 12 hyperparameter settings; baselines (Logistic Regression, Random Forest) |
| 6 | Loss/accuracy vs epoch, confusion matrix, ROC + AUC, precision-recall, score distribution, permutation feature importance |
| 7 | Conclusions, limitations, future scope |

## Key results (test set, 4,495 customers)
| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **MLP (PyTorch)** | **0.785** | 0.792 | 0.528 | 0.546 | 0.537 |
| Random Forest | 0.779 | 0.799 | 0.549 | 0.504 | 0.526 |
| Logistic Regression | 0.766 | 0.780 | 0.502 | 0.574 | 0.536 |

(Decision threshold tuned on the validation set; the MLP threshold is 0.327.)

**Main findings**
- Payment behaviour matters far more than demographics: default rises from ~13 % for customers paying on time to ~69 % for those two months late.
- The most important features are the latest repayment status (PAY_1), credit utilization, number of late months and average payment; age, gender and marital status contribute almost nothing.
- A deep network gives only a small gain over simple models on this tabular dataset; the ceiling (AUC ≈ 0.78) comes from overlapping classes, not from the model.

## How to run
```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/eda_dl_case_study.ipynb   # or open it in Jupyter / VS Code
```
All random seeds are fixed (42) and training runs on CPU, so the results are reproducible. Figures are written to `figures/`.

## References
1. Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications, 36*(2), 2473-2480.
2. Paszke, A. et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *NeurIPS 32*.
3. Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR, 12*, 2825-2830.
4. Fry, B. (2007). *Visualizing Data: Exploring and Explaining Data with the Processing Environment.* O'Reilly Media.
5. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering, 9*(3), 90-95.
