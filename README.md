# Customer Churn Prediction – ML Classification Project

## Project Overview
This project focuses on predicting customer churn using machine learning models.  
The goal is not just to maximize accuracy, but to correctly identify customers who are likely to leave (churn = 1), which is critical for business retention strategies.

The dataset contains 7,000+ customers with demographic, service, contract, and billing information.

---

## Data Preparation
- Encoded categorical variables using one-hot encoding
- Handled class imbalance (more non-churners than churners)
- Split data into train/test sets using stratified sampling
- Evaluated models using Accuracy, ROC-AUC, Precision, Recall, F1-score, and Confusion Matrix

---

## Models Implemented
- Decision Tree
- Random Forest (n_estimators=200)
- XGBoost (Gradient Boosting)

XGBoost delivered the best overall performance.

---

## Final Model: XGBoost (Optimized for Recall)

- **300 trees** → The model builds 300 sequential decision trees to improve predictions.
- **Learning rate = 0.05** → Controls how much each new tree contributes. A smaller value makes learning more stable and reduces overfitting.
- **Max depth = 4** → Limits how complex each tree can be. This prevents the model from memorizing the data.
- **Evaluation metric = log loss** → Measures how confident and accurate the probability predictions are.

The model was further optimized by lowering the classification threshold to improve **recall for churners**, which significantly reduced missed churn cases.

---

## Libraries Used

- **pandas**  
- **numpy** 
- **scikit-learn** 
- **xgboost** 
- **matplotlib** – Data visualization  

---

This approach prioritizes detecting churners (recall) over raw accuracy, making it more aligned with real-world business objectives.

