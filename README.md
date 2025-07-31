# 🏦 Loan Approval Prediction

This notebook is part of an internship project aimed at building a robust machine learning pipeline to predict whether a loan application will be approved based on applicant data.

## 📂 Dataset
- Source: [Kaggle Loan Approval Prediction Dataset](https://www.kaggle.com/datasets)
- Format: CSV
- Features: Income, loan amount, CIBIL score, asset values, education, self-employment, and more.

## 🧠 Problem Statement
Build a classification model that can predict loan approval status. Special focus is placed on handling imbalanced data and comparing different algorithms.

## 🛠️ Tools & Libraries
- Python
- Pandas, NumPy
- Scikit-learn (Classification, Evaluation)
- Seaborn, Matplotlib (Visualization)
- Imbalanced-learn (Handled via custom oversampling due to kernel issues)

## 📊 Steps Performed
1. **Data Preprocessing**
   - Handling missing values
   - Label encoding
   - Feature engineering (`Total_Income`, `Income_Loan_Ratio`)
2. **Data Visualization**
   - Distribution plots
   - Missing value heatmaps
   - Target distribution
3. **Handling Imbalance**
   - Applied Random Oversampling instead of SMOTE (environment compatibility)
4. **Model Training**
   - Logistic Regression
   - Decision Tree
5. **Model Evaluation**
   - Confusion Matrix
   - Classification Report (Precision, Recall, F1-score)
   - ROC Curve with AUC comparison

## 🔍 Results
- Both models were trained and compared on ROC AUC and classification metrics.
- The use of random oversampling balanced the dataset, improving recall for the minority class.

## 📌 Highlights
- Balanced the dataset without relying on heavy external libraries.
- Clean, interpretable models with explainable visual evaluation.
- ROC Curve to assess overall model performance.


