# End-to-End Documentation for Loan Default Prediction

## 1. Introduction

This document provides a detailed walkthrough of the loan default prediction project, from understanding the problem to deploying the final model.

### 1.1. Problem Statement

The goal of this project is to build a machine learning model to predict the likelihood of a borrower defaulting on a loan. This will help the financial institution make better lending decisions and mitigate financial risk.

### 1.2. Deliverables

*   A Jupyter notebook with the complete code for data analysis, feature engineering, and model building.
*   A presentation slide illustrating the system architecture for deployment.
*   This end-to-end documentation.

## 2. Exploratory Data Analysis (EDA)

The EDA was performed in the `eda.ipynb` notebook. The key findings are:

*   The dataset is imbalanced, with a small percentage of loan defaulters.
*   Several columns have missing values, which need to be handled.
*   The dataset contains a mix of numerical and categorical features.

## 3. Data Cleaning and Preprocessing

The following steps were taken to clean and preprocess the data:

*   **Handling Missing Values:**
    *   Columns with a high percentage of missing values (`Social_Circle_Default`, `Score_Source_2`, `Own_House_Age`) were dropped.
    *   Missing values in the `Client_Occupation` column were imputed with the mode.
    *   Missing values in numerical columns (`Score_Source_1`, `Score_Source_3`, `Credit_Bureau`, `Loan_Annuity`) were imputed with the median.
*   **Categorical Feature Encoding:**
    *   Categorical features were converted into numerical format using one-hot encoding.

## 4. Feature Engineering

The following new features were created to improve model performance:

*   `Age_Years`: The client's age in years.
*   `Employment_Years`: The client's employment duration in years.
*   `Credit_to_Income_Ratio`: The ratio of the credit amount to the client's income.
*   `Annuity_to_Income_Ratio`: The ratio of the loan annuity to the client's income.
*   `Credit_Term`: The credit term of the loan.

## 5. Model Building and Evaluation

Three models were trained and evaluated:

*   **Logistic Regression:** A baseline model.
*   **Random Forest Classifier:** An ensemble model.
*   **Gradient Boosting Classifier:** Another ensemble model.

The Gradient Boosting model performed the best, with the highest AUC-ROC score.

## 6. Hyperparameter Tuning

The Gradient Boosting model was further optimized by tuning its hyperparameters using `GridSearchCV`. This resulted in a more robust and accurate model.

## 7. Deployment

The final model can be deployed as a REST API using a web framework like Flask or FastAPI. The system architecture for deployment is described in the `system_architecture.md` file.

### 7.1. Business Solution

The loan default prediction model can be integrated into the loan application process to provide a real-time risk score for each applicant. This will help the financial institution:

*   **Make more informed lending decisions:** By identifying high-risk applicants, the institution can reduce the number of loan defaults.
*   **Automate the loan approval process:** The model can be used to automatically approve or reject loan applications based on a predefined risk threshold.
*   **Improve risk management:** The model can be used to monitor the overall risk of the loan portfolio.

### 7.2. Interpretation of Results

The model's predictions can be interpreted using techniques like SHAP (SHapley Additive exPlanations). This will help explain why the model made a particular prediction for a given applicant, which is important for transparency and fairness.
