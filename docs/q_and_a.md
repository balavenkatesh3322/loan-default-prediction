# Interview Evaluation & System Design Q&A

This document provides detailed answers to the evaluation criteria and system design questions outlined in the project description.

---

## Part 1: Data Science Interview Evaluation

### § Problem Understanding
My understanding is that the core problem is to mitigate financial risk for a lending institution by accurately predicting the probability of a loan applicant defaulting. This is a binary classification problem where the positive class (default) is rare, meaning we must handle significant class imbalance. The solution needs to be robust, interpretable, and deployable to provide real-world value.

### § Proposed Solution
My proposed approach, which I have implemented, follows a structured, end-to-end machine learning workflow:
1.  **Data Ingestion & EDA:** Load the dataset and perform a thorough Exploratory Data Analysis (EDA) to understand feature distributions, correlations, and identify initial challenges like missing data and class imbalance.
2.  **Data Preprocessing:** Systematically handle missing values using appropriate imputation techniques (median for numerical, mode for categorical) and drop columns where imputation is not feasible.
3.  **Feature Engineering:** Create new, more predictive features from existing ones, such as `Credit_to_Income_Ratio` and `Employment_Years`, to capture more complex relationships.
4.  **Model Selection:** Train and evaluate multiple baseline models (Logistic Regression, Random Forest, Gradient Boosting) to identify the best-performing algorithm for this specific dataset.
5.  **Handling Imbalance:** Implement the SMOTE (Synthetic Minority Over-sampling Technique) on the training data to correct for class imbalance and prevent the model from being biased towards the majority class.
6.  **Model Training & Tuning:** Train the selected model (Gradient Boosting) on the preprocessed, balanced dataset. Then, perform hyperparameter tuning using `GridSearchCV` to optimize its predictive performance.
7.  **Evaluation:** Evaluate the final model on an unseen test set using a comprehensive set of metrics, including AUC-ROC, Precision, and Recall, to ensure it performs well on both classes.

### § Model Development
My ability to develop a predictive model is demonstrated in the `eda.ipynb` notebook. Specifically:
*   **Handling Missing Values:** I identified columns with missing data and applied a clear strategy: dropping columns with over 50% missing values and imputing the rest with the median or mode to preserve data integrity.
*   **Handling Imbalanced Data:** I correctly identified the class imbalance from the target variable's distribution and used SMOTE to create a balanced training set. This is a standard and effective technique for this common problem.
*   **Handling Outliers:** By using median imputation and tree-based models like Gradient Boosting (which are naturally robust to outliers), the impact of extreme values is minimized without aggressive data removal.

### § Model Evaluation
My understanding of model evaluation is demonstrated by the selection and interpretation of the following metrics in the notebook:
*   **Precision:** Measures the accuracy of positive predictions. This is critical for the business, as it tells us how many of the applicants we flag as defaulters will *actually* default.
*   **Recall (Sensitivity):** Measures the model's ability to identify all actual defaulters. High recall is crucial to minimize the number of defaults that go undetected.
*   **F1-Score:** The harmonic mean of precision and recall, providing a single score that balances both concerns.
*   **AUC-ROC Score:** Measures the model's ability to distinguish between the positive and negative classes across all classification thresholds. It provides a robust, aggregate measure of model performance, especially on imbalanced datasets.

### § Knowledge and Experience
This project demonstrates my knowledge and experience in:
*   **Machine Learning Algorithms:** I implemented and compared Logistic Regression, Random Forest, and Gradient Boosting, explaining the trade-offs of each.
*   **Data Analysis:** I used libraries like Pandas, Matplotlib, and Seaborn to conduct a thorough EDA, which informed the entire modeling process.
*   **Python & Scikit-learn:** The entire solution is built using Python and the Scikit-learn, Imblearn, and Pandas libraries, demonstrating proficiency in the standard data science toolkit.

---

## Part 2: Evaluation Criteria (Detailed)

### § EDA and Pre-processing
This was a foundational step. The `eda.ipynb` notebook contains detailed code for exploring the data, visualizing distributions, and implementing the complete preprocessing pipeline, including imputation and one-hot encoding.

### § Feature Importance
The final Gradient Boosting model allows us to extract feature importances, which explain which factors are most influential in predicting loan defaults. This is crucial for business interpretation. I can add a code cell to the notebook to plot the top 10 most important features.

### § Modelling and Results
The notebook shows the end-to-end modeling process: splitting the data, applying SMOTE, training multiple models, and performing hyperparameter tuning. The results, including the final classification report and AUC-ROC score, are printed at the end of the notebook, providing a clear measure of the final model's accuracy.

### § Business Solution/Interpretation of Results Obtained
The model provides a risk score (a probability between 0 and 1) for each loan applicant. The business can use this score to:
1.  **Set a Risk Threshold:** Automatically approve applications below the threshold, reject those above it, and send borderline cases for manual review.
2.  **Prioritize Collections:** For active loans, the model can be used to predict the likelihood of future default, allowing the collections department to prioritize their efforts.
3.  **Inform Interest Rates:** Higher-risk applicants could be offered loans at higher interest rates to compensate for the increased risk.

### § How to Handle an Imbalance Dataset
I addressed this using the **SMOTE (Synthetic Minority Over-sampling Technique)**. This technique works by creating new, synthetic data points for the minority class (defaulters) in the training set. It does this by finding the k-nearest neighbors of a minority class instance and then creating a new instance at a random point along the line connecting the original instance and its neighbors. This balances the dataset and helps the model learn the characteristics of the minority class more effectively, leading to better recall and F1-scores.

---

## Part 3: System Design Tasks

### § Design System Architecture to Deploy this ML Model in Production
The architecture is detailed and visualized in `docs/system_architecture.md`. It consists of an ML pipeline for automated training and a serving infrastructure to host the model as a REST API, with continuous monitoring.

### § How do you perform canary build?
For an ML model, a canary build involves deploying the new model version (the "canary") alongside the current production model. A small percentage of live traffic (e.g., 5% of loan applications) is routed to the canary model. We then compare its performance (e.g., prediction distribution, latency, error rate) against the production model in real-time. If the canary performs as expected or better, we gradually increase the traffic it receives until it handles 100% and becomes the new production model. If it performs worse, we can immediately roll it back without affecting the majority of users.

### § What should be the strategy for ML Model Monitoring?
A robust monitoring strategy includes:
1.  **Data Drift Monitoring:** Track the statistical distribution of incoming data features and compare it to the training data's distribution. If the distributions diverge significantly (e.g., average income suddenly drops), the model's performance may degrade. This would trigger an alert to retrain the model.
2.  **Model Drift Monitoring:** Track the model's output (prediction distribution) over time. A sudden shift in the proportion of predicted defaults could indicate a problem.
3.  **Performance Monitoring:** Track the model's key metrics (like AUC-ROC or F1-score) on new, labeled data as it becomes available. A decline in performance is the most direct indicator that the model needs to be retrained.
4.  **Operational Monitoring:** Track system metrics like API latency, error rates, and CPU/memory usage to ensure the model is serving predictions efficiently.

### § How do you perform load and stress testing?
I would use a tool like **Locust** or **JMeter** to simulate a high volume of concurrent requests to the deployed model's API endpoint.
*   **Load Testing:** Start with an expected number of requests per second (e.g., the peak number of loan applications per hour) and gradually increase it to identify performance bottlenecks and ensure the system can handle the expected load while maintaining acceptable latency.
*   **Stress Testing:** Push the system beyond its expected capacity to find its breaking point. This helps us understand how the system fails and allows us to implement graceful degradation (e.g., returning a "system busy" message instead of crashing).

### § How do you track, monitor and audit ML training?
I would use a dedicated MLOps tool like **MLflow** or **Weights & Biases**. For each training run, I would automatically log:
*   **Code Version:** The Git commit hash of the training script.
*   **Data Version:** The version or hash of the dataset used for training.
*   **Parameters:** The hyperparameters used for the run.
*   **Metrics:** The resulting evaluation metrics (AUC-ROC, F1-score, etc.).
*   **Model Artifact:** The serialized, trained model file itself.
This creates a fully auditable and reproducible history of every model ever trained, allowing us to compare experiments, debug issues, and roll back to previous versions if needed.

### § Design framework for continuous delivery and automation of machine learning tasks.
This is an MLOps CI/CD pipeline. I would design it as follows:
1.  **CI (Continuous Integration):** When new code is pushed to the Git repository, a CI server (like Jenkins or GitHub Actions) automatically runs unit tests and code quality checks.
2.  **CT (Continuous Training):** If the CI pipeline passes, a CT pipeline is triggered. This pipeline automatically retrains the model on the latest version of the dataset. The newly trained model and its metrics are logged to our experiment tracking server (MLflow).
3.  **CD (Continuous Deployment):** If the new model's performance (from the CT step) is better than the currently deployed model, a CD pipeline is triggered. This pipeline would:
    *   Package the model into a Docker container.
    *   Deploy the container to a staging environment for final integration tests.
    *   If staging tests pass, automatically deploy the model to production using a canary release strategy.

This framework ensures that model updates are tested, validated, and deployed in a safe, automated, and repeatable manner.
