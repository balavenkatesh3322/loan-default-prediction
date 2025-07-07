# System Architecture for Loan Default Prediction

## Deployment Process

This diagram illustrates the end-to-end architecture for deploying the loan default prediction model.

```mermaid
graph TD
    A[Data Sources] --> B{Data Ingestion};
    B --> C[Data Preprocessing & Feature Engineering];
    C --> D[Model Training & Tuning];
    D --> E[Model Registry];
    E --> F[Model Deployment];
    F --> G[API Endpoint];
    G --> H[Loan Application];

    subgraph "ML Pipeline"
        B;
        C;
        D;
    end

    subgraph "Serving Infrastructure"
        E;
        F;
        G;
    end

    subgraph "Monitoring"
        M[Model Monitoring] --> N[Alerting];
        F --> M;
    end
```

### Components:

*   **Data Sources:** The raw data for loan applications.
*   **Data Ingestion:** A process to collect and store the data.
*   **Data Preprocessing & Feature Engineering:** The data is cleaned, transformed, and new features are created.
*   **Model Training & Tuning:** The Gradient Boosting model is trained and tuned on the prepared data.
*   **Model Registry:** The trained model is versioned and stored in a central registry.
*   **Model Deployment:** The model is deployed as a REST API for real-time predictions.
*   **API Endpoint:** The deployed model is exposed through an API endpoint.
*   **Loan Application:** The loan application system interacts with the API to get predictions.
*   **Model Monitoring:** The model's performance is continuously monitored for drift and other issues.
*   **Alerting:** Alerts are triggered if the model's performance degrades.
