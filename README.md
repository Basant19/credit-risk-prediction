Credit Risk Assessment Engine
An end-to-end, production-grade machine learning application designed to evaluate borrower default risk in real time. The platform features an automated startup pipeline that handles data ingestion, structural cleaning, stratified train-test splitting, class imbalance correction via SMOTE, and hyperparameter tuning, paired with an interactive Flask web interface for real-time inference.

🚀 Project Overview & Features
Automated Boot Pipeline: Automatically detects missing local artifacts on startup, triggers data ingestion, runs preprocessing, trains the model, and serializes outputs.

Robust Data Transformation: Filters out data anomalies (e.g., unrealistic ages or employment lengths), handles missing values with median imputation, and applies scikit-learn's ColumnTransformer (Standard Scaling for numerical attributes and One-Hot Encoding for categorical fields).

Class Imbalance Mitigation: Integrates SMOTE (Synthetic Minority Over-sampling Technique) strictly on the training partition to prevent data leakage and balance default classes.

Optimized XGBoost Classifier: Employs gradient boosting fine-tuned via RandomizedSearchCV optimizing for ROC-AUC.

Flask Inference Web App: Provides a clean user interface to input borrower parameters and instantly receive default risk assessments and precise default probabilities.

📂 Project Architecture & File Structure
Plaintext
E:\credit-risk-prediction\
│
├── artifacts/                  # Generated local persistence storage
│   ├── raw_data.csv            # Standardized raw dataset copy
│   ├── best_model.pkl          # Serialized XGBoost best estimator
│   └── preprocessor.pkl        # Serialized scikit-learn ColumnTransformer
│
├── dataset/                    # Source input repository
│   └── credit_risk_dataset.csv # Original credit risk raw data
│
├── templates/                  # Frontend views
│   └── index.html              # User interface form for parameter input
│
├── venv/                       # Isolated Python virtual environment
├── app.py                      # Core monolith script: Pipeline, Training, and Flask Server
├── logger.py                   # Centralized logging configuration module
└── exception.py                # Custom exception handling and error traceback wrapper
Core File Breakdown:
app.py: The central application script. It orchestrates ingestion, transformation, model training classes, and startup automation checks (run_training_pipeline()). It also hosts the Flask web server routes (/ for home UI and /predict for real-time scoring).

logger.py: Configures centralized pipeline logging to record milestones, model performance metrics, warning alerts, and execution errors with precise timestamps.

exception.py: Custom exception wrapper that intercepts standard runtime errors, extracts exact script and line numbers, and standardizes error tracing.

artifacts/: Local model registry containing serialized objects (best_model.pkl, preprocessor.pkl) loaded instantly by the inference server upon startup.

🛠️ Technical Stack & Dependencies
Language: Python 3.11+

Web Framework: Flask

Machine Learning: Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE)

Data Manipulation: Pandas, NumPy

Serialization: Pickle

📊 Model Performance & KPIs
Evaluated on a held-out stratified test dataset, the optimized pipeline achieves:

ROC-AUC Score: 0.9427 — Exceptional capability in distinguishing defaulting versus non-defaulting borrowers.

Model Accuracy: 93.94% — High overall correctness across risk categories.

F1-Score: 0.8405 — Strong harmonic balance between precision and recall for minority default cases.

⚙️ Installation & Setup Instructions
Clone or Navigate to the Project Directory:

Bash
cd E:\credit-risk-prediction
Activate the Virtual Environment:

Bash
venv\Scripts\activate
Install Required Packages:

Bash
pip install flask scikit-learn xgboost imbalanced-learn pandas numpy
Run the Application:

Bash
python app.py
(On initial boot, the application will automatically build the artifacts, train the model, and launch the development server).

Access the Web Interface:
Open your browser and navigate to:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

📝 Example Test Profiles
1. Expected Low Risk (Prime Borrower)
Person Age: 32

Annual Income ($): 85000

Employment Length (yrs): 6.0

Home Ownership: MORTGAGE

Loan Intent: VENTURE

Loan Grade: A

Loan Amount ($): 10000

Interest Rate (%): 7.5

Percent Income: 0.12

Default on File: N (No)

Credit History Length (yrs): 8.0

2. Expected High Risk (Default Expected)
Person Age: 22

Annual Income ($): 24000

Employment Length (yrs): 1.0

Home Ownership: RENT

Loan Intent: DEBTCONSOLIDATION

Loan Grade: D

Loan Amount ($): 15000

Interest Rate (%): 15.5

Percent Income: 0.62

Default on File: Y (Yes)

Credit History Length (yrs): 2.0