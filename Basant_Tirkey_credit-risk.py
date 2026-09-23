import os
import sys
import pickle
import pandas as pd
import numpy as np

from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from logger import get_logger
from exception import CustomException

logger = get_logger(__name__)
app = Flask(__name__)

ARTIFACTS_DIR = os.path.join(os.getcwd(), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.pkl")
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
RAW_DATA_PATH = os.path.join(ARTIFACTS_DIR, "raw_data.csv")


# ==========================================
# 1. DATA INGESTION MODULE
# ==========================================
class DataIngestion:
    def __init__(self):
        self.raw_data_path = RAW_DATA_PATH

    def initiate_data_ingestion(self, dataset_path: str):
        logger.info("Entered data ingestion method")
        try:
            df = pd.read_csv(dataset_path)
            logger.info("Successfully read dataset into pandas DataFrame")

            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
            df.to_csv(self.raw_data_path, index=False, header=True)
            logger.info(f"Saved raw data to {self.raw_data_path}")

            return self.raw_data_path
        except Exception as e:
            logger.error("Error occurred in data ingestion module")
            raise CustomException(e, sys)


# ==========================================
# 2 & 3. DATA PREPROCESSING & SMOTE MODULE
# ==========================================
class DataTransformation:
    def __init__(self):
        self.preprocessor_obj_file_path = PREPROCESSOR_PATH

    def get_data_transformer_object(self, numerical_columns, categorical_columns):
        try:
            logger.info("Building preprocessing pipeline...")
            num_pipeline = StandardScaler()
            cat_pipeline = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, raw_data_path):
        try:
            df = pd.read_csv(raw_data_path)
            logger.info("Read raw data successfully for preprocessing.")

            # Cleaning steps (filtering anomalies and handling missing values)
            df = df[df['person_age'] < 100]
            df = df[df['person_emp_length'] < 60]
            df.fillna(df.median(numeric_only=True), inplace=True)
            logger.info("Handled missing values and filtered anomalies.")

            target_column_name = "loan_status"
            X = df.drop(columns=[target_column_name])
            y = df[target_column_name]

            # Explicit lists prevent pandas/numpy dtype parsing issues entirely
            numerical_columns = [
                "person_age", "person_income", "person_emp_length", 
                "loan_amnt", "loan_int_rate", "loan_percent_income", 
                "cb_person_cred_hist_length"
            ]
            categorical_columns = [
                "person_home_ownership", "loan_intent", 
                "loan_grade", "cb_person_default_on_file"
            ]

            # Train-Test Split (Stratified)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            logger.info("Performed stratified train-test split.")

            # Preprocessing fit_transform
            preprocessor = self.get_data_transformer_object(numerical_columns, categorical_columns)
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            # Apply SMOTE strictly on training data
            logger.info("Applying SMOTE to handle class imbalance on training partition...")
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_arr, y_train)

            # Save preprocessor object
            os.makedirs(os.path.dirname(self.preprocessor_obj_file_path), exist_ok=True)
            with open(self.preprocessor_obj_file_path, "wb") as file_obj:
                pickle.dump(preprocessor, file_obj)
            logger.info("Preprocessor object serialized and saved successfully.")

            return X_train_resampled, y_train_resampled, X_test_arr, y_test, self.preprocessor_obj_file_path

        except Exception as e:
            logger.error("Error in data transformation pipeline.")
            raise CustomException(e, sys)

# ==========================================
# 4, 5 & 6. MODEL TRAINING, EVALUATION & SAVING
# ==========================================
class ModelTrainer:
    def __init__(self):
        self.model_file_path = MODEL_PATH

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logger.info("Initializing XGBoost classifier for hyperparameter tuning...")
            model = XGBClassifier(random_state=42, eval_metric="logloss")

            param_distributions = {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 1.0]
            }

            logger.info("Starting RandomizedSearchCV...")
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distributions,
                n_iter=5,
                scoring="roc_auc",
                cv=3,
                verbose=1,
                random_state=42
            )
            search.fit(X_train, y_train)

            best_model = search.best_estimator_
            logger.info(f"Best hyperparameters found: {search.best_params_}")

            # Model Evaluation
            predictions = best_model.predict(X_test)
            probabilities = best_model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            roc_auc = roc_auc_score(y_test, probabilities)

            logger.info(f"Model Evaluation Metrics -> Accuracy: {acc:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

            # Save Model Artifact
            os.makedirs(os.path.dirname(self.model_file_path), exist_ok=True)
            with open(self.model_file_path, "wb") as f:
                pickle.dump(best_model, f)
            logger.info("Best model artifact saved successfully.")

            return roc_auc

        except Exception as e:
            logger.error("Error occurred during model training and evaluation.")
            raise CustomException(e, sys)


# ==========================================
# AUTOMATED PIPELINE EXECUTION ON STARTUP
# ==========================================
def run_training_pipeline():
    try:
        dataset_filename = r"E:\credit-risk-prediction\dataset\credit_risk_dataset.csv" 
        if os.path.exists(dataset_filename) and not (os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH)):
            logger.info("Training artifacts missing. Starting automated training pipeline...")
            
            # 1. Ingestion
            ingestion = DataIngestion()
            raw_data = ingestion.initiate_data_ingestion(dataset_filename)
            
            # 2. Transformation
            transformation = DataTransformation()
            X_train_res, y_train_res, X_test, y_test, _ = transformation.initiate_data_transformation(raw_data)
            
            # 3. Training & Evaluation
            trainer = ModelTrainer()
            trainer.initiate_model_trainer(X_train_res, y_train_res, X_test, y_test)
            logger.info("Automated training pipeline completed successfully.")
    except Exception as e:
        logger.error("Pipeline execution failed during startup.")
        raise CustomException(e, sys)


# Run pipeline check before booting up Flask
run_training_pipeline()


# ==========================================
# 7. FLASK WEB APP & ENDPOINTS
# ==========================================
try:
    logger.info("Loading serialized artifacts for inference...")
    with open(MODEL_PATH, "rb") as mf, open(PREPROCESSOR_PATH, "rb") as pf:
        model = pickle.load(mf)
        preprocessor = pickle.load(pf)
    logger.info("Artifacts successfully loaded.")
except Exception as e:
    logger.error("Failed to load artifacts on startup. Ensure the training pipeline has run.")
    model = None
    preprocessor = None


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.method == "POST":
            logger.info("Received prediction request from UI form.")
            
            if model is None or preprocessor is None:
                raise Exception("Model or preprocessor artifacts are not loaded.")

            # Map input parameters from request form
            input_data = pd.DataFrame({
                "person_age": [float(request.form.get("person_age"))],
                "person_income": [float(request.form.get("person_income"))],
                "person_home_ownership": [request.form.get("person_home_ownership")],
                "person_emp_length": [float(request.form.get("person_emp_length"))],
                "loan_intent": [request.form.get("loan_intent")],
                "loan_grade": [request.form.get("loan_grade")],
                "loan_amnt": [float(request.form.get("loan_amnt"))],
                "loan_int_rate": [float(request.form.get("loan_int_rate"))],
                "loan_percent_income": [float(request.form.get("loan_percent_income"))],
                "cb_person_default_on_file": [request.form.get("cb_person_default_on_file")],
                "cb_person_cred_hist_length": [float(request.form.get("cb_person_cred_hist_length"))]
            })

            transformed_data = preprocessor.transform(input_data)
            prediction = model.predict(transformed_data)
            proba = model.predict_proba(transformed_data)[0][1]

            result = "High Risk (Default Expected)" if prediction[0] == 1 else "Low Risk (Non-Default)"
            logger.info(f"Inference result: {result} with probability {proba:.2f}")

            return render_template(
                "index.html",
                prediction_text=f"Assessment: {result}",
                probability=f"Default Probability: {proba:.2%}"
            )
    except Exception as e:
        logger.error("Error during prediction endpoint execution.")
        raise CustomException(e, sys)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)