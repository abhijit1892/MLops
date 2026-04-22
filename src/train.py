import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import os

# Set MLflow tracking URI to a local SQLite database or remote server
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
mlflow.set_experiment("Travel_Product_Taken_Prediction")

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    # Separate features and target
    X = df.drop(['CustomerID', 'ProdTaken'], axis=1) # CustomerID is not a feature
    y = df['ProdTaken']

    # Define categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # Preprocessing pipelines for both numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    return X, y, preprocessor

def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

def train_and_log_model(X_train, y_train, X_test, y_test, preprocessor, model, model_name):
    # Create a full pipeline with preprocessor and model
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])

    with mlflow.start_run(run_name=model_name):
        # Fit the pipeline
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)

        # Evaluate
        accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)

        # Log parameters
        mlflow.log_param("model_name", model_name)
        if hasattr(model, 'get_params'):
            params = model.get_params()
            # Only log basic params to avoid overwhelming MLflow
            for key in ['n_estimators', 'max_depth', 'learning_rate', 'random_state']:
                if key in params:
                    mlflow.log_param(key, params[key])

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log model components
        # Note: We log the whole pipeline to ensure preprocessing is included in inference
        mlflow.sklearn.log_model(clf, "model")

        print(f"[{model_name}] Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    print("Loading Data from S3 Bucket (predict-1)...")
    # Stream directly from S3!
    df = load_data("s3://predict-1/raw-data/Travel.csv")
    
    print("Preprocessing...")
    X, y, preprocessor = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    print("Training Random Forest...")
    train_and_log_model(X_train, y_train, X_test, y_test, preprocessor, rf_model, "RandomForest")

    # 2. XGBoost
    xgb_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
    print("Training XGBoost...")
    train_and_log_model(X_train, y_train, X_test, y_test, preprocessor, xgb_model, "XGBoost")

    print("Training Complete. Check MLflow UI for results.")
