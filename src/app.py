import streamlit as st
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import os

# Configure page
st.set_page_config(page_title="MLOps Model Dashboard", layout="wide")

st.title("🎯 MLOps Pipeline Dashboard")
st.markdown("Monitor and interact with the automated ML pipeline results.")

# Connect to MLflow
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient()

experiment_name = "Travel_Product_Taken_Prediction"

st.header("📊 Experiment Tracking")
try:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment:
        runs = client.search_runs(experiment_ids=[experiment.experiment_id], 
                                  order_by=["metrics.accuracy DESC"])
        if runs:
            st.success(f"Connected to MLflow. Found {len(runs)} runs in '{experiment_name}'.")
            
            # Extract run details
            run_data = []
            for run in runs:
                data = {
                    "Run ID": run.info.run_id,
                    "Model": run.data.params.get("model_name", "Unknown"),
                    "Accuracy": round(run.data.metrics.get("accuracy", 0.0), 4),
                    "F1 Score": round(run.data.metrics.get("f1_score", 0.0), 4),
                    "Status": run.info.status
                }
                run_data.append(data)
                
            df_runs = pd.DataFrame(run_data)
            
            # Display best model dynamically
            best_run = df_runs.iloc[0]
            st.metric(label=f"🥇 Best Model: {best_run['Model']}", 
                      value=f"{best_run['Accuracy'] * 100:.2f}% Accuracy", 
                      delta="F1: " + str(best_run['F1 Score']))

            # Display all runs
            st.subheader("All Training Runs")
            st.dataframe(df_runs.style.highlight_max(axis=0, subset=['Accuracy', 'F1 Score']), use_container_width=True)

            # --- Prediction UI Section ---
            st.markdown("---")
            st.header("🔮 Make a Prediction")
            st.markdown(f"Using the best automatically loaded model: **{best_run['Model']}**")
            
            with st.form("prediction_form"):
                st.write("Enter Customer Details:")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    age = st.number_input("Age", min_value=18, max_value=100, value=35)
                    city_tier = st.selectbox("City Tier", [1, 2, 3])
                    occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
                    gender = st.selectbox("Gender", ["Male", "Female"])
                    duration_pitch = st.number_input("Duration Of Pitch", min_value=1, max_value=100, value=10)
                    monthly_income = st.number_input("Monthly Income", min_value=0, value=20000)
                
                with col2:
                    typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
                    num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
                    num_followups = st.number_input("Number of Followups", min_value=1, max_value=10, value=3)
                    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
                    property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
                
                with col3:
                    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
                    num_trips = st.number_input("Number of Trips", min_value=1, max_value=20, value=2)
                    passport = st.selectbox("Passport", [0, 1])
                    pitch_sat_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
                    own_car = st.selectbox("Own Car", [0, 1])
                    num_children = st.number_input("Number Of Children Visiting", min_value=0, max_value=10, value=0)
                    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

                submitted = st.form_submit_button("Predict")

            if submitted:
                # Construct dataframe for prediction
                input_data = pd.DataFrame([{
                    "Age": age,
                    "TypeofContact": typeof_contact,
                    "CityTier": city_tier,
                    "DurationOfPitch": duration_pitch,
                    "Occupation": occupation,
                    "Gender": gender,
                    "NumberOfPersonVisiting": num_person_visiting,
                    "NumberOfFollowups": num_followups,
                    "ProductPitched": product_pitched,
                    "PreferredPropertyStar": property_star,
                    "MaritalStatus": marital_status,
                    "NumberOfTrips": num_trips,
                    "Passport": passport,
                    "PitchSatisfactionScore": pitch_sat_score,
                    "OwnCar": own_car,
                    "NumberOfChildrenVisiting": num_children,
                    "Designation": designation,
                    "MonthlyIncome": monthly_income
                }])
                
                with st.spinner("Downloading model from AWS S3 and predicting..."):
                    try:
                        model_uri = f"runs:/{best_run['Run ID']}/model"
                        # This automatically uses boto3 to download from your predict-1 S3 bucket 
                        loaded_model = mlflow.sklearn.load_model(model_uri)
                        
                        prediction = loaded_model.predict(input_data)[0]
                        
                        try:
                            prediction_prob = loaded_model.predict_proba(input_data)[0][1]
                        except:
                            prediction_prob = None
                        
                        st.markdown("### Prediction Result")
                        if prediction == 1:
                            st.success(f"🎊 **Will Purchase Product** (Probability: {prediction_prob:.2%} if available)")
                        else:
                            st.error(f"🛑 **Will NOT Purchase Product** (Probability of purchasing: {prediction_prob:.2%} if available)")
                    except Exception as e:
                        st.error(f"Error executing prediction: {e}")

        else:
            st.warning("No runs found in the experiment yet. Training might be in progress.")
    else:
        st.warning(f"Experiment '{experiment_name}' not found. Have you executed training yet?")

except Exception as e:
    st.error(f"Error connecting to MLflow: {e}")

st.markdown("---")
st.markdown("""
### Architecture Overview
1. **GitHub Actions**: Auto-triggers upon code commits.
2. **AWS EC2 (Docker Compose)**: Orchestrates MLflow Server, Training Script, and this UI.
3. **AWS S3**: Models and artifacts are pushed to the `predict-1` bucket natively.
4. **MLflow**: Tracks hyperparameter values, metrics, and serves models on-the-fly.
""")
