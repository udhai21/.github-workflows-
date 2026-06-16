
import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
from google.colab import userdata
import skops.io as sio # Import skops.io

# --- Configuration ---
# Hugging Face Token (from Colab secrets or environment variables)
hf_token = os.getenv("HF_TOKEN", userdata.get("HF_TOKEN"))

# Hugging Face Model Repository details
model_repo_id = "revathisrihari/tourism-model"
model_repo_type = "model"
model_file_name = "best_decision_tree_model.skops" # Change to .skops

# Hugging Face Dataset Repository details for preprocessor
dataset_repo_id = "revathisrihari/tourism-dataset"
dataset_repo_type = "dataset"
train_file = "tourism_train.csv"

@st.cache_resource
def load_model():
    """Downloads and loads the trained model from Hugging Face Hub."""
    print(f"Downloading model {model_file_name} from Hugging Face Hub (repo: {model_repo_id})...")
    model_path = hf_hub_download(
        repo_id=model_repo_id,
        repo_type=model_repo_type,
        filename=model_file_name,
        token=hf_token
    )
    print(f"Model downloaded to: {model_path}")
    model = sio.load(model_path) # Load using skops.io.load
    return model

@st.cache_resource
def get_preprocessor():
    """Downloads and loads training data to fit the preprocessor."""
    print(f"Downloading training data {train_file} from Hugging Face Hub (repo: {dataset_repo_id})...")
    local_train_path = hf_hub_download(
        repo_id=dataset_repo_id,
        repo_type=dataset_repo_type,
        filename=train_file,
        token=hf_token
    )
    train_df = pd.read_csv(local_train_path)

    # Re-create the preprocessor (OHE for categorical features)
    X_train = train_df.drop('ProdTaken', axis=1)
    categorical_features = X_train.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # Keep numerical features as they are
    )
    preprocessor.fit(X_train) # Fit on original training data
    return preprocessor

# Load model and preprocessor
model = load_model()
preprocessor = get_preprocessor()

st.title("Tourism Package Purchase Predictor")
st.write("Enter customer details to predict if they will purchase a tourism package.")

# --- Input Form ---
with st.form("prediction_form"):
    age = st.slider("Age", 18, 70, 30)
    typeofcontact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    citytier = st.selectbox("City Tier", [1, 2, 3])
    durationofpitch = st.slider("Duration of Pitch (minutes)", 1, 60, 10)
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female", "Fe Male"])
    numberofpersonvisiting = st.slider("Number of Persons Visiting", 1, 10, 2)
    numberoffollowups = st.slider("Number of Follow-ups", 0, 10, 3)
    productpitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    preferredpropertystar = st.slider("Preferred Property Star Rating", 1, 5, 3)
    maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    numberoftrips = st.slider("NumberOfTrips (yearly)", 1, 20, 5)
    passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    pitchsatisfactionscore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    owncar = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    numberofchildrenvisiting = st.slider("Number of Children Visiting", 0, 5, 1)
    designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP", "Director"])
    monthlyincome = st.slider("Monthly Income", 10000, 100000, 30000)

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame([{
            'Age': age,
            'TypeofContact': typeofcontact,
            'CityTier': citytier,
            'DurationOfPitch': durationofpitch,
            'Occupation': occupation,
            'Gender': gender,
            'NumberOfPersonVisiting': numberofpersonvisiting,
            'NumberOfFollowups': numberoffollowups,
            'ProductPitched': productpitched,
            'PreferredPropertyStar': preferredpropertystar,
            'MaritalStatus': maritalstatus,
            'NumberOfTrips': numberoftrips,
            'Passport': passport,
            'PitchSatisfactionScore': pitchsatisfactionscore,
            'OwnCar': owncar,
            'NumberOfChildrenVisiting': numberofchildrenvisiting,
            'Designation': designation,
            'MonthlyIncome': monthlyincome
        }])

        # Process input data using the loaded preprocessor
        processed_input_data = preprocessor.transform(input_data)

        # Convert to DataFrame with feature names (needed for prediction with some models)
        categorical_features_preprocessor = preprocessor.named_transformers_['cat'].get_feature_names_out()
        numerical_features_preprocessor = [col for col in input_data.columns if col not in preprocessor.named_transformers_['cat'].feature_names_in_]
        all_feature_names = list(categorical_features_preprocessor) + list(numerical_features_preprocessor)

        processed_input_df = pd.DataFrame(processed_input_data, columns=all_feature_names)

        prediction = model.predict(processed_input_df)
        prediction_proba = model.predict_proba(processed_input_df)[:, 1]

        if prediction[0] == 1:
            st.success(f"Prediction: Customer **will** purchase the package! (Probability: {prediction_proba[0]:.2f})")
        else:
            st.error(f"Prediction: Customer will **not** purchase the package. (Probability: {prediction_proba[0]:.2f})")
