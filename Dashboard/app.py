import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Configure dashboard
st.set_page_config(
    page_title="DeepKey",
    page_icon="🔐",
    layout="wide"
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "deepkey_mlp_model.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "deepkey_scaler.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "fixed_text_cleaned.csv"


# Load saved model and scaler
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


# Load dataset
@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


model, scaler = load_model()
df = load_dataset()

# Identify the 46 timing features in the same order used during training
feature_columns = list(scaler.feature_names_in_)

# Dashboard heading
st.title("🔐 DeepKey")
st.subheader("Keystroke Behavioral Biometrics")

st.write(
    "A student prototype that identifies participants "
    "using keystroke timing patterns."
)

# Model information
col1, col2, col3 = st.columns(3)

col1.metric("Model", "MLP Neural Network")
col2.metric("Input Features", len(feature_columns))
col3.metric("Participants", df["participant"].nunique())

st.divider()

# Automatic prediction
st.header("Participant Identification")

st.write(
    "Select a dataset record. DeepKey will automatically "
    "load its timing features and predict the participant."
)

# Select a participant
participants = sorted(df["participant"].unique())

selected_participant = st.selectbox(
    "Select participant",
    participants
)

# Show records for the selected participant
participant_records = df[
    df["participant"] == selected_participant
].copy()

# Select a repetition
selected_repetition = st.selectbox(
    "Select repetition",
    sorted(participant_records["repetition"].unique())
)

# Select a session
available_sessions = sorted(
    participant_records.loc[
        participant_records["repetition"] == selected_repetition,
        "session"
    ].unique()
)

selected_session = st.selectbox(
    "Select session",
    available_sessions
)

# Predict when the button is clicked
if st.button("Predict Participant"):

    selected_record = participant_records[
        (participant_records["repetition"] == selected_repetition)
        & (participant_records["session"] == selected_session)
    ]

    if selected_record.empty:
        st.error("No matching record was found.")
    else:
        # Extract timing features in the original training order
        input_data = selected_record[feature_columns]

        # Apply the saved scaler
        scaled_input = scaler.transform(input_data)

        # Predict participant
        prediction = model.predict(scaled_input)[0]

        actual_participant = selected_record.iloc[0]["participant"]

        # Display results
        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        col1.metric("Actual Participant", actual_participant)
        col2.metric("Predicted Participant", prediction)

        if prediction == actual_participant:
            st.success("Prediction matches the dataset label.")
        else:
            st.error("Prediction does not match the dataset label.")

        # Show timing values used for prediction
        with st.expander("View the 46 timing features"):
            st.dataframe(input_data.T.rename(columns={input_data.index[0]: "Value"}))


st.divider()

st.caption(
    "DeepKey student prototype | Dataset-based demonstration. "
    "This does not independently verify a live user's identity."
)