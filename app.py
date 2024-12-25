import streamlit as st
import torch
import torch.nn as nn
import base64
from PIL import Image
import io

# Must be the first Streamlit command
st.set_page_config(
    page_title="Breast Cancer Survival Prediction",
    page_icon="🎗️",
    layout="wide"
)

# Define the model architecture
class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 20px;
        color: #000000;
    }
    .stTextInput, .stNumberInput, .stSelectbox {
        background-color: #FF69B4;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    .stButton button {
        background-color: #FF69B4;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        width: 100%;
    }
    .stSuccess {
        padding: 20px;
        border-radius: 5px;
        background-color: rgba(255, 255, 255, 0.9);
    }
    
    </style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    model = PyTorchModel()
    model.load_state_dict(torch.load("model_state.pth"))
    model.eval()
    return model

try:
    model = load_model()
except:
    st.error("Error loading model. Please ensure model file exists.")
    st.stop()

# Title and description with custom styling
st.markdown("""
    <h1 style='text-align: center; color: #FF69B4; font-size: 3.2em;'>Breast Cancer Survival Prediction</h1>
    <p style='text-align: center; font-size: 1.2em; color: #666;'>
        This tool predicts breast cancer survival likelihood based on patient data.
        Please fill in all required information carefully.
    </p>
""", unsafe_allow_html=True)

# Create two columns for better layout
col1, col2 = st.columns(2)

validation_ranges = {
    "Age": (0, 120, 30),  # min, max, default
    "Protein1": (0, 100, 0),
    "Protein2": (0, 100, 0),
    "Protein3": (0, 100, 0),
    "Protein4": (0, 100, 0),
    "Tumour_Stage": (1, 3, 1),
    "Histology": (1, 5, 1),
    "Surgery_type": (0, 1, 0),
    "Gender_MALE": (0, 1, 0),
    "HER2 status_Positive": (0, 1, 0)
}

# Initialize session state for inputs with proper default values
if 'inputs' not in st.session_state:
    st.session_state.inputs = {k: validation_ranges[k][2] for k in validation_ranges.keys()}


# Input fields with validation
inputs = {}
with col1:
    st.markdown("### Patient Information")
    for feature in ["Age", "Gender_MALE", "Tumour_Stage", "Histology", "Surgery_type"]:
        min_val, max_val, default_val = validation_ranges[feature]
        help_text = f"Valid range: {min_val} to {max_val}"
        
        if feature == "Surgery_type":
            value = st.selectbox(
                f"{feature}",
                options=[0, 1],
                format_func=lambda x: "Lumpectomy" if x == 0 else "Mastectomy",
                help=help_text
            )
        elif feature == "Gender_MALE":
            value = st.selectbox(
                f"{feature}",
                options=[0, 1],
                format_func=lambda x: "Female" if x == 0 else "Male",
                help=help_text
            )
        else:
            value = st.number_input(
                feature,
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(st.session_state.inputs[feature]),
                help=help_text
            )
        inputs[feature] = value

with col2:
    st.markdown("### Clinical Markers")
    for feature in ["Protein1", "Protein2", "Protein3", "Protein4", "HER2 status_Positive"]:
        min_val, max_val, default_val = validation_ranges[feature]
        help_text = f"Valid range: {min_val} to {max_val}"
        
        if feature == "HER2 status_Positive":
            value = st.selectbox(
                f"{feature}",
                options=[0, 1],
                format_func=lambda x: "Negative" if x == 0 else "Positive",
                help=help_text
            )
        else:
            value = st.number_input(
                feature,
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(st.session_state.inputs[feature]),
                help=help_text
            )
        inputs[feature] = value

# Update session state
st.session_state.inputs = inputs

# Centered predict button
col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict_button = st.button("Predict Survival Likelihood", use_container_width=True)

# Prediction logic
if predict_button:
    try:
        # Convert inputs to tensor
        features = torch.tensor([[inputs[k] for k in validation_ranges.keys()]], dtype=torch.float32)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(features).item()
        
        # Display prediction with custom styling
        st.markdown("""
            <div style='background-color: rgba(255, 255, 255, 0.9); 
                        padding: 20px; 
                        border-radius: 10px; 
                        text-align: center; 
                        margin-top: 20px;'>
                <h2 style='color: #FF69B4;'>Prediction Results</h2>
                <p style='font-size: 1.2em;'>
                    Predicted survival score: {:.2f}
                </p>
                <p style='font-size: 1.2em; color: #666;'>
                    Note: This prediction is for informational purposes only and should not replace professional medical advice.
                </p>
            </div>
        """.format(prediction), unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Footer
st.markdown("""
    <div style='text-align: center; margin-top: 30px; padding: 20px; 
                background-color: rgba(255, 255, 255, 0.9); 
                border-radius: 10px;'>
        <p style='color: #666; font-size: 0.8em;'>
            This tool is for research purposes only. 
            Always consult with healthcare professionals for medical decisions.
        </p>
    </div>
""", unsafe_allow_html=True)