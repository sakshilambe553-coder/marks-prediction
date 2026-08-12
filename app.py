import streamlit as st
import pickle
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.25rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        text-align: center;
        color: #555555;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        border-radius: 8px;
        height: 3em;
        font-size: 1rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Title Header
st.markdown('<div class="main-header">🎓 Student Performance Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Estimate student performance score based on course load and study hours.</div>', unsafe_allow_html=True)

# Model Loader
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("model.pkl not found in the current directory. Please place your model file in the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar - Model Info
with st.sidebar:
    st.header("📌 Model Information")
    st.write("*Model Type:* KNeighborsRegressor")
    st.write("*Expected Features:*")
    st.markdown("- number_courses\n- time_study")
    st.divider()
    st.caption("Deployment powered by Streamlit")

# Main Input Section
st.subheader("⚙️ Input Parameters")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        number_courses = st.number_input(
            "Number of Courses",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
            help="Select the total number of enrolled courses."
        )
        
    with col2:
        time_study = st.number_input(
            "Daily Study Time (Hours)",
            min_value=0.0,
            max_value=24.0,
            value=4.5,
            step=0.5,
            help="Enter average daily study time in hours."
        )
        
    submit_button = st.form_submit_button("Predict Performance")

# Prediction Execution
if submit_button:
    # Construct input dataframe with precise feature names matching the model
    input_data = pd.DataFrame({
        'number_courses': [number_courses],
        'time_study': [time_study]
    })
    
    try:
        prediction = model.predict(input_data)[0]
        
        st.divider()
        st.subheader("🎯 Prediction Result")
        
        # Display results in metric card
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric(label="Predicted Score", value=f"{prediction:.2f}")
        with res_col2:
            st.success("Prediction generated successfully!")
            st.caption("The score is calculated using Nearest Neighbors regression based on your inputs.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
