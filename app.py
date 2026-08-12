import os
import pickle
import numpy as np
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Course Marks Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #555555;
        margin-bottom: 2rem;
    }
    .stMetric {
        background-color: #F0F4F8;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load Model Function with Caching
@st.cache_resource
def load_model(file_path: str):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as file:
        model = pickle.load(file)
    return model


def main():
    # Header Section
    st.markdown(
        "<h1 class='main-header'>🎓 Student Performance Predictor</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='sub-header'>Predict marks based on study time and course load using KNN Regression</p>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Fixed filename to match model.pkl
    model_filename = "model.pkl"
    model = load_model(model_filename)

    if model is None:
        st.error(
            f"❌ Model file `{model_filename}` not found in the current directory. "
            "Please place `model.pkl` in the same directory as `app.py`."
        )
        st.stop()

    # Sidebar Information
    with st.sidebar:
        st.header("⚙️ Model Details")
        st.info(
            f"""
            - **Algorithm**: KNN Regressor
            - **Neighbors (K)**: `{getattr(model, 'n_neighbors', 'N/A')}`
            - **Features Required**:
              1. Number of Courses (`number_courses`)
              2. Time Spent Studying (`time_study`)
            """
        )
        st.markdown("---")
        st.caption("Enter parameters in the form to generate a score prediction.")

    # Input Form Layout
    st.subheader("📋 Enter Input Details")

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        number_courses = st.number_input(
            "Number of Courses",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Select the number of courses taken.",
        )

    with col2:
        time_study = st.number_input(
            "Study Time (Hours per day)",
            min_value=0.0,
            max_value=24.0,
            value=4.5,
            step=0.5,
            help="Enter daily study time in hours.",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Predict Button
    if st.button("🔮 Predict Score", use_container_width=True, type="primary"):
        # Match order expected by model: [number_courses, time_study]
        input_data = np.array([[number_courses, time_study]])

        try:
            prediction = model.predict(input_data)[0]

            # Display Results
            st.divider()
            st.success("✅ Prediction Generated Successfully!")

            res_col1, res_col2 = st.columns([1, 1], gap="medium")

            with res_col1:
                st.metric(
                    label="Predicted Score / Marks",
                    value=f"{prediction:.2f}",
                )

            with res_col2:
                st.write("**Summary of Inputs:**")
                st.write(f"- **Courses Enrolled:** {number_courses}")
                st.write(f"- **Study Hours / Day:** {time_study} hrs")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")


if __name__ == "__main__":
    main()
