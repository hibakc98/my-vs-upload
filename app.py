import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from datetime import date

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Accident Detection System",
    page_icon="🚦",
    layout="wide"
)

# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = Path(__file__).parent / "accident_detection.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# Get categories learned by the fitted OneHotEncoder
preprocessor = model.named_steps["preprocessing"]
categorical_pipeline = preprocessor.named_transformers_["cat"]
encoder = categorical_pipeline.named_steps["encoder"]

categorical_columns = [
    "Day_of_Week",
    "Time_of_Day",
    "Area_Type",
    "Road_Type",
    "Intersection_Type",
    "Weather",
    "Road_Condition",
    "Vehicle_Type"
]

category_options = dict(
    zip(categorical_columns, encoder.categories_)
)

# ==========================================
# 3. APPLICATION TITLE
# ==========================================

st.title("🚦 Traffic Accident Detection System")

st.write(
    "Enter the road, weather, traffic and driver details "
    "to predict whether an accident is likely."
)

st.divider()

# ==========================================
# 4. USER INPUT FORM
# ==========================================

st.subheader("📋 Enter Details")

with st.form("accident_prediction_form"):

    # Date and time
    st.markdown("### 📅 Date and Time")

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_date = st.date_input(
            "Date",
            value=date.today()
        )

    with col2:
        hour = st.number_input(
            "Hour (0–23)",
            min_value=0,
            max_value=23,
            value=10
        )

    with col3:
        time_of_day = st.selectbox(
            "Time of Day",
            options=category_options["Time_of_Day"].tolist()
        )

    is_holiday = st.checkbox("Is Holiday?")

    st.divider()

    # Road details
    st.markdown("### 🛣️ Road Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        area_type = st.selectbox(
            "Area Type",
            category_options["Area_Type"].tolist()
        )

        road_type = st.selectbox(
            "Road Type",
            category_options["Road_Type"].tolist()
        )

    with col2:
        intersection_type = st.selectbox(
            "Intersection Type",
            category_options["Intersection_Type"].tolist()
        )

        speed_limit = st.number_input(
            "Speed Limit (km/h)",
            min_value=0.0,
            value=50.0
        )

    with col3:
        weather = st.selectbox(
            "Weather",
            category_options["Weather"].tolist()
        )

        road_condition = st.selectbox(
            "Road Condition",
            category_options["Road_Condition"].tolist()
        )

    visibility = st.number_input(
        "Visibility (m)",
        min_value=0.0,
        value=1000.0
    )

    congestion = st.number_input(
        "Congestion Index",
        min_value=0.0,
        value=0.5
    )

    st.divider()

    # Vehicle details
    st.markdown("### 🚗 Vehicle Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        vehicle_type = st.selectbox(
            "Vehicle Type",
            category_options["Vehicle_Type"].tolist()
        )

        vehicle_age = st.number_input(
            "Vehicle Age (years)",
            min_value=0.0,
            value=5.0
        )

    with col2:
        recorded_speed = st.number_input(
            "Recorded Speed (km/h)",
            min_value=0.0,
            value=40.0
        )

    with col3:
        response_time = st.number_input(
            "Baseline Expected Response Time (min)",
            min_value=0.0,
            value=10.0
        )

    st.divider()

    # Driver details
    st.markdown("### 👤 Driver Details")

    col1, col2 = st.columns(2)

    with col1:
        driver_age = st.number_input(
            "Driver Age",
            min_value=18,
            max_value=100,
            value=30
        )

    with col2:
        driver_experience = st.number_input(
            "Driver Experience (years)",
            min_value=0.0,
            value=5.0
        )

    st.divider()

    # Prediction threshold
    threshold = st.slider(
        "Prediction Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05
    )

    submit = st.form_submit_button(
        "🔍 Predict Accident",
        type="primary",
        use_container_width=True
    )

# ==========================================
# 5. PREDICTION
# ==========================================

if submit:

    # Derive date-related features exactly as in training
    weekday_names = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    input_data = {
        "Hour": int(hour),
        "Month": selected_date.month,
        "Day_of_Week": weekday_names[selected_date.weekday()],
        "Time_of_Day": time_of_day,
        "Is_Holiday": int(is_holiday),
        "Area_Type": area_type,
        "Road_Type": road_type,
        "Intersection_Type": intersection_type,
        "Speed_Limit_kmh": speed_limit,
        "Weather": weather,
        "Road_Condition": road_condition,
        "Visibility_m": visibility,
        "Congestion_Index": congestion,
        "Vehicle_Type": vehicle_type,
        "Driver_Age": driver_age,
        "Driver_Experience_yrs": driver_experience,
        "Vehicle_Age_yrs": vehicle_age,
        "Recorded_Speed_kmh": recorded_speed,
        "Baseline_Expected_Response_Time_min": response_time,
        "Year": selected_date.year,
        "Day": selected_date.day,
        "DayOfWeek": selected_date.weekday()
    }

    input_df = pd.DataFrame([input_data])

    # Make prediction
    try:
        probabilities = model.predict_proba(input_df)[0]

        class_1_index = list(model.classes_).index(1)
        accident_score = probabilities[class_1_index]

        prediction = int(accident_score >= threshold)

        st.divider()
        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Accident Score",
                f"{accident_score * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Selected Threshold",
                f"{threshold * 100:.0f}%"
            )

        if prediction == 1:
            st.error(
                "⚠️ Model Prediction: Accident"
            )
            st.write(
                "The model predicts the accident class."
            )
        else:
            st.success(
                "✅ Model Prediction: No Accident"
            )
            st.write(
                "The model predicts the non-accident class."
            )

        with st.expander("View Input Details"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ==========================================
# 6. FOOTER
# ==========================================

st.divider()

st.caption(
    "Accident Detection System | Machine Learning Project"
)

st.warning(
    "This is an educational prediction system trained on synthetic data. "
    "Its test precision for accident predictions was approximately 8.4%. "
    "It should not be used for real-world emergency or road-safety decisions."
)