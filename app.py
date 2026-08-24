import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="centered"
)


# ============================================================
# CUSTOM CSS — colorful theme
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;800&display=swap');

    /* ---------- App background: dark IDE / terminal vibe ---------- */
    .stApp {
        background-color: #0d1117;
        background-image:
            linear-gradient(rgba(0, 255, 170, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 170, 0.05) 1px, transparent 1px),
            radial-gradient(circle at 15% 10%, rgba(0, 255, 170, 0.08), transparent 40%),
            radial-gradient(circle at 85% 90%, rgba(80, 130, 255, 0.10), transparent 45%);
        background-size: 42px 42px, 42px 42px, cover, cover;
        font-family: 'Fira Code', monospace;
    }

    /* ---------- Hero header card ---------- */
    .hero-card {
        background: #161b22;
        padding: 26px 30px;
        border-radius: 12px;
        margin-bottom: 28px;
        border: 1px solid #30363d;
        border-left: 4px solid #39ff88;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
        position: relative;
    }
    .hero-card::before {
        content: "● ● ●";
        display: block;
        color: #f85149;
        letter-spacing: 6px;
        font-size: 11px;
        margin-bottom: 14px;
        opacity: 0.7;
    }
    .hero-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 44px;
        font-weight: 800;
        color: #39ff88;
        margin: 0;
        text-shadow: 0 0 18px rgba(57, 255, 136, 0.35);
    }
    .hero-title::before {
        content: "> ";
        color: #58a6ff;
    }
    .hero-sub {
        font-family: 'Fira Code', monospace;
        font-size: 15px;
        color: #8b949e;
        margin-top: 10px;
    }

    /* ---------- Section headers ---------- */
    .section-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 18px;
        font-weight: 700;
        color: #58a6ff;
        background: #161b22;
        padding: 10px 16px;
        border: 1px solid #30363d;
        border-left: 4px solid #58a6ff;
        border-radius: 6px;
        margin: 20px 0 14px 0;
    }
    .section-header::before {
        content: "// ";
        color: #6e7681;
    }

    /* ---------- Input widget containers ---------- */
    div[data-testid="stNumberInput"], div[data-testid="stSelectbox"] {
        background: #161b22;
        border-radius: 8px;
        padding: 10px 14px 4px 14px;
        margin-bottom: 6px;
        border: 1px solid #30363d;
        transition: 0.2s ease-in-out;
    }
    div[data-testid="stNumberInput"]:hover, div[data-testid="stSelectbox"]:hover {
        border: 1px solid #39ff88;
        box-shadow: 0 0 12px rgba(57, 255, 136, 0.2);
    }
    div[data-testid="stNumberInput"] label, div[data-testid="stSelectbox"] label {
        font-family: 'Fira Code', monospace;
        color: #c9d1d9 !important;
    }

    /* ---------- Predict button ---------- */
    div.stButton > button {
        background: #161b22;
        color: #39ff88;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 17px;
        padding: 12px 0;
        border-radius: 8px;
        border: 1px solid #39ff88;
        box-shadow: 0 0 14px rgba(57, 255, 136, 0.15);
        transition: 0.15s ease-in-out;
    }
    div.stButton > button:hover {
        background: #39ff88;
        color: #0d1117;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(57, 255, 136, 0.45);
    }

    /* ---------- Result banners ---------- */
    .result-placed {
        background: #0d2818;
        padding: 18px 20px;
        border-radius: 10px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 19px;
        font-weight: 700;
        color: #39ff88;
        text-align: center;
        border: 1px solid #39ff88;
        box-shadow: 0 0 24px rgba(57, 255, 136, 0.25);
        margin-bottom: 10px;
    }
    .result-not-placed {
        background: #2a0d0d;
        padding: 18px 20px;
        border-radius: 10px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 19px;
        font-weight: 700;
        color: #ff5c5c;
        text-align: center;
        border: 1px solid #ff5c5c;
        box-shadow: 0 0 24px rgba(255, 92, 92, 0.25);
        margin-bottom: 10px;
    }

    /* ---------- Confidence metric card ---------- */
    .confidence-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        margin-bottom: 18px;
    }
    .confidence-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 36px;
        font-weight: 800;
        color: #58a6ff;
        text-shadow: 0 0 16px rgba(88, 166, 255, 0.35);
    }
    .confidence-label {
        font-family: 'Fira Code', monospace;
        font-size: 13px;
        color: #8b949e;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* ---------- Probability bars ---------- */
    .prob-row {
        margin-bottom: 12px;
    }
    .prob-label {
        font-family: 'Fira Code', monospace;
        display: flex;
        justify-content: space-between;
        font-size: 14px;
        color: #c9d1d9;
        margin-bottom: 4px;
        font-weight: 600;
    }
    .prob-track {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        height: 14px;
        overflow: hidden;
    }
    .prob-fill-not-placed {
        height: 100%;
        border-radius: 10px;
        background: #ff5c5c;
        box-shadow: 0 0 10px rgba(255, 92, 92, 0.5);
    }
    .prob-fill-placed {
        height: 100%;
        border-radius: 10px;
        background: #39ff88;
        box-shadow: 0 0 10px rgba(57, 255, 136, 0.5);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL, SCALER AND FEATURE NAMES
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load("logistic_regression_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")

    return model, scaler, feature_names


model, scaler, feature_names = load_model()


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero-card">
        <p class="hero-title">🎓 Student Placement Prediction</p>
        <p class="hero-sub">by Muawiyah &nbsp;•&nbsp; Enter the student's academic and skill
        information to predict the placement status.</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown('<div class="section-header">📋 Academic Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    cgpa = st.number_input(
        "🎯 CGPA",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.1
    )

    ssc_marks = st.number_input(
        "📘 10th Marks(%)",
        min_value=0,
        max_value=100,
        value=70,
        step=1
    )

with col2:
    aptitude_score = st.number_input(
        "🧠 Aptitude Test Score",
        min_value=0,
        max_value=100,
        value=70,
        step=1
    )

    hsc_marks = st.number_input(
        "📗  12th Marks(%)",
        min_value=0,
        max_value=100,
        value=70,
        step=1
    )


st.markdown('<div class="section-header">💼 Experience & Skills</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    internships = st.number_input(
        "🏢 Number of Internships",
        min_value=0,
        max_value=10,
        value=1,
        step=1
    )

    workshops = st.number_input(
        "📜 Workshops / Certifications",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

with col4:
    projects = st.number_input(
        "🛠️ Number of Projects",
        min_value=0,
        max_value=10,
        value=2,
        step=1
    )

    soft_skills = st.number_input(
        "🗣️ Soft Skills Rating",
        min_value=0.0,
        max_value=5.0,
        value=3.5,
        step=0.1
    )


st.markdown('<div class="section-header">✨ Other Details</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)

with col5:
    extracurricular = st.selectbox(
        "🏅 Extracurricular Activities",
        ["No", "Yes"]
    )

with col6:
    placement_training = st.selectbox(
        "🎯 Placement Training",
        ["No", "Yes"]
    )


st.write("")


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🔮 Predict Placement",
    use_container_width=True
):

    # --------------------------------------------------------
    # Convert categorical values to the same representation
    # used during model training
    # --------------------------------------------------------

    extracurricular_yes = (
        extracurricular == "Yes"
    )

    placement_training_yes = (
        placement_training == "Yes"
    )


    # --------------------------------------------------------
    # Create input dataframe
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "CGPA": [cgpa],

        "Internships": [internships],

        "Projects": [projects],

        "Workshops/Certifications": [workshops],

        "AptitudeTestScore": [aptitude_score],

        "SoftSkillsRating": [soft_skills],

        "SSC_Marks": [ssc_marks],

        "HSC_Marks": [hsc_marks],

        "ExtracurricularActivities_Yes":
            [extracurricular_yes],

        "PlacementTraining_Yes":
            [placement_training_yes]
    })


    # --------------------------------------------------------
    # Ensure EXACT feature order
    # --------------------------------------------------------

    input_data = input_data[
        feature_names
    ]


    # --------------------------------------------------------
    # Scale input using the SAME scaler used during training
    # --------------------------------------------------------

    input_scaled = scaler.transform(
        input_data
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        input_scaled
    )


    # --------------------------------------------------------
    # Prediction probability
    # --------------------------------------------------------

    probability = model.predict_proba(
        input_scaled
    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.markdown('<div class="section-header">📊 Prediction Result</div>', unsafe_allow_html=True)


    # Get predicted class
    predicted_class = prediction[0]


    # Get probability of predicted class
    predicted_probability = probability[
        0
    ].max()


    # --------------------------------------------------------
    # Display prediction
    # --------------------------------------------------------

    if predicted_class == 1:

        st.markdown(
            '<div class="result-placed">🎉 Student is predicted to be PLACED</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="result-not-placed">❌ Student is predicted to be NOT PLACED</div>',
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # Confidence card
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="confidence-card">
            <div class="confidence-label">Prediction Confidence</div>
            <div class="confidence-value">{predicted_probability * 100:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # Show probability for both classes as colorful bars
    # --------------------------------------------------------

    st.write("### 📈 Prediction Probability")

    probability_df = pd.DataFrame({

        "Status": model.classes_,

        "Probability": probability[0]

    })

    probability_df["Probability"] = (
        probability_df["Probability"] * 100
    )

    probability_df["Probability"] = (
        probability_df["Probability"].round(2)
    )

    status_labels = {0: "❌ Not Placed", 1: "✅ Placed"}
    fill_classes = {0: "prob-fill-not-placed", 1: "prob-fill-placed"}

    for _, row in probability_df.iterrows():
        status = int(row["Status"])
        pct = row["Probability"]
        label = status_labels.get(status, str(status))
        fill_class = fill_classes.get(status, "prob-fill-placed")

        st.markdown(
            f"""
            <div class="prob-row">
                <div class="prob-label">
                    <span>{label}</span>
                    <span>{pct:.2f}%</span>
                </div>
                <div class="prob-track">
                    <div class="{fill_class}" style="width:{pct}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )