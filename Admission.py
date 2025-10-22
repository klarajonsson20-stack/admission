# Import necessary libraries
import streamlit as st
import pickle
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

password_guess = st.text_input("What is the Password?")
if password_guess != st.secrets["password"]:
    st.stop()

st.title("Graduate Admission Prediction🌟")
st.write("This app uses multiple inputs to predict the probability of a student getting admitted to a graduate school.")
# Display an image of admission
st.image('admission.jpg', width = 400)

# Create a sidebar for input collection
st.sidebar.header("Enter your profile details")
GRE_score = st.sidebar.number_input("GRE Score", min_value=0, max_value=340, value=300, step=1)
TOEFL_score = st.sidebar.number_input("TOEFL Score", min_value=0, max_value=120, value=100)
CGPA_score = st.sidebar.number_input("CGPA Score", min_value=0.0, max_value=10.0, value=8.0)
research_experience = st.sidebar.selectbox("Research Experience", options=["Yes", "No"])  

university_rating = st.sidebar.slider("University Rating", min_value=1, max_value=5, value=3)
statement_of_purpose = st.sidebar.slider("Statement of Purpose Strength", min_value=1, max_value=5, value=3)
letter_of_recommendation = st.sidebar.slider("Letter of Recommendation (LOR)", min_value=1, max_value=5, value=3)

# Convert research experience to binary
# one-hot to match training columns
Research_No = 1 if research_experience == "No" else 0
Research_Yes = 1 if research_experience == "Yes" else 0

# Predict button
if st.sidebar.button("Predict Admission Probability"):
    # Load the pre-trained model
    with open('reg_admission.pickle', 'rb') as file:
        model = pickle.load(file)
    
    # Create a DataFrame for the input features
    input_data = pd.DataFrame({
        'GRE Score': [GRE_score],
        'TOEFL Score': [TOEFL_score],
        'University Rating': [university_rating],
        'SOP': [statement_of_purpose],
        'LOR': [letter_of_recommendation],
        'CGPA': [CGPA_score],
        'Research_No': [Research_No],
        'Research_Yes': [Research_Yes]
    })
    
    # Make prediction
    prediction = float(model.predict(input_data)[0])
    pred_perc = prediction * 100
    
    # --- Styled output section ---
    st.markdown("<h2 style='color:#1E3A8A;'>Predicting Admission Chance...</h2>", unsafe_allow_html=True)

    # Text box with large prediction
    st.markdown(f"""
    <div style="
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 20px;
        background-color: #ffffff;
        ">
        <p style="font-weight:600; margin:0; font-size:16px;">Predicted Admission Probability</p>
        <p style="font-size:42px; font-weight:700; margin: 10px 0;">{pred_perc:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)


    if prediction >= 0.75:
        st.balloons()
        st.write("Congratulations! You have a high chance of getting admitted! 🎉")
        
    
    # Display prediction intervals using pickle
    with open('reg_admission.pickle', 'rb') as file:
        mapie = pickle.load(file)   
    # Define input features for prediction intervals
    
    # Get prediction intervals
    alpha = 0.1  # For 90% confidence level
    y_pred, y_pis = mapie.predict(input_data, alpha=alpha)
    st.write("")
    st.write("With a 90% confidence level:")
    st.write(f"Prediction Interval: [{float(y_pis[0][0]*100):.2f}%, {float(y_pis[0][1]*100):.2f}%]")

# Showing additional items in tabs
st.markdown("<h3 style='color:darkred;'>Model Insights</h3>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Histogram of residuals", "Predicted vs Actual", "Coverage Plot"])
with tab1:
    st.write("### Feature Importance")
    st.image('feature_importance.png', width=700)
with tab2:
    st.write("### Histogram of Residuals")
    st.image('histogram_of_residuals.png', width=700)
with tab3:
    st.write("### Predicted vs Actual")
    st.image('predicted_vs_actual.png', width=700)
with tab4:
    st.write("### Coverage Plot")
    st.image('coverage_plot.png', width=700)
