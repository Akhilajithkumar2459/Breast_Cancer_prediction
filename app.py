import streamlit as st
import pandas as pd
import pickle
import base64

# Function to set a background image
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded_image = base64.b64encode(image.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}
    .css-1v3fvcr {{
        background-color: rgba(255, 255, 255, 0.8);
    }}
    .stButton>button {{
        background-color: #23cc53;
        color: white;
        padding: 15px 32px;
        text-align: center;
        font-size: 18px;
        border-radius: 5px;
    }}
    /* Custom layout for input and result */
    .stApp .main .block-container {{
        display: flex;
        justify-content: space-between;
    }}
    .stApp .main .block-container .column {{
        flex: 0 0 45%;  /* Adjust width of the content and input */
    }}
    .input-container {{
        margin-top: 100px;
    }}
    .input-container .stButton>button {{
        width: 100%;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Set the background image
set_background("pp1.jpg")  # Replace with the filename of your image

# Load the pre-trained model
with open('breast_cancer_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataset to get the feature names (assuming it has a header)
data = pd.read_csv('Breast_cancer.csv')
feature_names = data.columns[:-1]  # Assuming the last column is the target (e.g., diagnosis)

# Function to create input fields for each feature
def get_user_input(features):
    input_data = {}
    for feature in features:
        input_data[feature] = st.number_input(f"{feature}:", step=1, format="%d", key=feature)
    return pd.DataFrame([input_data])

# Streamlit app setup
st.title("Breast Cancer Prediction App")
st.markdown("""
    <h2 style="color: ##23cc53;">Predict whether breast cancer is malignant or benign</h2>
    <p style="font-size: 18px; color: white; text-align: center;">Please fill in the details below and click "Predict" to get the result.</p>
""", unsafe_allow_html=True)

# Create two columns for layout (input on the right)
col1, col2 = st.columns([1, 2])


# Right Column (for input fields)
with col2:
    st.write("### Please enter the following details:")

    # Create a container for the input fields to keep them aligned on the right
    with st.container():
        user_input_df = get_user_input(feature_names)

        # Prediction button
        if st.button("Predict", key="predict_button"):
            # Display a loading spinner
            with st.spinner('Making prediction...'):
                import time
                time.sleep(2)  # Simulate some processing time

                # Make prediction
                try:
                    prediction = model.predict(user_input_df)[0]
                    result = "Malignant- You have cancer" if prediction == 1 else "Benign - You dont have cancer"
                    # Display the result with some styling
                    st.write(f"<h3 style='color: ##edf0ee;'>Prediction: <b>{result}</b></h3>", unsafe_allow_html=True)
                except Exception as e:
                    st.error("Error in making prediction: " + str(e))

# Optional: Add a footer or additional info
st.markdown("""
    <footer style="text-align: center; padding: 10px;">
        <p style="color: white; font-size: 14px;">ML project by Akhil,Rohith and Amelsha | Breast Cancer Prediction Model</p>
    </footer>
""", unsafe_allow_html=True)

