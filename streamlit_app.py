#-----------------------------------
# Import necessary libraries
#-----------------------------------
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

#-----------------------------------
# Load Trained Models (CNN - Plant Image Classification | CNN LSTM - Soil Moisture Prediction)
#-----------------------------------

@st.cache_resource
def load_cnn_model():
    model = tf.keras.models.load_model("image_classification_CNN.keras")
    return model

cnn_model = load_cnn_model()

def load_cnn_lstm_model():
    model = tf.keras.models.load_model("SM prediction_1D CNN LSTM.keras")
    return model

soilmoisturemodel = load_cnn_lstm_model()

#-----------------------------------
# Function to predict image input
#-----------------------------------

def predict_crop_image(image_file):

    image = Image.open(image_file).convert("RGB")
    image = image.resize((224, 168))
    image_array = np.array(image, dtype=np.float32)

    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # -------Run model inference-------
    prediction = cnn_model.predict(image_array, verbose=0)

    probability = float(np.asarray(prediction).reshape(-1)[0])

    # Assumption: 0 = Drought, 1 = Watered
    if probability >= 0.5:
        predicted_class = "Watered"
        confidence = probability
    else:
        predicted_class = "Drought"
        confidence = 1 - probability

    return predicted_class, confidence

#-----------------------------------
# Soil Moisture Prediction
#-----------------------------------

RAW_INPUT = ['Soil_Moisture_pct', 'Time', 'Temperature_C', 'Humidity_pct', 'Light_Intensity']

# -------Requrired features (with transformed time)-------
FEATURES_INPUT = ['Soil_Moisture_pct','Time_sine','Time_cosine', 'Temperature_C', 'Humidity_pct', 'Light_Intensity']

# -------Same lookback as defined during training-------
LOOKBACK = 24

# -------Load CNN LSTM scalers (features + target)-------
SCALER = joblib.load("sensor_scaler.pkl")
TGT_SCALER = joblib.load("SM_scaler.pkl")

#--------- Functions for Soil Moisture prediction-------

def SMprediction():

    # Execute the preprocessing & trained model
    df, last_24_rows, X_lstm = prepare_sensor_data(sensor_data)
    st.caption(f"LSTM input shape: {X_lstm.shape}")

    # 3. Run the saved LSTM model
    prediction = soilmoisturemodel.predict(X_lstm, verbose=0)
    predicted_moisture = TGT_SCALER.inverse_transform(prediction.reshape(-1, 1))[0, 0]

    return df, last_24_rows, X_lstm, predicted_moisture

# Preprocessing
def prepare_sensor_data(df):

    # Check if dataframe is empty
    if df.empty:
        raise ValueError("Empty *.csv file detected!")

    # Check for missing columns (mismatch with required fetaures)
    missing_col = [
        col for col in RAW_INPUT
        if col not in df.columns
    ]

    if missing_col:
        raise ValueError(
            "Check Missing Columns"
        )

    # At least 24 rows available for lookback
    if len(df) < LOOKBACK:
        raise ValueError("Insufficient rows for prediction, please reupload with >24 rows.")

    # Time transformation to sine/cosine encoding
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')

    df['Hour'] = df['Time'].dt.hour	+ df['Time'].dt.minute/60.0
    df['Time_sine'] = np.sin(2 * np.pi * df["Hour"]/24.0)
    df['Time_cosine'] = np.cos(2 * np.pi * df["Hour"]/24.0)

    # Convert model input features to numeric values
    for col in FEATURES_INPUT:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Select last 24 rows - assuming latest data are last 24 rows to predict next point
    last_24_rows = df[FEATURES_INPUT].tail(LOOKBACK).copy()
    X_scaled = SCALER.transform(last_24_rows)
    X_lstm = X_scaled.astype("float32")[None, :, :]

    return df, last_24_rows, X_lstm

#-----------------------------------
# Set config
#-----------------------------------

st.set_page_config(
    page_title="APU Smart Irrigation System",
    page_icon="🌱",
    layout="wide"
)

#-----------------------------------
# Header
#-----------------------------------

# ------- Define the styles (font)-------
st.markdown("""
<style>

.title {
    font-size: 30px;
    font-weight: 1000;
    text-align: center;
    font-family: 'Consolas';
}


.subtitle {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 60px;
    text-align: center;
    font-family: 'Consolas';
}

.section-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 30px;
}

.stButton > button {
        font-family: 'Consolas';
        font-size: 40px;
        font-weight: bold;
        background-color: #1B88CB;
        font-weight: bold;
        border-radius: 8px;

    }

.stButton > button p{
    font-family: 'Consolas';
    font-size: 25px;
    font-weight: bold;
    background-color: #1B88CB;
    font-weight: bold;
    border-radius: 8px;
    padding: 0.5rem 1rem;

}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="title">🌱 APU Smart Irrigation System</div>'
    '<div class="subtitle">''Plant Image Classification + Sensor Telemetry Soil Moisture Prediction''</div>'
    '<div class="section-title">📥 Input Data</div>',
    unsafe_allow_html=True
)

# ------- Upload Image & Sensor Telemetry CSV -------

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🖼️ Crop Image")

    st.markdown(
        "Please upload plant image in *.jpg format!"
    )

    image_file = st.file_uploader(
        "*.jpg",
        type=["jpg"],
        key="imgupload"
    )

    if image_file is not None:
        st.image(
            image_file,
            caption="Image Uploaded...",
            use_container_width=True
        )
        st.success("Image uploaded!")    

    else:
        st.info("Please upload plant image in *.jpg format!")

with col2:

    st.markdown("#### 📊 Sensor Data")
    st.markdown("Please upload sensor telemetry file in *.csv format!")
    csv_file = st.file_uploader(
        "*.csv",
        type=["csv"],
        key="sensorupload"
    )

    if csv_file is not None:

        try:
            sensor_data = pd.read_csv(csv_file)
            st.success("CSV uploaded successfully!")
        except Exception as e:
            st.error("Unable to read CSV")
    else:
        st.info("Please upload sensor telemetry file in *.csv format!")
    
    st.markdown('</div>',unsafe_allow_html=True)


# ------------ Late fusion
run_prediction = st.button("🚀 Execute Analysis")
if run_prediction:
    _, _, _, predicted_moisture = SMprediction()
    cnn_prediction, cnn_confidence = predict_crop_image(image_file)
    st.subheader("💧 Irrigation Decision")

    # Set your chosen soil moisture threshold (%)
    MOISTURE_THRESHOLD = 65.0

    # Late fusion rule
    if cnn_prediction == "Drought" or predicted_moisture < MOISTURE_THRESHOLD:
        irrigation_needed = True

    else:
        irrigation_needed = False

    # Display results
    if irrigation_needed:
        st.error("🚨 Irrigation Recommended")
    else:
        st.success("✅ Irrigation Not Required")
    st.markdown(
        f"""
        <table style="
            width:100%;
            font-size:18px;
            border-collapse:collapse;
            margin-bottom:80px;
        ">
            <tr>
                <th style="text-align:left; padding:10px; border:1px solid #ccc;">
                    Parameter
                </th>
                <th style="text-align:left; padding:10px; border:1px solid #ccc;">
                    Value
                </th>
            </tr>
            <tr>
                <td style="padding:10px; border:1px solid #ccc;">Moisture Threshold</td>
                <td style="padding:10px; border:1px solid #ccc;">{MOISTURE_THRESHOLD:.1f}%</td>
            </tr>
            <tr>
                <td style="padding:10px; border:1px solid #ccc;">Image Classification</td>
                <td style="padding:10px; border:1px solid #ccc;">{cnn_prediction}</td>
            </tr>
            <tr>
                <td style="padding:10px; border:1px solid #ccc;">Predicted Soil Moisture</td>
                <td style="padding:10px; border:1px solid #ccc;">{predicted_moisture:.2f}%</td>
            </tr>
        </table>
        """,
        unsafe_allow_html=True
    )

# ------- Execute analysis -------

    if image_file is None:
        st.warning("Missing Image!")

    else:
        with st.spinner("Analysing crop image..."):
            try:
                
                st.markdown("""
                    <div style="
                        background-color: #169FDC;
                        padding: 0px 16px;
                        border-radius: 22px;
                        margin-bottom: 15px;
                    ">
                        <h3 style="
                            color: white;
                            margin: 0;
                            font-size: 22px;
                        ">
                            CNN Crop Image Classification
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)

                st.metric("Predicted Condition", cnn_prediction)

                st.metric("Model Confidence",f"{cnn_confidence:.1%}")

            except Exception as e:
                st.error("Classification failed!")


    if csv_file is None:
        st.warning("Please upload sensor data.")

    else:
        try:

            st.markdown("""
                <div style="
                    background-color: #169FDC;
                    padding: 0px 16px;
                    border-radius: 22px;
                    margin-bottom: 15px;
                ">
                    <h3 style="
                        color: white;
                        margin: 0;
                        font-size: 22px;
                    ">
                        Soil Moisture Prediction
                    </h3>
                </div>
                """, unsafe_allow_html=True)
            
            _,last_24_rows,_, predicted_moisture = SMprediction()

            # Display soil moisture outcome
            st.metric(
                "Predicted Soil Moisture",
                f"{predicted_moisture:.2f}%"
            )

        except Exception as e:
            st.error(f"{e}")

    # Display raw data table used
    st.markdown("""
        <div style="
            background-color: #E6830C;
            padding: 8px 12px;
            border-radius: 6px;
            margin-bottom: 10px;
        ">
            <p style="
                color: white;
                font-size: 18px;
                font-weight: 600;
                margin: 0;
            ">
        Sensor Telemetry Data Retrieved (with transformed time)
        </p>""",
        unsafe_allow_html=True
    )
    st.dataframe(last_24_rows, use_container_width=True)
