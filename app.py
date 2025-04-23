import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import ast
import tempfile
from streamlit_mermaid import st_mermaid

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Set page config
st.set_page_config(page_title="Yoga Pose Classifier", layout="wide")

# Load models and scalers (with caching)
@st.cache_resource
def load_mlp_model():
    model = tf.keras.models.load_model("model_MLP.h5")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    scaler = joblib.load("scaler_mlp.pkl")
    return model, scaler

@st.cache_resource
def load_cnn_model():
    model = tf.keras.models.load_model("model_CNN.h5")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    scaler = joblib.load("scaler_cnn.pkl")
    return model, scaler

@st.cache_resource
def load_lstm_model():
    model = tf.keras.models.load_model("model_LSTM.h5")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    scaler = joblib.load("scaler_lstm.pkl")
    return model, scaler

# Load label encoder (replace with your actual labels)
label_encoder = {
    "adho mukha svanasana": 0,
    "adho mukha vriksasana": 1,
    "agnistambhasana": 2,
    "ananda balasana": 3,
    "anantasana": 4,
    "anjaneyasana": 5,
    "ardha bhekasana": 6,
    "ardha chandrasana": 7,
    "ardha matsyendrasana": 8,
    "ardha pincha mayurasana": 9,
    "ardha uttanasana": 10,
    "ashtanga namaskara": 11,
    "astavakrasana": 12,
    "baddha konasana": 13,
    "bakasana": 14,
    "balasana": 15,
    "bhairavasana": 16,
    "bharadvajasana i": 17,
    "bhekasana": 18,
    "bhujangasana": 19,
    "bhujapidasana": 20,
    "bitilasana": 21,
    "camatkarasana": 22,
    "chakravakasana": 23,
    "chaturanga dandasana": 24,
    "dandasana": 25,
    "dhanurasana": 26,
    "durvasasana": 27,
    "dwi pada viparita dandasana": 28,
    "eka pada koundinyanasana i": 29,
    "eka pada koundinyanasana ii": 30,
    "eka pada rajakapotasana": 31,
    "eka pada rajakapotasana ii": 32,
    "ganda bherundasana": 33,
    "garbha pindasana": 34,
    "garudasana": 35,
    "gomukhasana": 36,
    "halasana": 37,
    "hanumanasana": 38,
    "janu sirsasana": 39,
    "kapotasana": 40,
    "krounchasana": 41,
    "kurmasana": 42,
    "lolasana": 43,
    "makara adho mukha svanasana": 44,
    "makarasana": 45,
    "malasana": 46,
    "marichyasana i": 47,
    "marichyasana iii": 48,
    "marjaryasana": 49,
    "matsyasana": 50,
    "mayurasana": 51,
    "natarajasana": 52,
    "padangusthasana": 53,
    "padmasana": 54,
    "parighasana": 55,
    "paripurna navasana": 56,
    "parivrtta janu sirsasana": 57,
    "parivrtta parsvakonasana": 58,
    "parivrtta trikonasana": 59,
    "parsva bakasana": 60,
    "parsvottanasana": 61,
    "pasasana": 62,
    "paschimottanasana": 63,
    "phalakasana": 64,
    "pincha mayurasana": 65,
    "prasarita padottanasana": 66,
    "purvottanasana": 67,
    "salabhasana": 68,
    "salamba bhujangasana": 69,
    "salamba sarvangasana": 70,
    "salamba sirsasana": 71,
    "savasana": 72,
    "setu bandha sarvangasana": 73,
    "simhasana": 74,
    "sukhasana": 75,
    "supta baddha konasana": 76,
    "supta matsyendrasana": 77,
    "supta padangusthasana": 78,
    "supta virasana": 79,
    "tadasana": 80,
    "tittibhasana": 81,
    "tolasana": 82,
    "tulasana": 83,
    "upavistha konasana": 84,
    "urdhva dhanurasana": 85,
    "urdhva hastasana": 86,
    "urdhva mukha svanasana": 87,
    "urdhva prasarita eka padasana": 88,
    "ustrasana": 89,
    "utkatasana": 90,
    "uttana shishosana": 91,
    "uttanasana": 92,
    "utthita ashwa sanchalanasana": 93,
    "utthita hasta padangustasana": 94,
    "utthita parsvakonasana": 95,
    "utthita trikonasana": 96,
    "vajrasana": 97,
    "vasisthasana": 98
}

# Create inverse mapping
inverse_label_encoder = {v: k for k, v in label_encoder.items()}
inverse_label_encoder = {v: k for k, v in label_encoder.items()}

# Function to calculate angles between keypoints
def calculate_angle(a, b, c):
    a = np.array(a[:3])  # Only use x,y,z (ignore visibility)
    b = np.array(b[:3])
    c = np.array(c[:3])
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

# Function to extract keypoints and angles from an image
def extract_features(image):
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        return None
    
    # Extract keypoints (33 landmarks * 4 values each)
    keypoints = []
    for lm in results.pose_landmarks.landmark:
        keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])  # 132 features (33*4)
    
    # Define angles of interest
    angle_definitions = {
        "left_elbow": (11, 13, 15),   # Shoulder-Elbow-Wrist
        "right_elbow": (12, 14, 16),
        "left_knee": (23, 25, 27),    # Hip-Knee-Ankle
        "right_knee": (24, 26, 28),
        "left_hip": (11, 23, 25),     # Shoulder-Hip-Knee
        "right_hip": (12, 24, 26),
        "left_shoulder": (13, 11, 23),
        "right_shoulder": (14, 12, 24),
    }
    
    # Calculate angles (8 features)
    landmarks = results.pose_landmarks.landmark
    angles = []
    for name, (a, b, c) in angle_definitions.items():
        angle = calculate_angle(
            (landmarks[a].x, landmarks[a].y, landmarks[a].z),
            (landmarks[b].x, landmarks[b].y, landmarks[b].z),
            (landmarks[c].x, landmarks[c].y, landmarks[c].z)
        )
        angles.append(angle)
    
    # Pad features if needed to match expected dimension
    features = np.array(keypoints + angles, dtype=np.float32)
    
    # If your model expects 164 features, pad with zeros
    if len(features) < 164:
        features = np.pad(features, (0, 164 - len(features)), 'constant')
    
    return features

# Prediction functions
def predict_with_mlp(image, model, scaler):
    features = extract_features(image)
    if features is None:
        return "No pose detected", 0.0
    
    # Scale features
    features_scaled = scaler.transform([features])
    
    # Predict
    pred_probs = model.predict(features_scaled)
    predicted_class = np.argmax(pred_probs)
    confidence = pred_probs[0][predicted_class]
    
    return inverse_label_encoder[predicted_class], confidence

def predict_with_cnn(image, model, scaler):
    features = extract_features(image)
    if features is None:
        return "No pose detected", 0.0
    
    # Scale features and reshape for CNN
    features_scaled = scaler.transform([features])
    features_scaled = features_scaled.reshape(1, features_scaled.shape[1], 1)
    
    # Predict
    pred_probs = model.predict(features_scaled)
    predicted_class = np.argmax(pred_probs)
    confidence = pred_probs[0][predicted_class]
    
    return inverse_label_encoder[predicted_class], confidence

def predict_with_lstm(image, model, scaler):
    features = extract_features(image)
    if features is None:
        return "No pose detected", 0.0
    
    # Scale features
    features_scaled = scaler.transform([features])
    features_scaled = features_scaled.reshape(1, 164, 1)
    
    # Predict
    try:
        pred_probs = model.predict(features_scaled)
        predicted_class = np.argmax(pred_probs)
        confidence = pred_probs[0][predicted_class]
        
        if predicted_class not in inverse_label_encoder:
            return f"Unknown pose (class {predicted_class})", confidence
        
        return inverse_label_encoder[predicted_class], confidence
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Prediction failed", 0.0
    
# Load model performance graphs
mlp_graph = "MLP.png"
cnn_graph = "CNN.png"

# App title
st.title("Yoga Pose Classification")

# Main menu
menu = st.sidebar.selectbox("Select Option", 
                           ["Live Camera Classification", 
                            "Upload Image Classification", 
                            "Model Information"])

if menu == "Live Camera Classification":
    st.header("Live Camera Pose Classification")
    
    # Model selection
    model_choice = st.selectbox("Select Model", 
                               ["Multi-layer Perceptron (MLP)", 
                                "Convolutional Neural Network (CNN)"])
    
    # Load selected model
    if model_choice == "Multi-layer Perceptron (MLP)":
        model, scaler = load_mlp_model()
    elif model_choice == "Convolutional Neural Network (CNN)":
        model, scaler = load_cnn_model()
    
    # Initialize session state
    if 'previous_label' not in st.session_state:
        st.session_state.previous_label = None
    if 'current_label' not in st.session_state:
        st.session_state.current_label = None
    if 'capture' not in st.session_state:
        st.session_state.capture = False
    
    # Initialize webcam
    run = st.checkbox('Start Camera', key='camera_toggle')
    FRAME_WINDOW = st.image([])
    prediction_placeholder = st.empty()
    camera = cv2.VideoCapture(0)
    
    # Capture button
    if st.button('Capture and Predict', key='capture_button'):
        st.session_state.capture = True
    
    while run:
        # Capture frame-by-frame
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to capture frame from camera")
            break
        
        # Convert the image from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
        
        if st.session_state.capture:
            # Convert frame to BGR for processing
            image_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Make prediction
            if model_choice == "Multi-layer Perceptron (MLP)":
                current_label, confidence = predict_with_mlp(image_bgr, model, scaler)
            elif model_choice == "Convolutional Neural Network (CNN)":
                current_label, confidence = predict_with_cnn(image_bgr, model, scaler)
            
            # Only update if pose has changed
            if current_label != st.session_state.previous_label:
                st.session_state.current_label = current_label
                prediction_placeholder.success(f"Predicted Pose: {current_label} (Confidence: {confidence:.2%})")
                st.session_state.previous_label = current_label
            
            st.session_state.capture = False
    
    if not run:
        st.write('Camera is stopped')
        camera.release()

elif menu == "Upload Image Classification":
    st.header("Upload Image for Pose Classification")
    
    # Model selection
    model_choice = st.selectbox("Select Model", 
                               ["Multi-layer Perceptron (MLP)", 
                                "Convolutional Neural Network (CNN)"])
    
    # Load selected model
    if model_choice == "Multi-layer Perceptron (MLP)":
        model, scaler = load_mlp_model()
    elif model_choice == "Convolutional Neural Network (CNN)":
        model, scaler = load_cnn_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the image file
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display the uploaded image with use_container_width
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Uploaded Image', use_container_width=True)
        # Make prediction based on selected model
        if model_choice == "Multi-layer Perceptron (MLP)":
            label, confidence = predict_with_mlp(image, model, scaler)
        elif model_choice == "Convolutional Neural Network (CNN)":
            label, confidence = predict_with_cnn(image, model, scaler)
        else:
            label, confidence = predict_with_lstm(image, model, scaler)
        
        # Display prediction
        st.success(f"Predicted Pose: {label} (Confidence: {confidence:.2%})")

elif menu == "Model Information":
    st.header("Model Information")
    
    tab1, tab2= st.tabs(["MLP", "CNN"])
    
    with tab1:
        st.subheader("Multi-layer Perceptron (MLP)")
        
        st.image("MLP_arch.jpg", caption="MLP Architecture")
        
        st.markdown("""
        ### MLP for Pose Detection
        The Multi-Layer Perceptron is a basic but effective neural network architecture for pose classification:
        - The model accepts input shape (33 landmarks × 3 coordinates + 8 angles = 107 features) and outputs probabilities for all 107 yoga poses via softmax activation
        - The model is lightweight yet powerful as dense layers are more computationally efficient than convolutional operations and requires significantly fewer parameters than equivalent CNN architectures
        - Learns relationships between body joints through fully connected layers and maintains some position invariance through normalized landmark inputs
        - Highly dependent on accurate MediaPipe landmark detection and more interpretable than CNN models as weights directly correspond to biomechanical features
        - Follows a descending neuron pattern (256 → 128 → 64 → 32) to hierarchically compress features.
        - Avoids abrupt dimensionality reduction, preserving critical pose information.
        - Batch Normalization after each dense layer stabilizes training.
        - Dropout (0.3) mitigates overfitting by randomly deactivating neurons.
        - Activation function ReLU for hidden layers (efficient gradient flow) and softmax for multi-class output (107 poses).
        """)
        
        if os.path.exists(mlp_graph):
            st.image(mlp_graph, caption="MLP Training Performance")
        else:
            st.warning("MLP performance graph not found")
        
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", caption="MLP Training Performance")
        else:
            st.warning("MLP confusion matrix not found")
    
    with tab2:
        st.subheader("Convolutional Neural Network (CNN)")
        
        st.image("CNN_arch.jpg", caption="CNN Architecture")
        
        st.markdown("""
        **1D CNN Model for Yoga Pose Classification:**
        - The model accepts input shape (33+8, 1) (flattened landmarks + angles) and outputs probabilities for all 107 yoga poses via softmax activation
        - The model is lightweight yet powerful because 1D convolutions are computationally efficient compared to 2D CNNs
        - Dropout + L2 regularization prevent overfitting on small datasets
        - This model learns spatial relationships between body joints regardless of absolute positions
        - The model is dependent on accurate MediaPipe landmark detection and less interpretable compared to simpler ML models like multi layer perceptrons.
        - A 1D Convolutional Neural Network designed to process sequential data (landmark coordinates + angles as time-series-like features)
        - Uses three Conv1D blocks with increasing filters (32 → 64 → 128) to hierarchically extract spatial features
        -Implements Batch Normalization after each Conv1D layer for stable training
        - The accuracy curve shows steady improvement, with minor fluctuations due to aggressive dropout (0.3–0.5).
        - Loss curve indicates stable convergence, though slower than MLP due to deeper feature learning.

        """)
        
        if os.path.exists(cnn_graph):
            st.image(cnn_graph, caption="CNN Training Performance")
        else:
            st.warning("CNN performance graph not found")
        if os.path.exists("confusion_matrix_MODEL_CNN.png"):
            st.image("confusion_matrix_MODEL_CNN.png", caption="CNN Training Performance")
        else:
            st.warning("CNN performance graph not found")
    
    

