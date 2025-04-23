# Yoga-Pose-Detection
The goal of this project is to develop a multi-class classification system that can predict 107 different yoga poses from input images.

## Workflow
This project follows a multi-stage approach to tackle the classification problem. First, the input images are preprocessed:
- MediaPipe is used to detect and extract body landmarks.
- The landmarks are normalized for consistency. 
- The landmarks are used to calculate the angles in the joints
- These features are stored in a structured dataset, allowing for both image-based (CNN) and feature-based (MLP) classification approaches.
 
Then, two different models are trained and compared:
- 1D Convolutional Neural Network (CNN) Model
- Multilayer Perceptron (MLP) 

The trained models are saved in .h5 (for Keras models) or .pkl (for Scikit-learn models) formats for deployment.
To make the system accessible, a Streamlit-based web application is developed, offering three key functionalities:
- Live Camera Pose Detection: Real-time classification using a webcam feed.
- Uploaded Image Classification: Predicts poses from user-uploaded images.
- Model Performance Dashboard: Displays accuracy, confusion matrices, and other metrics.

## Preview
[![Video Preview](Video.png)](https://drive.google.com/file/d/1rVNMi8UYEs9T_Mle0LMCWjRgiMvod9mQ/view?usp=sharing)

## Project Setup
```bash
Yoga-Pose-Detection/
│── Models
     |── model_cnn.h5
     |── scalar_cnn.pkl
     |── model_mlp.h5
     |── scalar_mlp.pkl 
│── Exploration.ipynb
|── Preprocessing.ipynb
|── CNN.ipynb
|── MLP.ipynb
│── app.py            
│── requirements.txt       
│── README.md           
│── License
```

## Installation and setup

## Dataset


## 1D CNN Model
- The model accepts input shape (33+8, 1) (flattened landmarks + angles) and outputs probabilities for all 107 yoga poses via softmax activation
- The model is lightweight yet powerful because 1D convolutions are computationally efficient compared to 2D CNNs
- Dropout + L2 regularization prevent overfitting on small datasets
- This model learns spatial relationships between body joints regardless of absolute positions
- The model is dependent on accurate MediaPipe landmark detection and less interpretable compared to simpler ML models like multi layer perceptrons.
- Uses three Conv1D blocks with increasing filters (32 → 64 → 128) to hierarchically extract spatial features
- Implements Batch Normalization after each Conv1D layer for stable training
- ![cnn_arch](image/cnn_arch.png)
- The accuracy curve shows steady improvement, with minor fluctuations due to aggressive dropout (0.3–0.5).
- Loss curve indicates stable convergence, though slower than MLP due to deeper feature learning.
- The model achives an accuracy of 73.48% in 196 epochs
- ![cnn_arch](image/cnn.png)

## MLP Model
- The model accepts input shape (33 landmarks × 3 coordinates + 8 angles = 107 features) and outputs probabilities for all 107 yoga poses via softmax activation
- The model is lightweight yet powerful as dense layers are more computationally efficient than convolutional operations and requires significantly fewer parameters than equivalent CNN architectures
- Learns relationships between body joints through fully connected layers and maintains some position invariance through normalized landmark inputs
- Highly dependent on accurate MediaPipe landmark detection and more interpretable than CNN models as weights directly correspond to biomechanical features
- Follows a descending neuron pattern (256 → 128 → 64 → 32) to hierarchically compress features.
- Avoids abrupt dimensionality reduction, preserving critical pose information.
- Batch Normalization after each dense layer stabilizes training.
- Dropout (0.3) mitigates overfitting by randomly deactivating neurons.
- Activation function ReLU for hidden layers (efficient gradient flow) and softmax for multi-class output (107 poses).
- image[mlp_arch.png]
- Accuracy plateaued earlier (by epoch 40), suggesting efficient feature extraction from preprocessed landmarks.
- Loss dropped sharply in initial epochs but stabilized quickly, indicating the model learned key patterns early.
- The model achives an accuracy of 71.46% in just 65 epochs.
- image[mlp_accuracy.png]
