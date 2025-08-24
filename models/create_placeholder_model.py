"""
Script to create placeholder emotion detection model files.
This creates minimal model structure for development purposes.
"""
import os
import numpy as np
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - will create mock structure")

def create_placeholder_model():
    """Create a placeholder emotion detection model"""
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not installed - creating placeholder files")
        # Create a simple placeholder file
        with open('emotion_model_weights.h5', 'w') as f:
            f.write("# Placeholder model file - TensorFlow not available\n")
        return
    
    # Build the same model structure as in utils.py
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dense(7, activation='softmax')  # 7 emotion classes
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Save the model weights (will be random initialization)
    model.save_weights('emotion_model_weights.h5')
    print("Created placeholder emotion model weights at emotion_model_weights.h5")
    
    return model

if __name__ == "__main__":
    create_placeholder_model()