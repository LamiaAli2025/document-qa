# rag.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Arabic character set - adjust to match your model's classes
ARABIC_CHARS = "ابتةثجحخدذرزسشصضطظعغفقكلمنهوي"

def load_handwriting_model(model_path='.github/My_Model/RAG.h5'):
    """Load and return the pre-trained model"""
    return load_model(model_path)

def preprocess_image(image):
    """Preprocess image for model input"""
    # Convert to grayscale
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Apply preprocessing
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Resize to model input size (adjust dimensions to match your model)
    img = cv2.resize(img, (128, 32))
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    # Add channel dimension
    img = np.expand_dims(img, axis=-1)
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def predict_text(model, image):
    """Run model prediction on preprocessed image"""
    # Get model prediction
    prediction = model.predict(image)
    
    # Convert prediction to text (adjust based on your model output)
    # Example for CTC output:
    # decoded = K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1])[0][0]
    # text = decode_labels(decoded, ARABIC_CHARS)
    
    # Dummy implementation - replace with your actual decoding
    text = "النموذج المخصص يعمل بنجاح"
    return text