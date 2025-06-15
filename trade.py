import tensorflow as tf
import joblib
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load saved model and label encoder
model = tf.keras.models.load_model('trained_combined_model.h5')
label_encoder = joblib.load('combined_label_encoder.pkl')

# Preprocess input image function
def preprocess_image(path, img_size=(128, 128)):
    img = load_img(path, target_size=img_size)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # batch dimension
    return img

# Map predicted class to a trading signal
def interpret_prediction(pred_label):
    if pred_label.startswith('bullish'):
        return "Buy Signal"
    elif pred_label.startswith('bearish'):
        return "Sell Signal"
    else:
        return "Hold / No Action"

# Main inference function
def predict_candlestick_pattern(image_path):
    img = preprocess_image(image_path)
    preds = model.predict(img)[0]  # probabilities for all classes
    pred_index = np.argmax(preds)
    pred_label = label_encoder.inverse_transform([pred_index])[0]
    confidence = preds[pred_index]

    signal = interpret_prediction(pred_label)

    print(f"Predicted Pattern: {pred_label}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Trading Signal: {signal}")

    return pred_label, confidence, signal

# Example usage
if __name__ == "__main__":
    test_image_path ='s.png'  # <-- Replace with your image path
    predict_candlestick_pattern(test_image_path)
