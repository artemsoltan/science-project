import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

def load_and_preprocess_image(img_path, target_size):
    """Load and preprocess an image."""
    img = image.load_img(img_path, target_size=target_size, color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Scale the image pixels to [0, 1]
    return img_array

def predict_image(model, img_array):
    """Make a prediction on a single image."""
    predictions = model.predict(img_array)
    return predictions

def main():
    # Path to the saved model
    model_path = 'D:/Projects/Python/science-project/model/architecture.keras'
    
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Image settings
    target_size = (28, 28)  # Adjust based on your model's input size

    # Path to the image(s) you want to predict
    image_dir = 'C:/Users/USER/Desktop/input'
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]

    for img_path in image_paths:
        # Preprocess the image
        img_array = load_and_preprocess_image(img_path, target_size)
        
        # Predict the class of the image
        predictions = predict_image(model, img_array)
        
        # Post-process the predictions
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        
        print(f"Image: {img_path}, Predicted class: {predicted_class}, Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()