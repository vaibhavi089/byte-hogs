from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import io
import numpy as np
import joblib  # Use joblib to load .pkl files

app = Flask(__name__)

# Load your pre-trained model (adjust the path and loading process as per your model)
model = joblib.load(r'C:/Users/DELL/OneDrive/Desktop/python/Untitled2.pkl')

# Define the image preprocessing function for your model
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize the image to 128x128 as expected by the model
    image = np.array(image)  # Convert the image to a numpy array
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        print("No file part")
        return redirect(request.url)

    file = request.files['image']
    
    if file.filename == '':
        print("No selected file")
        return redirect(request.url)

    if file:
        try:
            img_bytes = file.read()
            image = Image.open(io.BytesIO(img_bytes))
            print(f"Image uploaded: {file.filename}, size: {len(img_bytes)} bytes")
        except Exception as e:
            print(f"Error opening image: {e}")
            return "Error processing image"

        # Preprocess the image for your model
        input_tensor = preprocess_image(image)

        # Make prediction using the loaded model
        predictions = model.predict(input_tensor)

        # Process the model output (this depends on your specific model)
        predicted_class = np.argmax(predictions, axis=1)[0]

        return f'Predicted Class: {predicted_class}'

if __name__ == '__main__':
    app.run(debug=True)
