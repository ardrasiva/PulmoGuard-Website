from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
from pymongo import MongoClient
import os

app = Flask(__name__)

client = MongoClient("mongodb+srv://ardrasiva123:ardmongo1612@cluster0.wqlfkjh.mongodb.net/pulmoguard?retryWrites=true&w=majority")

# Replace 'pulmoguard' with your database name
db = client["pulmoguard"]
results_collection = db["results"]

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="pneumonia_vgg19_classifier.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = ['NORMAL', 'PNEUMONIA']

def preprocess_image(image):
    img = image.resize((128, 128))  # Resize as per model requirement
    img_array = np.array(img).astype('float32') / 255.0
    if len(img_array.shape) == 2:  # grayscale handling
        img_array = np.stack((img_array,) * 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/results')
def show_results():
    results = list(results_collection.find().sort("_id", -1).limit(10))
    return render_template("results.html", results=results)
1

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('upload.html', prediction="No image uploaded.")

    image_file = request.files['image']
    if image_file.filename == '':
        return render_template('upload.html', prediction="No image selected.")

    image = Image.open(image_file).convert('RGB')
    processed_image = preprocess_image(image)

    interpreter.set_tensor(input_details[0]['index'], processed_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_index = int(np.argmax(output_data))
    result = labels[predicted_index]

    results_collection.insert_one({
    "filename": image_file.filename,
    "result": result,
    "confidence": float(output_data[0][predicted_index]),
})

    return render_template('upload.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True, port = 5000)