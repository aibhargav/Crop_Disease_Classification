import os
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import joblib

app = Flask(__name__)



# Load crop prediction model
crop_model = joblib.load("croppred.joblib")

# Load pepper model
pepper_model = tf.keras.models.load_model("my_pepper_model.h5")
pepper_class_names = ['Bacterial spot', 'Pepper bell healthy']

# Load potato model
potato_model = tf.keras.models.load_model("potatoes.h5")
potato_class_names = ['Early Blight', 'Late Blight', 'Healthy']

# Load tomato model
tomato_model = tf.keras.models.load_model("my_tomato_model.h5")
tomato_class_names = ['Tomato_Bacterial_spot',
 'Tomato_Late_blight',
 'Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato_healthy'
 ]

pesticide_recommendation = {
    'Bacterial spot': 'Use pesticide Copper sprays',
    'Pepper bell healthy': 'Don\'t use pesticide',
    'Early Blight': 'Use pesticide oxychloride and streptomycin solution',
    'Late Blight': 'Use pesticide oxychloride and streptomycin solution',
    'Healthy': 'Don\'t use pesticide',
    'Tomato_Bacterial_spot':'Use Pesticide Copper-containing bactericides',
 'Tomato_Late_blight':'Use fungicide sprays based on mandipropamid, chlorothalonil, fluazinam, mancozeb ',
 'Tomato__Tomato_YellowLeaf__Curl_Virus':'There is no treatment but try to reduce whiteflys by keeping the leaves clean',
 'Tomato_healthy':'Don\'t use pesticide'
}

@app.route('/')
def index():
    return render_template('indexall.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    if 'crop_type' not in request.form:
        return jsonify({'error': 'Crop type not specified'})

    crop_type = request.form['crop_type']
    if crop_type == 'pepper':
        model = pepper_model
        class_names = pepper_class_names
    elif crop_type == 'potatoes':
        model = potato_model
        class_names = potato_class_names
    elif crop_type == 'tomatoes':
        model = tomato_model
        class_names = tomato_class_names
    else:
        return jsonify({'error': 'Invalid crop type'})

    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image file'})

    try:
        img = Image.open(file)
        img = img.resize((256, 256))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        predicted_class = class_names[np.argmax(prediction)]
        pesticide = pesticide_recommendation.get(predicted_class, 'Unknown')
        return jsonify({'predicted_class': predicted_class, 'pesticide_recommendation': pesticide})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/crop-predict', methods=['POST'])
def crop_predict():
    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Make a prediction using the loaded model
        prediction = crop_model.predict([[N, P, K, temperature, humidity, ph, rainfall]])

        return jsonify({'predicted_label': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
