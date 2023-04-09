from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('snake_species_model.h5')

# Define a list of species names
species_names = ['cobra', 'rattlesnake', 'python', 'viper']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    image_file = request.files['image']

    # Preprocess the image
    image = tf.keras.preprocessing.image.load_img(image_file, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.

    # Make the prediction
    prediction = model.predict(image)[0]
    species_index = np.argmax(prediction)
    species_name = species_names[species_index]

    return species_name

if __name__ == '__main__':
    app.run()

