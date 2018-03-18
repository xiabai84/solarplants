import io, traceback

from flask import Flask, request, g,  url_for
from flask import send_file
from flask import jsonify
from flask_mako import MakoTemplates, render_template
from plim import preprocessor

from PIL import Image, ExifTags
from scipy.misc import imresize
import numpy as np
from keras.models import load_model
import tensorflow as tf

app = Flask(__name__, instance_relative_config=True)
# For Plim templates
mako = MakoTemplates(app)
app.config['MAKO_PREPROCESSOR'] = preprocessor
app.config.from_object('config.ProductionConfig')


# Preload our model
print("Loading model")
model = load_model('./model/solar_model.h5', compile=False)
graph = tf.get_default_graph()

def ml_predict(image):
    with graph.as_default():
        # Add a dimension for the batch
        prediction = model.predict([image[:, :, 0:3][None, :, :, :]])
    return prediction

@app.route('/prediction/<long>/<lat>')
def predict(long,lat):
    image = Image.open("test.png")
    resized_image = imresize(image, (64, 64)) / 255.0
    result =  ml_predict(resized_image)
    return jsonify({'result': result.tolist() })

@app.route('/')
def homepage():
    return url_for('static', filename='yes.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0')
