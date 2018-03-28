import io, traceback
import os

from flask import Flask, request, g,  url_for, redirect
from flask import send_file
from flask import jsonify
from flask_mako import MakoTemplates, render_template
from plim import preprocessor

from PIL import Image, ExifTags
from scipy.misc import imresize
import numpy as np
from keras.models import load_model
import tensorflow as tf

import urllib, cStringIO

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

@app.route('/predict/<lo>/<lat>')
def predict(lo,lat):
    thumb_size = 64
    size = 300
    zoom = 20 
    #url = "https://maps.googleapis.com/maps/api/staticmap?maptype=satellite&center=" + str(lo) + "," + str(lat) + "&zoom=" + str(zoom) + "&size=" + str(size) + "x" + str(size) + "&key=" + str(os.environ['MAPS_API_KEY'])
        #url = "https://maps.googleapis.com/maps/api/staticmap?maptype=satellite&center=" + str(lo) + "," + str(lat) + "&zoom=" + str(zoom) + "&size=" + str(size) + "x" + str(size) + "&key=" + str(os.environ['MAPS_API_KEY'])
    url = "https://maps.googleapis.com/maps/api/staticmap?maptype=satellite&format=png32&center=" + str(lo) + "," + str(lat) + "&zoom=20&size=300x300&key=AIzaSyA91yDb1_0u2a9_l-yzVgcqvtJg_RBbgl4"
    file = cStringIO.StringIO(urllib.urlopen(url).read())
    image = Image.open(file)
    resized_image = imresize(image, (64, 64)) / 255.0
    channel_means = [0.30500001, 0.30549607, 0.26879644]
    channel_stds = [0.17779149, 0.16334727, 0.14763094]
    for channel in range(3):
        resized_image[:, :, channel] -= channel_means[channel]
        resized_image[:, :, channel] /= channel_stds[channel]
    result =  ml_predict(resized_image)
    return jsonify({'result': result.tolist() })

@app.route('/')
def homepage():
    return redirect(url_for('static', filename='app.html'))



if __name__ == '__main__':
    app.run(host='0.0.0.0')
