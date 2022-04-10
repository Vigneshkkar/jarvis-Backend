import os

from flask import Flask
from flask import request,send_from_directory
from tensorflow.keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
from flask import jsonify
import json
from flask_cors import CORS, cross_origin


def create_app(test_config=None):
    loaded_model = load_model('NLP2.h5')
# print(loaded_model.summary())

    with (open("tokenizer2.pkl", "rb")) as openfile:
        # while True:
        try:
            tokenizer = pickle.load(openfile)
        except EOFError:
            print("Err")

    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    cors = CORS(app)
    app.config.from_mapping(
        SECRET_KEY='dev'
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/home/<path:path>')
    @cross_origin()
    def send_js(path):
        return send_from_directory('build', path)
    
    @app.route('/api/v1/getprediction')
    @cross_origin()
    def predictions():
        text = request.args.get('sentence')
        predicted = loaded_model.predict(pad_sequences( tokenizer.texts_to_sequences([text]), maxlen = 1000))
        # print(predicted.shape)
        # print(predicted)
        return jsonify({"data": predicted[0].tolist()})

    return app