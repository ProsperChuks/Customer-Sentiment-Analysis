import numpy as np
import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

train_data = pd.read_csv('../notebooks/datasets/train - train.csv')
model = keras.models.load_model('../models/LSTM/Hyperparam Tuning/my_model0.001.h5')

train_data = train_data['text'] + train_data['aspect']
tk = Tokenizer(len(train_data), filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', split=' ')
tk.fit_on_texts(train_data)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    lab = []
    label = []
    [lab.append(x) for x in request.form.values()]
    text_asp = lab[0] + lab[1]
    text_asp = tk.texts_to_sequences(text_asp)
    text_asp_padded =  pad_sequences(text_asp, maxlen=32, truncating='post', padding='post')

    classify = model.predict(text_asp_padded)
    label.append({'Negative': np.argmax(classify[:, 0]), 'Neutral': np.argmax(classify[:, 1]), 'Positive': np.argmax(classify[:, 2])})

    return render_template('index.html', prediction_text=label)

@app.route('/results', methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
