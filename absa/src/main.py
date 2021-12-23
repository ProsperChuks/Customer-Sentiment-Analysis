import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = pickle.load(open('C:/Users/Shallom/Desktop/do not open/Customer-Feedback-Analysis/absa/models/LSTM/Hyperparam Tuning/model0.001', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction

    return render_template('index.html', prediction_text='Rating of the Wine is {}'.format(output))

@app.route('/results', methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
