# -*- coding: utf-8 -*-
"""evaluation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lxoAcU_bEhqb-KRC1je48B0AawY4L9sB

### Dependecies
"""

import numpy as np
import pandas as pd
import pickle
import keras

"""### Model and Test Data"""

model = keras.models.load_model('/content/drive/MyDrive/Colab Notebooks/Customer Feedback Classification/absa/models/LSTM/Hyperparam Tuning/my_model0.001.h5')
test = pickle.load(open('/content/drive/MyDrive/Colab Notebooks/Customer Feedback Classification/absa/notebooks/pickled files/X_test.pkl', 'rb'))
test_text = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Customer Feedback Classification/absa/notebooks/dataset/test - test.csv')

"""### Prediction"""

test = pd.DataFrame(test)
label = []

for sen in range(len(test)):
  predict = model.predict(test.iloc[sen, :])
  label.append({'Negative': np.argmax(predict[:, 0]), 'Neutral': np.argmax(predict[:, 1]), 'Positive': np.argmax(predict[:, 2])})

label = pd.DataFrame(label).idxmax(axis=1)

test_text = pd.concat([test_text, label], axis=1)

test_text.rename(columns={0: 'label'}, inplace=True)
test_text.to_csv('/content/drive/MyDrive/Colab Notebooks/Customer Feedback Classification/absa/data/results/test.csv')

test_text