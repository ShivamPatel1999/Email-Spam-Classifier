from flask import Flask,render_template,request
from model import word_dict
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('new html.html')


@app.route('/predict', methods=["GET", "POST"])
def predict():
    email = request.form.get('email')
    sample = []
    for i in word_dict:
        sample.append(email.split(" ").count(i[0]))

    sample = np.array(sample)
    result = model.predict(sample.reshape(1, 3000))

    if result == 1:
        return render_template('new html.html', label=1)
    else:
        return render_template('new html.html', label=-1)


if __name__ == '__main__':
    app.run(debug=True)