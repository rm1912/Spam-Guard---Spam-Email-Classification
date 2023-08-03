from flask import Flask, render_template, request
import pickle
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

with open('stopwords.txt', 'r') as file:
    custom_stop_words = set(word.strip() for word in file)

def clean_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    lowercased_text = cleaned_text.lower()
    return ' '.join([word for word in lowercased_text.split() if word not in custom_stop_words])

with open('spam_classifier.pkl', 'rb') as f:
    final_model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        email = request.form['email']
        cleaned_email = clean_text(email)
        
        features = vectorizer.transform([cleaned_email])
        prediction = final_model.predict(features)

        result = "spam" if prediction[0] == 1 else "not spam"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
