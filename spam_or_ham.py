import pandas as pd
import re
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from bs4 import BeautifulSoup 

dataset = pd.read_csv('emails.csv')

missing_values = dataset['email'].isna().sum()
print("Number of missing values in 'email' column:", missing_values)

dataset.dropna(subset=['email'], inplace=True)

dataset['cleaned_email'] = dataset['email']

def clean_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator=' ')
    return text

dataset['cleaned_email'] = dataset['cleaned_email'].apply(clean_html)


def remove_special_characters(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    return cleaned_text

def convert_to_lowercase(text):
    lowercased_text = text.lower()
    return lowercased_text

with open('stopwords.txt', 'r') as file:
    custom_stop_words = set(word.strip() for word in file)

def remove_stop_words(text):
    words = text.split()
    filtered_words = [word for word in words if word not in custom_stop_words]
    return ' '.join(filtered_words)  

dataset['cleaned_email'] = dataset['cleaned_email'].apply(remove_special_characters)
dataset['cleaned_email'] = dataset['cleaned_email'].apply(convert_to_lowercase)
dataset['cleaned_email'] = dataset['cleaned_email'].apply(remove_stop_words)


dataset.to_csv('emails_cleaned.csv', index=False) 

X = dataset['cleaned_email']
y = dataset['label']

print(dataset.columns)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()

X_train_features = vectorizer.fit_transform(X_train)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

X_val_features = vectorizer.transform(X_val)

classifier = MultinomialNB()

classifier.fit(X_train_features, y_train)

y_pred_val = classifier.predict(X_val_features)

accuracy_val = accuracy_score(y_val, y_pred_val)
print("Accuracy on validation data:", accuracy_val)

with open('spam_classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)

param_grid = {
    'alpha': [0.1, 1, 10],
    'fit_prior': [True, False]
}

model = MultinomialNB()

grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=5)

grid_search.fit(X_train_features, y_train)

best_params_grid = grid_search.best_params_

param_dist = {
    'alpha': [0.1, 1, 10],
    'fit_prior': [True, False]
}

model = MultinomialNB()

random_search = RandomizedSearchCV(model, param_dist, n_iter=6, scoring='accuracy', cv=5)

random_search.fit(X_train_features, y_train)

best_params_random = random_search.best_params_

best_params = best_params_grid if grid_search.best_score_ > random_search.best_score_ else best_params_random

final_model = MultinomialNB(alpha=best_params['alpha'], fit_prior=best_params['fit_prior'])
final_model.fit(X_train_features, y_train)

X_test = np.array(dataset['cleaned_email'])
y_test = np.array(dataset['label'])

X_test_features = vectorizer.transform(X_test)

y_pred_test = final_model.predict(X_test_features)

y_test_binary = np.where(y_test == 'ham', 0, 1)
y_pred_test_binary = np.where(y_pred_test == 'ham', 0, 1)

accuracy_test = accuracy_score(y_test_binary, y_pred_test_binary)
print("Accuracy on test data:", accuracy_test)

precision_test = precision_score(y_test_binary, y_pred_test_binary)
recall_test = recall_score(y_test_binary, y_pred_test_binary)
f1_test = f1_score(y_test_binary, y_pred_test_binary)

print("Precision on test data:", precision_test)
print("Recall on test data:", recall_test)
print("F1-score on test data:", f1_test)
