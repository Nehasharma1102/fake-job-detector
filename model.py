import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

df = pd.read_csv('fake_job_postings.csv')
df = df[['title', 'location', 'description', 'requirements', 'fraudulent']].fillna('')
df['text'] = df['title'] + ' ' + df['location'] + ' ' + df['description'] + ' ' + df['requirements']

X = df['text']
y = df['fraudulent']

vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {acc * 100:.2f}%")

pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
print("Model saved!")