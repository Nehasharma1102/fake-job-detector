from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load model & vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']

    # Convert text
    text_vectorized = vectorizer.transform([text])

    # ML Prediction
    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized)[0]

    fake_score = round(probability[1] * 100, 2)
    real_score = round(probability[0] * 100, 2)

    text_lower = text.lower()

    # 🔥 RULE-BASED BOOST (important)
    is_fake_rule = False
    reasons = []

    if "registration fee" in text_lower:
        is_fake_rule = True
        reasons.append("Asking for money")

    if "earn money" in text_lower or "earn" in text_lower:
        is_fake_rule = True
        reasons.append("Unrealistic earning claims")

    if "no interview" in text_lower:
        is_fake_rule = True
        reasons.append("No interview process")

    if "urgent hiring" in text_lower:
        is_fake_rule = True
        reasons.append("Creates urgency")

    if "gmail.com" in text_lower:
        is_fake_rule = True
        reasons.append("Unprofessional email")

    if "work from home" in text_lower and "salary" in text_lower:
        is_fake_rule = True
        reasons.append("Too good to be true offer")

    # 🔥 FINAL DECISION
    if is_fake_rule:
        result = "FAKE"
        score = max(fake_score, 80)  # boost confidence
    else:
        if fake_score > 40:   # lowered threshold
            result = "FAKE"
            score = fake_score
        else:
            result = "REAL"
            score = real_score

    return jsonify({
        'result': result,
        'confidence': round(score, 2),
        'reasons': reasons
    })

if __name__ == '__main__':
    app.run(debug=True)