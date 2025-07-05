from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    confidence = ""
    user_input = ""
    if request.method == 'POST':
        user_input = request.form['news']
        if user_input.strip() == "":
            prediction = "❗ Please enter some news text."
        else:
            vect = vectorizer.transform([user_input])
            result = model.predict(vect)[0]
            prob = model.predict_proba(vect)[0]
            prediction = "✅ Real News" if result == 1 else "❌ Fake News"
            confidence = f"Confidence: {round(max(prob) * 100, 2)}%"
    return render_template('index.html', prediction=prediction, confidence=confidence, user_input=user_input)


if __name__ == '__main__':
    app.run(debug=True)
