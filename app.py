from flask import Flask, request, jsonify, render_template
import pickle
import os
import gdown

# Initialize the Flask app
app = Flask(__name__)

# Google Drive URLs for models and vectorizer
logistic_model_url = "https://drive.google.com/uc?export=download&id=1se-oijE0oVTdrDn9nzLw8UriwI3uUJWs"
naive_bayes_model_url = "https://drive.google.com/uc?export=download&id=1FxWHsYiG8sDFXdYz__e4n1oyl0GMeU7m"
vectorizer_url = "https://drive.google.com/uc?export=download&id=12UkcLB_7R3kc-YYQghSGsXX1k_wi2E36"

# Local file paths for downloaded files
logistic_model_path = 'logistic_regression_model.pkl'
naive_bayes_model_path = 'naive_bayes_model.pkl'
vectorizer_path = 'count_vectorizer.pkl'

# Function to download files from Google Drive
def download_file(url, output):
    gdown.download(url, output, quiet=False)

# Download files if they do not exist
if not os.path.exists(logistic_model_path):
    download_file(logistic_model_url, logistic_model_path)

if not os.path.exists(naive_bayes_model_path):
    download_file(naive_bayes_model_url, naive_bayes_model_path)

if not os.path.exists(vectorizer_path):
    download_file(vectorizer_url, vectorizer_path)

# Load models and vectorizer
try:
    logistic_model = pickle.load(open(logistic_model_path, "rb"))
    naive_bayes_model = pickle.load(open(naive_bayes_model_path, "rb"))
    vectorizer = pickle.load(open(vectorizer_path, "rb"))
except Exception as e:
    raise Exception(f"Error loading models or vectorizer: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text')
    model_choice = data.get('model')

    if not text:
        return jsonify({'error': 'No text provided'}), 400
    if not model_choice:
        return jsonify({'error': 'No model selected'}), 400

    transformed_text = vectorizer.transform([text])

    if model_choice == 'logistic':
        prediction = logistic_model.predict(transformed_text)
        model_used = "Logistic Regression"
    elif model_choice == 'naive_bayes':
        prediction = naive_bayes_model.predict(transformed_text)
        model_used = "Naive Bayes"
    else:
        return jsonify({'error': 'Invalid model selected'}), 400

    sentiment = "Positive" if prediction == 1 else "Negative"
    return jsonify({'sentiment': sentiment, 'model_used': model_used})

if __name__ == '__main__':
    app.run(debug=True)
