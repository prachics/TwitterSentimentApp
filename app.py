from flask import Flask, request, jsonify, render_template
import pickle
import os

# Initialize the Flask app
app = Flask(__name__)

# Define the paths for the models and vectorizer
logistic_model_path = os.getenv("LOGISTIC_MODEL_PATH", "/home/chinmay/Desktop/Projects Resume/Twitter sentiment/logistic_regression_model.pkl")
naive_bayes_model_path = os.getenv("NAIVE_BAYES_MODEL_PATH", "/home/chinmay/Desktop/Projects Resume/Twitter sentiment/naive_bayes_model.pkl")
vectorizer_path = os.getenv("VECTORIZER_PATH", "/home/chinmay/Desktop/Projects Resume/Twitter sentiment/count_vectorizer.pkl")

# Load both models and vectorizer
try:
    logistic_model = pickle.load(open(logistic_model_path, "rb"))
    naive_bayes_model = pickle.load(open(naive_bayes_model_path, "rb"))
    vectorizer = pickle.load(open(vectorizer_path, "rb"))
except Exception as e:
    raise Exception(f"Error loading models or vectorizer: {str(e)}")

@app.route('/')
def index():
    # Serve the main page with a form for text input and model selection
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from POST request
    data = request.get_json()
    text = data.get('text')
    model_choice = data.get('model')

    # Validate input data
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    if not model_choice:
        return jsonify({'error': 'No model selected'}), 400

    # Preprocess the text input
    transformed_text = vectorizer.transform([text])

    # Choose the model based on the request
    if model_choice == 'logistic':
        prediction = logistic_model.predict(transformed_text)
        model_used = "Logistic Regression"
    elif model_choice == 'naive_bayes':
        prediction = naive_bayes_model.predict(transformed_text)
        model_used = "Naive Bayes"
    else:
        return jsonify({'error': 'Invalid model selected'}), 400

    # Determine the sentiment from the prediction
    sentiment = "Positive" if prediction == 1 else "Negative"

    # Return the prediction as a JSON response
    return jsonify({'sentiment': sentiment, 'model_used': model_used})

if __name__ == '__main__':
    # Run the app
    app.run(debug=True)
