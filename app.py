from flask import Flask, render_template, request
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

# Initialize Flask application
app = Flask(__name__)

# Load the trained model and vectorizer from pickle files
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
with open('preprocessor.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def preprocess_text(text):
    """
    Preprocess the input text by performing lowercase conversion,
    tokenization, and stopword removal
    Args:
        text (str): Input text to be preprocessed
    Returns:
        str: Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text into individual words
    tokens = word_tokenize(text)
    # Remove common English stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back to string with spaces
    return ' '.join(tokens)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get message from the form submission
        message = request.form['message']
        # Preprocess the input text
        processed_message = preprocess_text(message)
        # Transform text using loaded vectorizer for model input
        message_vectorized = vectorizer.transform([processed_message])
        # Make prediction using the loaded model
        prediction = model.predict(message_vectorized)
        
        # Convert prediction to human-readable result
        result = "Spam" if prediction[0] == 1 else "Not Spam"
        return render_template('index.html', prediction=result, message=message)

if __name__ == '__main__':
    # Download required NLTK data before running the app
    nltk.download('punkt')
    nltk.download('stopwords')
    # Run the Flask application in debug mode
    app.run(debug=True)