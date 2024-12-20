# Email and SMS Spam Detector

This repository contains a machine learning-based **Email and SMS Spam Detector** that identifies whether a given text message (email or SMS) is spam or not. The project leverages natural language processing (NLP) techniques and classification models to make predictions. It also includes a simple web application for user interaction.

---

## Features
- **Spam Detection**: Accurately classify messages as spam or not.
- **Text Preprocessing**: Includes text cleaning, tokenization, and vectorization.
- **Machine Learning Models**: Utilizes classification algorithms such as Naive Bayes, Logistic Regression, or other customizable models.
- **Web Application**: Flask-based web app to interact with the model and predict messages.
- **Pre-trained Models**: Includes saved model and preprocessor for quick predictions.

---

## Project Structure

```plaintext
Email_SMS_Spam_Detector/
├── spam.csv             # Dataset containing labeled SMS data
├── main.ipynb           # Jupyter Notebook for model training and testing
├── app.py               # Flask web application for spam detection
├── preprocessor.pkl     # Saved preprocessor for text data
├── model.pkl            # Saved machine learning model
├── templates/           # HTML templates for the web application
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## Prerequisites

Before running this project, ensure you have the following installed:
- **Python 3.8+**
- **pip** (Python package manager)

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/samp1012/Email_SMS_Spam_Detector.git
cd Email_SMS_Spam_Detector
```

### 2. Dataset
- The dataset file `spam.csv` is already included in the repository.
- It contains labeled SMS data with the following structure:

| v1       | v2                               |
|----------|----------------------------------|
| ham      | "Go until jurong point..."       |
| spam     | "Free entry in 2 a weekly comp"  |

- `v1`: The label column (ham or spam).
- `v2`: The text message content.

### 3. Run the Web Application

The project includes a Flask web application for spam detection.

Run the application with:

```bash
python app.py
```

Access the web app in your browser at:
```
http://127.0.0.1:5000
```

You can input a message, and the model will predict whether it is spam or not.

---

## Web App Templates
The web app uses basic HTML templates stored in the `templates/` folder:
- `index.html`: Main page to input text and view predictions.

---

## Saved Models
The project includes pre-trained files for quick predictions:
- **preprocessor.pkl**: Text preprocessing pipeline.
- **model.pkl**: Machine learning model for spam detection.

These files are used by the `app.py` script to process input data and generate predictions.

---

## Results
Evaluation metrics such as accuracy, precision, recall, and F1-score can be evaluated in the notebook. Visualizations are also generated for better insights.

Example Output:
```
Accuracy: 98.5%
Precision: 95%
Recall: 92%
F1-Score: 93%
```

---

## Technologies Used
- **Python**
- **Flask**: Web framework for deployment
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning models
- **Numpy**: Numerical operations
- **Matplotlib/Seaborn**: Data visualization

---

## Contact
For queries or suggestions:
- **GitHub**: [samp1012](https://github.com/samp1012)
- **Email**: samparkadas@gmail.com
- **Linked In**: [Samparka Das](https://www.linkedin.com/in/samparka-das-b4317726b/)
