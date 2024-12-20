# Spam Email and SMS Detector

This project is a machine learning-based application designed to classify email and SMS messages as spam or not spam. It utilizes natural language processing (NLP) techniques and various classification algorithms to achieve accurate predictions.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Saved Models](#saved-models)
- [Technologies Used](#technologies-used)
- [Contact](#contact)

## Features

- **Spam Detection**: Classifies messages as spam or not spam with high accuracy.
- **Text Preprocessing**: Performs text cleaning, tokenization, and vectorization to prepare data for modeling.
- **Machine Learning Models**: Employs algorithms such as Naive Bayes and Logistic Regression.
- **Web Application**: Provides a user-friendly interface built with Flask for message classification.
- **Pre-trained Models**: Includes saved models and preprocessors for immediate use.

## Project Structure

```
Spam_Email_Spam_Detector/
├── spam.csv             # Dataset containing labeled SMS data
├── main.ipynb           # Jupyter Notebook for model training and testing
├── app.py               # Flask web application for spam detection
├── preprocessor.pkl     # Saved preprocessor for text data
├── model.pkl            # Saved machine learning model
├── templates/           # HTML templates for the web application
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/samp1012/Spam_Email_Detector.git
   cd Spam_Email_Detector
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the web application**:
   ```bash
   python app.py
   ```

2. **Access the application**:
   Open your web browser and navigate to `http://127.0.0.1:5000/`.

3. **Classify a message**:
   - Enter the email or SMS text into the input field.
   - Click the "Predict" button to determine if the message is spam.

## Model Training

To train the model on new data or adjust parameters:

1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook main.ipynb
   ```

2. **Follow the steps in the notebook** to preprocess data, train the model, and evaluate performance.

3. **Save the trained model and preprocessor**:
   - Update the `model.pkl` and `preprocessor.pkl` files with the new models.


## Saved Models

The project includes pre-trained files for quick predictions:
- **preprocessor.pkl**: Text preprocessing pipeline.
- **model.pkl**: Machine learning model for spam detection.

These files are used by the `app.py` script to process input data and generate predictions.


## Technologies Used
- **Python**
- **Flask**: Web framework for deployment
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning models
- **Numpy**: Numerical operations
- **Matplotlib/Seaborn**: Data visualization


## Contact
For queries or suggestions:
- **GitHub**: [samp1012](https://github.com/samp1012)
- **Email**: samparkadas@gmail.com
- **Linked In**: [Samparka Das](https://www.linkedin.com/in/samparka-das-b4317726b/)
