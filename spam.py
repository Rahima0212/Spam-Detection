import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import joblib

# Download NLTK stopwords
import os
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_path)

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))


# Constants
FILE_PATH = 'spam.csv'
MODEL_PATH = 'spam_model.joblib'

# Load and clean dataset
df = pd.read_csv(FILE_PATH, encoding='latin-1')
df = df[['v2', 'v1']]
df.columns = ['message', 'label']
df = df.dropna()
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Enhanced preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    
    text = text.lower()
    has_url = int('http' in text or 'www' in text)
    has_currency = int(any(c in text for c in ['$', '€', '£']))
    has_number = int(any(c.isdigit() for c in text))
    
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    words = text.split()

    important_words = {
        'urgent', 'verify', 'free', 'winner', 'win', 'click', 'account',
        'limited', 'prize', 'congratulations', 'claim', 'offer', 'discount',
        'reward', 'guaranteed', 'exclusive', 'riskfree', 'cash', 'bonus',
        'subscribe', 'password', 'deal', 'million', 'lottery', 'security',
        'payment', 'credit', 'loan', 'expire', 'instant', 'risk', 'access',
        'membership', 'confidential', 'important', 'action', 'required'
    }

    filtered_words = [
        word for word in words if word not in stop_words or word in important_words
    ]
    processed_text = ' '.join(filtered_words)
    return f'{processed_text} url_{has_url} currency_{has_currency} number_{has_number}'

# Apply preprocessing
df['processed_message'] = df['message'].apply(preprocess_text)
df = df.dropna(subset=['label'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_message'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
)

# Create pipeline and train model
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2),
    MultinomialNB(alpha=0.1)
)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f'Cross-validation scores: {cv_scores}')
print(f'Average CV score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})')

# Evaluate on test set
y_pred = model.predict(X_test)
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Interactive prediction
def predict_spam():
    # Get the class labels from the model
    # Assuming the model is a pipeline where the last step is the classifier
    classifier = model.steps[-1][1] # Get the actual classifier from the pipeline
    class_labels = classifier.classes_ # Get the class labels in the order they appear in predict_proba

    while True:
        user_input = input('\nEnter a message (or "quit" to exit): ')
        if user_input.lower() == 'quit':
            break
        processed_input = preprocess_text(user_input)
        prediction_label = model.predict([processed_input])[0] # This is the class label (e.g., 'ham', 'spam')
        probability = model.predict_proba([processed_input])[0] # This is the array of probabilities

        # Find the index of the predicted label in the class labels array
        try:
            prediction_index = list(class_labels).index(prediction_label)
            confidence = probability[prediction_index] # Use the integer index to get the confidence
        except ValueError:
            # Handle cases where the predicted label is not found in class_labels (unlikely with this model)
            print(f"Warning: Predicted label '{prediction_label}' not found in model classes.")
            confidence = 0.0 # Or handle as appropriate

        result = 'Spam' if prediction_label == 1 else 'Not Spam' # Check against the label string
        print(f'Prediction: {result} (Confidence: {confidence:.2f})')

if __name__ == '__main__':
    predict_spam()

