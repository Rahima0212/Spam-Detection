# 📧 Spam Message Classifier using Machine Learning

A simple and effective spam message detection system built using Python, Scikit-learn, and Streamlit. This project uses natural language processing (NLP) techniques to classify text messages as **Spam** or **Not Spam**.

---
## 🛠️ Features
- Text preprocessing (lowercasing, punctuation & stopword removal, stemming)

- TF-IDF vectorization

- Multinomial Naive Bayes classifier

- Model training & saving with joblib

- Interactive Streamlit web app

## 🧠 How It Works
- Data Preprocessing: Cleans text (removes punctuation, stopwords, etc.)

- Vectorization: Transforms text into numerical format using TF-IDF

- Model: A Naive Bayes classifier is trained to detect spam

- Web App: Streamlit provides a simple interface to test messages
  

📦 spam-message-classifier/
│
├── spam.py              # Model training script
├── spam_app.py          # Streamlit UI for prediction
├── spam_model.joblib    # Saved trained model
├── spam.csv             # Dataset (SMS messages)
├── README.md            # You're here!


## 🚀 Demo

Try it locally:
```bash
streamlit run spam_app.py




