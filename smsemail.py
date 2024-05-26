import streamlit as st
import pickle
import string
import re
import spacy

# Load spaCy model and stop words
nlp = spacy.load('en_core_web_sm')
from spacy.lang.en.stop_words import STOP_WORDS

# Text transformation function
def transform_text(text):
    text = text.lower()
    text = [word.text for word in nlp(text)]
    y = []
    for i in text:
        if i.isalnum() and (i not in STOP_WORDS) and (i not in string.punctuation):
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        for j in nlp(i):
            y.append(j.lemma_)
    return " ".join(y)

# Load the trained vectorizers and models
sms_tfidf = pickle.load(open('Smsvectorizer.pkl', 'rb'))
sms_model = pickle.load(open('Smsmodel.pkl', 'rb'))
email_tfidf = pickle.load(open('Emailvectorizer.pkl', 'rb'))
email_model = pickle.load(open('emailmodel.pkl', 'rb'))

# Streamlit app title
st.title("Spam Classifier for SMS and Email")

# User input for message type and message content
message_type = st.radio("Select the type of message:", ('SMS', 'Email'))
input_message = st.text_area("Enter the message")

# Prediction button
if st.button('Predict'):

    # Preprocess the input message
    transformed_message = transform_text(input_message)
    
    # Vectorize and predict based on message type
    if message_type == 'SMS':
        vector_input = sms_tfidf.transform([transformed_message])
        model = sms_model
    else:
        vector_input = email_tfidf.transform([transformed_message])
        model = email_model

    # Ensure model is fitted
    if not hasattr(model, 'classes_'):
        st.error("Model is not fitted. Please train the model first.")
    else:
        # Predict and display results
        prob_ham, prob_spam = model.predict_proba(vector_input)[0]  # Probabilities of being ham and spam
        result = model.predict(vector_input)[0]
        
        st.write(f"Probability of Ham: {prob_ham * 100:.2f}%")
        st.write(f"Probability of Spam: {prob_spam * 100:.2f}%")

        st.title("Probabilities of being spam and ham")
        # Visualization
        st.progress(int(prob_spam * 100))  # Show a progress bar for the spam probability
        
        # Optional: bar chart for a more detailed visualization
        st.bar_chart({"Probabilities": {"Ham": prob_ham, "Spam": prob_spam}})
