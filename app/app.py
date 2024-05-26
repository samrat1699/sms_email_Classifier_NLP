import streamlit as st
import pickle
import string
import re
import spacy


nlp = spacy.load('en_core_web_sm')
from spacy.lang.en.stop_words import STOP_WORDS

def transform_text(text):
    text = text.lower()
    text = [word.text for word in nlp(text)]
    y=[]
    for i in text:
        if i.isalnum() and (i not in STOP_WORDS) and (i not in string.punctuation):
            y.append(i)

    text=y[:]
    y.clear()
    for i in text:
        for j in nlp(i):
            y.append(j.lemma_)
    return " ".join(y)

# Load the trained vectorizer and model
tfidf = pickle.load(open('Smsvectorizer.pkl', 'rb'))
model = pickle.load(open('Smsmodel.pkl', 'rb'))

st.title("Email/SMS Spam Classifier App")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    
    # Ensure model is fitted
    if not hasattr(model, 'classes_'):
        st.error("Model is not fitted. Please train the model first.")
    else:
        # 3. predict
        prob_ham, prob_spam = model.predict_proba(vector_input)[0]  # Probabilities of being ham and spam
        result = model.predict(vector_input)[0]
        
        
        st.write(f"Probability of Ham: {prob_ham*100:.2f}%")
        st.write(f"Probability of Spam: {prob_spam*100:.2f}%")
