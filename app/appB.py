import streamlit as st
import pickle
import string
import re
from bnlp import BengaliCorpus as corpus
from banglakit.lemmatizer import BengaliLemmatizer

# Initialize the Bengali lemmatizer
try:
    lemmatizer = BengaliLemmatizer()
except UnicodeDecodeError as e:
    st.error(f"UnicodeDecodeError: {e}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while initializing the Bengali lemmatizer: {e}")
    st.stop()

def transform(text):
    try:
        # Get Bengali punctuations from the corpus
        punn = corpus.punctuations
        
        # Define regex patterns for emojis and English characters
        emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            u"\u00C0-\u017F"          # latin
                            u"\u2000-\u206F"          # generalPunctuations
                            "]+", flags=re.UNICODE)
        english_pattern = re.compile('[a-zA-Z0-9]+', flags=re.I)

        # Remove emojis, symbols, flags, English characters, and punctuations
        text = emoji_pattern.sub(r'', text)
        text = english_pattern.sub(r'', text)
        no_punct = ''.join([char for char in text if char not in punn])
        
        # Replace specified characters with spaces
        specified_chars = '''````£|¢|Ñ+-*/=EROero৳০১২৩৪৫৬৭৮৯012–34567•89।!()-[]{};:'"“\’,<>./?@#$%^&*_~‘—॥”‰🤣🥰🥚🥭⚽️🥰🥳✌�￰৷'''
        for char in specified_chars:
            no_punct = no_punct.replace(char, ' ')
        
        # Split the text at hyphens ('-')
        parts = no_punct.split('-')
        
        # Remove newline characters and replace them with spaces
        no_newlines = ' '.join(parts)
        
        # Remove specified Bengali characters
        specified_bengali_chars = ['লি', 'এ']
        for char in specified_bengali_chars:
            no_newlines = no_newlines.replace(char, ' ')
        
        # Get Bengali stopwords from the corpus
        stopwords_list = corpus.stopwords
        
        # Remove stopwords
        no_stopwords = ' '.join([word for word in no_newlines.split() if word not in stopwords_list])
        
        # Lemmatize each word in the text
        lemmatized_words = [lemmatizer.lemmatize(word) for word in no_stopwords.split()]
        
        # Join the lemmatized words back into a string
        lemmatized_text = ' '.join(lemmatized_words)
        
        return lemmatized_text
    except Exception as e:
        st.error(f"An error occurred during text transformation: {e}")
        st.stop()


# Load the trained vectorizer and model
try:
    tfidf = pickle.load(open('vectorizer1.pkl', 'rb'))
    model = pickle.load(open('model1.pkl', 'rb'))
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

st.title("Email/SMS Spam Classifier App")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    try:
        # 1. preprocess
        transformed_sms = transform(input_sms)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        
        # Ensure model is fitted
        if not hasattr(model, 'classes_'):
            st.error("Model is not fitted. Please train the model first.")
        else:
            # 3. predict
            prob_ham, prob_spam = model.predict_proba(vector_input)[0]  # Probabilities of being ham and spam
            result = model.predict(vector_input)[0]
            
            if result == 'ham':
                st.write("This message is likely not spam (ham).")
            else:
                st.write("This message is likely spam.")
            
            # Display confidence levels
            st.write(f"Probability of Ham: {prob_ham*100:.2f}%")
            st.write(f"Probability of Spam: {prob_spam*100:.2f}%")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
