import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('spam-sms/vectorizer.pkl','rb'))
model = pickle.load(open('spam-sms/model.pkl','rb'))

st.markdown(
    """
    <style>
    .stApp {
        background-color: #308ca1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Spam SMS Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Preprocess the input message
    transformed_sms = transform_text(input_sms)
    # Vectorize the preprocessed message
    vector_input = tfidf.transform([transformed_sms])
    # Predict the result
    result = model.predict(vector_input)[0]
    # Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
