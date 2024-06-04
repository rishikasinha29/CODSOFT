import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

nltk.download('stopwords')
nltk.download('punkt')

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
try:
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

except FileNotFoundError:
  print("Error: Could not find pickle files. Please ensure they exist in the same directory.")
    

new_text1 = "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's,,,"

transformed_text = transform_text(new_text1)  # Assuming the function is defined

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
