import re
import pandas as pd
import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#references:
# https://github.com/sebtheiler/tutorials/blob/main/twitter-sentiment/main.py

# set page title
#st.title('MI Analysis')
apptitle = st.container()
about_us = st.container()

with apptitle:
    st.title("Mood Journal")
    st.text("Tell us about your day and we'll tell you how you feel :)")
    
with about_us:
    st.header("About Us")
    about_us_text = "We're doing a  project for AI4Good!"
    st.write(about_us_text)

# load model
loaded_model = pickle.load(open('bert_emotions.pkl', 'rb'))
model = loaded_model['model']
labels = preprocessing.LabelEncoder()
labels = loaded_model['le_emotions']
corpus = loaded_model['corpus']


# clean user input as was done to get text_clean in training data
nltk.download('stopwords')
stop_words = stopwords.words('english')
more_stopwords = ['u', 'im', 'c']
stop_words = stop_words + more_stopwords
stemmer = nltk.SnowballStemmer("english")


def clean_user_input(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    #remove stopwords
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    #stemming
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    return text



st.subheader('Multi-class classification of mental illness from text input')
checkin =  st.container()

with checkin:
    st.header("Check In")
    st.text("Answer some questions here like it's your diary. Press enter when you're done and scroll down for your results!")
    question1 =  "What did you do today?"
    #answer1 = checkin.text_input(question1)
    # get text input and detect + classify mental illness
    user_input = st.text_input(question1)

if user_input != '':

    user_input = clean_user_input(user_input)

    user_input = [user_input]

    # now vectorize
    cv = CountVectorizer(max_features=5000)
    cv.fit_transform(corpus)
    sentence = cv.transform(user_input)

    # make prediction
    y_pred = model.predict(sentence)


    # inverse transform of predictions
    y_act = labels.inverse_transform(y_pred)

    results = st.container()
    with results:
        st.header("Results")
        st.write('It seems like you are showing symptoms of', y_act)


