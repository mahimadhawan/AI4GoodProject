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
st.title('MI Analysis')


# load model
loaded_model = pickle.load(open('mental_illness_rf_corpus.pkl', 'rb'))
model = loaded_model['model']
labels = preprocessing.LabelEncoder()
labels = loaded_model['le_mental_illness']
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

# get text input and detect + classify mental illness
user_input = st.text_input('Journal entry:')

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


    st.write('prediction is ', y_act)


