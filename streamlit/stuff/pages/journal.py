import streamlit as st
import pickle
import numpy as np
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


st.set_page_config(layout="wide")

markdown = """
Web App URL: <INSERT>

GitHub Repository: <INSERT>
"""

st.sidebar.title("New journal entry")
st.sidebar.info(markdown)
# logo = "IMAGE LINK"
# st.sidebar.image(logo)


# display app title & tagline
st.title("Mood Journal")
st.write("Tell us about your day and we'll tell you how you feel.")

# display about us section
st.header("About Us")
about_us_text = "We're doing a  project for AI4Good!"
st.write(about_us_text)
        


def load_emotion_model():
    """  load emotions model """
    
    with open('NAME OF PICKLE FILE', 'rb') as file:
        data = pickle.load(file)
    return data


def load_mi_model():
    """ load mental illnesses model """
    
    with open('mental_illness_rf_corpus.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


def predictor(): 
    """ predictor function takes text box as input and returns prediction(s) """

    txt = txt = st.text_area('Enter journal text here', '')

    # TOODO
    # need to return both emotions predictor as well as mental illness predictor IF mental illness prediction does not equal 'neither'

    return(prediction)

def clean_user_input_mi(text):
    """ clean user input for mental illness model """
    
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    more_stopwords = ['u', 'im', 'c']
    stop_words = stop_words + more_stopwords
    stemmer = nltk.SnowballStemmer("english")
    
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


def use_mi_model(text):
    """ use mental illness model & return results to user """
    if text != '':
        text = clean_user_input_mi(text)
        
        # use MI model
        loaded_model = load_mi_model()
        
        model = loaded_model['model']
        labels = preprocessing.LabelEncoder()
        labels = loaded_model['le_mental_illness']
        corpus = loaded_model['corpus']
        
        text = [text]

        # now vectorize
        cv = CountVectorizer(max_features=5000)
        cv.fit_transform(corpus)
        sentence = cv.transform(text)
    
        # make prediction
        y_pred = model.predict(sentence)
    
        # inverse transform of predictions
        y_actual = labels.inverse_transform(y_pred)
        
        # change to string & remove extra characters
        mental_illness = str(y_actual)
        mental_illness = mental_illness.replace("'","")
        mental_illness = mental_illness.replace("[","")
        mental_illness = mental_illness.replace("]","")
        
        # change adhd to ADHD
        if mental_illness == 'adhd':
            mental_illness = mental_illness.upper()
        
        return mental_illness   


#emotion_data = load_emotion_model()

#emotion_classifier  = data["model"]
#le_emotions = data["le_mental_illness"]

# def show_journal_page():
#     """ page display """    
    
        
#     # display journal section
#     if page == "Journal":
st.header("Journal")
st.subheader("Multi-class classification of mental illness from text input")

# collect journal entry from user
journal_entry = st.text_input("Journal Entry:")

# use mental illness model
mental_illness = use_mi_model(journal_entry)

st.header("Results")

# display emotion results to user
emotion = "anger"
st.write("You are displaying %s." %emotion)        

# display mental illness results to user
if mental_illness != 'neither':
    st.write("You are displaying symptoms of %s." % mental_illness)
    
    if mental_illness == 'depression':
        st.markdown('''
            Call 911 or other emergency services immediately if:
            
            - You or someone you know is thinking seriously of suicide or has recently tried suicide. Serious signs include these thoughts:
            - You have decided on how to kill yourself, such as with a weapon or medicines.
            - You have set a time and place to do it.
            - You think there is no other way to solve the problem or end the pain.
            - You feel you can't stop from hurting yourself or someone else.
            
            Keep the number for a suicide crisis centre on or near your phone. Go to the Canadian Association for Suicide Prevention web page at http://suicideprevention.ca/need-help to find a suicide crisis prevention centre in your area.
            
            Call a doctor now if:
            
            - You hear voices.
            - You have been thinking about death or suicide a lot, but you don't have a plan to harm yourself.
            - You are worried that your feelings of depression or thoughts of suicide aren't going away.
            
            https://myhealth.alberta.ca/health/pages/conditions.aspx?Hwid=hw30709
            Treatment for depression includes counselling, medicines, and lifestyle changes. Your treatment will depend on you and your symptoms. You and your health care team will work together to find the best treatment for you.
            
            - If you have moderate to severe symptoms, your doctor probably will suggest medicine or therapy or both.
            - If you are using medicine, your doctor may have you try different medicines or a combination of medicines.
            - You may need to go to the hospital if you show warning signs of suicide, such as having thoughts about harming yourself or another person, not being able to tell the difference between what is real and what is not (psychosis), or using a lot of alcohol or drugs.
            ''')        

        


            
            
        
        
             
        
    
        
    
    
