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

# %%capture
# pip install -U sentence-transformers
# pip install datasets

import transformers
from transformers import BertTokenizer
from transformers import TrainingArguments, Trainer
from transformers import EvalPrediction
import torch



st.set_page_config(layout="wide")

markdown = """
Web App URL: <INSERT>

GitHub Repository: <INSERT>
"""

st.sidebar.title("Journal")
st.sidebar.info(markdown)
# logo = "IMAGE LINK"
# st.sidebar.image(logo)


# display app title & tagline
st.title("MoodRing")
st.write("Tell us about your day and we'll help tell you figure out how you feel.")

# display about us section
st.subheader("About Us")
about_us_text = "We're doing a  project for AI4Good!"
st.write(about_us_text)
        



##Helper functions + global vars

# @st.cache(suppress_st_warning=True)
def load_mi_model():
    """ load mental illnesses model """
    
    with open('mental_illness_rf_final.pkl', 'rb') as file:
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



#labels + dictionaries
labels = ['anger', 'fear', 'joy', 'love','sadness','surprise','thankfulness','disgust','guilt']
id2label = {idx:label for idx, label in enumerate(labels)}

    

# @st.cache(suppress_st_warning=True)
def load_emotion_model():

    from transformers import BertForSequenceClassification
    import os
    output_dir = os.getcwd()
    model = BertForSequenceClassification.from_pretrained(output_dir)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


    test_args = TrainingArguments(
        output_dir = output_dir,
        do_train = False,
        do_predict = True,
        per_device_eval_batch_size = 8,   
        dataloader_drop_last = False    
    )

    # init trainer
    trainer = Trainer(
                  model = model, 
                  args = test_args) 

    return tokenizer, model, trainer



def get_emotion(text, tokenizer, model, trainer):
    # text = "This sucks and I'm grouchy and upset"
    encoding = tokenizer(text, return_tensors="pt")
    encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}
    outputs = trainer.model(**encoding)
    logits = outputs.logits
    # apply sigmoid + threshold
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.5)] = 1
    # turn predicted id's into actual label names
    predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
    # st.write(predicted_labels)
    return predicted_labels




# display mental illness results to user
def display_mi_results(mental_illness):
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








#################user display#########################

st.subheader("New Journal Entry:")

tokenizer, model, trainer = load_emotion_model()
load_mi_model()


# collect journal entry from user
journal_entry = st.text_input("Enter text here:")


if journal_entry != '':

    emotions = get_emotion(journal_entry, tokenizer, model, trainer)

    if len(emotions) > 0:
        for i in range(len(emotions)):
            temp = emotions[i]
            st.write("You are displaying %s." %emotions[i])
            st.write("\n")         


    mental_illness = use_mi_model(journal_entry)
    display_mi_results(mental_illness)

    st.header("Results")


















            


            
            
        
        
             
        
    
        
    
    
