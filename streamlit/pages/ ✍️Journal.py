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
MoodRing is a mood journal web app that helps track & detect emotions and flag mental health symptoms based on text. This project was done as part of the 2022 AI4Good Lab.
"""

st.sidebar.title("About")
st.sidebar.info(markdown)
logo = "https://github.com/mahimadhawan/AI4GoodProject/blob/main/logo.png?raw=true"
st.sidebar.image(logo, use_column_width='always')

# display app title & tagline
st.header("Sarah's Journal")
# st.write("What's on your mind?")


# @st.cache(suppress_st_warning=True)
def load_mi_model():
    """ load mental illnesses model """
    
    with open('mental_illness_rf_final.pkl', 'rb') as file:
        data = pickle.load(file)
    return data



# preprocessing functions for removing characters etc from: https://www.kaggle.com/code/oknashar/emotion-detection-deep-learning
def remove_hashtags(text):
    text = re.sub(r'@\w+', '', text)
    return text
def remove_emojis(text):
    text = [x for x in text.split(' ') if x.isalpha()]
    text = ' '.join(text)
    return text
def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def remove_urls(text):
    text = re.sub(r'http\S+', '', text)
    return text

def preprocess(text):
    text = remove_hashtags(text)
    text = remove_emoji(text)
    text = remove_urls(text)
    return text



# this function is from https://www.kaggle.com/code/andreshg/nlp-glove-bert-tf-idf-lstm-explained
def clean_user_input(text):
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
    
    return preprocess(text)






def get_mental_illness(text):
    """ use mental illness model & return results to user """
    if text != '':
        text = clean_user_input(text)
        
        # load model
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


def display_mi_results(user_mi):
    if user_mi == 'depression':
        st.markdown('''
            Your journal entry indiciates you might be experiencing symptoms of depression. You may want to discuss this with a mental health professional such as a physician or counselor.
            If you’re in crisis or having an emergency, please call your doctor or 911 immediately.  
            If you’re experiencing suicidal ideation, call 833-456-4566 anytime to speak to someone at the Canada Suicide Prevention Service. 
            If you’re experiencing any of the above and located outside Canada, please call your local emergency line.

            If you're not experiencing any of the above, there are other resources and information about depression you may find helpful:


            ''')    

    elif user_mi=='anxiety':
        st.markdown('''
            Your journal entry indiciates you might be experiencing symptoms of anxiety. You may want to discuss this with a mental health professional such as a physician or counselor.

            Other resources that you may find helpful include:


            ''')    

    elif user_mi=='ADHD':
        st.markdown('''
            Your journal entry indiciates you might be experiencing symptoms of ADHD. You may want to discuss this with a mental health professional such as a physician or counselor.

            Other resources that you may find helpful include:


            ''')  


#################user display#########################

st.text("")
st.text("")
st.text("")
# st.text("")

st.subheader("What's on your mind?")

tokenizer, model, trainer = load_emotion_model()
load_mi_model()


# collect journal entry from user
journal_entry = st.text_input("Enter text here:")


negative_emotions = ['anger', 'fear', 'disgust', 'guilt', 'sadness']
my_neg_emotions = []

if journal_entry != '':

    user_emotions = get_emotion(journal_entry, tokenizer, model, trainer)
    user_mi = get_mental_illness(journal_entry)

    if len(user_emotions) > 0:
        st.write("You seem to be experiencing the following emotions today: ")
        for i in range(len(user_emotions)):
            temp = user_emotions[i]
            st.write('-', temp)
            if temp in negative_emotions:
                my_neg_emotions.append(temp)


    if len(my_neg_emotions) > 0:
         st.markdown('''
            Based on your journal it seems you're feeling some negative emotions today. These feelings are unpleasant but also disruptive.

            It's important to note that everyone experiences negative emotions sometimes, even if they don't have a mental illness. Understanding and accepting your emotions can help manage these feelings.

            Some coping strategies for when you're feeling negative emotions are:



            ''')

    if user_mi != 'neither':
        display_mi_results(user_mi) 

    























            


            
            
        
        
             
        
    
        
    
    
