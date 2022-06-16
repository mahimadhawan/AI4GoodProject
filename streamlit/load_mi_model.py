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
        if y_act == ['depression']:
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
        if y_act == ['anxiety']:
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

                                    Feeling worried or nervous is a normal part of everyday life. Everyone frets or feels anxious from time to time. Mild to moderate anxiety can help you focus your attention, energy, and motivation. If anxiety is severe, you may have feelings of helplessness, confusion, and extreme worry that are out of proportion with the actual seriousness or likelihood of the feared event. Overwhelming anxiety that interferes with daily life is not normal. This type of anxiety may be a symptom of generalized anxiety disorder, or it may be a symptom of another problem, such as depression.
                                    Anxiety can cause physical and emotional symptoms. A specific situation or fear can cause some or all of these symptoms for a short time. When the situation passes, the symptoms usually go away.
                                    Physical symptoms of anxiety include:
                                        Trembling, twitching, or shaking.
                                        Feeling of fullness in the throat or chest.
                                        Breathlessness or rapid heartbeat.
                                        Light-headedness or dizziness.
                                        Sweating or cold, clammy hands.
                                        Feeling jumpy.
                                        Muscle tension, aches, or soreness (myalgias).
                                        Extreme tiredness.
                                    Sleep problems, such as the inability to fall asleep or stay asleep, early waking, or restlessness (not feeling rested when you wake up).
                                    Anxiety affects the part of the brain that helps control how you communicate. This makes it harder to express yourself creatively or function effectively in relationships. Emotional symptoms of anxiety include:
                                        Restlessness, irritability, or feeling on edge or keyed up.
                                        Worrying too much.
                                        Fearing that something bad is going to happen; feeling doomed.
                                        Inability to concentrate; feeling like your mind goes blank.

''') 

        if y_act == ['adhd']:
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

                                    https://myhealth.alberta.ca/Health/Pages/conditions.aspx?hwid=hw166083 
                                    The three types of ADHD symptoms include:
                                        - Trouble paying attention (inattention). People with ADHD are easily distracted. They have a hard time focusing on any one task.
                                        -  Trouble sitting still for even a short time (hyperactivity). Children with ADHD may squirm, fidget, or run around at the wrong times. Teens and adults often feel restless and fidgety. They aren't able to enjoy reading or other quiet activities.
                                        -  Acting before thinking (impulsivity). People with ADHD may talk too loud, laugh too loud, or become angrier than the situation calls for. Children may not be able to wait for their turn or to share. This makes it hard for them to play with other children. Teens and adults may make quick decisions that have a long-term impact on their lives. They may spend too much money or change jobs often.

                                    These symptoms affect all people who have ADHD. But typical behaviour varies by age.
                                        - In preschool-age children, symptoms are often the same as normal behaviour for young children.
                                        - In children ages 6 to 12, signs of ADHD are more obvious than in other age groups.
                                        - In teens ages 13 to 18, problems that began in earlier years may continue or get worse.
                                        - Symptoms of ADHD in adults may not be as noticeable as in other age groups.
                                        - Symptoms of ADHD in adults may not be as noticeable as in other age groups.

''') 
        if y_act == ['neither']:
            st.write('It seems like you are not showing symptoms of any major mental illness')
            #call emotions model

    


