# MoodRing - an AI4Good project


<p align="center">
  <img 
    src="https://github.com/mahimadhawan/AI4GoodProject/blob/main/logo.png?raw=true"
  >
</p>



##
## Description
AI4Good Project: A mood journal app to help detect & track emotions, and identify if mental health illness symptoms are present from User text. Providing user with resources for detected symptoms.


## Background
Our project is a mood journal which serves as a wellness tool. The app features emotion detection and mental illness detection based on user text (journal entries). This app provides an opportunity for users to cultivate emotional awareness by viewing their mood patterns (including charts and visualizations) and identify if negative thought patterns are present. It also can flag potential mental illness, but is not a diagnostic tool - this is simply an added feature to provide information and resources for people who may be at risk for depression, anxiety, or ADHD.

The goal of the project is to use machine learning to increase accessibility to mental health tools and information + promote mental well being via journaling (which is proven to have positive effects on mental health + help improve negative thought patterns). Many people are unable to invest time, energy, or money into their mental health. This app is important because it is a mental health tool that has a low barrier to entry and is also a low effort for the user. It’s a good use case for machine learning since it uses automation to help provide a mental health outlet to people who may otherwise not seek or be able to obtain help, and increase awareness if potential mental illness is flagged. The potential harms of this app could be causing people to self-diagnose mental illnesses, and there also needs to be strict measures to protect user privacy.


## Machine Learning

### Data preparation
* We prepared the data by removing punctation and stopwords, and using stemming and lemmatization. For the mental illness detection model, we concatenated multiple data sources together.
* We explored data via bar graphs and pie charts to view balance/imbalance and used word clouds and N-grams to identify common elements


### Emotion classification
* Data:
    * The Cleaned Balanced Emotional Tweets (CBET) dataset (see citations)
* Machine learning:
    * We used supervised learning via the labelled emotions tweets to generate a multi-label classifier
    * We tried a variety of machine learning algorithms in sklearn but achieved the highest accuracy with fine-tuning the pre-trained BERT model from Hugging Face transformers
    * Achieved an ROC score of 0.89, an f1 score of 0.83, and an accuracy of 0.78


### Mental illness detection
* Data: 
    * the inputs for this model were social media comments labeled as one of four categories: depression, anxiety, ADHD, or neither/control. The data was collected from the sources listed under ‘Citations and Resources’. 
    * The comments for the depression, anxiety, and control were from Twitter. We were unable to find a data source of labeled tweets for ADHD, so those comments came from reddit and we restricted the length to make them more equal with the length of tweets.
* Machine learning:
    * We tried machine learning algorithms in sklearn such as SVM and random forest, as well as Hugging Face transformers/BERT to generate a multi-class classifier via supervised learning
    * We chose random forest as the best model
    * We did feature extraction using bag of words to train our model
    * Achieved an f1 score of 0.98 and an accuracy score of 0.93

### Challenges we encountered along the way:
* Looking for data sources and pre-processing it was by far the most time consuming task in our project.
* Tuning hyperparameters to avoid overfitting, especially with random forest

## Instructions
1) Make sure requirements (listed below) are installed on your machine
2) Download this 'streamlit' folder
3) Unzip 'model2.zip' and 'pytorch_model_bin.zip'
4) Navigate to the folder on your terminal
5) type 'streamlit run Home.py'

Requirements: streamlit, transformers, torch, nltk, pandas, sklearn


## Future directions
* If we had more time we would want to improve our models and add additional features to the app (listed below), as well as include more emotions and mental illnesses in our classification categories.
* Improve our UI & also make it available as a mobile app
* Improving our prediction models and also broadening the scope by gathering new data. We would like to add more emotions to the prediction categories such as lonely, stressed, excited, etc. We would also like to add more mental illnesses to the mental illness detection model. We chose to 
* Some features we would like to app to our app:
    * Adding in-app exercises if the user is identified to be experiencing negative emotions ex) meditation or other soothing techniques
    * Improving the copy/adding more resources info based on more research or even consultation with the relevant professionals 




## Citations

* Cleaned Balanced Emotional Tweets (CBET) Dataset

> @inproceedings{shahraki2017lexical,  
> title={Lexical and learning-based emotion mining from text},  
> author={Shahraki, Ameneh Gholipour and Za\"{i}ane, Osmar R},  
> booktitle={Proceedings of the International Conference on Computational Linguistics and Intelligent Text Processing},  
> year={2017}
> }

* big_anxiety.csv and big_normal.csv: https://github.com/caciitg/UTrack/tree/main/Extracted%20Tweets 
* tweets_random.csv:  https://github.com/weronikazak/depression-tweets 
* tweets_combined.csv: https://github.com/swcwang/depression-detection/tree/master/data 
* mi_twitter_data_clean.csv: https://github.com/emiburns/text_analysis_of_depression_anxiety_tweets/tree/main/data 
* ADHD-comment.csv: https://www.kaggle.com/datasets/jerseyneo/reddit-adhd-dataset?select=ADHD-comment.csv 

The Streamlit template used for the web app is: https://github.com/giswqs/streamlit-multipage-template


