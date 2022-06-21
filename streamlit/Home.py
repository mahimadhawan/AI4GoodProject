import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


st.title('MoodRing')



# st.set_page_config(layout="wide")

# Customize the sidebar
markdown = """
MoodRing is a mood journal web app that helps track & detect emotions and flag mental health symptoms based on text. This project was done as part of the 2022 AI4Good Lab.

"""

st.sidebar.title("About")
st.sidebar.info(markdown)
logo = "https://github.com/mahimadhawan/AI4GoodProject/blob/main/logo.png?raw=true"
st.sidebar.image(logo, use_column_width='always')



# col1, col2 = st.columns(2)

# col1, col2= st.columns([3, 4])
col1, col2, col3 = st.columns([4, 0.1, 2])

with col3:
    st.text("Today's affirmations:")
    st.caption(" 'It does not matter how slowly you go as long as you do not stop.' - Confucius")
    # st.caption("- Confucius")

with col2:
    st.write('')


with col1:
    y = ([20,30, 10, 10, 15, 5, 5, 15, 2])
    mylabels = ['anger','fear','joy','love','sadness','surprise','thankfulness','disgust','guilt']
    st.text("Your month so far:")
    fig = go.Figure(data=[go.Pie(labels=mylabels, values=y, hole=.3)])
    st.plotly_chart(fig, use_container_width=True)




with st.container():
    st.subheader('Quick Links')
    # link = '[New Journal Entry](http://google.com)'
    # st.markdown(link, unsafe_allow_html=True)
    link2 = '[Sleep Tracker](http://google.com)'
    st.markdown(link2, unsafe_allow_html=True)


