import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(layout="wide")

markdown = """
MoodRing is a mood journal web app that helps track & detect emotions and flag mental health symptoms based on text. This project was done as part of the 2022 AI4Good Lab.
"""

st.sidebar.title("About")
st.sidebar.info(markdown)
logo = "https://github.com/mahimadhawan/AI4GoodProject/blob/main/logo.png?raw=true"
st.sidebar.image(logo, use_column_width='always')

st.header("📈 Analytics")



mylabels = ['anger',
 'fear',
 'joy',
 'love',
 'sadness',
 'surprise',
 'thankfulness',
 'disgust',
 'guilt']

# expander1 = st.expander("Visualizationsasda,lkm")


with st.expander("Visualizations"):
# expander1.write('asdnla')
    with st.container():
        st.write('Mood distribution over past week:')

        options = ['Donut chart', 'Bar chart', 'Patterns over time']
        basemap = st.selectbox("View previous journal entries:", options)

        #dummy in data
        y = ([20,30, 10, 10, 15, 5, 5, 15, 2])

        if (basemap=='Donut chart'):
            fig = go.Figure(data=[go.Pie(labels=mylabels, values=y, hole=.3)])
            st.plotly_chart(fig)

        if (basemap=='Bar chart'):

            d = {'Emotions' : mylabels, 'Frequency' : y}
            chart_data = pd.DataFrame(d)
            fig = px.bar(chart_data, x='Emotions', y='Frequency', color='Emotions')
            st.plotly_chart(fig)


        # if (basemap=='Patterns over time'):
            # fake line chart
            # fig = px.scatter(x=, y=)
            # st.plotly_chart(fig)






with st.expander("Statistics"):
    st.write('')

