import streamlit as st

st.set_page_config(layout="wide")

markdown = """
MoodRing is a mood journal web app that helps track & detect emotions and flag mental health symptoms based on text. This project was done as part of the 2022 AI4Good Lab.
"""

st.sidebar.title("About")
st.sidebar.info(markdown)
logo = "https://github.com/mahimadhawan/AI4GoodProject/blob/main/logo.png?raw=true"
st.sidebar.image(logo, use_column_width='always')


st.header('💫 Resources & Links')




