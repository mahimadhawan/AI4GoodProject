import streamlit as st

markdown = """
MoodRing is a mood journal web app that helps track & detect emotions and flag mental health symptoms based on text. This project was done as part of the 2022 AI4Good Lab.
"""


st.sidebar.title("About")
st.sidebar.info(markdown)
logo = "https://github.com/mahimadhawan/AI4GoodProject/blob/main/logo.png?raw=true"
st.sidebar.image(logo, use_column_width='always')

st.header("📚 View past entries:")

col1, col2 = st.columns([4, 2])
options = ['Most recent', 'Past 7 days', 'Past month', 'All time']

with col2:





    basemap = st.selectbox("View previous journal entries:", options)


with col1:
    st.write("")
    # st.write('[INSERT TEXT HERE]')
