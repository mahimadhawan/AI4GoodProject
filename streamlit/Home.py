import streamlit as st
import numpy as np


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



col1, col2 = st.columns(2)

with col1:
    st.text("Today's affirmations:")
    st.caption(" 'It does not matter how slowly you go as long as you do not stop.' - Confucius")
    # st.caption("- Confucius")


with col2:
    st.text("Your month so far:")
    st.image("https://plotly-r.com/images/economics.svg")


with st.container():
    # st.write("This is inside the container")

    # You can call any Streamlit command, including custom components:
    # st.bar_chart(np.random.randn(50, 3))

    st.markdown(markdown)
    st.subheader('Quick Links')
    link = '[Google](http://google.com)'
    st.markdown(link, unsafe_allow_html=True)


