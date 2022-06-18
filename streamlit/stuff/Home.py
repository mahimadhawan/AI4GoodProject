import streamlit as st


st.title('MoodRing')


def display_home_page():
    # st.set_page_config(layout="wide")

    # Customize the sidebar
    markdown = """
    MoodRing is a mood journal web app that helps track & detect emotions and flag mental health symptoms based on text. This project was done as part of the 2022 AI4Good Lab.

    """

    st.sidebar.title("About")
    st.sidebar.info(markdown)
    logo = "https://github.com/mahimadhawan/AI4GoodProject/blob/main/logo.png?raw=true"
    st.sidebar.image(logo, use_column_width='always')













    st.markdown(markdown)
    st.subheader('Links')
    link = '[Google](http://google.com)'
    st.markdown(link, unsafe_allow_html=True)


display_home_page()