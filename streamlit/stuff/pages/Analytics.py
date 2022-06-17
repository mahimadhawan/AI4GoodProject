import streamlit as st
# from Home import display_home_page

st.set_page_config(layout="wide")

markdown = """
Web App URL: <INSERT>

GitHub Repository: <INSERT>
"""

st.sidebar.title("Analytics")
st.sidebar.info(markdown)
# logo = "IMAGE LINK"
# st.sidebar.image(logo)

st.title("Analytics")

with st.expander("Visual chart of moods over time"):
    st.write('add here')


with st.expander("Statistics (percent of each emotion etc"):
    st.write('add here')

