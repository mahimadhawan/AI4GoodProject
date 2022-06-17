import streamlit as st

markdown = """
Web App URL: <INSERT>

GitHub Repository: <INSERT>
"""

st.sidebar.title("Past entries")
st.sidebar.info(markdown)
# logo = "IMAGE LINK"
# st.sidebar.image(logo)


st.title("Past entries")

col1, col2 = st.columns([4, 2])
options = ['Most recent', 'Past 7 days', 'Past month', 'All time']

with col2:

    basemap = st.selectbox("View previous journal entries:", options)


with col1:
    st.write('[INSERT TEXT HERE]')
