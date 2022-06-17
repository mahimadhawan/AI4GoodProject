import streamlit as st

st.set_page_config(layout="wide")

markdown = """
Web App URL: <INSERT>

GitHub Repository: <INSERT>
"""

st.sidebar.title("Results & Resources")
st.sidebar.info(markdown)
# logo = "IMAGE LINK"
# st.sidebar.image(logo)


st.title('Links')

# with st.expander("Visual chart of moods over time"):
#     link = '[GitHub](http://google.com)'
#     st.markdown(link, unsafe_allow_html=True)


# with st.expander("Statistics (percent of each emotion etc"):
#     st.write('add here')




