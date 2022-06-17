import streamlit as st


st.title('Home')


def display_home_page():
    # st.set_page_config(layout="wide")

    # Customize the sidebar
    markdown = """
    Web App URL: <INSERT HERE>

    GitHub Repository: <INSERT HERE>
    """

    st.sidebar.title("About")
    st.sidebar.info(markdown)
    # logo = "IMAGE LINK"
    # st.sidebar.image(logo)

    # Customize page title
    st.title("Mood journal")

    st.markdown(
        """
        This is a mood journal
        """
    )

    st.header("Instructions")

    markdown = """
    1. For the [GitHub repository](https://github.com/giswqs/streamlit-multipage-template) or [use it as a template](https://github.com/giswqs/streamlit-multipage-template/generate) for your own project.
    2. Customize the sidebar by changing the sidebar text and logo in each Python files.
    3. Find your favorite emoji from https://emojipedia.org.
    4. Add a new app to the `pages/` directory with an emoji in the file name, e.g., `1_🚀_Chart.py`.

    """

    st.markdown(markdown)



    st.title('Links')
    link = '[Google](http://google.com)'
    st.markdown(link, unsafe_allow_html=True)


display_home_page()