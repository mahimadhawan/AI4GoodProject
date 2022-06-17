import streamlit as st

st.set_page_config(layout="wide")

markdown = """
Web App URL: <INSERT>

GitHub Repository: <INSERT>
"""

st.sidebar.title("New journal entry")
st.sidebar.info(markdown)
# logo = "IMAGE LINK"
# st.sidebar.image(logo)


st.title('New journal entry')






sentence = st.text_input('Enter question here:') 


def display_emotions_result(emotion):
	if emotion=='anger':
		st.write('anger')
	if emotion=='fear':
		st.write('fear')



if sentence:
	st.write('got sentence')
    # mi_result = my_mi_model.predict(sentence)
    # emotions_result = my_emotions_model.predict(sentence)
    # display_emotions_result(emotions_result)



if mi_result!='neither':
	st.write('neither')


if mi_result=='adhd':
	st.write('adhd')


if mi_result=='anxiety':
	st.write('neither')


if mi_result=='depression':
	st.write('depression')






# def app():
#     st.title("Searching Basemaps")
#     st.markdown(
#         """
#     This app is a demonstration of searching and loading basemaps from [xyzservices](https://github.com/geopandas/xyzservices) and [Quick Map Services (QMS)](https://github.com/nextgis/quickmapservices). Selecting from 1000+ basemaps with a few clicks.  
#     """
#     )

#     with st.expander("See demo"):
#         st.image("https://i.imgur.com/0SkUhZh.gif")

#     row1_col1, row1_col2 = st.columns([3, 1])
#     width = 800
#     height = 600
#     tiles = None

#     with row1_col2:

#         checkbox = st.checkbox("Search Quick Map Services (QMS)")
#         keyword = st.text_input("Enter a keyword to search and press Enter:")
#         empty = st.empty()

#         if keyword:
#             options = leafmap.search_xyz_services(keyword=keyword)
#             if checkbox:
#                 options = options + leafmap.search_qms(keyword=keyword)

#             tiles = empty.multiselect("Select XYZ tiles to add to the map:", options)

#         with row1_col1:
#             m = leafmap.Map()

#             if tiles is not None:
#                 for tile in tiles:
#                     m.add_xyz_service(tile)

#             m.to_streamlit(width, height)


# app()
