import streamlit as st
from streamlit_theme import st_theme
from util.navigation import link_to_pages

st.set_page_config(layout='wide', initial_sidebar_state='expanded', page_title='optimizer by Yelnar')

theme = st_theme()
dark_mode = False
if theme is not None and theme['base']=='dark':
    dark_mode = True


st.toast('Development in progress...', icon=':material/info:')

st.title("Thermodynamic optimization of GSHP")
st.warning("Work in process...")
st.info(r'''
        Ready to use:
        - Imposed $q_0$ page    
        - Feedback section  
        ''')
st.markdown('***')
st.title("About")
st.write("Here will be some information about the project.")

_, col_img, _ = st.columns([1, 6, 1])
image_path = "img/gshp-dark.png" if dark_mode else "img/gshp.png"
col_img.image(image_path, caption="Ground source heat pump")

st.markdown('***')
st.markdown("### Navigation:")
link_to_pages(pages=[1, 2, 3])