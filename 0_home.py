import streamlit as st
from util.navigation import link_to_pages

st.set_page_config(layout='wide', initial_sidebar_state='expanded', page_title='optimizer by Yelnar')

st.toast('Development in progress...', icon=':material/info:')

st.title("Welcome to Home Page")
st.write("This is the home page content.")
st.warning("Will be updated soon...")
st.info(r'''
        Ready to use:
        - Imposed $q_0$ page    
        - Feedback section  
        ''')
st.markdown('***')


st.markdown("### Navigation:")
link_to_pages(pages=[1, 2, 3])
st.markdown('***')


st.title("About")
st.write("This is the about section.")