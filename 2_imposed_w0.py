import streamlit as st
from util.navigation import link_to_pages

st.set_page_config(layout='wide', initial_sidebar_state='expanded', page_title='soon...')


st.write("Welcome to Page 2")
st.warning("Coming soon...")
st.markdown('***')


st.markdown("### Navigation:")
link_to_pages(pages=[0, 1, 3])