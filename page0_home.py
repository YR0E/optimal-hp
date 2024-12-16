import streamlit as st

st.set_page_config(layout='wide', initial_sidebar_state='expanded', page_title='optimizer by Yelnar')

st.toast('Development in progress...', icon=':material/info:')

st.title("Welcome to Home Page")
st.write("This is the home page content.")

st.markdown("### Navigation:")
st.page_link("page1.py", label="Find min(w)", icon=":material/function:")
st.page_link("page2.py", label="Page 2", icon=":material/function:")


st.markdown('***')
st.title("About")
st.write("This is the about section.")