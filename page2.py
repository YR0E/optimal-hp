import streamlit as st

st.set_page_config(layout='wide', initial_sidebar_state='expanded', page_title='later')


st.write("Welcome to Page 2")

st.markdown("### Navigation:")
st.page_link("page0_home.py", label="Home page", icon=":material/home:")
st.page_link("page1.py", label="Find min(w)", icon=":material/function:")