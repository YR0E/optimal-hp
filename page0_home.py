import streamlit as st

st.title("Welcome to Home Page")
st.write("This is the home page content.")

st.page_link("page1.py", label="Page 1")
st.page_link("page2.py", label="Page 2")


st.markdown('***')
st.title("About")
st.write("This is the about section.")

