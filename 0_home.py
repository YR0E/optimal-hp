import streamlit as st

st.set_page_config(layout='wide', initial_sidebar_state='expanded', page_title='optimizer by Yelnar')

st.toast('Development in progress...', icon=':material/info:')

st.title("Welcome to Home Page")
st.write("This is the home page content.")
st.warning("Will be updated soon...")
st.markdown('***')

st.markdown("### Navigation:")
st.page_link("1_imposed_q0.py", label="Imposed q0: Find minimum power consumption min(w)", icon=":material/function:")
st.page_link("2_imposed_w0.py", label="Imposed w0: Find maximum heat extraction max(q)", icon=":material/function:")
st.page_link("3_imposed_COP0.py", label="Imposed COP0: Find maximum heat extraction max(q)", icon=":material/function:")


st.markdown('***')
st.title("About")
st.write("This is the about section.")