import streamlit as st

st.set_page_config(layout='wide', initial_sidebar_state='expanded', page_title='soon...')


st.write("Welcome to Page 3")
st.warning("Coming soon...")
st.markdown('***')

st.markdown("### Navigation:")
st.page_link("0_home.py", label="Home page", icon=":material/home:")
st.page_link("1_imposed_q0.py", label="Imposed q0: Find minimum power consumption min(w)", icon=":material/function:")
st.page_link("2_imposed_w0.py", label="Imposed w0: Find maximum heat extraction max(q)", icon=":material/function:")
