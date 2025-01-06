import streamlit as st
from util.feedback import feedback_form


# Navigation
PAGES = [
    st.Page("0_home.py", title="Home", icon=":material/home:"),
    st.Page("1_imposed_q0.py", title="Imposed q0: min(w)", icon=":material/function:"),
    st.Page("2_imposed_w0.py", title="Imposed w0: max(q)", icon=":material/function:"),
    st.Page("3_imposed_COP0.py", title="Imposed COP0: max(q)", icon=":material/function:")
]

pg = st.navigation(PAGES)
pg.run()


# Sidebar: Feedback Form
with st.sidebar:
    feedback_form()

    st.markdown("---")
    st.markdown("by [Yelnar Yerdesh](https://github.com/YR0E/)")