import streamlit as st
from util.feedback import feedback_form
from util.navigation import PAGES as PG


# Navigation
pages = [
    st.Page(PG['page'][i], title=PG['title'][i], icon=PG['icon'][i]) for i in range(len(PG))
]

pg = st.navigation(pages)
pg.run()


# Sidebar: Feedback Form
with st.sidebar:
    feedback_form()

    st.markdown("---")
    st.markdown("by [Yelnar Yerdesh](https://github.com/YR0E/)")