import streamlit as st

PAGES = {
     'page': ["0_home.py", 
              "1_imposed_q0.py", 
              "2_imposed_w0.py", 
              "3_imposed_COP0.py"],
     'title': ["Home",                         # for sidebar navigation
               "Imposed q0: min(w)", 
               "Imposed w0: max(q)", 
               "Imposed COP0: max(q)"],
     'label': ["Home page", 
               "Imposed q0: Find minimum power consumption min(w)", 
               "Imposed w0: Find maximum heat extraction max(q)", 
               "Imposed COP0: Find maximum heat extraction max(q)"],
     'icon': [":material/home:", 
              ":material/function:", 
              ":material/function:", 
              ":material/function:"]
}


def link_to_pages(pages):
    for page in pages:
        st.page_link(PAGES['page'][page], label=PAGES['label'][page], icon=PAGES['icon'][page])