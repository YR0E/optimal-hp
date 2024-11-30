import streamlit as st
from streamlit_gsheets import GSheetsConnection
from datetime import datetime
import pandas as pd


@st.cache_resource
def st_connection():
    # Connect to Google Sheets
    conn = st.connection("gsheets", type=GSheetsConnection)

    data = conn.read(worksheet="HP_optima", usecols=list(range(5)), ttl=5)
    data = data.dropna(how='all')
    return conn, data

def submit_feedback(submitting_date, name, section_type, feedback_type, comment):
    # Update Google Sheets with the new feedback
    conn, existing_data = st_connection()
    feedback_data = pd.DataFrame(
        [{ 
                "Date": submitting_date.strftime("%Y-%m-%d %H:%M:%S"),
                "Name": name,
                "Section": section_type,
                "CommentType": feedback_type,
                "Comment": comment,
        }]
    )

    updated_df = pd.concat([existing_data, feedback_data], ignore_index=True)
    conn.update(worksheet="HP_optima", data=updated_df)
    return None



pages = [
        st.Page("page0_home.py", title="Home", icon=":material/home:"),
        st.Page("page1.py", title="Page 1", icon=":material/function:"),
        st.Page("page2.py", title="Page 2", icon=":material/function:"),
        # st.Page("feedback.py", title="Feedback", icon=":material/feedback:"),
]

pg = st.navigation(pages)
pg.run()


SECTION_TYPES = [
    "Concept",
    "Documentation",
    "Calculation",
    "Visualization",
    "Other"
]

FEEDBACK_TYPES = [
    "Question",
    "Suggestion",
    "Request",
    "Error",
    "Bug",
    "Other",
]

with st.sidebar:
    with st.expander("Please leave your feedback here"):
        with st.form(key="feedback_form", border=False, clear_on_submit=True):

            submitting_date = datetime.now()
            name = st.text_input(label="Name* (*required*)")
            col1, col2 = st.columns(2)
            section_type = col1.selectbox("Section*", options=SECTION_TYPES, index=None)
            feedback_type = col2.selectbox("Feedback Type", options=FEEDBACK_TYPES, index=None)
            comment = st.text_area(label="Comments")
            
            submit_button = st.form_submit_button(label="Leave feedback")
            if submit_button:

                # Check if all mandatory fields are filled
                if not name or not section_type:
                    st.warning("Ensure all mandatory fields are filled.")
                    st.stop()


                submit_feedback(submitting_date, name, section_type, feedback_type, comment)
                st.success("Feedback successfully submitted!")