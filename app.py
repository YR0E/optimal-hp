import streamlit as st
from streamlit_gsheets import GSheetsConnection
from datetime import datetime
import pandas as pd

# Establish Google Sheets connection once
@st.cache_resource
def get_connection():
    return st.connection("gsheets", type=GSheetsConnection)

# Fetch data from Google Sheets and store in session state
def initialize_data(conn, worksheet_name):
    if 'existing_data' not in st.session_state:
        data = conn.read(worksheet=worksheet_name, usecols=list(range(5)))
        st.session_state.existing_data = data.dropna(how="all")

# Append new feedback to session state data
def append_feedback_to_session(feedback_row):
    st.session_state.existing_data = pd.concat(
        [st.session_state.existing_data, feedback_row], ignore_index=True
    )

# Update Google Sheets with the data from session state
def update_gsheet(conn, worksheet_name):
    conn.update(worksheet=worksheet_name, data=st.session_state.existing_data)


# Page/Sidebar configuration
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
    "Other"
]

with st.sidebar:
    st.markdown("## Feedback Submission")
    with st.expander("Please leave your feedback here", expanded=True):
        with st.form(key="feedback_form", clear_on_submit=True, border=False):
            conn = get_connection()
            worksheet_name = "HP_optima"
            initialize_data(conn, worksheet_name)

            submitting_date = datetime.now()
            name = st.text_input(label="Name* (*required*)")
            col1, col2 = st.columns(2)
            section_type = col1.selectbox("Section*", options=SECTION_TYPES, index=None)
            feedback_type = col2.selectbox("Feedback Type", options=FEEDBACK_TYPES, index=None)
            comment = st.text_area(label="Comments")

            submit_button = st.form_submit_button(label="Leave feedback")

            if submit_button:
                # Ensure mandatory fields are filled
                if not name or not section_type:
                    st.warning("Ensure all mandatory fields are filled!")
                    st.stop()

                # Prepare feedback data
                feedback_info = pd.DataFrame([{
                    "Date": submitting_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "Name": name,
                    "Section": section_type,
                    "CommentType": feedback_type,
                    "Comment": comment,
                }])

                # Append new feedback to session state
                append_feedback_to_session(feedback_info)

                # Update Google Sheets
                try:
                    update_gsheet(conn, worksheet_name)
                    st.success("Feedback successfully submitted!")
                except Exception as e:
                    st.error(f"Error updating Google Sheets: {e}")
