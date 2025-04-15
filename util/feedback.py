from datetime import datetime

import gspread
import pytz
import streamlit as st
from google.oauth2.service_account import Credentials

# Define constants
# SECTION_TYPES = ["Concept", "Documentation", "Calculation", "Visualization", "Other"]
FEEDBACK_TYPES = [
    "Question",
    "Suggestion",
    "Feature Request",
    "Report a Problem",
    "Other",
]
TIMEZONE = pytz.timezone("Asia/Almaty")

# Define the scope for Google API
SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
]


@st.cache_resource
def get_google_sheet():
    """
    Initialize and cache the Google Sheets client and worksheet.

    This function establishes a connection to Google Sheets using service account credentials
    and returns the first worksheet of the "Feedback" spreadsheet.

    Returns:
        gspread.models.Spreadsheet: The first worksheet of the "Feedback" spreadsheet.
    """
    # Load credentials from Streamlit secrets
    credentials = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=SCOPES
    )

    # Authorize the client using the credentials
    client = gspread.authorize(credentials)

    # Open the "Feedback" spreadsheet and access the first sheet
    return client.open("Feedback").sheet1


def validate_and_submit_feedback(sheet, feedback_data):
    """
    Validate feedback inputs and submit them to a Google Sheets document.

    Parameters:
    - sheet (gspread.models.Spreadsheet): The Google Sheets worksheet to append the feedback data to.
    - feedback_data (list): A list containing feedback details, including submitting date, name,
      section type, feedback type, and comments.

    This function checks if the name and feedback type fields are filled. If either is missing,
    a warning is displayed to the user. If both fields are filled, the function attempts to append
    the feedback data to the specified Google Sheets document. A success message is shown upon
    successful submission, or an error message is displayed if the submission fails.
    """

    name = feedback_data[1]
    type = feedback_data[3]
    if not name or not type:
        st.warning("Please fill out all required fields.")
    else:
        with st.spinner("Sending feedback..."):
            try:
                sheet.append_rows([feedback_data])
                st.success("Feedback sent successfully!")
            except Exception as e:
                st.error(f"Error updating Google Sheets: {e}")


@st.fragment
def feedback_form():
    """
    Render a Streamlit form for submitting user feedback.

    This function displays a Streamlit form with input fields for the user's name, feedback type, and comments.
    The form is wrapped in a Streamlit expander with a heading of "Drop a line here".
    The function also retrieves the cached Google Sheet to write the feedback data to.
    """
    st.markdown("## Feedback")
    with st.expander("Drop a line here", expanded=True):
        with st.form(key="feedback_form", clear_on_submit=True, border=False):
            # Retrieve the cached Google Sheet
            sheet = get_google_sheet()

            # Collecting feedback inputs
            submitting_date = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
            name = st.text_input(label="Name* (*required*)")
            # col1, col2 = st.columns(2)
            # section_type = col1.selectbox("Section*", options=SECTION_TYPES, index=None)
            section_type = "General"
            feedback_type = st.selectbox(
                "Feedback Type*", options=FEEDBACK_TYPES, index=None
            )
            comment = st.text_area(label="Comments")
            feedback_data = [
                submitting_date,
                name,
                section_type,
                feedback_type,
                comment,
            ]

            submit_button = st.form_submit_button(label="Send")
            if submit_button:
                validate_and_submit_feedback(sheet, feedback_data)
