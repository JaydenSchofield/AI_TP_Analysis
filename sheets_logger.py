import streamlit as st
import gspread
from datetime import datetime
import uuid

# Define the scope (permissions) we need
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

def get_session_id():
    """Returns a unique ID for the current user's session."""
    if 'user_session_id' not in st.session_state:
        st.session_state.user_session_id = str(uuid.uuid4())[:8] # Short 8-char ID
    return st.session_state.user_session_id

def log_interaction(question, condensed_question, answer, username="TestUser"):
    """
    Logs the interaction to Google Sheets with the new fields.
    Order: Timestamp, Name, Session ID, Question, Condensed Question, Answer
    """
    # 1. Check if secrets exist (Prevent crash if not set up)
    if "gcp_service_account" not in st.secrets:
        return

    try:
        # 2. Load Credentials
        creds_dict = dict(st.secrets["gcp_service_account"])
        
        # 3. Authenticate
        client = gspread.service_account_from_dict(creds_dict)

        # 4. Open the Sheet
        sheet = client.open("TrainingCoach_Logs").sheet1

        # 5. Append the Row
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session_id = get_session_id()
        
        # NEW ROW FORMAT
        row_data = [timestamp, username, session_id, question, condensed_question, answer]
        
        sheet.append_row(row_data)
        
    except Exception as e:
        print(f"Analytics Error: {e}")