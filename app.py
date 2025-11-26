"""
Yearly Workout Analysis Script using Google Gemini AI
(Router-Enhanced Version with Instant UI Feedback & Logging)
"""
import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import json
import re
import random
from datetime import datetime
from dotenv import load_dotenv

# --- IMPORT YOUR LOCAL LOGGER ---
import sheets_logger 

# --- API Key Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(env_path)

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    if not api_key:
        st.error("GOOGLE_API_KEY not found. Please set it in .env or secrets.")
    else:
        genai.configure(api_key=api_key)
else:
    genai.configure(api_key=api_key)

# --- Hardcoded User Profile ---
USER_PROFILE = {
    # "power_zones": {
    #     "1": "0-142w", "2": "143-193w", "3": "194-231w",
    #     "4": "232-270w", "5": "271-308w", "6": "309-2000w"
    # },
    # "bike_heart_rate_zones": {
    #     "1": "38-144bpm", "2": "145-152bpm", "3": "153-160bpm",
    #     "4": "161-169bpm", "5": "170-178bpm", "6": "179-190bpm"
    # },
    # "run_heart_rate_zones": {
    #     "1": "38-147bpm", "2": "148-156bpm", "3": "157-164bpm",
    #     "4": "165-173bpm", "5": "174-182bpm", "6": "183-195bpm"
    # }
}

# --- Suggested Questions List ---
SUGGESTED_QUESTIONS = [
    "What was my highest volume training month, and can you provide a weekly breakdown?",
    "Which month presented the highest risk of overtraining based on acute load spikes?",
    "What specific changes would you recommend to improve my Lactate Threshold 2 (LT2)?",
    "What is the ratio of time spent in Zone 2 vs. Zone 1 this year?",
    "Based on recent trends, project a recommended structure for next week's training.",
    "How consistent has my swim frequency and volume been week-over-week?",
    "What critical data fields are missing that would significantly enhance your analysis?",
    "Identify sessions in the last 60 days where poor sleep correlated with missed targets.",
    "What is the long-term consistency trend of my sleep duration and quality?",
    "Is there a correlation between my lower sleep scores and higher RPE?"
]

def parse_value(value):
    """Helper to parse 'Value' column from metrics.csv."""
    if pd.isna(value): return None
    value_str = str(value).strip()
    if 'Avg :' in value_str:
        try:
            parts = value_str.split('Avg :')
            if len(parts) > 1:
                return float(parts[1].strip().split()[0])
        except (ValueError, IndexError): pass
    try:
        return float(value_str)
    except ValueError: return None

def process_metrics(metrics_csv_path):
    """Process metrics.csv to wide-format DataFrame."""
    df_metrics = pd.read_csv(metrics_csv_path)
    if 'Timestamp' in df_metrics.columns:
        df_metrics['date'] = pd.to_datetime(df_metrics['Timestamp'], errors='coerce').dt.date
        df_metrics['date'] = df_metrics['date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None)
        df_metrics = df_metrics[df_metrics['date'].notna()]
    else: raise ValueError("'Timestamp' column missing")
    
    if 'Value' in df_metrics.columns:
        df_metrics['value_numeric'] = df_metrics['Value'].apply(parse_value)
    else: raise ValueError("'Value' column missing")
    
    df_pivoted = df_metrics.pivot_table(index='date', columns='Type', values='value_numeric', aggfunc='first')
    column_mapping = {
        'Sleep Hours': 'Sleep_Hours', 'Body Battery': 'BodyBattery_Avg',
        'Stress Level': 'StressLevel_Avg', 'Time In Deep Sleep': 'Sleep_Deep_Hours'
    }
    return df_pivoted.rename(columns=column_mapping)

def create_yearly_summary(yearly_csv_path, metrics_csv_path=None):
    """Read CSVs, merge, and create summary dict + source DF."""
    df_metrics = None
    if metrics_csv_path is not None:
        try:
            df_metrics = process_metrics(metrics_csv_path)
        except Exception as e:
            st.warning(f"Metric load error: {e}")
            df_metrics = None
    
    df_yearly = pd.read_csv(yearly_csv_path)
    if 'WorkoutDay' in df_yearly.columns:
        df_yearly['WorkoutDay_parsed'] = pd.to_datetime(df_yearly['WorkoutDay'], errors='coerce')
        df_yearly['date'] = df_yearly['WorkoutDay_parsed'].dt.strftime('%Y-%m-%d')
        date_range = {
            'start': df_yearly['WorkoutDay_parsed'].min().strftime('%Y-%m-%d') if not pd.isna(df_yearly['WorkoutDay_parsed'].min()) else None,
            'end': df_yearly['WorkoutDay_parsed'].max().strftime('%Y-%m-%d') if not pd.isna(df_yearly['WorkoutDay_parsed'].max()) else None
        }
    else:
        date_range = None; df_yearly['date'] = None
    
    if df_metrics is not None and 'date' in df_yearly.columns:
        df_yearly = pd.merge(df_yearly, df_metrics, left_on='date', right_index=True, how='left')
    
    # Identify completion
    has_actual = pd.Series([False] * len(df_yearly), index=df_yearly.index)
    if 'TimeTotalInHours' in df_yearly.columns: has_actual |= df_yearly['TimeTotalInHours'].notna()
    if 'DistanceInMeters' in df_yearly.columns: has_actual |= df_yearly['DistanceInMeters'].notna()
    
    completed = df_yearly[has_actual]
    
    # Statistics Generation
    def safe_stats(series):
        non_null = series.dropna()
        if len(non_null) == 0: return None
        return {
            'count': int(len(non_null)), 'mean': float(non_null.mean()),
            'median': float(non_null.median()), 'max': float(non_null.max())
        }
    
    actual_cols = ['DistanceInMeters', 'TimeTotalInHours', 'PowerAverage', 'HeartRateAverage', 'TSS', 'IF', 'Sleep_Hours', 'BodyBattery_Avg']
    actual_stats = {col: safe_stats(completed[col]) for col in actual_cols if col in completed.columns}
    
    yearly_summary = {
        'total_workouts': len(df_yearly),
        'completed_workouts': int(len(completed)),
        'date_range': date_range,
        'workout_types': df_yearly['WorkoutType'].value_counts().to_dict() if 'WorkoutType' in df_yearly.columns else {},
        'actual_stats': actual_stats,
        'data_quality': {col: {'non_null': int(df_yearly[col].notna().sum())} for col in df_yearly.columns},
        'data_preview': df_yearly.head(20).to_dict('records'),
        'source_df': df_yearly.drop(columns=['WorkoutDay_parsed'], errors='ignore')
    }
    
    # Default Systematic Sample
    max_records = 200
    if len(df_yearly) > max_records:
        indices = [int(i * len(df_yearly) / max_records) for i in range(max_records)]
        yearly_summary['default_sample'] = df_yearly.iloc[indices].to_dict('records')
    else:
        yearly_summary['default_sample'] = df_yearly.to_dict('records')
        
    return yearly_summary

# --- ROUTER LOGIC ---

def get_filtering_params(user_question, date_range):
    """Router: Asks AI to extract filtering parameters."""
    system_prompt = f"""
    You are a data query generator.
    The user has a dataset of workouts ranging from {date_range.get('start')} to {date_range.get('end')}.
    Analyze the user's question and return ONLY a JSON object with these keys:
    - "start_date": "YYYY-MM-DD" or null
    - "end_date": "YYYY-MM-DD" or null
    - "workout_types": ["Run", "Bike", etc.] or null
    
    User Question: "{user_question}"
    JSON Output:
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(system_prompt)
        text = response.text.strip().replace('```json', '').replace('```', '')
        return json.loads(text)
    except: return None

def get_relevant_data(yearly_summary, user_question):
    """Router: Filters data based on AI parameters."""
    df = yearly_summary['source_df']
    date_range = yearly_summary['date_range']
    params = get_filtering_params(user_question, date_range)
    
    if not params: return yearly_summary['default_sample']
    
    filtered_df = df.copy()
    if params.get('start_date'): filtered_df = filtered_df[filtered_df['date'] >= params['start_date']]
    if params.get('end_date'): filtered_df = filtered_df[filtered_df['date'] <= params['end_date']]
    if params.get('workout_types'):
        types = [t.lower() for t in params['workout_types']]
        filtered_df = filtered_df[filtered_df['WorkoutType'].astype(str).str.lower().isin(types)]
    
    row_count = len(filtered_df)
    if row_count == 0: return yearly_summary['default_sample']
    elif row_count <= 300: return filtered_df.to_dict('records')
    else:
        indices = [int(i * row_count / 300) for i in range(300)]
        return filtered_df.iloc[indices].to_dict('records')

def get_condensed_question(chat_history, new_question):
    """Condenses chat history."""
    if not chat_history: return new_question
    history_string = ""
    for entry in chat_history: history_string += f"{entry['role']}: {entry['content']}\n"
    prompt = f"""
    Rewrite to be standalone.
    History: {history_string}
    New Question: {new_question}
    Standalone Question:
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else new_question
    except: return new_question

def get_yearly_ai_analysis(yearly_summary, user_question, user_profile=None):
    """Main Analysis Function."""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        relevant_data_rows = get_relevant_data(yearly_summary, user_question)
        data_context_str = json.dumps(relevant_data_rows, indent=2, default=str)
        stats_section = json.dumps(yearly_summary['actual_stats'], indent=2)
        
        prompt = f"""You are an expert triathlon coach.
        USER PROFILE ZONES: {json.dumps(user_profile, indent=2)}
        YEARLY AGGREGATE STATS: {stats_section}
        SPECIFIC DATA FOR QUERY: {data_context_str}
        
        INSTRUCTIONS:
        - Answer directly using the SPECIFIC DATA.
        - Be concise and conversational (approx 4-5 sentences).
        - If specific days are in data, refer to them.
        - Ask 1 relevant follow-up question.
        
        USER QUESTION: {user_question}
        """
        response = model.generate_content(prompt)
        return response.text if response.text else "Error: Empty response."
    except Exception as e: return f"Error: {str(e)}"

# --- UI LOGIC HANDLER ---
def handle_user_query(query_text, method="chat_input"):
    """
    Handles the AI interaction loop with immediate UI feedback.
    Args:
        query_text: The text string the user wants to ask.
        method: 'chat_input' or 'random_button' (for analytics)
    """
    
    # 1. Save User Message to History
    st.session_state.messages.append({"role": "user", "content": query_text})
    
    # 2. RENDER USER MESSAGE IMMEDIATELY (UI Feedback)
    with st.chat_message("user"):
        st.markdown(query_text)
    
    # 3. Generate AI Response (with visible spinner)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # We capture 'condensed' here so we can log it later
            condensed = get_condensed_question(st.session_state.messages[:-1], query_text)
            
            answer = get_yearly_ai_analysis(st.session_state.yearly_summary, condensed, USER_PROFILE)
            st.markdown(answer)
    
    # 4. Save AI Message to History
    st.session_state.messages.append({"role": "model", "content": answer})

    # Retrieve the user parameter from the URL query params again, or default to "TestUser"
    current_user = st.query_params.get("user", "TestUser")
    sheets_logger.log_interaction(query_text, condensed, answer, username=current_user)


# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Training Coach", layout="centered")

if "analysis_active" not in st.session_state: st.session_state.analysis_active = False
if "yearly_summary" not in st.session_state: st.session_state.yearly_summary = None
if "messages" not in st.session_state: st.session_state.messages = []

# Debug - what files we are getting from server_data
import os
st.write(f"Current Working Directory: {os.getcwd()}")
if os.path.exists("server_data"):
    st.write(f"Files in server_data: {os.listdir('server_data')}")
else:
    st.write("server_data folder NOT found!")

# --- AUTO-LOAD LOGIC ---
# Get 'user' parameter from URL (e.g. ?user=johnsmith)
query_params = st.query_params
user_param = query_params.get("user", None)

# Directory where you will store the user files
SERVER_DATA_DIR = os.path.join(os.path.dirname(__file__), "server_data")

# Try to auto-load if user param exists
if user_param:
    # Construct file paths (e.g. johnsmithYearlySummary.csv)
    yearly_server_path = os.path.join(SERVER_DATA_DIR, f"{user_param}YearlySummary.csv")
    metrics_server_path = os.path.join(SERVER_DATA_DIR, f"{user_param}Metrics.csv")
    
    # Check if the yearly summary exists for this user
    if os.path.exists(yearly_server_path):
        try:
            # Load the data automatically
            # We pass the metrics path only if it exists
            metrics_path = metrics_server_path if os.path.exists(metrics_server_path) else None
            
            st.session_state.yearly_summary = create_yearly_summary(yearly_server_path, metrics_path)
            st.session_state.analysis_active = True
            # Optional: Track who logged in via analytics
            # sheets_logger.log_login(user_param) 
        except Exception as e:
            st.error(f"Error loading data for user '{user_param}': {e}")
    else:
        # If file doesn't exist, just do nothing and show standard upload screen
        # (Or show a specific error like "User data not found")
        pass

@st.cache_data
def load_data(yearly_file, metrics_file):
    return create_yearly_summary(yearly_file, metrics_file)

# STATE 1: Upload
if not st.session_state.analysis_active:
    st.title("🚴‍♂️ AI Analysis Chat")
    with st.container(border=True):
        f1 = st.file_uploader("Upload yearlySummary.csv", type="csv")
        f2 = st.file_uploader("Upload metrics.csv (Optional)", type="csv")
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            if f1:
                with st.spinner("Processing..."):
                    st.session_state.yearly_summary = load_data(f1, f2)
                    st.session_state.analysis_active = True
                    st.rerun()
            else: st.error("Missing yearlySummary.csv")

# STATE 2: Chat
else:
    # Sidebar
    with st.sidebar:
        if st.button("🔄 Upload New Files", use_container_width=True):
            st.session_state.analysis_active = False
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.caption(f"Loaded {st.session_state.yearly_summary['total_workouts']} workouts")

    st.title("💬 Chat with your TrainingPeaks Data")
    
    # History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
        
    # --- Random Question Button ---
    if st.button("🎲 Ask a Random Example Question", use_container_width=True):
        q = random.choice(SUGGESTED_QUESTIONS)
        handle_user_query(q, method="random_button")

    # Chat Input
    if prompt := st.chat_input("Ask about your training..."):
        handle_user_query(prompt, method="chat_input")