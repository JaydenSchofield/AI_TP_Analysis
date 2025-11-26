#!/usr/bin/env python3
"""
Yearly Workout Analysis Script using Google Gemini AI

This script reads a yearly summary CSV file and allows users to ask
questions about their yearly training data using Google Gemini AI.
"""

import pandas as pd
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

# Load API key from .env file
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(env_path)
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


def parse_value(value):
    """
    Helper function to parse the 'Value' column from metrics.csv.
    Handles two cases:
    1. Composite string like "Min : 29 / Max : 96 / Avg : 55" -> extracts Avg (55)
    2. Simple number like "6.07" -> converts to float (6.07)
    
    Args:
        value: String or numeric value from the Value column
        
    Returns:
        Float value or None if parsing fails
    """
    if pd.isna(value):
        return None
    
    # Convert to string if not already
    value_str = str(value).strip()
    
    # Check if it's a composite string with "Avg :"
    if 'Avg :' in value_str:
        try:
            # Extract the Avg value
            parts = value_str.split('Avg :')
            if len(parts) > 1:
                avg_part = parts[1].strip()
                # Remove any trailing text after the number
                avg_value = avg_part.split()[0] if avg_part.split() else None
                if avg_value:
                    return float(avg_value)
        except (ValueError, IndexError):
            pass
    
    # Try to convert as simple number
    try:
        return float(value_str)
    except ValueError:
        return None


def process_metrics(metrics_csv_path):
    """
    Process the metrics.csv file to create a wide-format DataFrame with daily metrics.
    
    Args:
        metrics_csv_path: Path to the metrics.csv file
        
    Returns:
        DataFrame with date as index and metric types as columns
    """
    # Read metrics CSV
    df_metrics = pd.read_csv(metrics_csv_path)
    
    # Create 'date' column from 'Timestamp'
    if 'Timestamp' in df_metrics.columns:
        df_metrics['date'] = pd.to_datetime(df_metrics['Timestamp'], errors='coerce').dt.date
        # Convert to string, handling NaT values
        df_metrics['date'] = df_metrics['date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) and x is not None else None)
        # Filter out rows with invalid dates
        df_metrics = df_metrics[df_metrics['date'].notna()]
    else:
        raise ValueError("'Timestamp' column not found in metrics.csv")
    
    # Create 'value_numeric' column by parsing 'Value'
    if 'Value' in df_metrics.columns:
        df_metrics['value_numeric'] = df_metrics['Value'].apply(parse_value)
    else:
        raise ValueError("'Value' column not found in metrics.csv")
    
    # Pivot the DataFrame: date as index, Type as columns, value_numeric as values
    if 'Type' not in df_metrics.columns:
        raise ValueError("'Type' column not found in metrics.csv")
    
    df_pivoted = df_metrics.pivot_table(
        index='date',
        columns='Type',
        values='value_numeric',
        aggfunc='first'  # Use first value if duplicates exist
    )
    
    # Rename columns to be descriptive
    column_mapping = {
        'Sleep Hours': 'Sleep_Hours',
        'Body Battery': 'BodyBattery_Avg',
        'Stress Level': 'StressLevel_Avg',
        'Time In Deep Sleep': 'Sleep_Deep_Hours'
    }
    
    # Apply renaming for columns that exist
    df_pivoted = df_pivoted.rename(columns=column_mapping)
    
    return df_pivoted


def create_yearly_summary(yearly_csv_path, metrics_csv_path=None):
    """
    Read yearly summary CSV file and create a comprehensive summary dictionary.
    Handles planned vs actual data, sparse data, date parsing, and merges with metrics data.
    
    Args:
        yearly_csv_path: Path to the yearly summary CSV file
        metrics_csv_path: Path to the metrics.csv file (optional)
        
    Returns:
        Dictionary containing yearly summary data with planned/actual distinction and metrics
    """
    # Process metrics data first (if provided)
    df_metrics = None
    if metrics_csv_path and os.path.exists(metrics_csv_path):
        try:
            df_metrics = process_metrics(metrics_csv_path)
            print(f"  ✓ Loaded daily metrics data ({len(df_metrics)} days)")
        except Exception as e:
            print(f"  ⚠ Warning: Could not load metrics data: {str(e)}")
            df_metrics = None
    
    # Read Yearly Summary CSV
    df_yearly = pd.read_csv(yearly_csv_path)
    
    # Parse WorkoutDay as datetime and create 'date' column for merging
    if 'WorkoutDay' in df_yearly.columns:
        df_yearly['WorkoutDay_parsed'] = pd.to_datetime(df_yearly['WorkoutDay'], errors='coerce')
        # Create 'date' column as YYYY-MM-DD string for merging
        df_yearly['date'] = df_yearly['WorkoutDay_parsed'].dt.strftime('%Y-%m-%d')
        date_range = {
            'start': df_yearly['WorkoutDay_parsed'].min().strftime('%Y-%m-%d') if not pd.isna(df_yearly['WorkoutDay_parsed'].min()) else None,
            'end': df_yearly['WorkoutDay_parsed'].max().strftime('%Y-%m-%d') if not pd.isna(df_yearly['WorkoutDay_parsed'].max()) else None
        }
    else:
        date_range = None
        df_yearly['date'] = None
    
    # Merge with metrics data if available
    if df_metrics is not None and 'date' in df_yearly.columns and df_yearly['date'].notna().any():
        df_yearly = pd.merge(df_yearly, df_metrics, left_on='date', right_index=True, how='left')
        print(f"  ✓ Merged metrics data with workout data")
    elif df_metrics is not None:
        print(f"  ⚠ Warning: Cannot merge metrics - 'date' column missing or invalid in workout data")
    
    # Use df_yearly for all subsequent operations (now includes merged metrics)
    
    # Identify planned vs actual workouts
    # A workout is "completed" if it has actual data (e.g., TimeTotalInHours, DistanceInMeters)
    has_actual_data = pd.Series([False] * len(df_yearly), index=df_yearly.index)
    if 'TimeTotalInHours' in df_yearly.columns:
        has_actual_data = has_actual_data | df_yearly['TimeTotalInHours'].notna()
    if 'DistanceInMeters' in df_yearly.columns:
        has_actual_data = has_actual_data | df_yearly['DistanceInMeters'].notna()
    
    completed_workouts = df_yearly[has_actual_data]
    planned_only_workouts = df_yearly[~has_actual_data]
    
    # Define column groups
    planned_columns = ['PlannedDuration', 'PlannedDistanceInMeters', 'WorkoutDescription']
    actual_core_columns = ['DistanceInMeters', 'TimeTotalInHours', 'Energy', 'VelocityAverage', 'VelocityMax']
    power_columns = ['PowerAverage', 'PowerMax'] + [f'PWRZone{i}Minutes' for i in range(1, 11)]
    hr_columns = ['HeartRateAverage', 'HeartRateMax'] + [f'HRZone{i}Minutes' for i in range(1, 11)]
    # Add metrics columns to other_columns so they're included in statistics
    metrics_columns = ['Sleep_Hours', 'BodyBattery_Avg', 'StressLevel_Avg', 'Sleep_Deep_Hours']
    other_columns = ['CadenceAverage', 'CadenceMax', 'TorqueAverage', 'TorqueMax'] + metrics_columns
    calculated_columns = ['IF', 'TSS', 'Rpe', 'Feeling']
    
    # Calculate statistics only for non-NaN values (handle sparse data)
    def safe_stats(series):
        """Calculate statistics only on non-NaN values."""
        non_null = series.dropna()
        if len(non_null) == 0:
            return None
        return {
            'count': int(len(non_null)),
            'mean': float(non_null.mean()),
            'median': float(non_null.median()),
            'min': float(non_null.min()),
            'max': float(non_null.max()),
            'std': float(non_null.std()) if len(non_null) > 1 else None
        }
    
    # Create summary dictionary
    yearly_summary = {
        'total_workouts': len(df_yearly),
        'completed_workouts': int(len(completed_workouts)),
        'planned_only_workouts': int(len(planned_only_workouts)),
        'completion_rate': float(len(completed_workouts) / len(df_yearly)) if len(df_yearly) > 0 else 0,
        'date_range': date_range,
        'workout_types': df_yearly['WorkoutType'].value_counts().to_dict() if 'WorkoutType' in df_yearly.columns else {},
        'columns': list(df_yearly.columns),
        'data_quality': {}
    }
    
    # Calculate data quality metrics (non-null percentages)
    for col in df_yearly.columns:
        non_null_count = df_yearly[col].notna().sum()
        yearly_summary['data_quality'][col] = {
            'non_null_count': int(non_null_count),
            'non_null_percentage': float(non_null_count / len(df_yearly) * 100) if len(df_yearly) > 0 else 0
        }
    
    # Get numeric columns only (to avoid errors with string columns)
    numeric_columns = set(df_yearly.select_dtypes(include=['number']).columns)
    
    # Calculate statistics for completed workouts only (actual data)
    yearly_summary['actual_stats'] = {}
    for col in actual_core_columns + power_columns + hr_columns + other_columns + calculated_columns:
        if col in df_yearly.columns and col in numeric_columns:
            stats = safe_stats(completed_workouts[col])
            if stats:
                yearly_summary['actual_stats'][col] = stats
    
    # Calculate statistics for planned data
    yearly_summary['planned_stats'] = {}
    for col in planned_columns:
        if col in df_yearly.columns and col in numeric_columns:
            stats = safe_stats(df_yearly[col])
            if stats:
                yearly_summary['planned_stats'][col] = stats
    
    # Group by workout type for completed workouts
    if 'WorkoutType' in df_yearly.columns:
        yearly_summary['by_workout_type'] = {}
        for workout_type in completed_workouts['WorkoutType'].dropna().unique():
            type_data = completed_workouts[completed_workouts['WorkoutType'] == workout_type]
            yearly_summary['by_workout_type'][workout_type] = {
                'count': int(len(type_data)),
                'avg_duration_hours': float(type_data['TimeTotalInHours'].mean()) if 'TimeTotalInHours' in type_data.columns else None,
                'avg_distance_meters': float(type_data['DistanceInMeters'].mean()) if 'DistanceInMeters' in type_data.columns else None,
                'avg_power': float(type_data['PowerAverage'].mean()) if 'PowerAverage' in type_data.columns else None,
                'avg_heart_rate': float(type_data['HeartRateAverage'].mean()) if 'HeartRateAverage' in type_data.columns else None,
                'avg_tss': float(type_data['TSS'].mean()) if 'TSS' in type_data.columns else None
            }
    
    # Sample data (first 20 rows) - cleaned for JSON
    yearly_summary['data_preview'] = df_yearly.head(20).to_dict('records')
    for record in yearly_summary['data_preview']:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
            elif isinstance(value, (int, float)):
                record[key] = float(value) if not pd.isna(value) else None
            elif isinstance(value, pd.Timestamp):
                record[key] = value.strftime('%Y-%m-%d') if pd.notna(value) else None
    
    # Full data - cleaned for JSON (limit to prevent token overflow)
    # For very large datasets, we'll include a representative sample
    max_records = 200  # Limit full data to prevent token limits
    if len(df_yearly) > max_records:
        # Sample evenly across the dataset
        sample_indices = [int(i * len(df_yearly) / max_records) for i in range(max_records)]
        df_sample = df_yearly.iloc[sample_indices]
    else:
        df_sample = df_yearly
    
    yearly_summary['full_data'] = df_sample.to_dict('records')
    for record in yearly_summary['full_data']:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
            elif isinstance(value, (int, float)):
                record[key] = float(value) if not pd.isna(value) else None
            elif isinstance(value, pd.Timestamp):
                record[key] = value.strftime('%Y-%m-%d') if pd.notna(value) else None
    
    return yearly_summary


def get_yearly_ai_analysis(yearly_summary, user_question, user_profile=None):
    """
    Send yearly summary to Google Gemini AI for analysis.
    
    Args:
        yearly_summary: Dictionary containing yearly summary data
        user_question: String question from the user
        user_profile: Dictionary containing athlete's power and heart rate zones (optional)
        
    Returns:
        String response from the AI model
    """
    try:
        # Initialize the gemini-2.5-flash model
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Build context sections
        context_sections = []
        
        # Add athlete profile if provided
        if user_profile:
            context_sections.append(f"""Athlete's Training Zones:
{json.dumps(user_profile, indent=2)}""")
        
        # Add yearly summary with planned vs actual distinction
        date_range_str = "N/A"
        if yearly_summary['date_range'] and yearly_summary['date_range'].get('start'):
            date_range_str = f"{yearly_summary['date_range']['start']} to {yearly_summary['date_range'].get('end', 'N/A')}"
        
        context_sections.append(f"""Yearly Training Summary:
- Total Workouts (Planned + Completed): {yearly_summary['total_workouts']}
- Completed Workouts (with actual data): {yearly_summary['completed_workouts']}
- Planned Only (future workouts): {yearly_summary['planned_only_workouts']}
- Completion Rate: {yearly_summary['completion_rate']:.1%}
- Date Range: {date_range_str}
- Workout Types Distribution: {json.dumps(yearly_summary['workout_types'], indent=2)}

IMPORTANT DATA STRUCTURE:
This dataset contains BOTH planned and actual workout data:
- PLANNED data: PlannedDuration, PlannedDistanceInMeters, WorkoutDescription (workouts scheduled but not yet completed)
- ACTUAL data: TimeTotalInHours, DistanceInMeters, PowerAverage, HeartRateAverage, TSS, etc. (completed workouts)
- A completed workout may have both planned and actual data. A future workout only has planned data.

WELLNESS METRICS (merged daily data):
The dataset now includes daily recovery and wellness metrics merged by date:
- Sleep_Hours: Daily sleep duration
- BodyBattery_Avg: Average body battery level (recovery indicator)
- StressLevel_Avg: Average daily stress level
- Sleep_Deep_Hours: Time spent in deep sleep
These metrics are linked to workout days and should be used to analyze the relationship between recovery and training performance.

Statistics for COMPLETED workouts (actual performance metrics):
{json.dumps(yearly_summary['actual_stats'], indent=2)}

Statistics for PLANNED workouts:
{json.dumps(yearly_summary['planned_stats'], indent=2)}

Performance by Workout Type (completed workouts only):
{json.dumps(yearly_summary.get('by_workout_type', {}), indent=2)}

Data Quality (non-null percentages - note: data is sparse, many columns have NaN):
Key columns data availability:
{json.dumps({k: v for k, v in list(yearly_summary['data_quality'].items())[:20]}, indent=2)}

Sample Data (first 20 workouts showing planned vs actual):
{json.dumps(yearly_summary['data_preview'], indent=2)}""")
        
        # Include full data sample (limited to 200 records to prevent token overflow)
        context_sections.append(f"""Full Yearly Data Sample (up to 200 workouts, evenly sampled):
{json.dumps(yearly_summary['full_data'], indent=2)}""")
        
        # Construct the Smart Prompt
        prompt = f"""You are an expert triathlon and cycling coach analyzing an athlete's yearly training data. 

CRITICAL UNDERSTANDING:
- The data contains BOTH planned workouts (future/scheduled) and completed workouts (actual performance)
- When analyzing performance, focus on COMPLETED workouts (those with actual data like TimeTotalInHours, PowerAverage, TSS)
- Planned workouts (PlannedDuration, PlannedDistanceInMeters) represent scheduled training, not completed training
- The data is SPARSE - many columns have NaN values, especially for non-bike workouts (power data is bike-specific)
- WorkoutType is key: "Bike", "Swim", "Run", "Strength" - each has different relevant metrics
- WELLNESS METRICS: The dataset includes daily recovery metrics (Sleep_Hours, BodyBattery_Avg, StressLevel_Avg, Sleep_Deep_Hours) merged by date. Use these to analyze how recovery affects training performance, workout completion, and overall training load management.

Focus on identifying patterns, trends, and insights across completed workouts AND their relationship to recovery metrics if avaliable. Be insightful and provide actionable feedback.

IMPORTANT: Keep your response concise - limit your answer to 8 sentences maximum. Be direct and to the point. 

Context:

{chr(10).join(context_sections)}

Task:

Answer the following question. Keep your response concise and within 7 sentences. Where possible, reference specific statistics or data points from the summary and actional data must be in line with latest sports science principles.

Question:
{user_question}
"""
        
        # Call the model (relying on prompt instructions for length control)
        response = model.generate_content(prompt)
        
        # Handle response safely - check if there's valid text
        if response.text:
            return response.text
        else:
            # If response.text is empty, try to get text from candidates
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    return candidate.content.parts[0].text
            # Fallback error message
            return "Error: The AI response was empty or incomplete. Please try again."
        
    except Exception as e:
        print(f"Error calling Google Gemini API: {str(e)}")
        raise


def get_condensed_question(chat_history: list, new_question: str) -> str:
    """
    Condenses chat history and a new question into a single, standalone question
    using a fast AI call.
    """
    if not chat_history:
        # This is the first question, no history to condense
        return new_question
    
    # Format the chat history into a simple string
    history_string = ""
    for entry in chat_history:
        role = "User" if entry['role'] == 'user' else 'AI'
        history_string += f"{role}: {entry['parts'][0]}\n"
        
    system_prompt = f"""
    You are a prompt re-writer. Your task is to analyze the given chat history
    and the user's new question.
    Condense this information into a single, standalone question that
    preserves all necessary context from the history.
    
    The new question must make sense *without* the chat history.
    
    ---
    CHAT HISTORY:
    {history_string}
    ---
    
    NEW QUESTION:
    "{new_question}"
    ---
    
    CONDENSED, STANDALONE QUESTION:
    """
    
    try:
        # Use the same model (it's fast) for the cheap "re-write" call
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(system_prompt)
        
        # Safely extract text
        condensed_question = ""
        if response.text:
            condensed_question = response.text
        elif response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                condensed_question = candidate.content.parts[0].text

        if not condensed_question:
             # Fallback in case of empty response
            return new_question
            
        return condensed_question.strip()
        
    except Exception as e:
        print(f"Error in condensing question: {e}")
        # Fallback to the original question on error
        return new_question


if __name__ == "__main__":
    
    # --- 1. SETUP & DATA LOADING (Run ONCE) ---
    print("Loading and processing data, please wait...")

    # --- Athlete's Profile (Hardcoded) ---
    USER_PROFILE = {
        "power_zones": {
            "1": "0-142w",
            "2": "143-193w",
            "3": "194-231w",
            "4": "232-270w",
            "5": "271-308w",
            "6": "309-2000w"
        },
        "bike_heart_rate_zones": {
            "1": "38-144bpm",
            "2": "145-152bpm",
            "3": "153-160bpm",
            "4": "161-169bpm",
            "5": "170-178bpm",
            "6": "179-190bpm"
        },
        "run_heart_rate_zones": {
            "1": "38-147bpm",
            "2": "148-156bpm",
            "3": "157-164bpm",
            "4": "165-173bpm",
            "5": "174-182bpm",
            "6": "183-195bpm"
        }
    }
    
    # Set file paths
    base_dir = '/Users/jaydenschofield/Projects/TrainingFile_Analysis/source_csv'
    yearly_csv = os.path.join(base_dir, 'yearlySummary.csv')
    metrics_csv = os.path.join(base_dir, 'metrics.csv')
    
    # Check if required file exists
    if not os.path.exists(yearly_csv):
        print(f"Error: Required data file not found.")
        print(f"Checked for: {yearly_csv}")
        print("Please ensure the file exists at the specified path.")
        exit(1)
    
    # Check if metrics file exists (optional)
    if not os.path.exists(metrics_csv):
        print(f"⚠ Warning: Metrics file not found: {metrics_csv}")
        print("Continuing without recovery metrics (Sleep, Body Battery, etc.).")
        print("Analysis will be based on workout data only.")
        metrics_csv = None
    else:
        print(f"✓ Found metrics file: {metrics_csv}")
    
    # --- Process and cache the data ONCE ---
    # (This now includes the merged metrics data if available)
    yearly_summary = create_yearly_summary(yearly_csv, metrics_csv)
    # Debug: expose the activity date range used by router sampling
    dr = yearly_summary.get('date_range') if isinstance(yearly_summary, dict) else None
    if dr:
        print(f"[Debug] Yearly data date range: start={dr.get('start')} end={dr.get('end')}")
    else:
        print(f"[Debug] Yearly data date range: None")
    
    print("\n" + "=" * 80)
    print(" AI ANALYSIS CHATBOT INITIALIZED")
    print("=" * 80)
    print(f"✓ Loaded {yearly_summary['total_workouts']} total workouts ({yearly_summary['completed_workouts']} completed).")
    if metrics_csv:
        print(f"✓ Loaded recovery metrics (Sleep, Body Battery, etc.).")
    else:
        print(f"⚠ Recovery metrics not available (metrics.csv not found).")
    print("Type 'quit' or 'exit' to end the chat.")
    print("=" * 80)

    # --- 2. CHAT HISTORY (Initialize) ---
    chat_history = []

    # --- 3. CHAT LOOP (Run Continuously) ---
    while True:
        # Get new question from user
        new_question = input("\nAsk a question: ")
        
        if new_question.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        print("\nThinking...")
        
        # 1. Condense the question
        condensed_question = get_condensed_question(chat_history, new_question)

        # Debug: always print condensed standalone question for easier debugging
        print(f"[Debug] Condensed question: \"{condensed_question}\"")

        # 2. Get the main AI analysis
        ai_analysis = get_yearly_ai_analysis(yearly_summary, condensed_question, USER_PROFILE)
        
        # 3. Print the answer
        print("\n" + "--- AI Analysis ---")
        print(ai_analysis)
        print("-" * 80)

        # 4. Update the chat history
        # (Append the *original* question and the final answer)
        chat_history.append({"role": "user", "parts": [new_question]})
        chat_history.append({"role": "model", "parts": [ai_analysis]})

