# AI Training Analysis (Streamlit)

This repository contains a Streamlit app that analyzes yearly workout CSVs and lets users ask natural-language questions about their training data.

Quick start (local)
1. Clone the repo.
2. Create a virtual environment and activate it:

```bash
cd TrainingYearAnalysis
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file (copy `.env.example`) and add your `GOOGLE_API_KEY`.

```bash
cp .env.example .env
# edit .env and add your key
```

5. (Optional) Place per-user CSVs in `server_data/` named like `FirstLastYearlySummary.csv` and optional `FirstLastMetrics.csv`.

6. Run the app:

```bash
streamlit run app.py --server.port 8501
```

7. Open the app in your browser. To auto-load a user's data, use:

```
http://localhost:8501/?user=FirstLast
```

Notes
- Do not commit `.env` or your service account JSON to GitHub. Add them to `.gitignore`.
- If you want persistent logging of user prompts to Google Sheets, configure the service account JSON path and spreadsheet ID in `.env` and enable `LOG_TO_SHEETS=true`.

Files to upload to GitHub
- `app.py`, `requirements.txt`, `README.md`, `sheets_logger.py`, `.env.example`, and other source files
- Do NOT upload: `.env`, `.venv/`, `server_data/`, `__pycache__/`
