# AI Product Inventor

Beginner-friendly Streamlit app for turning uploaded marketplace reviews into:
- pain-point insights
- sentiment and theme summaries
- LLM-generated product concept briefs

## Current Flow

The app follows a forward journey:
1. `Upload Data`
2. `Insights`
3. `Product Concepts`

## Input Format

Only Marketplace upload is supported.

- File type: `.xlsx`
- Required columns: exactly `Title` and `Body`
- No extra columns allowed

You can download a ready template directly in the app.

## What You See

### 1) Upload Data
- Upload XLSX file
- Dataset summary:
  - row count
  - column count
  - first 5 rows preview
- Optional: upload Google Trends CSV for search-volume citation support
  - direct link in app to Google Trends Explore
  - download CSV there, then upload in app
- `Next -> Insights` activates after valid upload

### 2) Insights
- Top recurring pain phrases (complaint-focused)
- Sentiment breakdown:
  - `% Negative`, `% Neutral`, `% Positive`
  - compact pie chart
- Top pain themes:
  - mention count
  - up to 5 supporting quotes per theme
- `Next -> Product Concepts`

### 3) Product Concepts
- Auto-targets 8-10 concepts based on detected theme count
- Generation mode toggle:
  - `Deterministic`: uses built-in women hair-care knowledge base (profiles, ingredients, pricing, positioning)
  - `LLM`: uses local model generation
- Generates concepts sequentially (one by one)
- Shows generation progress at top
- Renders each concept as soon as it is ready
- Each concept contains only:
  - Product Name
  - Target Consumer Profile
  - Key Ingredients / Formulation Direction
  - Suggested Price Point
  - Format
  - Competitive Positioning

## Windows Setup

From repository root (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the URL shown in terminal (usually `http://localhost:8501`).

## Notes on Local LLM

- App uses local Transformers inference.
- Default model is `google/flan-t5-base`.
- On first run, model download may be required.
- Keep internet available for first download when `local_files_only=False`.

## Streamlit Community Cloud Deployment

1. Push this repo to GitHub.
2. Create a new app in Streamlit Community Cloud.
3. Set main file to `app.py`.
4. Deploy.

Notes:
- Dependencies are listed in `requirements.txt`.
- Model download time can impact cold starts on free tier.
