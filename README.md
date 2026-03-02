# Hair Care Opportunity Studio (India)

Streamlit app to turn women hair-care consumer data into:
- pain insights
- sentiment + theme clusters
- product opportunity concepts

## App Flow

1. `Upload Data`
2. `Insights`
3. `Product Concepts`

## Step 1: Required Inputs (All 3 Mandatory)

You must complete all three before moving to Insights:

1. Marketplace Reviews (`.xlsx`)
- Exact columns required: `Title`, `Body`
- Template download is available in-app

2. Reddit Discussions
- Option A: fetch from Reddit public endpoints (no API key)
- Option B: upload prepared Reddit `.xlsx` (`Title`, `Body`)
- Fetched/uploaded Reddit data is cleaned before use:
  - removes junk/bot/auto messages
  - removes duplicates
  - removes male-context rows for this women-focused workflow

3. Google Trends
- App generates multiple Trends batches (5 terms per batch)
- Each batch has:
  - `Open Google Trends Explore (Batch N)`
  - `Upload Google Trends CSV - Batch N`
- Uploaded batches are auto-combined into one trends dataset

## Step 2: Insights

- Top recurring pain keywords as bubble chart
  - each bubble maps to a theme
  - hover shows unique example snippets
- Sentiment breakdown (`% Negative`, `% Neutral`, `% Positive`) + pie chart
- Top pain themes with mentions + supporting quotes

## Step 3: Product Concepts

- Deterministic generation (LLM mode removed)
- Generates 8-10 concepts (based on detected theme count)
- Opportunity cards + `View Brief` details
- Evidence section shows short, relevant snippets pulled from Step 1 Marketplace + Reddit data

## Local Run (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

## Streamlit Community Cloud Deployment

1. Push repo to GitHub
2. Create app in Streamlit Community Cloud
3. Set entrypoint to `app.py`
4. Deploy

Notes:
- Reddit fetch can hit network/rate-limit restrictions in cloud environments.
- If that happens, use prepared Reddit XLSX upload route.
