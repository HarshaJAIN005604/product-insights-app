# AI Product Inventor (MVP)

Beginner-friendly Streamlit app scaffold for the **AI Product Inventor** competition task.

Current MVP status:
- `Upload Data` tab is implemented with two source options:
  - Reddit: enter subreddit name and fetch posts + comments from last 3 months (subject to API limits)
  - Marketplace (Amazon/Nykaa/Flipkart etc.): upload one XLSX with exactly `Title` and `Body` columns (no extras)
- `Insights` tab is implemented with keyword, sentiment, and theme preview analytics.
- `Product Concepts` tab is implemented with scored concept brief generation.

## Project Structure

```text
product-insights-app/
|-- app.py
|-- requirements.txt
`-- README.md
```

## What This App Does Right Now

1. Upload Data uses a source toggle:
   - Reddit
   - Marketplace (Amazon/Nykaa/Flipkart etc.)
2. Reddit import:
   - input: subreddit name
   - output fields include `title`, `content`, and comment content
   - scope: last 3 months (or as many records as API returns within limits)
3. Marketplace XLSX import:
   - accepts one `.xlsx` file
   - required columns: exactly `Title`, `Body` (no extra columns)
   - includes one-click template download in the UI
4. After successful upload/import, `Next -> Insights` button becomes active.
5. Shows per-dataset summary:
   - row count
   - column count
   - first 5 rows preview
6. Insights tab shows:
   - top recurring pain phrases (top 15) with counts and evidence snippets
   - sentiment breakdown (`% Negative`, `% Neutral`, `% Positive`) + pie chart
   - top pain themes (3-5 clusters) with mention counts and 2 example quotes each
7. Product Concepts tab:
   - generates product concept briefs using the selected local LLM model (LLM-only mode)
   - each brief includes: name, target consumer, formulation direction, price/format, positioning
   - includes consumer-data citation snippets and mention counts
   - scores concepts on market size, competition intensity, and brand-capability alignment
   - marks top 2-3 as `Explore Now`
   - optional lightweight Transformers refinement (`google/flan-t5-base`) for polishing brief wording

## Prerequisites (Windows)

- Python 3.10+ (recommended: Python 3.11)
- `pip` available in terminal

Check versions:

```powershell
python --version
pip --version
```

## Local Setup (Windows PowerShell)

From this repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## Reddit API Setup

To use Reddit import, create Reddit app credentials:

1. Sign in to Reddit and open: `https://www.reddit.com/prefs/apps`
2. Click **create another app...**
3. Choose **script**
4. Save the generated values:
   - `client_id`
   - `client_secret`
   - your custom `user_agent` string

You can provide these directly in the app UI, or store them in Streamlit secrets.

## Streamlit Secrets (Optional, Recommended)

Create `.streamlit/secrets.toml`:

```toml
REDDIT_CLIENT_ID = "your_client_id"
REDDIT_CLIENT_SECRET = "your_client_secret"
REDDIT_USER_AGENT = "ai-product-inventor-app/0.1 by your_reddit_username"
```

If present, the app auto-fills Reddit credentials in the UI.

## Quick Test

1. Open **Upload Data** tab.
2. Choose `Marketplace (Amazon/Nykaa/Flipkart etc.)`.
3. Click **Download Marketplace XLSX Template**.
4. Fill sample rows in the template and upload it back.

## Common Errors and Fixes

1. File is not XLSX (Marketplace mode)
- Error: upload fails or parsing error.
- Fix: upload a `.xlsx` file generated from the template.

2. Reddit credentials missing/invalid
- Error: Reddit fetch fails.
- Fix: verify `client_id`, `client_secret`, `user_agent`, and subreddit name.

3. `Title` or `Body` column missing, or extra columns present (Marketplace XLSX)
- Error: app cannot process the file.
- Fix: keep exactly two columns in the file: `Title`, `Body`.

4. Empty XLSX
- Error: app reports empty/invalid file.
- Fix: include headers and at least one data row.

## Deploy to Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Go to Streamlit Community Cloud and click **New app**.
3. Select your repo, branch, and set main file path to `app.py`.
4. Deploy.

Notes:
- `requirements.txt` is included.
- Marketplace mode includes a downloadable XLSX template so the app can be demonstrated immediately.
- To avoid entering Reddit credentials every run, add them in **App settings -> Secrets**.
- For Streamlit free tier, keep Local LLM refinement OFF by default.
- If you enable Transformers refinement with `local_files_only=False`, the model will auto-download on first run (internet required).

## Next Stage (Planned)

- Evidence citations linking concepts back to uploaded consumer data
