# AI Product Inventor (India)

Streamlit app to convert consumer signals into ranked product opportunities for:
- `Hair Care`
- `Vitamin C Serum`

The app combines Marketplace reviews, Reddit discussions, and Google Trends data.

## Product Modes

Use the in-app selector:
- `Hair Care`
- `Vitamin C Serum`

Each product keeps isolated session data (`marketplace`, `reddit`, `trends`) so inputs do not mix across categories.

## App Flow

1. `Upload Data`
2. `Insights`
3. `Product Concepts`

## Step 1: Required Inputs (All 3 Mandatory)

You must complete all three before moving to Insights:

1. Marketplace Reviews (`.xlsx`)
- Required schema: exactly two columns: `Title`, `Body`
- Template download is available in-app

2. Reddit Discussions
- `Fetch Reddit Data` (public Reddit JSON, no API key)
- `Upload Prepared Reddit XLSX` (`Title`, `Body`)
- Data cleanup includes junk/mod/bot filtering, deduplication, and context filtering

3. Google Trends
- Category-aware trend links (Hair Care vs Vitamin C queries)
- Trends are split into 5-term batches
- Upload each batch CSV and app combines them

## Step 2: Insights

- Pain keyword bubble chart with theme mapping
- Sentiment split (`Negative`, `Neutral`, `Positive`)
- Theme cluster preview with quote evidence

## Step 3: Product Concepts

- Generates ranked opportunity concepts
- Concept cards include:
  - Opportunity score + components
  - Suggested price (`price_point_inr`, `price_band_inr`)
  - Review/forum/search evidence
- `View Brief` expands full concept details

For `Vitamin C Serum`, Stage 3 display content is adapted to skincare-relevant naming/format wording.

## Local Run (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

## Streamlit Cloud Notes

- Reddit fetch can be throttled in hosted environments (403/429).
- If fetch fails, use the prepared Reddit upload route.

## Repository

https://github.com/HarshaJAIN005604/product-insights-app
