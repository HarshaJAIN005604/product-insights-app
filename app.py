import io
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import quote
from urllib.request import Request, urlopen

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# -------------------------------
# Page configuration and title
# -------------------------------
st.set_page_config(
    page_title="Hair Care Opportunity Studio - India",
    page_icon=":bulb:",
    layout="wide",
)

st.markdown(
    """
    <style>
      :root {
        --brand-bg-1: #f6f7fb;
        --brand-bg-2: #eef2ff;
        --brand-bg-3: #ffffff;
        --brand-ink: #111827;
        --brand-muted: #4b5563;
        --brand-accent: #d97706;
        --brand-accent-soft: #fff7ed;
      }
      .stApp, [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 0% 0%, #dbeafe 0%, var(--brand-bg-2) 42%, var(--brand-bg-1) 100%);
        color: var(--brand-ink);
      }
      [data-testid="stHeader"] {
        background: transparent;
      }
      div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dbe3ef;
        border-radius: 14px;
        padding: 10px 12px;
      }
      .stAlert {
        border-radius: 12px;
      }
      .hero-wrap {
        background: linear-gradient(130deg, #ffffff 0%, #eef2ff 60%, #dbeafe 100%);
        border: 1px solid #d6dff0;
        border-radius: 18px;
        padding: 18px 20px;
        margin-bottom: 10px;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.08);
      }
      .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: var(--brand-ink);
        margin-bottom: 4px;
      }
      .hero-sub {
        color: var(--brand-muted);
        font-size: 1rem;
        margin-bottom: 10px;
      }
      .chip-row {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
      }
      .chip {
        background: var(--brand-accent-soft);
        color: #9a4f0e;
        border: 1px solid #fed7aa;
        border-radius: 999px;
        padding: 4px 10px;
        font-size: 0.82rem;
        font-weight: 600;
      }
      .ingest-card {
        background: linear-gradient(165deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #dbe3ef;
        border-radius: 16px;
        padding: 14px 14px 10px 14px;
        margin-bottom: 10px;
      }
      .stTabs [data-baseweb="tab"] {
        font-weight: 600;
      }
      @media (max-width: 900px) {
        .hero-title { font-size: 1.5rem; }
      }
    </style>
    <div class="hero-wrap">
      <div class="hero-title">Women’s Hair Care Opportunity Studio (India)</div>
      <div class="hero-sub">
        Turn consumer pain signals into evidence-backed product opportunities for the Indian women’s hair-care market.
      </div>
      <div class="chip-row">
        <div class="chip">Focus: Women in India</div>
        <div class="chip">Category: Hair Care</div>
        <div class="chip">Inputs: Reviews + Trends</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------
# Session state defaults
# -------------------------------
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Upload Data"
if "current_dataset" not in st.session_state:
    st.session_state.current_dataset = None
if "current_dataset_name" not in st.session_state:
    st.session_state.current_dataset_name = ""
if "current_dataset_source" not in st.session_state:
    st.session_state.current_dataset_source = ""
if "marketplace_dataset" not in st.session_state:
    st.session_state.marketplace_dataset = None
if "marketplace_dataset_name" not in st.session_state:
    st.session_state.marketplace_dataset_name = ""
if "reddit_dataset" not in st.session_state:
    st.session_state.reddit_dataset = None
if "reddit_dataset_name" not in st.session_state:
    st.session_state.reddit_dataset_name = ""
if "current_trends_dataset" not in st.session_state:
    st.session_state.current_trends_dataset = None
if "current_trends_dataset_name" not in st.session_state:
    st.session_state.current_trends_dataset_name = ""

FLOW_STEPS = ["Upload Data", "Insights", "Product Concepts"]


# -------------------------------
# Text analysis configuration
# -------------------------------
GENERIC_WORDS = {
    "product",
    "products",
    "item",
    "items",
    "amazon",
    "flipkart",
    "nykaa",
    "buy",
    "bought",
    "purchase",
    "use",
    "using",
    "used",
    "one",
    "get",
    "got",
    "really",
    "very",
    "also",
    "would",
    "could",
    "even",
    "like",
    "review",
    "reviews",
    "brand",
    "price",
}
CATEGORY_NOUNS = {
    "hair",
    "skin",
    "face",
    "scalp",
    "product",
    "products",
    "shampoo",
    "serum",
    "conditioner",
    "oil",
    "cream",
    "gel",
    "lotion",
    "mask",
    "toner",
    "cleanser",
    "moisturizer",
}
DOMAIN_STOPWORDS = set(ENGLISH_STOP_WORDS).union(GENERIC_WORDS).union(CATEGORY_NOUNS)

NEGATION_CUES = {
    "not",
    "no",
    "never",
    "dont",
    "didnt",
    "doesnt",
    "cant",
    "wont",
    "isnt",
    "arent",
}
COMPLAINT_CUES = {
    "waste",
    "worst",
    "irritated",
    "irritation",
    "itchy",
    "itching",
    "rash",
    "dry",
    "dryness",
    "greasy",
    "oily",
    "sticky",
    "breakout",
    "breakouts",
    "acne",
    "pimples",
    "dandruff",
    "hairfall",
    "shedding",
    "fall",
    "slow",
    "burning",
    "redness",
}
COMPLAINT_PATTERN = re.compile(
    r"\b(not|no|never|didn't|doesn't|don't|can't|won't|waste|worst|irritated|"
    r"itching|itchy|rash|dry|dryness|greasy|sticky|breakouts?|acne|pimples?|"
    r"dandruff|hair\s*fall|hairfall|shedding|slow)\b",
    re.IGNORECASE,
)
VARIANT_MAP = {
    "hairfall": "hair_fall",
    "shedding": "hair_fall",
    "itchy": "itching",
    "breakout": "breakouts",
    "acne": "breakouts",
    "pimple": "breakouts",
    "pimples": "breakouts",
    "oily": "greasy",
    "greasiness": "greasy",
}

THEME_RULES = [
    {
        "title": "No Visible Results After Long-Term Use",
        "patterns": [
            r"\b(no|zero|not (seeing|noticed|much))\s+(result|results|improvement|regrowth)\b",
            r"\b(no|not).{0,25}(work|working|effect|effective|difference)\b",
            r"\b(after|for)\s+\d+\s*(day|days|week|weeks|month|months).{0,30}\b(no|not).{0,20}(result|improvement|regrowth)\b",
        ],
    },
    {
        "title": "Rebound Hair Fall After Stopping Usage",
        "patterns": [
            r"\b(after|when).{0,20}(stop|stopped|stopping|discontinued|quit).{0,30}(hair\s*fall|shedding|fall)\b",
            r"\b(hair\s*fall|shedding).{0,30}(after|when).{0,20}(stop|stopped|stopping|discontinued|quit)\b",
            r"\bincreased?\s+(hair\s*fall|shedding)\b",
        ],
    },
    {
        "title": "Product Runs Out Too Fast",
        "patterns": [
            r"\b(bottle|pack|container|tube).{0,30}(lasted|lasts|finished|empty|ran out)\b",
            r"\b(only|just)\s+\d+\s*(day|days|week|weeks)\b",
            r"\bran out too fast\b",
        ],
    },
    {
        "title": "Scalp Dryness & Irritation",
        "patterns": [
            r"\b(scalp\s+)?(dry|dryness|flaky|flakes|itchy|itching|irritated|irritation|rash|burning|redness)\b",
            r"\bcaused?\s+(dryness|itching|irritation|rash|burning)\b",
        ],
    },
    {
        "title": "Breakouts & Acne Reactions",
        "patterns": [
            r"\b(breakouts?|acne|pimples?|comedones?)\b",
            r"\bcaused?\s+(breakouts?|acne|pimples?)\b",
            r"\b(skin|forehead|face).{0,20}(breakouts?|acne|pimples?)\b",
        ],
    },
    {
        "title": "Sticky Texture & Greasiness",
        "patterns": [
            r"\b(sticky|greasy|oily|greasiness)\b",
            r"\b(residue|heavy|chipchip|chip chipa)\b",
            r"\btoo\s+(greasy|oily|sticky)\b",
        ],
    },
    {
        "title": "Packaging & Dispensing Failure",
        "patterns": [
            r"\b(pump|bottle|packaging|dropper|cap|leak|leaking)\b",
            r"\b(not working|broken|defective|stopped working)\b",
            r"\b(package|packaging).{0,20}(bad|poor|damaged|faulty)\b",
        ],
    },
    {
        "title": "Perceived Waste of Money",
        "patterns": [
            r"\bwaste of money\b",
            r"\bmoney wasted\b",
            r"\bnot worth\b",
            r"\b(overpriced|too expensive|very expensive|expensive)\b",
        ],
    },
]

THEME_PRIORITY = {rule["title"]: idx for idx, rule in enumerate(THEME_RULES)}

COMPETITION_BY_FORMAT = {
    "Serum": 72,
    "Shampoo": 82,
    "Tablet": 68,
    "Gummy": 78,
    "Scalp Tonic": 64,
    "Leave-In Mist": 66,
}

FORMAT_DEMAND_PRIOR = {
    "Serum": 74,
    "Shampoo": 68,
    "Tablet": 61,
    "Gummy": 58,
}

FORMAT_OPTIONS = ["Serum", "Shampoo", "Tablet", "Gummy"]

THEME_FORMAT_HINTS = {
    "No Visible Results After Long-Term Use": ["Serum", "Tablet", "Gummy"],
    "Rebound Hair Fall After Stopping Usage": ["Serum", "Gummy", "Tablet"],
    "Product Runs Out Too Fast": ["Serum", "Shampoo", "Tablet"],
    "Scalp Dryness & Irritation": ["Shampoo", "Serum", "Gummy"],
    "Perceived Waste of Money": ["Shampoo", "Tablet", "Serum"],
}

# Women-focused hair-care concept database for deterministic generation.
WOMEN_HAIRCARE_THEME_DB = {
    "No Visible Results After Long-Term Use": {
        "name_stems": ["Follica", "Reviva", "Rootrise", "Antra", "Denselle", "Regaina"],
        "consumer_profiles": [
            "Women aged 23-40 with visible thinning around the part line after using growth products for 8-16 weeks without clear progress.",
            "Urban working women with stress-led diffuse thinning who want measurable regrowth milestones and easy daily adherence.",
            "Women with early-stage widening part who have already tried one or two products and now demand stronger evidence-backed efficacy.",
        ],
        "positioning_angles": [
            "Outcome-tracking regimen with visible progress checkpoints by week 4, 8, and 12.",
            "High-active yet lightweight system designed for faster perceived cosmetic density.",
            "Clinically framed responder-focused solution for women disappointed by low-efficacy routines.",
        ],
        "price_tier": "premium",
    },
    "Rebound Hair Fall After Stopping Usage": {
        "name_stems": ["Stabilia", "HoldRoot", "StayGrow", "Sustaina", "Anchora", "RootLock"],
        "consumer_profiles": [
            "Women aged 25-42 who notice renewed shedding within 2-6 weeks after discontinuing prior hair-fall treatments.",
            "Women with cyclical hair-fall patterns seeking a taper-friendly regimen that avoids rebound after stopping.",
            "Post-treatment users who need maintenance support to preserve gains and reduce drop-off shedding.",
        ],
        "positioning_angles": [
            "Built for transition and maintenance phases to minimize rebound hair-fall after stop-use.",
            "Retention-first formulation focused on stabilizing follicles across on/off usage cycles.",
            "Long-horizon fall-control strategy vs short-term cosmetic fixes.",
        ],
        "price_tier": "mid",
    },
    "Product Runs Out Too Fast": {
        "name_stems": ["LongUse", "Endura", "DailyDose", "MonthMax", "ValueFlow", "SustainDrop"],
        "consumer_profiles": [
            "Value-conscious women who are willing to pay for efficacy but expect 30-day minimum usage from one pack.",
            "Routine-driven women seeking controlled-dose packaging that avoids over-dispensing and wastage.",
            "Frequent repurchasers frustrated by products that run out too quickly relative to claims.",
        ],
        "positioning_angles": [
            "Dose-optimized delivery system that stretches usage without diluting active performance.",
            "High-efficiency format engineered for lower monthly cost-to-benefit.",
            "Designed for complete monthly adherence with predictable consumption.",
        ],
        "price_tier": "value",
    },
    "Scalp Dryness & Irritation": {
        "name_stems": ["Calmi", "Barriera", "DermaRoot", "SootheScalp", "Ceracalm", "HydraScalp"],
        "consumer_profiles": [
            "Women with sensitive scalp prone to itching, flaking, or tightness after active hair products.",
            "Women aged 22-38 managing dryness and irritation alongside hair-fall concerns.",
            "Users who discontinue treatments due to scalp discomfort and need a barrier-safe alternative.",
        ],
        "positioning_angles": [
            "Barrier-first scalp care that combines comfort and anti-shedding support.",
            "Low-irritation performance formula for sensitive scalp users who still need efficacy.",
            "Dermal-comfort positioning: reduce itch and dryness while supporting fuller-looking hair.",
        ],
        "price_tier": "mid",
    },
    "Perceived Waste of Money": {
        "name_stems": ["SmartRoot", "WorthIt", "ProofGrow", "ValueClin", "Evida", "BudgetPro"],
        "consumer_profiles": [
            "Price-sensitive women comparing alternatives and demanding evidence per rupee spent.",
            "Women who felt prior solutions under-delivered and now seek transparent value-for-money claims.",
            "First-time buyers needing affordable but credible entry products for hair-fall and thinning.",
        ],
        "positioning_angles": [
            "Efficacy-per-rupee proposition with transparent active dosages and realistic timelines.",
            "Performance-value hybrid designed to beat overpriced incumbents.",
            "Affordable clinical logic with no-frills but high-impact actives.",
        ],
        "price_tier": "value",
    },
}

WOMEN_HAIRCARE_INGREDIENT_LIBRARY = {
    "Serum": [
        "Redensyl",
        "Capixyl",
        "Procapil",
        "Caffeine",
        "Niacinamide",
        "Pea Sprout Extract (AnaGain)",
        "Copper Peptides",
        "Biotinyl Tripeptide-1",
        "Panthenol",
        "Ceramide Complex",
    ],
    "Shampoo": [
        "Piroctone Olamine",
        "Salicylic Acid",
        "Zinc PCA",
        "Ketoconazole Alternative Complex",
        "Tea Tree Fraction",
        "Panthenol",
        "Allantoin",
        "Ceramide NP",
        "Oat Beta-Glucan",
    ],
    "Tablet": [
        "Biotin",
        "Zinc",
        "Vitamin D3",
        "Iron Bisglycinate",
        "Folic Acid",
        "Selenium",
        "Marine Collagen Peptides",
        "Saw Palmetto",
        "Ashwagandha",
        "Pumpkin Seed Extract",
    ],
    "Gummy": [
        "Biotin",
        "Zinc",
        "Vitamin C",
        "Vitamin D3",
        "Folic Acid",
        "B12",
        "Amla Extract",
        "Bamboo Silica",
    ],
}

WOMEN_HAIRCARE_FORMAT_NOTES = {
    "Serum": "lightweight non-sticky leave-on scalp serum for nightly use",
    "Shampoo": "sulfate-controlled cleansing base for frequent use",
    "Tablet": "once-daily oral tablet protocol for 90-day adherence",
    "Gummy": "easy-compliance daily gummy format with taste-masked actives",
}

WOMEN_HAIRCARE_PRICE_DB = {
    "Serum": {"value": (599, 749, "50 ml"), "mid": (799, 999, "50 ml"), "premium": (1049, 1399, "50 ml")},
    "Shampoo": {"value": (449, 599, "200 ml"), "mid": (649, 799, "200 ml"), "premium": (849, 1099, "200 ml")},
    "Tablet": {"value": (499, 649, "30 tablets"), "mid": (699, 899, "30 tablets"), "premium": (949, 1199, "30 tablets")},
    "Gummy": {"value": (549, 699, "45 gummies"), "mid": (749, 899, "45 gummies"), "premium": (949, 1149, "45 gummies")},
}

WOMEN_HAIRCARE_BENEFIT_WORDS = [
    "Root Renewal",
    "Density Restore",
    "Fall Control",
    "Scalp Comfort",
    "Growth Support",
    "Shed Defense",
    "Barrier Balance",
    "Volume Recovery",
]

THEME_NAME_DESCRIPTORS = {
    "No Visible Results After Long-Term Use": ["Progress", "Visible", "Responder", "12-Week"],
    "Rebound Hair Fall After Stopping Usage": ["Stabilize", "Retention", "Sustain", "Transition"],
    "Product Runs Out Too Fast": ["Value", "Daily", "Month", "Dose"],
    "Scalp Dryness & Irritation": ["Barrier", "Calm", "Comfort", "Hydra"],
    "Breakouts & Acne Reactions": ["Clear", "Gentle", "Non-Comedo", "Derma"],
    "Sticky Texture & Greasiness": ["Light", "Air", "Fresh", "Feather"],
    "Packaging & Dispensing Failure": ["SmartPack", "EasyDose", "Flow", "Lock"],
    "Perceived Waste of Money": ["Proof", "Smart", "Worth", "Save"],
}

INGREDIENT_BANK = {
    "regrowth": ["Caffeine", "Peptide Complex", "Niacinamide", "Redensyl"],
    "hair_fall": ["Anagain", "Amino Complex", "Saw Palmetto", "Pumpkin Seed Extract"],
    "dryness_irritation": ["Ceramide", "Oat Extract", "Panthenol", "Allantoin"],
    "dandruff_flakes": ["Piroctone Olamine", "Zinc PCA", "Salicylic Acid", "Tea Tree Fraction"],
    "budget_value": ["High-concentration Active Base", "Optimized Dosage System", "Stabilized Blend"],
}
INGREDIENT_KEYWORDS = {
    term.lower()
    for terms in INGREDIENT_BANK.values()
    for term in terms
}

POSITIVE_WORDS = {
    "good",
    "great",
    "amazing",
    "best",
    "love",
    "liked",
    "effective",
    "works",
    "helpful",
    "smooth",
    "fast",
    "improved",
    "healthy",
}
NEGATIVE_WORDS = {
    "bad",
    "worst",
    "hate",
    "dryness",
    "dry",
    "dandruff",
    "sticky",
    "greasy",
    "greasiness",
    "hairfall",
    "hair_fall",
    "hair",
    "fall",
    "itchy",
    "itching",
    "slow",
    "waste",
    "expensive",
    "irritation",
    "rough",
    "breakout",
    "breakouts",
    "flakes",
    "flaky",
}


# -------------------------------
# Helper functions
# -------------------------------
def read_xlsx_file(uploaded_file) -> pd.DataFrame:
    """Read an uploaded XLSX file into a pandas DataFrame."""
    uploaded_file.seek(0)
    return pd.read_excel(uploaded_file)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names by trimming spaces and lowercasing for easier matching."""
    normalized_df = df.copy()
    normalized_df.columns = [str(col).strip().lower() for col in normalized_df.columns]
    return normalized_df


def build_dataset_summary(df: pd.DataFrame) -> Dict[str, int]:
    """Create a minimal summary for one dataset."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
    }


def render_dataset_summary_block(df: pd.DataFrame) -> Optional[Dict[str, int]]:
    """Render dataset summary with only rows, columns, and preview."""
    if df.empty:
        st.warning("This dataset has no rows. Add data and try again.")
        return None

    summary = build_dataset_summary(df)
    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Rows", summary["rows"])
    metric_col2.metric("Columns", summary["columns"])

    st.markdown("Preview (first 5 rows):")
    st.dataframe(df.head(5), use_container_width=True)
    return summary


def prepare_marketplace_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize marketplace XLSX (expects only Title and Body columns)."""
    normalized_df = normalize_columns(df)
    expected_cols = {"title", "body"}
    actual_cols = set(normalized_df.columns)

    if actual_cols != expected_cols or len(normalized_df.columns) != 2:
        raise ValueError(
            "Marketplace upload must contain exactly two columns: Title and Body, "
            "with no extra columns."
        )

    cleaned_df = normalized_df.copy()
    combined = (
        cleaned_df["title"].astype(str).fillna("")
        + " "
        + cleaned_df["body"].astype(str).fillna("")
    )
    male_mask = combined.apply(is_male_authored_or_male_context)
    filtered_count = int(male_mask.sum())
    cleaned_df = cleaned_df.loc[~male_mask].reset_index(drop=True)
    cleaned_df.attrs["male_filtered"] = filtered_count
    return cleaned_df


def build_marketplace_template_xlsx_bytes() -> bytes:
    """Create a starter XLSX template for marketplace uploads."""
    template_df = pd.DataFrame(
        [
            {
                "Title": "Leakproof travel bottle is hard to find",
                "Body": "Most bottles leak in my backpack after a few days of use.",
            },
            {
                "Title": "Hair serum bottle pump stops working",
                "Body": "The product is good but the pump fails and wastes serum.",
            },
        ]
    )
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        template_df.to_excel(writer, index=False, sheet_name="template")
    output.seek(0)
    return output.getvalue()


def build_dataset_xlsx_bytes(df: pd.DataFrame) -> bytes:
    """Export a dataset into Shareable XLSX with Title/Body columns."""
    if df is None or df.empty:
        raise ValueError("Dataset is empty. Nothing to export.")
    export_df = df.copy()
    export_df.columns = [str(col).strip().lower() for col in export_df.columns]
    if "title" not in export_df.columns:
        export_df["title"] = ""
    if "body" not in export_df.columns:
        export_df["body"] = ""
    export_df = export_df[["title", "body"]].fillna("")
    export_df = export_df.rename(columns={"title": "Title", "body": "Body"})
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="reddit_dataset")
    output.seek(0)
    return output.getvalue()


def clean_reddit_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Apply Reddit cleanup: remove junk/bot, male-context, empties, and duplicates."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["title", "body"]), "Reddit cleaning: no rows to clean."
    working = normalize_columns(df)
    if "title" not in working.columns:
        working["title"] = ""
    if "body" not in working.columns:
        working["body"] = ""
    working = working[["title", "body"]].fillna("")
    before = len(working)

    junk_mask = working.apply(
        lambda row: is_reddit_junk_text(str(row.get("title", "")), str(row.get("body", ""))),
        axis=1,
    )
    male_mask = (
        working["title"].astype(str).str.cat(working["body"].astype(str), sep=" ").apply(is_male_authored_or_male_context)
    )
    empty_mask = (
        working["title"].astype(str).str.strip().eq("")
        & working["body"].astype(str).str.strip().eq("")
    )

    filtered = working.loc[~(junk_mask | male_mask | empty_mask)].copy()
    after_filter = len(filtered)

    filtered["dedupe_key"] = (
        filtered["title"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        + " || "
        + filtered["body"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    deduped = filtered.drop_duplicates(subset=["dedupe_key"]).drop(columns=["dedupe_key"]).reset_index(drop=True)
    after_dedupe = len(deduped)

    summary = (
        f"Reddit cleaning applied: kept {after_dedupe}/{before} rows "
        f"(removed junk/bot={int(junk_mask.sum())}, male-context={int(male_mask.sum())}, "
        f"empty={int(empty_mask.sum())}, duplicates={max(0, after_filter - after_dedupe)})."
    )
    return deduped, summary


def combine_trends_batches(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Combine multiple Google Trends batch CSVs into one table."""
    if not frames:
        return pd.DataFrame()
    combined = frames[0].copy().reset_index(drop=True)
    for idx, frame in enumerate(frames[1:], start=2):
        current = frame.copy().reset_index(drop=True)
        if current.empty:
            continue
        if not current.columns.empty:
            first_col = str(current.columns[0])
            combined_first = str(combined.columns[0]) if not combined.columns.empty else ""
            if (
                first_col
                and combined_first
                and first_col.lower() == combined_first.lower()
                and len(current) == len(combined)
                and current.iloc[:, 0].astype(str).equals(combined.iloc[:, 0].astype(str))
            ):
                current = current.iloc[:, 1:]
        renamed_cols = []
        for col in current.columns:
            c = str(col)
            if c in combined.columns:
                c = f"{c} [batch {idx}]"
            renamed_cols.append(c)
        current.columns = renamed_cols
        combined = pd.concat([combined, current], axis=1)
    return combined


def build_combined_required_dataset() -> Optional[pd.DataFrame]:
    """Combine mandatory text sources (marketplace + reddit) into one analysis dataset."""
    parts: List[pd.DataFrame] = []
    marketplace_df = st.session_state.marketplace_dataset
    reddit_df = st.session_state.reddit_dataset
    if isinstance(marketplace_df, pd.DataFrame) and not marketplace_df.empty:
        parts.append(marketplace_df[["title", "body"]].copy())
    if isinstance(reddit_df, pd.DataFrame) and not reddit_df.empty:
        parts.append(reddit_df[["title", "body"]].copy())
    if not parts:
        return None
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.fillna("")
    combined = combined.drop_duplicates(subset=["title", "body"]).reset_index(drop=True)
    return combined


def refresh_combined_dataset_state() -> None:
    """Refresh current dataset state from mandatory source datasets."""
    combined = build_combined_required_dataset()
    if combined is None or combined.empty:
        st.session_state.current_dataset = None
        st.session_state.current_dataset_name = ""
        st.session_state.current_dataset_source = ""
    else:
        st.session_state.current_dataset = combined
        st.session_state.current_dataset_name = "combined_marketplace_reddit"
        st.session_state.current_dataset_source = "combined"


def reddit_public_get_json(url: str, timeout_sec: int = 25) -> Dict[str, object]:
    """Fetch Reddit public JSON endpoint with a browser-like User-Agent."""
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; HairCareOpportunityStudio/1.0; +https://example.com)",
            "Accept": "application/json",
        },
    )
    with urlopen(req, timeout=timeout_sec) as response:
        payload = response.read().decode("utf-8", errors="ignore")
    return json.loads(payload)


def is_reddit_junk_text(title: str, body: str, author: str = "") -> bool:
    """Detect moderation/admin boilerplate that should not be treated as consumer voice."""
    combined = f"{title} {body}".lower()
    author_lower = author.lower().strip()
    junk_patterns = [
        r"\byour post (was|has been) removed\b",
        r"\bawaiting moderator approval\b",
        r"\bpending moderator approval\b",
        r"\bsubmitted and is awaiting\b",
        r"\bthis post has been removed\b",
        r"\bremoved by moderators\b",
        r"\bautomoderator\b",
        r"\bmoderator message\b",
        r"\bplease contact the moderators\b",
        r"\bkindly post in (the )?weekly\b",
        r"\bthis thread is for\b",
    ]
    if author_lower in {"automoderator", "mod", "moderator"}:
        return True
    return any(re.search(pattern, combined) for pattern in junk_patterns)


def is_male_authored_or_male_context(text: str) -> bool:
    """Best-effort filter for male-authored/male-context text in a women-focused workflow."""
    t = str(text or "").lower().strip()
    if not t:
        return False
    male_patterns = [
        r"\bmale\b",
        r"\bman\b",
        r"\bguy\b",
        r"\bboy\b",
        r"\bhusband\b",
        r"\b\d{1,2}\s*m\b",   # e.g., 28 M
        r"\b\d{1,2}m\b",      # e.g., 28M
        r"\bm\s*\d{1,2}\b",   # e.g., M28 / M 28
        r"\b(?:i am|i'm|im|as a)\s+(?:male|man|guy|boy)\b",
        r"\b(?:male|man|guy)\s+here\b",
    ]
    return any(re.search(pattern, t) for pattern in male_patterns)


def iter_reddit_comment_bodies(children: List[Dict[str, object]]) -> List[str]:
    """Flatten Reddit nested comments from listing children."""
    collected: List[str] = []
    for child in children:
        if not isinstance(child, dict):
            continue
        if child.get("kind") != "t1":
            continue
        data = child.get("data", {}) if isinstance(child.get("data"), dict) else {}
        body = str(data.get("body", "")).strip()
        author = str(data.get("author", "")).strip()
        if (
            body
            and body not in {"[deleted]", "[removed]"}
            and not is_reddit_junk_text("", body, author)
            and not is_male_authored_or_male_context(body)
        ):
            collected.append(body)
        replies = data.get("replies")
        if isinstance(replies, dict):
            reply_data = replies.get("data", {}) if isinstance(replies.get("data"), dict) else {}
            reply_children = reply_data.get("children", []) if isinstance(reply_data.get("children"), list) else []
            collected.extend(iter_reddit_comment_bodies(reply_children))
    return collected


def fetch_reddit_public_discussions(
    subreddits: List[str],
    keywords: List[str],
    days_back: int = 90,
    max_posts_per_subreddit: int = 80,
    max_comments_per_post: int = 40,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> pd.DataFrame:
    """Fetch Reddit posts/comments without API keys using public JSON endpoints."""
    if not subreddits:
        raise ValueError("Please provide at least one subreddit.")
    if not keywords:
        raise ValueError("Please provide at least one keyword.")

    after_ts = int((datetime.now(timezone.utc).timestamp()) - (days_back * 86400))
    keyword_tokens = {
        tok
        for phrase in keywords
        for tok in re.findall(r"[a-z0-9]+", phrase.lower())
        if len(tok) >= 3
    }
    rows: List[Dict[str, str]] = []
    diagnostics = {
        "subreddits_requested": len(subreddits),
        "subreddits_processed": 0,
        "listing_requests": 0,
        "comment_requests": 0,
        "posts_seen": 0,
        "posts_in_window": 0,
        "posts_collected": 0,
        "comments_collected": 0,
        "skipped_old_posts": 0,
        "fallback_used_subreddits": 0,
        "fallback_posts_seen": 0,
        "fallback_posts_matched": 0,
        "request_errors": 0,
        "junk_filtered": 0,
        "male_filtered": 0,
        "error_examples": [],
    }
    subreddit_comment_counts: Dict[str, int] = {}

    for subreddit in subreddits:
        normalized_sub = subreddit.strip().replace("r/", "").replace("/", "")
        if not normalized_sub:
            continue
        subreddit_comment_counts.setdefault(normalized_sub, 0)
        diagnostics["subreddits_processed"] += 1
        query = quote(" OR ".join(keywords))
        after_token = ""
        fetched_posts = 0
        request_pages = 0

        while fetched_posts < max_posts_per_subreddit and request_pages < 12:
            listing_url = (
                f"https://www.reddit.com/r/{normalized_sub}/search.json"
                f"?q={query}&restrict_sr=1&sort=new&t=year&limit=100"
                + (f"&after={after_token}" if after_token else "")
            )
            try:
                payload = reddit_public_get_json(listing_url)
                diagnostics["listing_requests"] += 1
            except Exception as exc:
                diagnostics["request_errors"] += 1
                if len(diagnostics["error_examples"]) < 5:
                    diagnostics["error_examples"].append(f"listing r/{normalized_sub}: {exc}")
                break
            data_block = payload.get("data", {}) if isinstance(payload.get("data"), dict) else {}
            children = data_block.get("children", []) if isinstance(data_block.get("children"), list) else []
            after_token = str(data_block.get("after", "") or "")
            request_pages += 1

            if not children:
                break

            stop_for_time = False
            for child in children:
                if fetched_posts >= max_posts_per_subreddit:
                    break
                child_data = child.get("data", {}) if isinstance(child.get("data"), dict) else {}
                diagnostics["posts_seen"] += 1
                if progress_callback:
                    progress_callback(normalized_sub, int(subreddit_comment_counts.get(normalized_sub, 0)))
                created_utc = int(child_data.get("created_utc", 0) or 0)
                if created_utc < after_ts:
                    stop_for_time = True
                    diagnostics["skipped_old_posts"] += 1
                    continue
                diagnostics["posts_in_window"] += 1
                post_id = str(child_data.get("id", "")).strip()
                title = str(child_data.get("title", "")).strip()
                body = str(child_data.get("selftext", "")).strip()
                author = str(child_data.get("author", "")).strip()
                if is_reddit_junk_text(title, body, author):
                    diagnostics["junk_filtered"] += 1
                    continue
                if is_male_authored_or_male_context(f"{title} {body}"):
                    diagnostics["male_filtered"] += 1
                    continue
                if title or body:
                    rows.append({"title": title, "body": body})
                    diagnostics["posts_collected"] += 1

                if post_id:
                    comments_url = (
                        f"https://www.reddit.com/comments/{post_id}.json?limit=500&sort=top"
                    )
                    try:
                        comments_payload = reddit_public_get_json(comments_url)
                        diagnostics["comment_requests"] += 1
                        if isinstance(comments_payload, list) and len(comments_payload) > 1:
                            comments_listing = comments_payload[1]
                            comments_data = comments_listing.get("data", {}) if isinstance(comments_listing, dict) else {}
                            comments_children = comments_data.get("children", []) if isinstance(comments_data.get("children"), list) else []
                            comment_bodies = iter_reddit_comment_bodies(comments_children)[:max_comments_per_post]
                            for comment_body in comment_bodies:
                                rows.append({"title": title, "body": comment_body})
                                diagnostics["comments_collected"] += 1
                                subreddit_comment_counts[normalized_sub] = subreddit_comment_counts.get(normalized_sub, 0) + 1
                                if progress_callback:
                                    progress_callback(normalized_sub, int(subreddit_comment_counts.get(normalized_sub, 0)))
                    except Exception:
                        # Best effort; continue if comment fetch fails.
                        diagnostics["request_errors"] += 1
                        if len(diagnostics["error_examples"]) < 5:
                            diagnostics["error_examples"].append(f"comments post_id={post_id}: fetch_failed")
                fetched_posts += 1
                time.sleep(0.35)

            if stop_for_time or not after_token:
                break
            time.sleep(0.45)

        # Fallback for subreddits where search endpoint returns no children.
        # Pull recent posts and apply local keyword matching on title/body.
        if fetched_posts == 0:
            diagnostics["fallback_used_subreddits"] += 1
            fallback_after = ""
            fallback_pages = 0
            while fetched_posts < max_posts_per_subreddit and fallback_pages < 8:
                fallback_url = (
                    f"https://www.reddit.com/r/{normalized_sub}/new.json?limit=100"
                    + (f"&after={fallback_after}" if fallback_after else "")
                )
                try:
                    payload = reddit_public_get_json(fallback_url)
                    diagnostics["listing_requests"] += 1
                except Exception as exc:
                    diagnostics["request_errors"] += 1
                    if len(diagnostics["error_examples"]) < 5:
                        diagnostics["error_examples"].append(f"fallback r/{normalized_sub}: {exc}")
                    break

                data_block = payload.get("data", {}) if isinstance(payload.get("data"), dict) else {}
                children = data_block.get("children", []) if isinstance(data_block.get("children"), list) else []
                fallback_after = str(data_block.get("after", "") or "")
                fallback_pages += 1
                if not children:
                    break

                stop_for_time = False
                for child in children:
                    if fetched_posts >= max_posts_per_subreddit:
                        break
                    child_data = child.get("data", {}) if isinstance(child.get("data"), dict) else {}
                    diagnostics["posts_seen"] += 1
                    diagnostics["fallback_posts_seen"] += 1
                    if progress_callback:
                        progress_callback(normalized_sub, int(subreddit_comment_counts.get(normalized_sub, 0)))
                    created_utc = int(child_data.get("created_utc", 0) or 0)
                    if created_utc < after_ts:
                        stop_for_time = True
                        diagnostics["skipped_old_posts"] += 1
                        continue
                    diagnostics["posts_in_window"] += 1

                    title = str(child_data.get("title", "")).strip()
                    body = str(child_data.get("selftext", "")).strip()
                    author = str(child_data.get("author", "")).strip()
                    post_id = str(child_data.get("id", "")).strip()
                    combined = f"{title} {body}".lower()
                    if keyword_tokens and not any(tok in combined for tok in keyword_tokens):
                        continue
                    if is_reddit_junk_text(title, body, author):
                        diagnostics["junk_filtered"] += 1
                        continue
                    if is_male_authored_or_male_context(f"{title} {body}"):
                        diagnostics["male_filtered"] += 1
                        continue
                    diagnostics["fallback_posts_matched"] += 1

                    if title or body:
                        rows.append({"title": title, "body": body})
                        diagnostics["posts_collected"] += 1

                    if post_id:
                        comments_url = f"https://www.reddit.com/comments/{post_id}.json?limit=500&sort=top"
                        try:
                            comments_payload = reddit_public_get_json(comments_url)
                            diagnostics["comment_requests"] += 1
                            if isinstance(comments_payload, list) and len(comments_payload) > 1:
                                comments_listing = comments_payload[1]
                                comments_data = comments_listing.get("data", {}) if isinstance(comments_listing, dict) else {}
                                comments_children = comments_data.get("children", []) if isinstance(comments_data.get("children"), list) else []
                                comment_bodies = iter_reddit_comment_bodies(comments_children)[:max_comments_per_post]
                                for comment_body in comment_bodies:
                                    rows.append({"title": title, "body": comment_body})
                                    diagnostics["comments_collected"] += 1
                                    subreddit_comment_counts[normalized_sub] = subreddit_comment_counts.get(normalized_sub, 0) + 1
                                    if progress_callback:
                                        progress_callback(normalized_sub, int(subreddit_comment_counts.get(normalized_sub, 0)))
                        except Exception:
                            diagnostics["request_errors"] += 1
                            if len(diagnostics["error_examples"]) < 5:
                                diagnostics["error_examples"].append(f"comments post_id={post_id}: fetch_failed")

                    fetched_posts += 1
                    time.sleep(0.35)

                if stop_for_time or not fallback_after:
                    break
                time.sleep(0.45)

    if not rows:
        empty_df = pd.DataFrame(columns=["title", "body"])
        empty_df.attrs["reddit_diagnostics"] = diagnostics
        return empty_df

    df = pd.DataFrame(rows)
    df = df[["title", "body"]].fillna("")
    df = df[(df["title"].astype(str).str.len() > 0) | (df["body"].astype(str).str.len() > 0)]
    df = df.reset_index(drop=True)
    diagnostics["rows_returned"] = int(len(df))
    df.attrs["reddit_diagnostics"] = diagnostics
    return df


def render_reddit_fetch_diagnostics(diagnostics: Dict[str, object]) -> None:
    """Render Reddit fetch diagnostics in a compact, beginner-friendly panel."""
    if not diagnostics:
        return
    with st.expander("Reddit fetch diagnostics", expanded=False):
        dcol1, dcol2, dcol3 = st.columns(3)
        dcol1.metric("Subreddits processed", int(diagnostics.get("subreddits_processed", 0)))
        dcol2.metric("Posts scanned", int(diagnostics.get("posts_seen", 0)))
        dcol3.metric("Rows returned", int(diagnostics.get("rows_returned", 0)))

        dcol4, dcol5, dcol6 = st.columns(3)
        dcol4.metric("Posts in time window", int(diagnostics.get("posts_in_window", 0)))
        dcol5.metric("Posts collected", int(diagnostics.get("posts_collected", 0)))
        dcol6.metric("Comments collected", int(diagnostics.get("comments_collected", 0)))

        dcol7, dcol8, dcol9 = st.columns(3)
        dcol7.metric("Listing requests", int(diagnostics.get("listing_requests", 0)))
        dcol8.metric("Comment requests", int(diagnostics.get("comment_requests", 0)))
        dcol9.metric("Request errors", int(diagnostics.get("request_errors", 0)))

        dcol10, dcol11, dcol12 = st.columns(3)
        dcol10.metric("Fallback subreddits", int(diagnostics.get("fallback_used_subreddits", 0)))
        dcol11.metric("Fallback posts seen", int(diagnostics.get("fallback_posts_seen", 0)))
        dcol12.metric("Fallback matches", int(diagnostics.get("fallback_posts_matched", 0)))
        junk_filtered = int(diagnostics.get("junk_filtered", 0))
        if junk_filtered > 0:
            st.caption(f"Filtered moderation/admin junk rows: {junk_filtered}")
        male_filtered = int(diagnostics.get("male_filtered", 0))
        if male_filtered > 0:
            st.caption(f"Filtered male-authored/male-context rows: {male_filtered}")

        skipped_old = int(diagnostics.get("skipped_old_posts", 0))
        if skipped_old > 0:
            st.caption(f"Skipped old posts (outside selected lookback): {skipped_old}")
        error_examples = diagnostics.get("error_examples", [])
        if isinstance(error_examples, list) and error_examples:
            st.caption("Sample errors (best-effort fetch continued):")
            for err in error_examples[:5]:
                st.write(f"- {err}")


def read_google_trends_csv(uploaded_file) -> pd.DataFrame:
    """Read Google Trends CSV with best-effort parsing and metadata row handling."""
    uploaded_file.seek(0)
    csv_bytes = uploaded_file.getvalue()
    parse_errors: List[str] = []

    for encoding in ["utf-8-sig", "utf-8", "latin-1"]:
        for skip_rows in [0, 1, 2, 3]:
            try:
                df = pd.read_csv(io.BytesIO(csv_bytes), encoding=encoding, skiprows=skip_rows)
                if df.empty or len(df.columns) < 2:
                    continue
                return df
            except Exception as exc:
                parse_errors.append(f"{encoding}/skip={skip_rows}: {exc}")
                continue

    raise ValueError(
        "Could not parse Google Trends CSV. Please download the CSV directly from Google Trends "
        "and upload without editing."
    )


def coerce_trend_series(series: pd.Series) -> pd.Series:
    """Loosely coerce trend column values to numeric, handling '<1', commas and symbols."""
    text = series.astype(str).str.strip()
    text = text.str.replace("<1", "0.5", regex=False)
    text = text.str.replace(",", "", regex=False)
    text = text.str.replace("%", "", regex=False)
    text = text.str.replace(r"[^0-9\.\-]", "", regex=True)
    text = text.replace("", pd.NA)
    return pd.to_numeric(text, errors="coerce")


def extract_numeric_trends_matrix(trends_df: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
    """Extract high-confidence numeric trend columns from uploaded trends CSV."""
    if trends_df is None or trends_df.empty:
        return pd.DataFrame(), []
    working = trends_df.copy()
    kept_cols: List[str] = []
    for col in working.columns:
        col_name = str(col).strip()
        lower_name = col_name.lower()
        if any(token in lower_name for token in ["date", "day", "week", "month"]):
            continue
        converted = coerce_trend_series(working[col])
        valid_ratio = float(converted.notna().sum()) / float(max(1, len(converted)))
        has_spread = float(converted.dropna().nunique()) >= 2
        mean_val = float(converted.dropna().mean()) if converted.notna().sum() > 0 else 0.0
        if valid_ratio >= 0.45 and has_spread and mean_val > 0.0:
            working[col_name] = converted
            kept_cols.append(col_name)
    if not kept_cols:
        return pd.DataFrame(), []
    return working[kept_cols].copy(), kept_cols


def build_trends_signal_summary(trends_df: Optional[pd.DataFrame]) -> Dict[str, object]:
    """Build search-volume summary from uploaded Google Trends CSV."""
    if trends_df is None or trends_df.empty:
        return {
            "search_interest_avg": 0.0,
            "search_data_points": 0,
            "top_search_term": "N/A",
        }

    numeric_df, numeric_cols = extract_numeric_trends_matrix(trends_df)
    if not numeric_cols:
        return {
            "search_interest_avg": 0.0,
            "search_data_points": 0,
            "top_search_term": "N/A",
        }
    avg_interest = float(numeric_df.stack().mean()) if not numeric_df.empty else 0.0
    row_means = numeric_df.mean(axis=0)
    top_col = str(row_means.idxmax()) if not row_means.empty else "N/A"
    data_points = int(numeric_df.count().sum())

    return {
        "search_interest_avg": round(avg_interest, 2),
        "search_data_points": data_points,
        "top_search_term": top_col,
    }


def build_dynamic_trends_seed_terms(
    pain_df: pd.DataFrame,
    themes: List[Dict[str, object]],
) -> List[str]:
    """Build a comprehensive Google Trends seed set for women hair-care in India."""
    terms: List[str] = [
        "women hair fall",
        "hair growth serum women",
        "anti hair fall shampoo women",
        "scalp irritation women",
        "hair regrowth for women",
        "best hair serum for women india",
        "best shampoo for hair fall women",
        "women dandruff treatment",
        "dry scalp treatment women",
        "itchy scalp women remedy",
        "postpartum hair fall women",
        "hair thinning in women",
        "female pattern hair loss india",
        "hair breakage women",
        "frizzy hair serum women",
        "oily scalp shampoo women",
        "sulfate free shampoo women india",
        "paraben free shampoo women india",
        "keratin shampoo women india",
        "biotin for hair growth women",
        "rosemary oil for hair growth women",
        "onion hair oil women india",
        "scalp exfoliation women",
        "non sticky hair serum women",
        "hair care routine women india",
    ]

    theme_query_map = {
        "No Visible Results After Long-Term Use": "hair regrowth not working women",
        "Rebound Hair Fall After Stopping Usage": "hair fall after stopping serum",
        "Product Runs Out Too Fast": "best value hair serum women",
        "Scalp Dryness & Irritation": "scalp irritation from hair serum",
        "Perceived Waste of Money": "best affordable hair fall treatment women",
    }

    if themes:
        for theme in themes[:3]:
            label = str(theme.get("label", "")).strip()
            if label:
                mapped = theme_query_map.get(label, label.lower())
                terms.append(mapped)

    if not pain_df.empty and "pain_point" in pain_df.columns:
        for phrase in pain_df["pain_point"].astype(str).head(5).tolist():
            cleaned = phrase.replace("_", " ").strip().lower()
            if cleaned:
                if "hair" not in cleaned:
                    cleaned = f"{cleaned} hair women"
                terms.append(cleaned)

    deduped: List[str] = []
    seen = set()
    for term in terms:
        key = re.sub(r"\s+", " ", term).strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(key)

    return deduped[:30]


def build_trends_query_batches(terms: List[str], batch_size: int = 5) -> List[List[str]]:
    """Split a comprehensive term list into Google Trends-compatible batches."""
    cleaned_terms = [str(term).strip() for term in terms if str(term).strip()]
    return [cleaned_terms[i : i + batch_size] for i in range(0, len(cleaned_terms), batch_size)]


def build_trends_explore_url_from_terms(terms: List[str]) -> str:
    """Build a pre-filled Google Trends URL from explicit terms."""
    selected = [term for term in terms if term][:5]
    encoded = ",".join([quote(term) for term in selected])
    return f"https://trends.google.com/trends/explore?date=today%203-m&geo=IN&q={encoded}"


def build_dynamic_trends_explore_url(
    pain_df: pd.DataFrame,
    themes: List[Dict[str, object]],
) -> str:
    """Build a pre-filled Google Trends URL from detected high-intent terms."""
    selected = build_dynamic_trends_seed_terms(pain_df, themes)
    return build_trends_explore_url_from_terms(selected)


def get_analysis_texts(df: pd.DataFrame) -> List[str]:
    """Build analyzable text strings from title/body columns."""
    normalized_df = normalize_columns(df)
    title_series = (
        normalized_df["title"].fillna("").astype(str) if "title" in normalized_df.columns else ""
    )
    body_series = (
        normalized_df["body"].fillna("").astype(str) if "body" in normalized_df.columns else ""
    )

    if isinstance(title_series, str) and isinstance(body_series, str):
        return []

    if isinstance(title_series, str):
        texts = body_series.tolist()
    elif isinstance(body_series, str):
        texts = title_series.tolist()
    else:
        texts = (title_series.str.strip() + " " + body_series.str.strip()).str.strip().tolist()

    return [text for text in texts if text]


def tokenize_words(text: str) -> List[str]:
    """Tokenize to alphabetic words for generic analysis use-cases."""
    return re.findall(r"[a-zA-Z]{3,}", text.lower())


def normalize_token(token: str) -> str:
    """Normalize common variant tokens into canonical forms."""
    token = token.lower().strip()
    return VARIANT_MAP.get(token, token)


def normalize_pain_phrase(phrase: str) -> str:
    """Normalize pain-phrase variants for consistent grouping."""
    normalized = phrase.lower()
    normalized = re.sub(r"\bhair[_\s-]*fall\b|\bshedding\b", "hair fall", normalized)
    normalized = re.sub(r"\bitchy\b|\bitching\b", "itching", normalized)
    normalized = re.sub(r"\bbreakouts?\b|\bacne\b|\bpimples?\b", "breakouts", normalized)
    normalized = re.sub(r"\bgreasy\b|\boily\b|\bgreasiness\b", "greasy", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def sentence_split(text: str) -> List[str]:
    """Split text into sentences with lightweight regex for speed."""
    parts = re.split(r"[.!?\n\r]+", text)
    return [part.strip() for part in parts if part and part.strip()]


def complaint_cue_hits(sentence: str) -> int:
    """Count complaint/negation cues in a sentence."""
    tokens = tokenize_words(sentence)
    normalized_tokens = [normalize_token(token) for token in tokens]
    token_hits = sum(token in COMPLAINT_CUES or token in NEGATION_CUES for token in normalized_tokens)
    regex_hit = 1 if COMPLAINT_PATTERN.search(sentence) else 0
    return token_hits + regex_hit


def extract_complaint_sentences(texts: List[str]) -> List[str]:
    """Keep only complaint-like sentences."""
    complaint_sentences: List[str] = []
    for text in texts:
        for sentence in sentence_split(text):
            if complaint_cue_hits(sentence) > 0:
                complaint_sentences.append(sentence)
    return complaint_sentences


def clean_for_pain_phrases(sentence: str) -> List[str]:
    """Tokenize and clean a complaint sentence for pain phrase extraction."""
    raw_tokens = re.findall(r"[a-zA-Z]{2,}", sentence.lower())
    normalized_tokens = [normalize_token(token) for token in raw_tokens]
    cleaned = [
        token
        for token in normalized_tokens
        if token not in DOMAIN_STOPWORDS and len(token) >= 3 and token.isalpha()
    ]
    return cleaned


def build_pain_point_table(texts: List[str], top_n: int = 15) -> pd.DataFrame:
    """Extract pain-point phrases from complaint sentences and score them."""
    complaint_sentences = extract_complaint_sentences(texts)
    if not complaint_sentences:
        return pd.DataFrame(columns=["pain_point", "count", "score", "example_1", "example_2"])

    phrase_stats: Dict[str, Dict[str, object]] = {}

    for sentence in complaint_sentences:
        tokens = clean_for_pain_phrases(sentence)
        if len(tokens) < 2:
            continue

        cues = complaint_cue_hits(sentence)
        local_phrases = []

        for n in [2, 3]:
            if len(tokens) >= n:
                for i in range(len(tokens) - n + 1):
                    phrase = " ".join(tokens[i : i + n])
                    phrase = normalize_pain_phrase(phrase)
                    local_phrases.append(phrase)

        for phrase in local_phrases:
            if len(phrase.split()) < 2:
                continue
            if phrase not in phrase_stats:
                phrase_stats[phrase] = {
                    "count": 0,
                    "score": 0.0,
                    "snippets": [],
                }
            phrase_stats[phrase]["count"] += 1
            phrase_stats[phrase]["score"] += 1.0 + (0.3 * cues)
            if len(phrase_stats[phrase]["snippets"]) < 2:
                snippet = sentence.strip()
                if len(snippet) > 200:
                    snippet = f"{snippet[:197]}..."
                if snippet not in phrase_stats[phrase]["snippets"]:
                    phrase_stats[phrase]["snippets"].append(snippet)

    if not phrase_stats:
        return pd.DataFrame(columns=["pain_point", "count", "score", "example_1", "example_2"])

    rows = []
    for phrase, data in phrase_stats.items():
        examples = data["snippets"]
        rows.append(
            {
                "pain_point": phrase,
                "count": int(data["count"]),
                "score": round(float(data["score"]), 2),
                "example_1": examples[0] if len(examples) > 0 else "",
                "example_2": examples[1] if len(examples) > 1 else "",
            }
        )

    pain_df = pd.DataFrame(rows)
    pain_df = pain_df.sort_values(by=["score", "count"], ascending=[False, False])
    return pain_df.head(top_n).reset_index(drop=True)


def classify_sentiment(text: str) -> str:
    """Very simple lexicon-based sentiment classification."""
    tokens = [normalize_token(token) for token in tokenize_words(text)]
    pos_hits = sum(token in POSITIVE_WORDS for token in tokens)
    neg_hits = sum(token in NEGATIVE_WORDS for token in tokens)
    score = pos_hits - neg_hits

    if score > 0:
        return "Positive"
    if score < 0:
        return "Negative"
    return "Neutral"


def build_sentiment_table(texts: List[str]) -> pd.DataFrame:
    """Build percentage breakdown for Negative/Neutral/Positive."""
    if not texts:
        return pd.DataFrame(
            {
                "sentiment": ["Negative", "Neutral", "Positive"],
                "count": [0, 0, 0],
                "percentage": [0.0, 0.0, 0.0],
            }
        )

    labels = [classify_sentiment(text) for text in texts]
    counts = Counter(labels)
    total = len(labels)

    rows = []
    for label in ["Negative", "Neutral", "Positive"]:
        count = counts.get(label, 0)
        rows.append(
            {
                "sentiment": label,
                "count": count,
                "percentage": round((count / total) * 100, 1),
            }
        )
    return pd.DataFrame(rows)


def match_theme_title_from_text(text: str) -> Optional[str]:
    """Match a sentence/phrase to the first matching static theme rule."""
    normalized_text = normalize_pain_phrase(str(text).lower())
    matched_titles: List[str] = []
    for rule in THEME_RULES:
        if any(re.search(pattern, normalized_text, flags=re.IGNORECASE) for pattern in rule["patterns"]):
            matched_titles.append(rule["title"])
    if not matched_titles:
        return None
    matched_titles.sort(key=lambda title: THEME_PRIORITY.get(title, 999))
    return matched_titles[0]


def build_dynamic_theme_title_from_phrase(phrase: str) -> str:
    """Generate a readable dynamic theme title from an uncovered pain phrase."""
    tokens = [tok for tok in tokenize_words(str(phrase)) if tok not in DOMAIN_STOPWORDS]
    if not tokens:
        return "Emerging Consumer Pain Cluster"
    words = [token.title() for token in tokens[:3]]
    return "Emerging: " + " ".join(words)


def assign_pain_phrase_theme_label(phrase: str, themes: List[Dict[str, object]]) -> str:
    """Map a pain phrase to the closest available theme label."""
    direct = match_theme_title_from_text(phrase)
    if direct and any(str(theme.get("label", "")) == direct for theme in themes):
        return direct

    phrase_tokens = set(tokenize_words(str(phrase)))
    best_label = "Other Emerging Issues"
    best_overlap = 0
    for theme in themes:
        label = str(theme.get("label", ""))
        candidate_tokens = set(tokenize_words(label))
        for kw in theme.get("keywords", []) if isinstance(theme.get("keywords", []), list) else []:
            candidate_tokens.update(tokenize_words(str(kw)))
        overlap = len(phrase_tokens.intersection(candidate_tokens))
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label
    return best_label


def build_theme_previews(
    texts: List[str],
    min_themes: int = 3,
    max_themes: int = 5,
    pain_df: Optional[pd.DataFrame] = None,
) -> List[Dict[str, object]]:
    """Map complaint-like sentences to normalized consumer pain themes."""
    complaint_sentences = extract_complaint_sentences(texts)
    if not complaint_sentences:
        return []

    theme_buckets: Dict[str, Dict[str, object]] = {
        rule["title"]: {"mentions": 0, "quotes": [], "keywords": set()} for rule in THEME_RULES
    }

    for sentence in complaint_sentences:
        matched_theme = match_theme_title_from_text(sentence)
        if matched_theme:
            bucket = theme_buckets[matched_theme]
            bucket["mentions"] += 1
            if len(bucket["quotes"]) < 5:
                quote = sentence.strip()
                if len(quote) > 200:
                    quote = f"{quote[:197]}..."
                if quote not in bucket["quotes"]:
                    bucket["quotes"].append(quote)
            continue

    if pain_df is None:
        pain_df = build_pain_point_table(texts, top_n=25)

    dynamic_themes: List[Dict[str, object]] = []
    if pain_df is not None and not pain_df.empty:
        for _, row in pain_df.head(25).iterrows():
            phrase = str(row.get("pain_point", "")).strip()
            count = int(row.get("count", 0))
            if not phrase or count <= 0:
                continue
            matched_theme = match_theme_title_from_text(phrase)
            if matched_theme:
                bucket = theme_buckets[matched_theme]
                bucket["keywords"].add(phrase)
                continue
            if count < 4:
                continue
            dyn_label = build_dynamic_theme_title_from_phrase(phrase)
            dynamic_themes.append(
                {
                    "label": dyn_label,
                    "mentions": count,
                    "quotes": list(
                        dict.fromkeys(
                            [
                                str(row.get("example_1", "")).strip(),
                                str(row.get("example_2", "")).strip(),
                            ]
                        )
                    ),
                    "keywords": [phrase],
                }
            )

    themes = []
    for title, data in theme_buckets.items():
        if data["mentions"] == 0 and not data["keywords"]:
            continue
        quote_list = [q for q in data["quotes"] if q]
        keyword_list = list(data["keywords"])[:8]
        themes.append(
            {
                "label": title,
                "mentions": data["mentions"],
                "quotes": quote_list,
                "keywords": keyword_list,
            }
        )

    for dyn in dynamic_themes:
        if any(existing["label"] == dyn["label"] for existing in themes):
            continue
        dyn_quotes = [q for q in dyn.get("quotes", []) if q]
        themes.append(
            {
                "label": dyn["label"],
                "mentions": int(dyn.get("mentions", 0)),
                "quotes": dyn_quotes[:5],
                "keywords": dyn.get("keywords", [])[:8],
            }
        )

    themes.sort(key=lambda item: item["mentions"], reverse=True)
    if len(themes) < min_themes and dynamic_themes:
        for dyn in dynamic_themes:
            if len(themes) >= min_themes:
                break
            if any(existing["label"] == dyn["label"] for existing in themes):
                continue
            themes.append(dyn)
    return themes[:max_themes]


def build_data_signal_summary(
    df: pd.DataFrame,
    trends_df: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """Compute lightweight source signals for concept scoring and citations."""
    normalized_df = normalize_columns(df)
    total_rows = len(normalized_df)
    forum_mentions = 0
    review_mentions = total_rows
    if {"title", "body"}.issubset(set(normalized_df.columns)):
        combined = (
            normalized_df["title"].fillna("").astype(str).str.lower()
            + " "
            + normalized_df["body"].fillna("").astype(str).str.lower()
        )
        forum_pattern = r"\b(reddit|subreddit|r\/[a-z0-9_]+|forum|thread|op)\b"
        forum_mentions = int(combined.str.contains(forum_pattern, regex=True).sum())
        review_mentions = max(total_rows - forum_mentions, 0)

    trends_summary = build_trends_signal_summary(trends_df)

    return {
        "review_mentions": float(review_mentions),
        "forum_mentions": float(forum_mentions),
        "search_interest_avg": float(trends_summary["search_interest_avg"]),
        "search_data_points": int(trends_summary["search_data_points"]),
        "top_search_term": str(trends_summary["top_search_term"]),
    }


def clamp_score(value: float, min_value: float = 0.0, max_value: float = 100.0) -> float:
    """Clamp score into 0-100 range."""
    return max(min_value, min(value, max_value))


PLACEHOLDER_FIELD_VALUES = {
    "",
    "na",
    "n/a",
    "none",
    "null",
    "no",
    "yes",
    "ingredients",
    "target_consumer",
    "positioning",
    "price_point",
    "product_name",
    "format",
}


def normalize_simple_value(value: str) -> str:
    """Normalize free text for placeholder checks."""
    return re.sub(r"[\s\-_]+", " ", value.lower()).strip(" .,:;")


def is_valid_price_point(value: str) -> bool:
    """Check if price point looks meaningful."""
    text = value.strip().lower()
    if not text:
        return False
    if text in {"price", "pricing", "no", "n/a", "na"}:
        return False
    return bool(
        re.search(r"(inr|usd|\$|₹)\s*\d+", text)
        or re.search(r"\b\d{2,5}\b", text)
        or re.search(r"(premium|mid|affordable|budget)\s+(range|pricing)", text)
    )


def is_valid_ingredients_text(value: str) -> bool:
    """Check if ingredients text is formulation-like and not just format words."""
    text = value.strip().lower()
    if not text:
        return False
    format_hits = sum(fmt.lower() in text for fmt in FORMAT_OPTIONS)
    has_delimiter = any(sep in text for sep in ["+", ",", " with ", ";"])
    has_keyword = any(keyword in text for keyword in INGREDIENT_KEYWORDS)
    if format_hits >= 2 and not has_keyword:
        return False
    return has_keyword or has_delimiter


def validate_llm_concept_fields(parsed_fields: Dict[str, str]) -> Tuple[bool, List[str]]:
    """Hard-validate LLM generated concept fields."""
    reasons: List[str] = []
    product_name = str(parsed_fields.get("product_name", "")).strip()
    target_consumer = str(parsed_fields.get("target_consumer", "")).strip()
    ingredients = str(parsed_fields.get("ingredients", "")).strip()
    price_point = str(parsed_fields.get("price_point", "")).strip()
    positioning = str(parsed_fields.get("positioning", "")).strip()

    for field_name, raw in {
        "product_name": product_name,
        "target_consumer": target_consumer,
        "ingredients": ingredients,
        "price_point": price_point,
        "positioning": positioning,
    }.items():
        normalized = normalize_simple_value(raw)
        if normalized in PLACEHOLDER_FIELD_VALUES:
            reasons.append(f"{field_name}_placeholder")
        if re.search(
            r"\b(target_consumer|ingredients|price_point|positioning|product_name|text)\s*:",
            raw,
            re.IGNORECASE,
        ):
            reasons.append(f"{field_name}_echo_artifact")
        # Reject obvious format-list artifacts like "Serum, Shampoo, Tablet".
        raw_lower = raw.lower()
        format_hits = sum(fmt.lower() in raw_lower for fmt in FORMAT_OPTIONS)
        if format_hits >= 2 and ("," in raw or "/" in raw):
            reasons.append(f"{field_name}_formatlist_artifact")

    if len(product_name) < 3 or "target_consumer" in product_name.lower():
        reasons.append("product_name_invalid")
    if normalize_simple_value(product_name) in {fmt.lower() for fmt in FORMAT_OPTIONS}:
        reasons.append("product_name_format_only")
    if len(target_consumer) < 12 or target_consumer.lower().startswith("no"):
        reasons.append("target_consumer_invalid")
    if len(ingredients) < 8 or not is_valid_ingredients_text(ingredients):
        reasons.append("ingredients_invalid")
    if not is_valid_price_point(price_point):
        reasons.append("price_point_invalid")
    if len(positioning) < 8:
        reasons.append("positioning_invalid")

    return len(reasons) == 0, reasons


def targeted_repair_prompt(
    field_name: str,
    context: str,
    current_value: str,
) -> str:
    """Build field-specific repair instructions."""
    base = (
        "You are fixing one field in a product concept brief.\n"
        f"Field: {field_name}\n"
        "Return only one line value. No labels, no markdown.\n"
        f"Current invalid value: {current_value}\n"
        f"Context:\n{context}\n"
    )
    if field_name == "price_point":
        return (
            base
            + "Format strictly as: INR <number> for <pack size>.\n"
            + "Example: INR 899 for 50 ml\n"
        )
    if field_name == "product_name":
        return base + "Generate a short brandable name (2-4 words).\n"
    if field_name == "target_consumer":
        return base + "Write one specific consumer profile sentence (at least 12 characters).\n"
    if field_name == "ingredients":
        return base + "Write ingredients/formulation direction with at least two actives.\n"
    if field_name == "positioning":
        return base + "Write one competitive positioning sentence.\n"
    if field_name == "format":
        return base + f"Return one of these only: {', '.join(FORMAT_OPTIONS)}.\n"
    return base


def safe_title_fragment(text: str, fallback: str = "Care") -> str:
    """Build a short title-case fragment from text for naming."""
    tokens = [token.title() for token in re.findall(r"[a-zA-Z]{3,}", text)[:2]]
    return " ".join(tokens) if tokens else fallback


def phrase_to_name_token(phrase: str) -> str:
    """Convert pain phrase to a compact title token for naming diversity."""
    tokens = [tok.title() for tok in re.findall(r"[a-zA-Z]{3,}", str(phrase)) if tok.lower() not in DOMAIN_STOPWORDS]
    if not tokens:
        return "Care"
    return tokens[0][:10]


def infer_theme_tags(theme_label: str, top_pain_phrase: str) -> set:
    """Infer coarse concept tags from theme label and pain phrase."""
    combined = f"{theme_label} {top_pain_phrase}".lower()
    tags = set()
    if any(word in combined for word in ["result", "regrowth", "growth"]):
        tags.add("regrowth")
    if any(word in combined for word in ["hair fall", "hairfall", "shedding", "fall"]):
        tags.add("hair_fall")
    if any(word in combined for word in ["dry", "itch", "irrit", "rash", "redness"]):
        tags.add("dryness_irritation")
    if any(word in combined for word in ["dandruff", "flakes", "flaky"]):
        tags.add("dandruff_flakes")
    if any(word in combined for word in ["waste", "expensive", "cost", "money", "value"]):
        tags.add("budget_value")
    if not tags:
        tags.add("regrowth")
    return tags


def choose_formats_for_theme(theme_label: str, concept_index: int) -> str:
    """Choose a format based on theme hints and concept index."""
    hinted = THEME_FORMAT_HINTS.get(theme_label, FORMAT_OPTIONS)
    return hinted[concept_index % len(hinted)]


def capabilities_for_format(format_name: str) -> set:
    """Assign required capabilities by format."""
    if format_name in {"Serum", "Shampoo", "Scalp Tonic", "Leave-In Mist"}:
        return {"Topical Formulation", "Packaging Innovation"}
    if format_name in {"Tablet", "Gummy"}:
        return {"Supplement Formulation", "Clinical Claims"}
    return {"D2C Community"}


def build_ingredient_direction(tags: set, format_name: str) -> str:
    """Build ingredient/formulation direction from inferred pain tags."""
    ingredients: List[str] = []
    for tag in sorted(tags):
        ingredients.extend(INGREDIENT_BANK.get(tag, []))
    if not ingredients:
        ingredients = ["Peptide Complex", "Niacinamide", "Panthenol"]
    selected = ingredients[:4]

    if format_name in {"Tablet", "Gummy"}:
        return f"{' + '.join(selected)}; oral daily support format"
    if format_name == "Shampoo":
        return f"{' + '.join(selected)}; low-irritant cleansing base"
    return f"{' + '.join(selected)}; lightweight leave-on scalp format"


def build_price_point(format_name: str, signals: Dict[str, float], theme_mentions: int) -> str:
    """Suggest price point by format, demand signal, and theme intensity."""
    base = {
        "Serum": 899,
        "Shampoo": 749,
        "Tablet": 799,
        "Gummy": 699,
        "Scalp Tonic": 649,
        "Leave-In Mist": 599,
    }.get(format_name, 799)
    uplift = int(min(200, (signals["search_interest_avg"] * 1.5) + (theme_mentions * 5)))
    price = base + uplift
    pack = {
        "Serum": "50 ml",
        "Shampoo": "200 ml",
        "Tablet": "30 tablets",
        "Gummy": "45 gummies",
        "Scalp Tonic": "120 ml",
        "Leave-In Mist": "80 ml",
    }.get(format_name, "1 unit")
    return f"INR {price} for {pack}"


def build_target_profile(theme_label: str, top_pain_phrase: str) -> str:
    """Generate target consumer profile sentence from detected pain theme."""
    return (
        f"Consumers reporting '{top_pain_phrase}' under the theme '{theme_label}', "
        "seeking reliable outcomes and better perceived value."
    )


def build_positioning(theme_label: str, format_name: str, top_pain_phrase: str) -> str:
    """Generate competitive positioning from pain signal."""
    return (
        f"{format_name}-first solution designed to reduce '{top_pain_phrase}' "
        f"and outperform generic alternatives on the '{theme_label}' pain point."
    )


def get_theme_db_profile(theme_label: str) -> Dict[str, object]:
    """Return deterministic theme profile for women hair-care."""
    return WOMEN_HAIRCARE_THEME_DB.get(
        theme_label,
        {
            "name_stems": ["RootCare", "HairSense", "GrowWell"],
            "consumer_profiles": [
                "Women with recurring hair-fall and scalp discomfort seeking reliable, easy-to-follow outcomes."
            ],
            "positioning_angles": [
                "Evidence-first women hair-care proposition with practical daily use."
            ],
            "price_tier": "mid",
        },
    )


def build_deterministic_product_name(
    theme_label: str,
    format_name: str,
    focus_phrase: str,
    concept_index: int,
) -> str:
    """Build brandable deterministic product name for women hair-care."""
    profile = get_theme_db_profile(theme_label)
    stems = list(profile.get("name_stems", [])) or ["RootCare"]
    stem = stems[concept_index % len(stems)]
    descriptor_pool = THEME_NAME_DESCRIPTORS.get(theme_label, ["Core", "Prime", "Advance", "Focus"])
    descriptor = descriptor_pool[concept_index % len(descriptor_pool)]
    benefit = WOMEN_HAIRCARE_BENEFIT_WORDS[concept_index % len(WOMEN_HAIRCARE_BENEFIT_WORDS)]
    phrase_token = phrase_to_name_token(focus_phrase)
    naming_pattern = concept_index % 3
    if naming_pattern == 0:
        return f"{stem} {descriptor} {format_name}"
    if naming_pattern == 1:
        return f"{stem} {benefit} {format_name}"
    return f"{stem} {phrase_token} {format_name}"


def build_deterministic_target_profile(
    theme_label: str,
    focus_phrase: str,
    concept_index: int,
) -> str:
    """Build detailed women-focused target consumer profile."""
    profile = get_theme_db_profile(theme_label)
    profiles = list(profile.get("consumer_profiles", [])) or [
        "Women with persistent hair-fall concerns seeking a practical and effective routine."
    ]
    base_profile = profiles[concept_index % len(profiles)]
    return f"{base_profile} Primary complaint observed: {focus_phrase}."


def select_ingredients_for_theme_and_format(
    theme_label: str,
    format_name: str,
    concept_index: int,
) -> List[str]:
    """Pick ingredient stack from women hair-care library with theme emphasis."""
    library = WOMEN_HAIRCARE_INGREDIENT_LIBRARY.get(format_name, WOMEN_HAIRCARE_INGREDIENT_LIBRARY["Serum"])
    start = concept_index % max(1, len(library))
    rotated = library[start:] + library[:start]
    selected = rotated[:4]

    lower_theme = theme_label.lower()
    if "dryness" in lower_theme or "irritation" in lower_theme:
        priority = ["Ceramide Complex", "Panthenol", "Allantoin", "Oat Beta-Glucan"]
        for ingredient in reversed(priority):
            if ingredient in library and ingredient not in selected:
                selected[-1] = ingredient
    if "rebound" in lower_theme or "hair fall" in lower_theme:
        priority = ["Pea Sprout Extract (AnaGain)", "Saw Palmetto", "Pumpkin Seed Extract", "Redensyl"]
        for ingredient in reversed(priority):
            if ingredient in library and ingredient not in selected:
                selected[0] = ingredient
    return selected


def build_deterministic_ingredient_direction(
    theme_label: str,
    format_name: str,
    concept_index: int,
) -> str:
    """Build detailed ingredient + formulation direction for deterministic mode."""
    selected = select_ingredients_for_theme_and_format(theme_label, format_name, concept_index)
    format_note = WOMEN_HAIRCARE_FORMAT_NOTES.get(format_name, "daily women hair-care format")
    return f"{' + '.join(selected)}; {format_note}; free from heavy residue and optimized for women scalp compatibility."


def build_deterministic_price_point(
    theme_label: str,
    format_name: str,
    signals: Dict[str, float],
    theme_mentions: int,
    concept_index: int,
) -> str:
    """Build price point from women hair-care tier matrix."""
    profile = get_theme_db_profile(theme_label)
    tier = str(profile.get("price_tier", "mid"))
    format_prices = WOMEN_HAIRCARE_PRICE_DB.get(format_name, WOMEN_HAIRCARE_PRICE_DB["Serum"])
    low, high, pack = format_prices.get(tier, format_prices["mid"])
    demand_bump = int(min(80, (signals["search_interest_avg"] * 0.6) + (theme_mentions * 1.5)))
    span = max(1, high - low)
    stepped = low + ((concept_index * 37) % span)
    price = min(high, stepped + demand_bump)
    rounded_price = int(round(price / 10.0) * 10)
    return f"INR {rounded_price} for {pack}"


def build_deterministic_positioning(
    theme_label: str,
    format_name: str,
    focus_phrase: str,
    concept_index: int,
) -> str:
    """Build clear competitive positioning for deterministic mode."""
    profile = get_theme_db_profile(theme_label)
    angles = list(profile.get("positioning_angles", [])) or ["Evidence-led women hair-care differentiation."]
    angle = angles[concept_index % len(angles)]
    return (
        f"{angle} Positioned as a {format_name.lower()} built for women facing '{focus_phrase}', "
        "with stronger value clarity and lower drop-off risk than generic alternatives."
    )


@st.cache_resource(show_spinner=False)
def get_transformers_text2text_pipeline(model_id: str, local_files_only: bool = False):
    """Load and cache a lightweight transformers model bundle."""
    try:
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)

        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, local_files_only=local_files_only)
            return {"tokenizer": tokenizer, "model": model, "kind": "seq2seq"}, ""
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=local_files_only)
            return {"tokenizer": tokenizer, "model": model, "kind": "causal"}, ""
    except Exception as exc:
        return None, str(exc)


def generate_with_local_model(
    prompt: str,
    llm_model: str,
    local_files_only: bool,
    max_new_tokens: int = 320,
) -> Tuple[Optional[str], str]:
    """Generate text with local transformers model."""
    model_bundle, load_error = get_transformers_text2text_pipeline(
        model_id=llm_model,
        local_files_only=local_files_only,
    )
    if model_bundle is None:
        return None, f"model_load_failed: {load_error[:120]}"

    try:
        tokenizer = model_bundle["tokenizer"]
        model = model_bundle["model"]
        kind = model_bundle["kind"]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if kind == "causal" and decoded.startswith(prompt.strip()):
            decoded = decoded[len(prompt.strip()) :].strip()
        return decoded, "ok"
    except Exception:
        return None, "generation_failed"


def extract_first_text_line(text: str) -> str:
    """Return first meaningful line from model output."""
    cleaned_lines = [line.strip(" -*\t") for line in text.splitlines() if line.strip()]
    return cleaned_lines[0].strip() if cleaned_lines else ""


def clean_generated_field_value(field_name: str, value: str) -> str:
    """Remove prompt-echo artifacts and chained labels from generated field values."""
    text = value.strip().strip('"').strip("'")
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()

    # Remove self label prefix if model echoes it.
    text = re.sub(rf"^{re.escape(field_name)}\s*:\s*", "", text, flags=re.IGNORECASE).strip()

    # If model echoes other labels in same line, cut at first foreign label.
    foreign_label_pattern = re.compile(
        r"\b(product_name|target_consumer|ingredients|price_point|positioning|text|allowed formats?)\s*:",
        re.IGNORECASE,
    )
    match = foreign_label_pattern.search(text)
    if match:
        label = match.group(1).lower()
        if label != field_name.lower():
            text = text[: match.start()].strip()

    # Clean separators and repeated spaces.
    text = re.sub(r"\s+", " ", text).strip(" -|,:;")
    return text


def generate_field_with_local_model(
    field_name: str,
    context: str,
    llm_model: str,
    local_files_only: bool,
    max_new_tokens: int = 80,
) -> Tuple[str, str]:
    """Generate one field value with a focused prompt."""
    prompt = f"""
You are generating one field value for a product concept brief.
Return only the value for this field: {field_name}
No labels, no markdown, one line only.

Context:
{context}
"""
    output, reason = generate_with_local_model(
        prompt=prompt,
        llm_model=llm_model,
        local_files_only=local_files_only,
        max_new_tokens=max_new_tokens,
    )
    if not output:
        return "", reason
    line = extract_first_text_line(output)
    line = clean_generated_field_value(field_name, line)
    return line, "ok"


def score_concept(
    concept: Dict[str, object],
    theme_mentions: int,
    signals: Dict[str, float],
    selected_capabilities: set,
) -> Dict[str, float]:
    """Score a concept on market size, competition, and capability alignment."""
    market_size = (
        35.0
        + (theme_mentions * 4.0)
        + (signals["forum_mentions"] * 0.03)
        + (signals["review_mentions"] * 0.04)
        + (signals["search_interest_avg"] * 0.35)
    )
    market_size = clamp_score(market_size)

    base_competition = COMPETITION_BY_FORMAT.get(str(concept["format"]), 70)
    competition_intensity = clamp_score(base_competition + (theme_mentions * 0.6))

    required = concept.get("capabilities", set())
    if not required:
        alignment = 50.0
    else:
        matched = len(set(required).intersection(selected_capabilities))
        alignment = clamp_score((matched / len(required)) * 100.0)

    overall = (
        (0.45 * market_size)
        + (0.25 * (100.0 - competition_intensity))
        + (0.30 * alignment)
    )
    overall = clamp_score(overall)

    return {
        "market_size": round(market_size, 1),
        "competition_intensity": round(competition_intensity, 1),
        "alignment_score": round(alignment, 1),
        "overall_score": round(overall, 1),
    }


def build_concept_evidence(theme: Dict[str, object], signals: Dict[str, object]) -> str:
    """Build short citation text from available consumer data signals."""
    quotes = theme.get("quotes", [])
    theme_mentions = int(theme.get("mentions", 0))
    theme_forum_mentions = int(min(theme_mentions, int(signals.get("forum_mentions", 0))))
    quote_parts = []
    for idx, quote in enumerate(quotes[:2], start=1):
        quote_parts.append(f'Q{idx}: "{quote}"')
    quote_text = " | ".join(quote_parts) if quote_parts else "No direct quotes available."

    return (
        f"Mentions in uploaded data: {theme_mentions}; "
        f"Review mentions: {theme_mentions}; "
        f"Forum mentions: {theme_forum_mentions}; "
        f"Search interest avg: {signals['search_interest_avg']}; "
        f"Search data points: {int(signals.get('search_data_points', 0))}; "
        f"Top search term: {signals.get('top_search_term', 'N/A')}."
        f" {quote_text}"
    )


def parse_quotes_from_evidence(evidence: str) -> List[str]:
    """Extract Q1/Q2 style quotes from evidence text."""
    if not evidence:
        return []
    return re.findall(r'Q\d+:\s*"([^"]+)"', evidence)


def normalize_ingredient_terms(ingredients_text: str) -> set:
    """Normalize ingredient terms for novelty overlap scoring."""
    prefix = str(ingredients_text).split(";", 1)[0]
    terms = re.split(r"\+|,|/| and ", prefix, flags=re.IGNORECASE)
    normalized = {re.sub(r"[^a-z0-9 ]", "", term.lower()).strip() for term in terms}
    return {term for term in normalized if term}


def compute_novelty_scores(concepts_df: pd.DataFrame) -> List[float]:
    """Compute novelty score based on ingredient overlap and theme uniqueness."""
    if concepts_df.empty:
        return []
    ingredient_sets = [normalize_ingredient_terms(val) for val in concepts_df["ingredients"].astype(str)]
    theme_counts = concepts_df["theme"].value_counts().to_dict() if "theme" in concepts_df.columns else {}
    novelty_scores: List[float] = []

    for idx, this_set in enumerate(ingredient_sets):
        max_overlap = 0.0
        for jdx, other_set in enumerate(ingredient_sets):
            if idx == jdx:
                continue
            union = len(this_set.union(other_set))
            if union == 0:
                continue
            overlap = len(this_set.intersection(other_set)) / union
            max_overlap = max(max_overlap, overlap)
        base_novelty = 100.0 * (1.0 - max_overlap)
        theme = str(concepts_df.iloc[idx].get("theme", ""))
        theme_bonus = 8.0 if theme_counts.get(theme, 0) <= 2 else 0.0
        novelty_scores.append(round(clamp_score(base_novelty + theme_bonus), 1))
    return novelty_scores


def compute_text_uniqueness_scores(values: List[str]) -> List[float]:
    """Compute uniqueness score from token overlap across texts."""
    if not values:
        return []
    token_sets = []
    for value in values:
        tokens = set(re.findall(r"[a-zA-Z]{3,}", str(value).lower()))
        token_sets.append(tokens)

    scores: List[float] = []
    for idx, this_set in enumerate(token_sets):
        max_overlap = 0.0
        for jdx, other_set in enumerate(token_sets):
            if idx == jdx:
                continue
            union = len(this_set.union(other_set))
            if union == 0:
                continue
            overlap = len(this_set.intersection(other_set)) / union
            max_overlap = max(max_overlap, overlap)
        scores.append(round(clamp_score((1.0 - max_overlap) * 100.0), 1))
    return scores


def extract_formulation_archetype(row: Dict[str, object]) -> str:
    """Infer a coarse formulation archetype from ingredients/format/theme text."""
    text = " ".join(
        [
            str(row.get("ingredients", "")).lower(),
            str(row.get("format", "")).lower(),
            str(row.get("theme", "")).lower(),
            str(row.get("positioning", "")).lower(),
        ]
    )
    if any(token in text for token in ["redensyl", "procapil", "capixyl", "peptide", "anagain", "caffeine"]):
        return "follicle_stimulation"
    if any(token in text for token in ["ceramide", "panthenol", "allantoin", "oat", "barrier"]):
        return "barrier_repair"
    if any(token in text for token in ["piroctone", "salicylic", "zinc pca", "dandruff", "flake"]):
        return "anti_flake_scalp"
    if any(token in text for token in ["saw palmetto", "biotin", "collagen", "vitamin", "tablet", "gummy"]):
        return "nutri_oral_support"
    if any(token in text for token in ["value", "budget", "affordable", "cost", "waste"]):
        return "value_engineered"
    return "general_haircare"


def compute_template_similarity_penalty(concepts_df: pd.DataFrame) -> List[float]:
    """Penalty for repeated naming/positioning templates across concepts."""
    penalties: List[float] = []
    names = concepts_df["product_name"].astype(str).str.lower().tolist()
    positions = concepts_df["positioning"].astype(str).str.lower().tolist()
    archetypes = [extract_formulation_archetype(row.to_dict()) for _, row in concepts_df.iterrows()]
    archetype_counts = Counter(archetypes)

    for idx in range(len(concepts_df)):
        name_tokens = set(re.findall(r"[a-zA-Z]{3,}", names[idx]))
        pos_tokens = set(re.findall(r"[a-zA-Z]{3,}", positions[idx]))
        max_name_overlap = 0.0
        max_pos_overlap = 0.0
        for jdx in range(len(concepts_df)):
            if idx == jdx:
                continue
            other_name_tokens = set(re.findall(r"[a-zA-Z]{3,}", names[jdx]))
            other_pos_tokens = set(re.findall(r"[a-zA-Z]{3,}", positions[jdx]))
            name_union = len(name_tokens.union(other_name_tokens))
            pos_union = len(pos_tokens.union(other_pos_tokens))
            if name_union > 0:
                max_name_overlap = max(max_name_overlap, len(name_tokens.intersection(other_name_tokens)) / name_union)
            if pos_union > 0:
                max_pos_overlap = max(max_pos_overlap, len(pos_tokens.intersection(other_pos_tokens)) / pos_union)

        archetype_penalty = max(0.0, float(archetype_counts.get(archetypes[idx], 1) - 1) * 6.0)
        combined_penalty = (max_name_overlap * 28.0) + (max_pos_overlap * 24.0) + archetype_penalty
        penalties.append(round(clamp_score(combined_penalty), 1))
    return penalties


def minmax_to_100(values: List[float]) -> List[float]:
    """Scale values to [0,100] with stable handling for flat lists."""
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        return [50.0 for _ in values]
    return [round(((val - vmin) / (vmax - vmin)) * 100.0, 1) for val in values]


def harden_concept_display_numbers(enriched: pd.DataFrame) -> pd.DataFrame:
    """Derive robust per-concept display numbers for mentions/search/competition."""
    hardened = enriched.copy()
    if hardened.empty:
        return hardened

    if "concept_id" not in hardened.columns:
        hardened["concept_id"] = [f"concept_{idx+1}" for idx in range(len(hardened))]

    hardened["quote_count"] = hardened["theme_quotes"].apply(
        lambda quotes: len(quotes) if isinstance(quotes, list) else 0
    )

    # Theme-level redistribution: avoid flat repeated mention counts within same theme.
    display_review = []
    display_forum = []
    for theme, group_df in hardened.groupby("theme", sort=False):
        base_reviews = group_df["review_mentions"].astype(float).tolist()
        base_forums = group_df["forum_mentions"].astype(float).tolist()
        fit_vals = group_df["theme_fit"].astype(float).tolist() if "theme_fit" in group_df.columns else [50.0] * len(group_df)
        intensity_vals = group_df["pain_intensity"].astype(float).tolist() if "pain_intensity" in group_df.columns else [50.0] * len(group_df)
        quote_vals = group_df["quote_count"].astype(float).tolist()
        trend_vals = group_df["trend_momentum"].astype(float).tolist() if "trend_momentum" in group_df.columns else [50.0] * len(group_df)

        weights = []
        for idx in range(len(group_df)):
            weight = (
                0.55
                + 0.20 * (fit_vals[idx] / 100.0)
                + 0.15 * (intensity_vals[idx] / 100.0)
                + 0.05 * min(1.0, quote_vals[idx] / 4.0)
                + 0.05 * (trend_vals[idx] / 100.0)
            )
            weights.append(max(0.25, weight))

        weight_mean = sum(weights) / max(1, len(weights))
        for idx in range(len(group_df)):
            multiplier = max(0.65, min(1.45, weights[idx] / max(0.001, weight_mean)))
            reviews_adj = max(1, int(round(base_reviews[idx] * multiplier)))
            forums_adj = max(0, int(round(base_forums[idx] * multiplier * 0.9)))
            forums_adj = min(reviews_adj, forums_adj)
            display_review.append(reviews_adj)
            display_forum.append(forums_adj)

    hardened["display_review_mentions"] = display_review
    hardened["display_forum_mentions"] = display_forum
    hardened["display_search_interest"] = hardened["concept_search_interest"].astype(float).round(1)

    # Harden competition intensity displayed in risk logic using crowding context.
    format_counts = hardened["format"].astype(str).value_counts().to_dict()
    theme_counts = hardened["theme"].astype(str).value_counts().to_dict()
    comp_display = []
    for _, row in hardened.iterrows():
        base = float(row.get("competition_intensity", 70.0))
        format_crowd = max(0.0, float(format_counts.get(str(row.get("format", "")), 1) - 1) * 4.0)
        theme_crowd = max(0.0, float(theme_counts.get(str(row.get("theme", "")), 1) - 1) * 3.0)
        novelty_relief = float(row.get("novelty_component", 50.0)) * 0.08
        adjusted = clamp_score(base + format_crowd + theme_crowd - novelty_relief)
        comp_display.append(round(adjusted, 1))
    hardened["competition_intensity_display"] = comp_display
    return hardened


def infer_best_trend_column_for_concept(
    row: Dict[str, object],
    trend_columns: List[str],
) -> str:
    """Pick the most relevant trend column for one concept by token overlap."""
    if not trend_columns:
        return "N/A"
    context = " ".join(
        [
            str(row.get("product_name", "")),
            str(row.get("theme", "")),
            str(row.get("format", "")),
            str(row.get("ingredients", "")),
            str(row.get("positioning", "")),
        ]
    ).lower()
    context_tokens = set(re.findall(r"[a-zA-Z]{3,}", context))
    if not context_tokens:
        return trend_columns[0]

    best_col = trend_columns[0]
    best_score = -1
    for col in trend_columns:
        col_tokens = set(re.findall(r"[a-zA-Z]{3,}", str(col).lower()))
        overlap = len(context_tokens.intersection(col_tokens))
        if overlap > best_score:
            best_score = overlap
            best_col = col
    return best_col


def rank_trend_columns_for_concept(
    row: Dict[str, object],
    trend_columns: List[str],
) -> List[Tuple[str, float]]:
    """Rank trend columns for a concept using token overlap and phrase hints."""
    if not trend_columns:
        return []
    context = " ".join(
        [
            str(row.get("product_name", "")),
            str(row.get("theme", "")),
            str(row.get("format", "")),
            str(row.get("ingredients", "")),
            str(row.get("positioning", "")),
        ]
    ).lower()
    context_tokens = set(re.findall(r"[a-zA-Z]{3,}", context))
    ranked: List[Tuple[str, float]] = []
    for col in trend_columns:
        col_text = str(col).lower()
        col_tokens = set(re.findall(r"[a-zA-Z]{3,}", col_text))
        overlap = float(len(context_tokens.intersection(col_tokens)))
        phrase_bonus = 0.0
        if col_text in context:
            phrase_bonus += 1.5
        if any(term in context for term in col_text.split()):
            phrase_bonus += 0.8
        score = overlap + phrase_bonus
        ranked.append((str(col), score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def build_concept_trend_signals(
    concepts_df: pd.DataFrame,
    trends_df: Optional[pd.DataFrame],
) -> Tuple[List[float], List[float], List[str]]:
    """Build per-concept search average and trend momentum from trends data."""
    n = len(concepts_df)
    if trends_df is None or trends_df.empty:
        fallback_avg = concepts_df["search_interest_avg"].astype(float).tolist() if "search_interest_avg" in concepts_df.columns else [12.0] * n
        return fallback_avg, [50.0] * n, ["N/A"] * n

    working, numeric_cols = extract_numeric_trends_matrix(trends_df)
    if not numeric_cols:
        fallback_avg = concepts_df["search_interest_avg"].astype(float).tolist() if "search_interest_avg" in concepts_df.columns else [12.0] * n
        return fallback_avg, [50.0] * n, ["N/A"] * n

    global_avg = float(working.stack().mean()) if not working.empty else 12.0
    if global_avg <= 0:
        global_avg = 12.0
    global_median = float(working.stack().median()) if not working.empty else global_avg
    if global_median <= 0:
        global_median = global_avg

    # Collision-aware assignment so concepts do not all map to the same trend column.
    max_use_per_column = max(1, (n + len(numeric_cols) - 1) // len(numeric_cols))
    usage_counter: Counter = Counter()
    assigned_column_by_index: Dict[int, str] = {}
    confidence_by_index: Dict[int, float] = {}
    candidate_rows: List[Tuple[int, List[Tuple[str, float]]]] = []
    for idx, (_, row) in enumerate(concepts_df.iterrows()):
        ranked_candidates = rank_trend_columns_for_concept(row.to_dict(), numeric_cols)
        candidate_rows.append((idx, ranked_candidates))
    candidate_rows.sort(key=lambda item: item[1][0][1] if item[1] else -1, reverse=True)

    for idx, ranked_candidates in candidate_rows:
        chosen_col = ranked_candidates[0][0] if ranked_candidates else numeric_cols[0]
        for candidate_col, _ in ranked_candidates:
            if usage_counter[candidate_col] < max_use_per_column:
                chosen_col = candidate_col
                break
        usage_counter[chosen_col] += 1
        assigned_column_by_index[idx] = chosen_col
        confidence_by_index[idx] = float(ranked_candidates[0][1]) if ranked_candidates else 0.0

    trend_avgs: List[float] = []
    trend_momentums: List[float] = []
    trend_terms: List[str] = []
    for idx, (_, row) in enumerate(concepts_df.iterrows()):
        chosen = assigned_column_by_index.get(
            idx,
            infer_best_trend_column_for_concept(row.to_dict(), numeric_cols),
        )
        confidence = float(confidence_by_index.get(idx, 0.0))
        series = working[chosen].dropna()
        if series.empty or confidence <= 0.0:
            trend_avgs.append(round(global_median, 2))
            trend_momentums.append(50.0)
            trend_terms.append(str(chosen) if confidence > 0.0 else "N/A (fallback)")
            continue
        avg_val = float(series.mean())
        if avg_val <= 0.0:
            avg_val = global_avg
        first = float(series.iloc[0]) if float(series.iloc[0]) != 0 else max(1.0, global_median)
        last = float(series.iloc[-1])
        growth = (last - first) / abs(first)
        growth = max(-1.0, min(1.0, growth))
        momentum = 50.0 + (50.0 * growth)
        trend_avgs.append(round(avg_val, 2))
        trend_momentums.append(round(clamp_score(momentum), 1))
        trend_terms.append(str(chosen))
    return trend_avgs, trend_momentums, trend_terms


def compute_concept_intensity_signals(concepts_df: pd.DataFrame) -> Tuple[List[float], List[float], List[float]]:
    """Compute concept-level intensity signals from available local evidence/text."""
    pain_intensity_scores: List[float] = []
    theme_fit_scores: List[float] = []
    format_prior_scores: List[float] = []

    for _, row in concepts_df.iterrows():
        theme_text = str(row.get("theme", "")).lower()
        evidence_text = str(row.get("evidence", "")).lower()
        positioning_text = str(row.get("positioning", "")).lower()
        target_text = str(row.get("target_consumer", "")).lower()
        ingredient_text = str(row.get("ingredients", "")).lower()
        full_text = " ".join([theme_text, evidence_text, positioning_text, target_text, ingredient_text])

        tokens = tokenize_words(full_text)
        normalized_tokens = [normalize_token(token) for token in tokens]
        complaint_hits = sum(token in COMPLAINT_CUES for token in normalized_tokens)
        negation_hits = sum(token in NEGATION_CUES for token in normalized_tokens)
        regex_hit = 1 if COMPLAINT_PATTERN.search(full_text) else 0
        quote_count = len(row.get("theme_quotes", [])) if isinstance(row.get("theme_quotes", []), list) else 0
        pain_intensity = clamp_score(
            20.0
            + (complaint_hits * 4.0)
            + (negation_hits * 3.0)
            + (regex_hit * 8.0)
            + (quote_count * 5.0)
        )
        pain_intensity_scores.append(round(pain_intensity, 1))

        theme_tokens = set(tokenize_words(theme_text))
        concept_tokens = set(tokenize_words(" ".join([positioning_text, target_text, ingredient_text])))
        if not theme_tokens or not concept_tokens:
            fit_score = 45.0
        else:
            overlap_ratio = len(theme_tokens.intersection(concept_tokens)) / max(1, len(theme_tokens))
            fit_score = clamp_score(25.0 + (overlap_ratio * 75.0))
        theme_fit_scores.append(round(fit_score, 1))

        format_name = str(row.get("format", "Serum"))
        format_prior = float(FORMAT_DEMAND_PRIOR.get(format_name, 62))
        format_prior_scores.append(round(clamp_score(format_prior), 1))

    return pain_intensity_scores, theme_fit_scores, format_prior_scores


def enrich_opportunity_scores(
    concepts_df: pd.DataFrame,
    trends_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Add demand/competition/novelty/opportunity score columns."""
    if concepts_df.empty:
        return concepts_df
    enriched = concepts_df.copy()
    for col in ["review_mentions", "forum_mentions", "search_interest_avg", "competition_intensity"]:
        if col not in enriched.columns:
            enriched[col] = 0

    trend_avgs, trend_momentum, trend_terms = build_concept_trend_signals(enriched, trends_df)
    enriched["concept_search_interest"] = trend_avgs
    enriched["trend_momentum"] = trend_momentum
    enriched["concept_trend_term"] = trend_terms
    pain_intensity, theme_fit, format_prior = compute_concept_intensity_signals(enriched)
    enriched["pain_intensity"] = pain_intensity
    enriched["theme_fit"] = theme_fit
    enriched["format_demand_prior"] = format_prior

    demand_raw = (
        enriched["review_mentions"].astype(float) * 1.0
        + enriched["forum_mentions"].astype(float) * 1.2
        + enriched["concept_search_interest"].astype(float) * 2.1
        + enriched["trend_momentum"].astype(float) * 0.6
        + enriched["pain_intensity"].astype(float) * 0.7
        + enriched["theme_fit"].astype(float) * 0.8
        + enriched["format_demand_prior"].astype(float) * 0.4
    ).tolist()
    demand_component = minmax_to_100(demand_raw)
    base_comp_adv = [round(clamp_score(100.0 - float(val)), 1) for val in enriched["competition_intensity"].tolist()]
    format_counts = enriched["format"].astype(str).value_counts().to_dict() if "format" in enriched.columns else {}
    format_rarity_raw = [100.0 / max(1.0, float(format_counts.get(str(fmt), 1))) for fmt in enriched["format"].astype(str).tolist()]
    format_rarity = minmax_to_100(format_rarity_raw)
    theme_counts = enriched["theme"].astype(str).value_counts().to_dict() if "theme" in enriched.columns else {}
    theme_rarity_raw = [100.0 / max(1.0, float(theme_counts.get(str(theme), 1))) for theme in enriched["theme"].astype(str).tolist()]
    theme_rarity = minmax_to_100(theme_rarity_raw)
    competition_advantage = [
        round(
            clamp_score(
                (0.65 * base_comp_adv[idx])
                + (0.20 * format_rarity[idx])
                + (0.15 * theme_rarity[idx])
            ),
            1,
        )
        for idx in range(len(enriched))
    ]

    ingredient_novelty = compute_novelty_scores(enriched)
    name_uniqueness = compute_text_uniqueness_scores(enriched["product_name"].astype(str).tolist())
    positioning_uniqueness = compute_text_uniqueness_scores(enriched["positioning"].astype(str).tolist())
    archetypes = [extract_formulation_archetype(row.to_dict()) for _, row in enriched.iterrows()]
    archetype_counts = Counter(archetypes)
    archetype_uniqueness = [
        round(clamp_score(100.0 / max(1.0, float(archetype_counts.get(archetype, 1)))) , 1)
        for archetype in archetypes
    ]
    archetype_uniqueness = minmax_to_100(archetype_uniqueness)
    template_penalty = compute_template_similarity_penalty(enriched)
    novelty_component = [
        round(
            clamp_score(
                (0.50 * ingredient_novelty[idx])
                + (0.20 * name_uniqueness[idx])
                + (0.15 * positioning_uniqueness[idx])
                + (0.15 * archetype_uniqueness[idx])
                - (0.25 * template_penalty[idx])
            ),
            1,
        )
        for idx in range(len(enriched))
    ]
    opportunity_scores = []
    for idx in range(len(enriched)):
        score = (
            0.5 * demand_component[idx]
            + 0.25 * competition_advantage[idx]
            + 0.25 * novelty_component[idx]
        )
        opportunity_scores.append(round(clamp_score(score), 1))

    enriched["demand_component"] = demand_component
    enriched["competition_component"] = competition_advantage
    enriched["novelty_component"] = novelty_component
    enriched["opportunity_score"] = opportunity_scores
    enriched = harden_concept_display_numbers(enriched)
    return enriched


def build_why_win_bullets(row: Dict[str, object]) -> List[str]:
    """Create concise decision bullets from score and evidence."""
    review_val = int(row.get("display_review_mentions", row.get("review_mentions", 0)))
    forum_val = int(row.get("display_forum_mentions", row.get("forum_mentions", 0)))
    search_val = float(row.get("display_search_interest", row.get("search_interest_avg", 0)))
    bullets = []
    bullets.append(
        f"Strong demand signal: {review_val} reviews, {forum_val} forum mentions."
    )
    bullets.append(
        f"Search traction: avg interest {search_val}"
        f" (top term: {row.get('top_search_term', 'N/A')})."
    )
    bullets.append(
        f"Novelty {row.get('novelty_component', 0):.1f}/100 with pain-theme fit on "
        f"'{row.get('theme', 'Consumer Pain')}'."
    )
    return bullets


def split_profile_bullets(target_consumer: str) -> List[str]:
    """Split target consumer text into up to 3 readable bullets."""
    chunks = re.split(r"[.;]", str(target_consumer))
    cleaned = [chunk.strip() for chunk in chunks if chunk.strip()]
    if len(cleaned) >= 3:
        return cleaned[:3]
    if len(cleaned) == 2:
        return cleaned + ["Needs an easy, evidence-backed routine with visible milestones."]
    if len(cleaned) == 1:
        return [
            cleaned[0],
            "Wants high-confidence outcomes without complicated regimen changes.",
            "Prefers clear value-for-money and transparent claim boundaries.",
        ]
    return [
        "Women with persistent hair-care pain points.",
        "Needs reliable and practical outcomes.",
        "Looks for strong value and evidence clarity.",
    ]


def get_usage_and_claims(format_name: str) -> List[str]:
    """Return usage and claim boundary notes by format."""
    notes = {
        "Serum": [
            "Daily leave-on scalp serum; 1 ml nightly on thinning zones.",
            "Best positioned for progressive visible density support over 8-12 weeks.",
            "Avoid over-claiming regrowth speed; position as consistent routine support.",
        ],
        "Shampoo": [
            "Rinse-off scalp shampoo; 3-4 uses/week with 2-minute contact time.",
            "Best for comfort, flake control, and improving scalp environment.",
            "Avoid claiming standalone regrowth; frame as scalp-foundation support.",
        ],
        "Tablet": [
            "Once-daily oral tablet with food; 90-day protocol.",
            "Best for internal nutrition support in thinning and shedding patterns.",
            "Avoid treatment claims; position as nutritional support with consistency.",
        ],
        "Gummy": [
            "Daily chewable supplement; 1-2 gummies/day.",
            "Best for adherence and routine consistency among first-time users.",
            "Avoid curative claims; position as habit-friendly nutrition support.",
        ],
    }
    return notes.get(format_name, notes["Serum"])


def build_formulation_rationale(ingredients_text: str, theme: str) -> List[str]:
    """Create rationale bullets from ingredient stack and pain theme."""
    terms = list(normalize_ingredient_terms(ingredients_text))[:4]
    term_display = ", ".join([term.title() for term in terms]) if terms else "Selected active blend"
    return [
        f"Primary actives: {term_display}.",
        f"Selected to address '{theme}' with multi-pathway support (follicle/scalp/barrier).",
        "Keep sensory profile light and non-greasy to improve adherence and repeat usage.",
    ]


def build_risk_assumptions(row: Dict[str, object]) -> List[str]:
    """Generate concise risks and assumptions list."""
    comp_signal = float(
        row.get("competition_intensity_display", row.get("competition_intensity", 70))
    )
    return [
        "Perceived efficacy may vary by baseline severity and adherence quality.",
        f"Competition is {'high' if comp_signal > 75 else 'moderate'} in this format; differentiation must stay clear.",
        "Claims must be tightly aligned to substantiation and local regulatory boundaries.",
    ]


def get_step1_comment_pool() -> List[Dict[str, str]]:
    """Build a deduplicated source-aware comment pool from Marketplace + Reddit datasets."""
    pool: List[Dict[str, str]] = []
    source_map = {
        "marketplace_dataset": "Marketplace",
        "reddit_dataset": "Reddit",
    }
    for key, source_label in source_map.items():
        ds = st.session_state.get(key)
        if isinstance(ds, pd.DataFrame) and not ds.empty:
            normalized = normalize_columns(ds)
            series = normalized["body"].fillna("").astype(str) if "body" in normalized.columns else normalized.get("title", pd.Series([], dtype=str)).fillna("").astype(str)
            for text in series.tolist():
                pool.append({"source": source_label, "text": str(text)})

    deduped: List[Dict[str, str]] = []
    seen = set()
    for item in pool:
        cleaned = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"source": str(item.get("source", "Input")), "text": cleaned})
    return deduped


def comment_to_relevant_snippet(comment: str, focus_tokens: set) -> str:
    """Extract a short, relevant snippet from comment text."""
    sentences = sentence_split(comment)
    if not sentences:
        snippet = comment
    else:
        best_sentence = sentences[0]
        best_score = -1.0
        for sent in sentences:
            sent_tokens = set(re.findall(r"[a-z]{3,}", sent.lower()))
            overlap = len(sent_tokens.intersection(focus_tokens))
            cue = complaint_cue_hits(sent)
            score = (overlap * 2.0) + (cue * 1.6)
            if score > best_score:
                best_score = score
                best_sentence = sent
        snippet = best_sentence

    snippet = re.sub(r"\s+", " ", snippet).strip()
    if len(snippet) > 180:
        snippet = f"{snippet[:177]}..."
    return snippet


def snippet_jaccard(a: str, b: str) -> float:
    """Compute token Jaccard similarity for snippet diversity filtering."""
    ta = set(re.findall(r"[a-z]{3,}", a.lower()))
    tb = set(re.findall(r"[a-z]{3,}", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta.intersection(tb)) / max(1, len(ta.union(tb)))


def select_relevant_evidence_comments(row_dict: Dict[str, object], max_quotes: int = 5) -> List[str]:
    """Select short, concept-relevant and diverse snippets from Step 1 sources."""
    pool = get_step1_comment_pool()
    if not pool:
        return []

    focus_text = " ".join(
        [
            str(row_dict.get("theme", "")),
            str(row_dict.get("product_name", "")),
            str(row_dict.get("ingredients", "")),
            str(row_dict.get("format", "")),
            str(row_dict.get("positioning", "")),
            str(row_dict.get("evidence", "")),
        ]
    ).lower()
    focus_tokens = set(re.findall(r"[a-z]{3,}", focus_text))
    scored: List[Tuple[float, str, str]] = []
    for item in pool:
        comment = str(item.get("text", ""))
        source = str(item.get("source", "Input"))
        lc = comment.lower()
        tokens = set(re.findall(r"[a-z]{3,}", lc))
        overlap = len(tokens.intersection(focus_tokens))
        cue_score = complaint_cue_hits(comment)
        length = len(comment)
        quality = 1.0 if 35 <= length <= 420 else 0.3
        score = (overlap * 2.0) + (cue_score * 1.5) + quality
        if score > 0:
            snippet = comment_to_relevant_snippet(comment, focus_tokens)
            snippet_bonus = 0.8 if 30 <= len(snippet) <= 180 else 0.0
            scored.append((score + snippet_bonus, source, snippet))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected: List[str] = []
    for _, source, snippet in scored:
        formatted = f"[{source}] {snippet}"
        too_similar = any(snippet_jaccard(formatted, prev) >= 0.6 for prev in selected)
        if too_similar:
            continue
        if formatted not in selected:
            selected.append(formatted)
        if len(selected) >= max_quotes:
            break
    return selected


def build_next_experiments(row: Dict[str, object]) -> List[str]:
    """Generate practical next-step experiments."""
    return [
        f"Landing-page A/B test for '{row.get('theme', 'pain theme')}' messaging and CTR-to-add-to-cart.",
        "Rapid bench feasibility check for active stability + texture + fragrance tolerance.",
        "Claim substantiation pilot: 8-12 week user study with before/after and drop-off tracking.",
    ]


def render_full_concept_brief(row_dict: Dict[str, object]) -> None:
    """Render one complete concept brief block."""
    pitch = (
        f"{row_dict['product_name']} is a {str(row_dict['format']).lower()} concept for women "
        f"facing {str(row_dict['theme']).lower()}, designed to deliver clearer outcomes with stronger adherence."
    )
    st.markdown("**1-line pitch**")
    st.write(pitch)

    st.markdown("**Who it's for**")
    for bullet in split_profile_bullets(str(row_dict.get("target_consumer", "")))[:3]:
        st.write(f"- {bullet}")

    st.markdown("**What it is**")
    for bullet in get_usage_and_claims(str(row_dict.get("format", "Serum"))):
        st.write(f"- {bullet}")

    st.markdown("**Formulation direction**")
    st.write(str(row_dict.get("ingredients", "")))
    for bullet in build_formulation_rationale(
        str(row_dict.get("ingredients", "")),
        str(row_dict.get("theme", "")),
    ):
        st.write(f"- {bullet}")

    st.markdown("**Positioning**")
    st.write(f"- {row_dict.get('positioning', '')}")
    st.write(
        f"- Better pain-theme focus on '{row_dict.get('theme', '')}' than generic multi-claim offerings."
    )
    st.write("- Designed for stronger adherence through format-specific user experience.")

    st.markdown("**Evidence**")
    comments = select_relevant_evidence_comments(row_dict, max_quotes=5)
    if not comments:
        fallback_quotes = row_dict.get("theme_quotes", [])
        if not isinstance(fallback_quotes, list):
            fallback_quotes = []
        if not fallback_quotes:
            fallback_quotes = parse_quotes_from_evidence(str(row_dict.get("evidence", "")))
        comments = [str(q).strip() for q in fallback_quotes if str(q).strip()][:5]
    if comments:
        for quote in comments:
            st.write(f"- {quote}")
    else:
        st.write("- No relevant comments available yet from current inputs.")

    st.markdown("**Risks & assumptions**")
    for bullet in build_risk_assumptions(row_dict):
        st.write(f"- {bullet}")


def get_trend_series(trends_df: Optional[pd.DataFrame], preferred_term: str) -> Optional[pd.Series]:
    """Extract a numeric trend series for sparkline display."""
    if trends_df is None or trends_df.empty:
        return None
    working = trends_df.copy()
    numeric_cols = []
    for col in working.columns:
        converted = pd.to_numeric(working[col], errors="coerce")
        if converted.notna().sum() > 0:
            working[col] = converted
            numeric_cols.append(col)
    if not numeric_cols:
        return None
    if preferred_term in numeric_cols:
        series = working[preferred_term].dropna()
        return series.rename("interest") if not series.empty else None
    series = working[numeric_cols[0]].dropna()
    return series.rename("interest") if not series.empty else None


def summarize_concept_trend_metrics(
    row_dict: Dict[str, object],
    trends_df: Optional[pd.DataFrame],
) -> Optional[Dict[str, object]]:
    """Build concept-level trend summary: current, 3M avg, momentum, peak, sparkline series."""
    preferred_term = str(
        row_dict.get("concept_trend_term", row_dict.get("top_search_term", ""))
    ).strip()
    series = get_trend_series(trends_df, preferred_term)
    if series is None or series.empty:
        return None

    current_val = float(series.iloc[-1])
    avg_val = float(series.mean())
    peak_val = float(series.max())

    if len(series) >= 5:
        base_val = float(series.iloc[-5])
    else:
        base_val = float(series.iloc[0])
    if base_val == 0:
        momentum_pct = 0.0
    else:
        momentum_pct = ((current_val - base_val) / abs(base_val)) * 100.0

    return {
        "term": preferred_term if preferred_term else "N/A",
        "current": round(current_val, 1),
        "avg_3m": round(avg_val, 1),
        "momentum_pct": round(momentum_pct, 1),
        "peak": round(peak_val, 1),
        "series": series.reset_index(drop=True),
    }

def infer_target_concept_count(theme_count: int) -> int:
    """Infer concept count in [8, 10] from number of detected pain themes."""
    if theme_count <= 0:
        return 8
    # 2 themes -> 8, 3 themes -> 9, 4+ themes -> 10
    return max(8, min(10, theme_count + 6))


def generate_product_concepts(
    themes: List[Dict[str, object]],
    pain_df: pd.DataFrame,
    data_df: pd.DataFrame,
    trends_df: Optional[pd.DataFrame],
    selected_capabilities: set,
    use_local_llm: bool = False,
    llm_model: str = "google/flan-t5-base",
    local_files_only: bool = False,
    min_concepts: int = 5,
    max_concepts: int = 10,
    concept_index_seed: int = 0,
) -> pd.DataFrame:
    """Generate concept briefs from pain themes using LLM or deterministic mode."""
    if not themes:
        df = pd.DataFrame()
        df.attrs["llm_attempted_count"] = 0
        df.attrs["llm_generated_count"] = 0
        df.attrs["llm_failure_reasons"] = {"no_themes": 1}
        return df

    signals = build_data_signal_summary(data_df, trends_df=trends_df)
    concepts: List[Dict[str, object]] = []
    llm_attempted_count = 0
    llm_generated_count = 0
    llm_failure_reasons: Counter = Counter()

    target_briefs = max(1, int(max_concepts))

    theme_count = len(themes)
    if theme_count == 0:
        df = pd.DataFrame()
        df.attrs["llm_attempted_count"] = int(llm_attempted_count)
        df.attrs["llm_generated_count"] = int(llm_generated_count)
        df.attrs["llm_failure_reasons"] = dict(llm_failure_reasons)
        return df

    base_per_theme = target_briefs // theme_count
    remainder = target_briefs % theme_count
    per_theme_targets = [base_per_theme + (1 if i < remainder else 0) for i in range(theme_count)]

    if not use_local_llm:
        deterministic_counter = int(concept_index_seed)
        used_names: set = set()
        for theme_idx, theme in enumerate(themes):
            theme_label = str(theme.get("label", "Consumer Pain"))
            theme_mentions = int(theme.get("mentions", 0))
            theme_quotes = theme.get("quotes", [])
            theme_phrase_pool: List[str] = []

            if not pain_df.empty:
                for _, row in pain_df.iterrows():
                    phrase = str(row.get("pain_point", "")).strip()
                    if phrase and phrase.lower() in " ".join(theme_quotes).lower():
                        theme_phrase_pool.append(phrase)
            if not theme_phrase_pool:
                theme_phrase_pool = [safe_title_fragment(theme_label, fallback="pain").lower()]

            theme_target = max(1, per_theme_targets[theme_idx])
            for idx in range(theme_target):
                deterministic_counter += 1
                focus_phrase = theme_phrase_pool[idx % len(theme_phrase_pool)]
                # Use global deterministic counter so formats rotate across sequential generation calls.
                format_name = choose_formats_for_theme(theme_label, deterministic_counter)
                product_name = build_deterministic_product_name(
                    theme_label=theme_label,
                    format_name=format_name,
                    focus_phrase=focus_phrase,
                    concept_index=deterministic_counter,
                )
                if product_name in used_names:
                    product_name = build_deterministic_product_name(
                        theme_label=theme_label,
                        format_name=format_name,
                        focus_phrase=focus_phrase,
                        concept_index=deterministic_counter + 7,
                    )
                used_names.add(product_name)
                concept_obj = {
                    "name": product_name,
                    "format": format_name,
                    "target_consumer": build_deterministic_target_profile(
                        theme_label=theme_label,
                        focus_phrase=focus_phrase,
                        concept_index=deterministic_counter,
                    ),
                    "ingredients": build_deterministic_ingredient_direction(
                        theme_label=theme_label,
                        format_name=format_name,
                        concept_index=deterministic_counter,
                    ),
                    "price_point": build_deterministic_price_point(
                        theme_label=theme_label,
                        format_name=format_name,
                        signals=signals,
                        theme_mentions=theme_mentions,
                        concept_index=deterministic_counter,
                    ),
                    "positioning": build_deterministic_positioning(
                        theme_label=theme_label,
                        format_name=format_name,
                        focus_phrase=focus_phrase,
                        concept_index=deterministic_counter,
                    ),
                    "capabilities": capabilities_for_format(format_name),
                }
                evidence_text = build_concept_evidence(theme, signals)
                scores = score_concept(
                    concept=concept_obj,
                    theme_mentions=theme_mentions,
                    signals=signals,
                    selected_capabilities=selected_capabilities,
                )
                concepts.append(
                    {
                        "theme": theme_label,
                        "product_name": concept_obj["name"],
                        "format": concept_obj["format"],
                        "target_consumer": concept_obj["target_consumer"],
                        "ingredients": concept_obj["ingredients"],
                        "price_point": concept_obj["price_point"],
                        "positioning": concept_obj["positioning"],
                        "evidence": evidence_text,
                        "review_mentions": int(theme_mentions),
                        "forum_mentions": int(min(theme_mentions, int(signals["forum_mentions"]))),
                        "search_interest_avg": float(signals["search_interest_avg"]),
                        "search_data_points": int(signals.get("search_data_points", 0)),
                        "top_search_term": str(signals.get("top_search_term", "N/A")),
                        "theme_quotes": list(theme.get("quotes", [])),
                        **scores,
                    }
                )
        concepts_df = pd.DataFrame(concepts)
        concepts_df = concepts_df.sort_values(by="overall_score", ascending=False).reset_index(drop=True)
        concepts_df = concepts_df.head(min(target_briefs, len(concepts_df))).copy()
        concepts_df.attrs["llm_attempted_count"] = 0
        concepts_df.attrs["llm_generated_count"] = len(concepts_df)
        concepts_df.attrs["llm_failure_reasons"] = {}
        return concepts_df

    for theme_idx, theme in enumerate(themes):
        if llm_generated_count >= target_briefs:
            break
        theme_label = str(theme.get("label", "Consumer Pain"))
        theme_mentions = int(theme.get("mentions", 0))
        theme_quotes = theme.get("quotes", [])
        theme_phrase_pool: List[str] = []

        if not pain_df.empty:
            for _, row in pain_df.iterrows():
                phrase = str(row.get("pain_point", "")).strip()
                if phrase and phrase.lower() in " ".join(theme_quotes).lower():
                    theme_phrase_pool.append(phrase)

        if not theme_phrase_pool:
            theme_phrase_pool = [safe_title_fragment(theme_label, fallback="pain").lower()]

        theme_target = max(1, per_theme_targets[theme_idx])
        for idx in range(theme_target):
            if llm_generated_count >= target_briefs:
                break
            focus_phrase = theme_phrase_pool[idx % len(theme_phrase_pool)]
            field_context = (
                f"Pain theme: {theme_label}\n"
                f"Primary pain phrase: {focus_phrase}\n"
                f"Quotes: {' | '.join([str(q) for q in theme_quotes[:2]])}\n"
                f"Allowed formats: {', '.join(FORMAT_OPTIONS)}\n"
            )
            concept_created = False
            retry_reasons: List[str] = []
            last_candidate_fields: Optional[Dict[str, str]] = None
            last_invalid_reasons: List[str] = []
            for attempt in range(3):
                parsed_fields: Dict[str, str] = {}
                preferred_format = choose_formats_for_theme(theme_label, idx)
                format_context = field_context + f"Preferred format hint: {preferred_format}\n"
                llm_attempted_count += 1
                format_value, format_reason = generate_field_with_local_model(
                    field_name="format",
                    context=format_context,
                    llm_model=llm_model,
                    local_files_only=local_files_only,
                    max_new_tokens=40,
                )
                if not format_value and format_reason != "ok":
                    llm_failure_reasons[f"field_format_{format_reason}"] += 1
                format_name = format_value.title().strip() if format_value else preferred_format
                if format_name not in FORMAT_OPTIONS:
                    format_name = preferred_format
                parsed_fields["format"] = format_name

                for field_name in ["product_name", "target_consumer", "ingredients", "price_point", "positioning"]:
                    llm_attempted_count += 1
                    value, field_reason = generate_field_with_local_model(
                        field_name=field_name,
                        context=field_context + f"Chosen format: {format_name}\n",
                        llm_model=llm_model,
                        local_files_only=local_files_only,
                        max_new_tokens=70 if field_name != "positioning" else 120,
                    )
                    if value:
                        parsed_fields[field_name] = value
                    else:
                        parsed_fields[field_name] = ""
                        if field_reason != "ok":
                            llm_failure_reasons[f"field_{field_name}_{field_reason}"] += 1

                format_name = str(parsed_fields.get("format", "")).strip().title()
                if format_name not in FORMAT_OPTIONS:
                    format_name = choose_formats_for_theme(theme_label, idx)
                parsed_fields["format"] = format_name

                is_valid, invalid_reasons = validate_llm_concept_fields(parsed_fields)
                if not is_valid:
                    last_candidate_fields = {
                        "product_name": str(parsed_fields.get("product_name", "")).strip(),
                        "format": str(parsed_fields.get("format", "")).strip(),
                        "target_consumer": str(parsed_fields.get("target_consumer", "")).strip(),
                        "ingredients": str(parsed_fields.get("ingredients", "")).strip(),
                        "price_point": str(parsed_fields.get("price_point", "")).strip(),
                        "positioning": str(parsed_fields.get("positioning", "")).strip(),
                    }
                    last_invalid_reasons = invalid_reasons.copy()
                    # Targeted field repair before full retry.
                    repaired_any = False
                    for reason in invalid_reasons:
                        field_name = reason.replace("_invalid", "")
                        if field_name not in parsed_fields:
                            continue
                        for _ in range(2):
                            repair_value_prompt = targeted_repair_prompt(
                                field_name=field_name,
                                context=field_context,
                                current_value=str(parsed_fields.get(field_name, "")),
                            )
                            field_value_text, field_reason = generate_with_local_model(
                                prompt=repair_value_prompt,
                                llm_model=llm_model,
                                local_files_only=local_files_only,
                                max_new_tokens=90,
                            )
                            if not field_value_text:
                                if field_reason != "ok":
                                    llm_failure_reasons[f"repair_{field_name}_{field_reason}"] += 1
                                continue
                            parsed_value = extract_first_text_line(field_value_text)
                            if parsed_value:
                                parsed_fields[field_name] = parsed_value
                                repaired_any = True
                                break

                    # Re-validate after targeted repairs.
                    is_valid, invalid_reasons = validate_llm_concept_fields(parsed_fields)
                    if is_valid:
                        format_name = str(parsed_fields.get("format", "")).strip().title()
                        if format_name not in FORMAT_OPTIONS:
                            format_name = choose_formats_for_theme(theme_label, idx)
                        parsed_fields["format"] = format_name
                    elif repaired_any:
                        retry_reasons = invalid_reasons
                        continue

                    for reason in invalid_reasons:
                        llm_failure_reasons[reason] += 1
                    retry_reasons = invalid_reasons
                    continue

                concept_obj = {
                    "name": str(parsed_fields.get("product_name", "")).strip(),
                    "format": format_name,
                    "target_consumer": str(parsed_fields.get("target_consumer", "")).strip(),
                    "ingredients": str(parsed_fields.get("ingredients", "")).strip(),
                    "price_point": str(parsed_fields.get("price_point", "")).strip(),
                    "positioning": str(parsed_fields.get("positioning", "")).strip(),
                    "capabilities": capabilities_for_format(format_name),
                }
                evidence_text = build_concept_evidence(theme, signals)
                scores = score_concept(
                    concept=concept_obj,
                    theme_mentions=theme_mentions,
                    signals=signals,
                    selected_capabilities=selected_capabilities,
                )
                concepts.append(
                    {
                        "theme": theme_label,
                        "product_name": concept_obj["name"],
                        "format": concept_obj["format"],
                        "target_consumer": concept_obj["target_consumer"],
                        "ingredients": concept_obj["ingredients"],
                        "price_point": concept_obj["price_point"],
                        "positioning": concept_obj["positioning"],
                        "evidence": evidence_text,
                        "review_mentions": int(theme_mentions),
                        "forum_mentions": int(min(theme_mentions, int(signals["forum_mentions"]))),
                        "search_interest_avg": float(signals["search_interest_avg"]),
                        "search_data_points": int(signals.get("search_data_points", 0)),
                        "top_search_term": str(signals.get("top_search_term", "N/A")),
                        "theme_quotes": list(theme.get("quotes", [])),
                        **scores,
                    }
                )
                llm_generated_count += 1
                concept_created = True
                break

            if not concept_created:
                # Final accept-with-warning fallback so user gets output.
                if last_candidate_fields:
                    fallback_format = str(last_candidate_fields.get("format", "")).title().strip()
                    if fallback_format not in FORMAT_OPTIONS:
                        fallback_format = choose_formats_for_theme(theme_label, idx)

                    fallback_price = str(last_candidate_fields.get("price_point", "")).strip()
                    if not is_valid_price_point(fallback_price):
                        fallback_price = build_price_point(fallback_format, signals, theme_mentions)

                    fallback_positioning = str(last_candidate_fields.get("positioning", "")).strip()
                    if len(fallback_positioning) < 8:
                        fallback_positioning = build_positioning(theme_label, fallback_format, focus_phrase)

                    fallback_name = str(last_candidate_fields.get("product_name", "")).strip()
                    if not fallback_name:
                        fallback_name = f"{safe_title_fragment(theme_label)} {fallback_format}"

                    fallback_consumer = str(last_candidate_fields.get("target_consumer", "")).strip()
                    if len(fallback_consumer) < 12:
                        fallback_consumer = build_target_profile(theme_label, focus_phrase)

                    fallback_ingredients = str(last_candidate_fields.get("ingredients", "")).strip()
                    if len(fallback_ingredients) < 8 or not is_valid_ingredients_text(fallback_ingredients):
                        fallback_ingredients = build_ingredient_direction(
                            infer_theme_tags(theme_label, focus_phrase),
                            fallback_format,
                        )

                    fallback_concept = {
                        "name": fallback_name,
                        "format": fallback_format,
                        "target_consumer": fallback_consumer,
                        "ingredients": fallback_ingredients,
                        "price_point": fallback_price,
                        "positioning": fallback_positioning,
                        "capabilities": capabilities_for_format(fallback_format),
                    }
                    final_validation_payload = {
                        "product_name": fallback_concept["name"],
                        "format": fallback_concept["format"],
                        "target_consumer": fallback_concept["target_consumer"],
                        "ingredients": fallback_concept["ingredients"],
                        "price_point": fallback_concept["price_point"],
                        "positioning": fallback_concept["positioning"],
                    }
                    _, unresolved_reasons = validate_llm_concept_fields(final_validation_payload)
                    warning_text = ""
                    if unresolved_reasons:
                        warning_text = (
                            "Accepted with warning after retries; "
                            + ", ".join(unresolved_reasons[:3])
                        )
                    evidence_text = build_concept_evidence(theme, signals)
                    scores = score_concept(
                        concept=fallback_concept,
                        theme_mentions=theme_mentions,
                        signals=signals,
                        selected_capabilities=selected_capabilities,
                    )
                    concepts.append(
                        {
                            "theme": theme_label,
                            "product_name": fallback_concept["name"],
                            "format": fallback_concept["format"],
                            "target_consumer": fallback_concept["target_consumer"],
                            "ingredients": fallback_concept["ingredients"],
                            "price_point": fallback_concept["price_point"],
                            "positioning": fallback_concept["positioning"],
                            "evidence": evidence_text,
                            "generation_warning": warning_text,
                            "review_mentions": int(theme_mentions),
                            "forum_mentions": int(min(theme_mentions, int(signals["forum_mentions"]))),
                            "search_interest_avg": float(signals["search_interest_avg"]),
                            "search_data_points": int(signals.get("search_data_points", 0)),
                            "top_search_term": str(signals.get("top_search_term", "N/A")),
                            "theme_quotes": list(theme.get("quotes", [])),
                            **scores,
                        }
                    )
                    llm_generated_count += 1
                    if warning_text:
                        llm_failure_reasons["accepted_with_warning"] += 1
                    concept_created = True
                if not concept_created:
                    llm_failure_reasons["concept_retry_exhausted"] += 1

    if not concepts:
        df = pd.DataFrame()
        df.attrs["llm_attempted_count"] = int(llm_attempted_count)
        df.attrs["llm_generated_count"] = int(llm_generated_count)
        df.attrs["llm_failure_reasons"] = dict(llm_failure_reasons)
        return df

    concepts_df = pd.DataFrame(concepts)
    concepts_df = concepts_df.sort_values(by="overall_score", ascending=False).reset_index(drop=True)

    # Keep output count capped by requested target.
    target_count = min(target_briefs, len(concepts_df))
    concepts_df = concepts_df.head(target_count).copy()

    # Mark top concepts as worth serious exploration.
    concepts_df["exploration_priority"] = "Monitor"
    top_explore_count = min(3, len(concepts_df))
    concepts_df.loc[: top_explore_count - 1, "exploration_priority"] = "Explore Now"
    concepts_df.attrs["llm_attempted_count"] = int(llm_attempted_count)
    concepts_df.attrs["llm_generated_count"] = int(llm_generated_count)
    concepts_df.attrs["llm_failure_reasons"] = dict(llm_failure_reasons)
    return concepts_df


# -------------------------------
# Forward journey header
# -------------------------------
current_index = FLOW_STEPS.index(st.session_state.active_tab)
st.caption(
    f"Journey: {' -> '.join(FLOW_STEPS)} | Market Lens: Indian women’s hair-care"
)
st.progress((current_index + 1) / len(FLOW_STEPS))


# -------------------------------
# Upload Data section
# -------------------------------
if st.session_state.active_tab == "Upload Data":
    st.subheader("Step 1: Required Data Inputs (All 3 Needed)")
    st.caption(
        "Complete all three sources below to unlock Insights: Marketplace Reviews + Reddit Discussions + Google Trends."
    )
    status_col1, status_col2, status_col3 = st.columns(3)
    status_col1.metric("Marketplace", "Ready" if st.session_state.marketplace_dataset is not None else "Pending")
    status_col2.metric("Reddit", "Ready" if st.session_state.reddit_dataset is not None else "Pending")
    status_col3.metric("Google Trends", "Ready" if st.session_state.current_trends_dataset is not None else "Pending")

    st.markdown('<div class="ingest-card">', unsafe_allow_html=True)
    st.markdown("### 1) Marketplace Reviews (Required)")
    st.write("Upload one XLSX file with exactly two columns: `Title` and `Body`.")
    st.caption(
        "Quick prep tip: copy review text from top 5 relevant product listings, paste once into ChatGPT, "
        "ask for a clean XLSX in this template, then upload."
    )
    with st.expander("Copy-Paste Prompt for ChatGPT (Marketplace Data Prep)", expanded=False):
        st.code(
            """I am building a women hair-care insights app for India.
Convert the raw marketplace review text I provide into a downloadable Excel file (.xlsx)
with EXACTLY these two columns and names:
1) Title
2) Body

Rules:
- Keep only meaningful consumer review content.
- Remove spam, duplicate lines, and non-review text.
- Keep one review per row.
- If a review has no clear title, create a short title from the sentence.
- Do not add any extra columns.
- Preserve original meaning; do not rewrite sentiment.

Output needed:
- Provide a downloadable .xlsx file named: marketplace_reviews_prepared.xlsx
- Also show a preview of first 10 rows in table format.

Raw review text starts below:
[PASTE AMAZON/NYKAA/FLIPKART REVIEWS FROM TOP 5 PRODUCTS HERE]""",
            language="text",
        )
    st.download_button(
        label="Download Marketplace XLSX Template",
        data=build_marketplace_template_xlsx_bytes(),
        file_name="marketplace_reviews_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Use this template format: exactly two columns, Title and Body.",
    )
    uploaded_file = st.file_uploader(
        "Upload Marketplace XLSX",
        type=["xlsx"],
        accept_multiple_files=False,
        key="marketplace_xlsx_uploader_required",
        help="Required columns: Title and Body only.",
    )
    if uploaded_file:
        try:
            df = read_xlsx_file(uploaded_file)
            prepared_df = prepare_marketplace_dataframe(df)
            st.session_state.marketplace_dataset = prepared_df.copy()
            st.session_state.marketplace_dataset_name = uploaded_file.name
            render_dataset_summary_block(prepared_df)
            filtered_count = int(prepared_df.attrs.get("male_filtered", 0))
            if filtered_count > 0:
                st.caption(f"Filtered male-authored/male-context rows: {filtered_count}")
            st.success("Marketplace dataset saved.")
        except Exception as exc:
            st.error(f"Marketplace upload failed. {exc}")
    elif st.session_state.marketplace_dataset is not None:
        st.info(f"Using marketplace file: {st.session_state.marketplace_dataset_name}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="ingest-card">', unsafe_allow_html=True)
    st.markdown("### 2) Reddit Discussions (Required)")
    st.markdown(
        """
        - `Fetch Reddit Data`: best for local runs.
        - `Upload Prepared Reddit XLSX`: best for cloud if fetch hits 403/429.
        """
    )
    reddit_route = st.radio(
        "Reddit input route",
        ["Fetch Reddit Data", "Upload Prepared Reddit XLSX"],
        horizontal=True,
        key="reddit_route_required",
    )
    if reddit_route == "Fetch Reddit Data":
        default_subs = "IndianSkincareAddicts, HaircareScience, FemaleHairAdvice"
        default_keywords = "women hair fall, hair thinning women, scalp irritation, dandruff women, hair regrowth"
        reddit_subreddits_text = st.text_input("Subreddits (comma-separated)", value=default_subs)
        reddit_keywords_text = st.text_input("Keywords (comma-separated)", value=default_keywords)
        col_r1, col_r2, col_r3 = st.columns(3)
        days_back = int(col_r1.slider("Lookback (days)", min_value=30, max_value=365, value=120, step=10))
        max_posts = int(col_r2.slider("Max posts/subreddit", min_value=20, max_value=120, value=60, step=10))
        max_comments = int(col_r3.slider("Max comments/post", min_value=5, max_value=80, value=25, step=5))
        if st.button("Fetch Reddit Discussions"):
            subreddits = [item.strip() for item in reddit_subreddits_text.split(",") if item.strip()]
            keywords = [item.strip() for item in reddit_keywords_text.split(",") if item.strip()]
            if not subreddits:
                st.error("Please enter at least one subreddit.")
            elif not keywords:
                st.error("Please enter at least one keyword.")
            else:
                progress_caption = st.empty()

                def update_reddit_progress(subreddit_name: str, comments_count: int) -> None:
                    progress_caption.caption(
                        f"Fetching r/{subreddit_name} | comments fetched so far: {comments_count}"
                    )

                with st.spinner("Fetching Reddit posts and comments..."):
                    try:
                        reddit_df = fetch_reddit_public_discussions(
                            subreddits=subreddits,
                            keywords=keywords,
                            days_back=days_back,
                            max_posts_per_subreddit=max_posts,
                            max_comments_per_post=max_comments,
                            progress_callback=update_reddit_progress,
                        )
                        progress_caption.caption("Reddit fetch complete.")
                        reddit_diag = reddit_df.attrs.get("reddit_diagnostics", {})
                        if reddit_df.empty:
                            st.warning("No Reddit data fetched for current filters.")
                            render_reddit_fetch_diagnostics(reddit_diag)
                        else:
                            cleaned_reddit_df, cleaning_summary = clean_reddit_dataframe(reddit_df)
                            st.session_state.reddit_dataset = cleaned_reddit_df.copy()
                            st.session_state.reddit_dataset_name = (
                                f"reddit_public_{len(subreddits)}subs_{len(keywords)}terms"
                            )
                            render_dataset_summary_block(cleaned_reddit_df)
                            render_reddit_fetch_diagnostics(reddit_diag)
                            st.caption(cleaning_summary)
                            reddit_export_bytes = build_dataset_xlsx_bytes(cleaned_reddit_df)
                            st.download_button(
                                label="Download Reddit Data as XLSX",
                                data=reddit_export_bytes,
                                file_name="reddit_prepared_dataset.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                help="This file is upload-safe in the manual Reddit route.",
                            )
                            st.success("Reddit dataset saved.")
                    except Exception as exc:
                        st.error(
                            "Reddit fetch failed. Try fewer subreddits or upload prepared Reddit XLSX. "
                            f"Technical detail: {exc}"
                        )
    else:
        st.download_button(
            label="Download Reddit XLSX Template",
            data=build_marketplace_template_xlsx_bytes(),
            file_name="reddit_prepared_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Use exact schema: Title and Body.",
        )
        reddit_uploaded_file = st.file_uploader(
            "Upload Prepared Reddit XLSX",
            type=["xlsx"],
            accept_multiple_files=False,
            key="reddit_manual_xlsx_uploader_required",
            help="Required columns: Title and Body only.",
        )
        if reddit_uploaded_file:
            try:
                reddit_manual_df = read_xlsx_file(reddit_uploaded_file)
                prepared_reddit_df = prepare_marketplace_dataframe(reddit_manual_df)
                cleaned_reddit_df, cleaning_summary = clean_reddit_dataframe(prepared_reddit_df)
                st.session_state.reddit_dataset = cleaned_reddit_df.copy()
                st.session_state.reddit_dataset_name = reddit_uploaded_file.name
                render_dataset_summary_block(cleaned_reddit_df)
                reddit_male_filtered = int(prepared_reddit_df.attrs.get("male_filtered", 0))
                if reddit_male_filtered > 0:
                    st.caption(f"Filtered male-authored/male-context rows: {reddit_male_filtered}")
                st.caption(cleaning_summary)
                st.success("Reddit dataset saved.")
            except Exception as exc:
                st.error(f"Reddit upload failed. {exc}")
        elif st.session_state.reddit_dataset is not None:
            st.info(f"Using reddit file: {st.session_state.reddit_dataset_name}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="ingest-card">', unsafe_allow_html=True)
    st.markdown("### 3) Google Trends (Required)")
    base_for_terms = st.session_state.marketplace_dataset if st.session_state.marketplace_dataset is not None else pd.DataFrame(columns=["title", "body"])
    base_texts = get_analysis_texts(base_for_terms)
    base_pain_df = build_pain_point_table(base_texts, top_n=20) if base_texts else pd.DataFrame()
    base_themes = build_theme_previews(base_texts, min_themes=3, max_themes=5, pain_df=base_pain_df) if base_texts else []
    trend_terms = build_dynamic_trends_seed_terms(base_pain_df, base_themes)
    trend_batches = build_trends_query_batches(trend_terms, batch_size=5)
    st.caption("Google Trends is split into 5-term batches. Open each batch and upload its CSV.")
    uploaded_trend_frames: List[pd.DataFrame] = []
    uploaded_trend_names: List[str] = []
    for idx, batch in enumerate(trend_batches[:8], start=1):
        row_left, row_right = st.columns(2)
        with row_left:
            st.caption(f"Batch {idx}")
            st.link_button(
                f"Open Google Trends Explore (Batch {idx})",
                build_trends_explore_url_from_terms(batch),
                use_container_width=True,
            )
        with row_right:
            batch_file = st.file_uploader(
                f"Upload Google Trends CSV - Batch {idx}",
                type=["csv"],
                accept_multiple_files=False,
                key=f"trends_csv_batch_{idx}",
            )
        if batch_file:
            try:
                batch_df = read_google_trends_csv(batch_file)
                uploaded_trend_frames.append(batch_df)
                uploaded_trend_names.append(batch_file.name)
            except Exception as exc:
                st.error(f"Batch {idx} could not be read. {exc}")

    if uploaded_trend_frames:
        trends_df = combine_trends_batches(uploaded_trend_frames)
        st.session_state.current_trends_dataset = trends_df.copy()
        st.session_state.current_trends_dataset_name = " + ".join(uploaded_trend_names)
        trends_summary = build_trends_signal_summary(trends_df)
        tcol1, tcol2, tcol3 = st.columns(3)
        tcol1.metric("Rows", len(trends_df))
        tcol2.metric("Columns", len(trends_df.columns))
        tcol3.metric("Search Data Points", trends_summary["search_data_points"])
        st.caption(
            f"Avg Search Interest: {trends_summary['search_interest_avg']} | "
            f"Top Search Term: {trends_summary['top_search_term']}"
        )
        st.dataframe(trends_df.head(5), use_container_width=True)
        st.success(f"Google Trends batches combined and saved ({len(uploaded_trend_frames)} batch file(s)).")
    elif st.session_state.current_trends_dataset is not None:
        st.info(f"Using trends file(s): {st.session_state.current_trends_dataset_name}")
    st.markdown("</div>", unsafe_allow_html=True)

    refresh_combined_dataset_state()
    all_required_ready = (
        st.session_state.marketplace_dataset is not None
        and st.session_state.reddit_dataset is not None
        and st.session_state.current_trends_dataset is not None
        and st.session_state.current_dataset is not None
    )
    if not all_required_ready:
        st.warning("All 3 sources are mandatory. Please complete Marketplace, Reddit, and Google Trends inputs.")

    next_disabled = not all_required_ready
    if st.button("Next -> Insights", disabled=next_disabled):
        st.session_state.active_tab = "Insights"
        st.rerun()


# -------------------------------
# Insights section
# -------------------------------
if st.session_state.active_tab == "Insights":
    st.subheader("Step 2: Consumer Insight Signals (Women Hair-Care, India)")
    required_missing = (
        st.session_state.marketplace_dataset is None
        or st.session_state.reddit_dataset is None
        or st.session_state.current_trends_dataset is None
    )
    if required_missing or st.session_state.current_dataset is None:
        st.info("Complete all required Step 1 inputs (Marketplace + Reddit + Google Trends), then click Next.")
    else:
        data_df = st.session_state.current_dataset.copy()
        data_name = st.session_state.current_dataset_name or "current dataset"

        st.caption(f"Source dataset: {data_name}")
        texts = get_analysis_texts(data_df)

        if not texts:
            st.warning("No usable text found for analysis. Ensure title/body have content.")
        else:
            pain_df = build_pain_point_table(texts, top_n=15)
            themes = build_theme_previews(texts, min_themes=3, max_themes=5, pain_df=pain_df)
            st.markdown("### Top Recurring Keywords (Pain Signals)")
            if pain_df.empty:
                st.info("Not enough complaint-like text to extract pain phrases yet.")
            else:
                bubble_df = pain_df[["pain_point", "count", "example_1", "example_2"]].copy()
                if themes:
                    bubble_df["theme"] = bubble_df["pain_point"].apply(
                        lambda phrase: assign_pain_phrase_theme_label(str(phrase), themes)
                    )
                else:
                    bubble_df["theme"] = "Other Emerging Issues"
                bubble_df["examples"] = bubble_df.apply(
                    lambda row: " | ".join(
                        list(
                            dict.fromkeys(
                                [
                                    str(row.get("example_1", "")).strip(),
                                    str(row.get("example_2", "")).strip(),
                                ]
                            )
                        )[:2]
                    ),
                    axis=1,
                )
                bubble_df["examples"] = bubble_df["examples"].replace("", "No example available")
                bubble_chart = (
                    alt.Chart(bubble_df)
                    .mark_circle(opacity=0.82, stroke="#1f2937", strokeWidth=0.5)
                    .encode(
                        x=alt.X("count:Q", title="Mention Count"),
                        y=alt.Y("pain_point:N", sort="-x", title="Pain Point"),
                        size=alt.Size("count:Q", scale=alt.Scale(range=[260, 2600]), legend=None),
                        color=alt.Color("theme:N", title="Mapped Theme"),
                        tooltip=[
                            alt.Tooltip("pain_point:N", title="Pain Point"),
                            alt.Tooltip("theme:N", title="Mapped Theme"),
                            alt.Tooltip("count:Q", title="Count"),
                            alt.Tooltip("examples:N", title="Unique Examples"),
                        ],
                    )
                    .properties(height=420)
                )
                st.altair_chart(bubble_chart, use_container_width=True)

            st.markdown("### Sentiment Breakdown")
            sentiment_df = build_sentiment_table(texts)
            sent_left, sent_right = st.columns([1, 1.25])
            with sent_left:
                st.metric(
                    "% Negative",
                    f"{sentiment_df.loc[sentiment_df['sentiment'] == 'Negative', 'percentage'].iloc[0]}%",
                )
                st.metric(
                    "% Neutral",
                    f"{sentiment_df.loc[sentiment_df['sentiment'] == 'Neutral', 'percentage'].iloc[0]}%",
                )
                st.metric(
                    "% Positive",
                    f"{sentiment_df.loc[sentiment_df['sentiment'] == 'Positive', 'percentage'].iloc[0]}%",
                )
            with sent_right:
                fig, ax = plt.subplots(figsize=(3.2, 3.2))
                ax.pie(
                    sentiment_df["count"],
                    labels=sentiment_df["sentiment"],
                    autopct="%1.1f%%",
                    startangle=90,
                )
                ax.axis("equal")
                st.pyplot(fig)

            st.markdown("### Top Pain Themes (Cluster Preview)")
            if not themes:
                st.info("Not enough data to build themes yet.")
            else:
                for idx, theme in enumerate(themes, start=1):
                    st.markdown(f"**Theme {idx}: {theme['label']}**")
                    st.write(f"Mentions: {theme['mentions']}")
                    quotes = theme.get("quotes", [])
                    if quotes:
                        for quote_idx, quote in enumerate(quotes[:5], start=1):
                            st.write(f"Quote {quote_idx}: \"{quote}\"")
                    st.divider()

    next_disabled = (
        st.session_state.current_dataset is None
        or st.session_state.current_trends_dataset is None
        or st.session_state.marketplace_dataset is None
        or st.session_state.reddit_dataset is None
    )
    if st.button("Next -> Product Concepts", disabled=next_disabled):
        st.session_state.active_tab = "Product Concepts"
        st.rerun()


# -------------------------------
# Placeholder sections
# -------------------------------
if st.session_state.active_tab == "Product Concepts":
    st.subheader("Step 3: Product Opportunity Concepts")
    required_missing = (
        st.session_state.marketplace_dataset is None
        or st.session_state.reddit_dataset is None
        or st.session_state.current_trends_dataset is None
    )
    if required_missing or st.session_state.current_dataset is None:
        st.info("Complete all required Step 1 inputs before opening Product Concepts.")
    else:
        concept_df_source = st.session_state.current_dataset.copy()
        concept_texts = get_analysis_texts(concept_df_source)
        concept_pain_df = build_pain_point_table(concept_texts, top_n=20)
        concept_themes = build_theme_previews(concept_texts, min_themes=3, max_themes=5, pain_df=concept_pain_df)
        selected_capabilities = {
            "Topical Formulation",
            "Clinical Claims",
            "D2C Community",
            "Packaging Innovation",
        }
        concept_count = infer_target_concept_count(len(concept_themes))
        st.caption(
            f"Auto concept count: {concept_count} (derived from {len(concept_themes)} pain themes)."
        )
        

        if not concept_themes:
            st.info("No pain themes detected yet. Refine data quality or add more complaint-heavy data.")
        else:
            theme_count = len(concept_themes)
            base_per_theme = concept_count // theme_count
            remainder = concept_count % theme_count
            per_theme_targets = [base_per_theme + (1 if i < remainder else 0) for i in range(theme_count)]

            sequential_targets: List[Dict[str, object]] = []
            for theme_idx, theme in enumerate(concept_themes):
                for _ in range(per_theme_targets[theme_idx]):
                    sequential_targets.append(theme)

            progress_bar = st.progress(0.0)

            total_targets = max(1, len(sequential_targets))
            total_attempted = 0
            total_generated = 0
            all_failures: Counter = Counter()
            rendered_concepts: List[Dict[str, object]] = []

            for seq_idx, theme in enumerate(sequential_targets, start=1):
                single_df = generate_product_concepts(
                    themes=[theme],
                    pain_df=concept_pain_df,
                    data_df=concept_df_source,
                    trends_df=st.session_state.current_trends_dataset,
                    selected_capabilities=selected_capabilities,
                    use_local_llm=False,
                    min_concepts=1,
                    max_concepts=1,
                    concept_index_seed=seq_idx - 1,
                )

                total_attempted += int(single_df.attrs.get("llm_attempted_count", 0))
                total_generated += int(single_df.attrs.get("llm_generated_count", 0))
                all_failures.update(single_df.attrs.get("llm_failure_reasons", {}))

                if not single_df.empty:
                    rendered_concepts.extend(single_df.to_dict(orient="records"))

                progress_bar.progress(seq_idx / total_targets)

            if not rendered_concepts:
                st.warning("Could not generate concept briefs from current themes in deterministic mode.")
            else:
                concepts_df = pd.DataFrame(rendered_concepts).copy()
                concepts_df = enrich_opportunity_scores(
                    concepts_df,
                    trends_df=st.session_state.current_trends_dataset,
                )
                concepts_df["concept_id"] = [
                    f"concept_{idx + 1}_{re.sub(r'[^a-z0-9]+', '_', str(name).lower())[:24]}"
                    for idx, name in enumerate(concepts_df["product_name"].tolist())
                ]
                concepts_df = concepts_df.sort_values(by="opportunity_score", ascending=False).reset_index(drop=True)

                # Decision summary: top 3 opportunities
                st.markdown("### All Identified Women Hair-Care Opportunities")
                opportunity_help = (
                    "Opportunity Score = 50% Demand + 25% Competition Advantage + 25% Novelty. "
                    "Demand uses review/forum mentions plus concept-specific search traction and trend momentum. "
                    "Competition advantage uses format competition and format rarity. "
                    "Novelty uses ingredient overlap plus name/positioning uniqueness."
                )
                for start_idx in range(0, len(concepts_df), 3):
                    row_slice = concepts_df.iloc[start_idx : start_idx + 3]
                    card_cols = st.columns(3)
                    for col_idx, (_, row) in enumerate(row_slice.iterrows()):
                        with card_cols[col_idx]:
                            st.markdown(f"**{row['product_name']}**")
                            st.caption(f"`{row['format']}` | Theme: {row['theme']}")
                            st.metric(
                                "Opportunity Score",
                                f"{row['opportunity_score']:.1f}/100",
                                help=opportunity_help,
                            )
                            score_col1, score_col2, score_col3 = st.columns(3)
                            score_col1.metric("Demand", f"{float(row.get('demand_component', 0)):.1f}")
                            score_col2.metric("Comp. Adv.", f"{float(row.get('competition_component', 0)):.1f}")
                            score_col3.metric("Novelty", f"{float(row.get('novelty_component', 0)):.1f}")
                            st.progress(float(row["opportunity_score"]) / 100.0)
                            st.caption(
                                f"Reviews: {int(row.get('display_review_mentions', row['review_mentions']))} | "
                                f"Forums: {int(row.get('display_forum_mentions', row['forum_mentions']))} | "
                                f"Search Avg: {float(row.get('display_search_interest', row['search_interest_avg']))}"
                            )
                            if st.button("View Brief", key=f"view_{row['concept_id']}"):
                                st.session_state.focus_concept_id = row["concept_id"]
                            if str(st.session_state.get("focus_concept_id", "")) == str(row["concept_id"]):
                                st.markdown("**Selected Product Concept Brief**")
                                render_full_concept_brief(row.to_dict())

