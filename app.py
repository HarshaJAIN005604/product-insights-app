import io
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import praw
import streamlit as st
from prawcore.exceptions import PrawcoreException
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# -------------------------------
# Page configuration and title
# -------------------------------
st.set_page_config(
    page_title="AI Product Inventor",
    page_icon=":bulb:",
    layout="wide",
)

st.title("AI Product Inventor")
st.caption("Upload consumer data and build evidence-backed product ideas.")


# -------------------------------
# Session state defaults
# -------------------------------
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Upload Data"
if "current_dataset" not in st.session_state:
    st.session_state.current_dataset = None
if "current_dataset_name" not in st.session_state:
    st.session_state.current_dataset_name = ""


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
        "title": "Perceived Waste of Money",
        "patterns": [
            r"\bwaste of money\b",
            r"\bmoney wasted\b",
            r"\bnot worth\b",
            r"\b(overpriced|too expensive|very expensive|expensive)\b",
        ],
    },
]

COMPETITION_BY_FORMAT = {
    "Serum": 72,
    "Shampoo": 82,
    "Tablet": 68,
    "Gummy": 78,
    "Scalp Tonic": 64,
    "Leave-In Mist": 66,
}

FORMAT_OPTIONS = ["Serum", "Shampoo", "Tablet", "Gummy"]

THEME_FORMAT_HINTS = {
    "No Visible Results After Long-Term Use": ["Serum", "Tablet", "Gummy"],
    "Rebound Hair Fall After Stopping Usage": ["Serum", "Gummy", "Tablet"],
    "Product Runs Out Too Fast": ["Serum", "Shampoo", "Tablet"],
    "Scalp Dryness & Irritation": ["Shampoo", "Serum", "Gummy"],
    "Perceived Waste of Money": ["Shampoo", "Tablet", "Serum"],
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

    return normalized_df.copy()


def build_reddit_client(client_id: str, client_secret: str, user_agent: str) -> praw.Reddit:
    """Create a Reddit client from provided credentials."""
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        check_for_async=False,
    )


def fetch_reddit_data(
    reddit_client: praw.Reddit,
    subreddit_name: str,
    max_posts: int,
    max_comments_per_post: int,
) -> pd.DataFrame:
    """Fetch recent posts and comments from a subreddit and return a unified dataset."""
    three_months_ago = datetime.now(timezone.utc) - timedelta(days=90)
    rows: List[Dict[str, object]] = []

    subreddit = reddit_client.subreddit(subreddit_name)

    for submission in subreddit.new(limit=max_posts):
        created_dt = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
        if created_dt < three_months_ago:
            continue

        rows.append(
            {
                "source_type": "reddit_post",
                "subreddit": subreddit_name,
                "title": submission.title or "",
                "body": submission.selftext or "",
                "author": str(submission.author) if submission.author else "[deleted]",
                "score": submission.score,
                "created_utc": created_dt.isoformat(),
                "permalink": f"https://reddit.com{submission.permalink}",
            }
        )

        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list()[:max_comments_per_post]:
            comment_dt = datetime.fromtimestamp(comment.created_utc, tz=timezone.utc)
            rows.append(
                {
                    "source_type": "reddit_comment",
                    "subreddit": subreddit_name,
                    "title": submission.title or "",
                    "body": comment.body or "",
                    "author": str(comment.author) if comment.author else "[deleted]",
                    "score": comment.score,
                    "created_utc": comment_dt.isoformat(),
                    "permalink": f"https://reddit.com{comment.permalink}",
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "source_type",
                "subreddit",
                "title",
                "body",
                "author",
                "score",
                "created_utc",
                "permalink",
            ]
        )

    return pd.DataFrame(rows)


def dataframe_to_xlsx_bytes(df: pd.DataFrame, sheet_name: str = "data") -> bytes:
    """Convert a DataFrame to XLSX bytes for Streamlit downloads."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    output.seek(0)
    return output.getvalue()


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


def build_theme_previews(texts: List[str], min_themes: int = 3, max_themes: int = 5) -> List[Dict[str, object]]:
    """Map complaint-like sentences to normalized consumer pain themes."""
    complaint_sentences = extract_complaint_sentences(texts)
    if not complaint_sentences:
        return []

    theme_buckets: Dict[str, Dict[str, object]] = {
        rule["title"]: {"mentions": 0, "quotes": []} for rule in THEME_RULES
    }

    for sentence in complaint_sentences:
        normalized_sentence = normalize_pain_phrase(sentence.lower())

        for rule in THEME_RULES:
            matched = any(
                re.search(pattern, normalized_sentence, flags=re.IGNORECASE)
                for pattern in rule["patterns"]
            )
            if not matched:
                continue

            bucket = theme_buckets[rule["title"]]
            bucket["mentions"] += 1
            if len(bucket["quotes"]) < 2:
                quote = sentence.strip()
                if len(quote) > 200:
                    quote = f"{quote[:197]}..."
                if quote not in bucket["quotes"]:
                    bucket["quotes"].append(quote)
            break

    themes = []
    for title, data in theme_buckets.items():
        if data["mentions"] == 0:
            continue
        themes.append(
            {
                "label": title,
                "mentions": data["mentions"],
                "quotes": data["quotes"],
            }
        )

    themes.sort(key=lambda item: item["mentions"], reverse=True)
    return themes[:max_themes]


def build_data_signal_summary(df: pd.DataFrame) -> Dict[str, float]:
    """Compute lightweight source signals for concept scoring and citations."""
    normalized_df = normalize_columns(df)
    total_rows = len(normalized_df)

    if "source_type" in normalized_df.columns:
        forum_mentions = int(normalized_df["source_type"].astype(str).str.contains("reddit").sum())
        review_mentions = max(total_rows - forum_mentions, 0)
    else:
        forum_mentions = 0
        review_mentions = total_rows

    search_interest_avg = 0.0
    if "search interest" in normalized_df.columns:
        search_interest_avg = float(
            pd.to_numeric(normalized_df["search interest"], errors="coerce").fillna(0).mean()
        )

    return {
        "review_mentions": float(review_mentions),
        "forum_mentions": float(forum_mentions),
        "search_interest_avg": round(search_interest_avg, 2),
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


def build_dynamic_concept_blueprint(
    theme: Dict[str, object],
    top_pain_phrase: str,
    concept_index: int,
    signals: Dict[str, float],
) -> Dict[str, object]:
    """Construct one concept brief dynamically from current data signals."""
    theme_label = str(theme["label"])
    format_name = choose_formats_for_theme(theme_label, concept_index)
    tags = infer_theme_tags(theme_label, top_pain_phrase)
    name_core = safe_title_fragment(top_pain_phrase, fallback="Relief")
    product_name = f"{name_core} {format_name}"

    return {
        "name": product_name,
        "format": format_name,
        "target_consumer": build_target_profile(theme_label, top_pain_phrase),
        "ingredients": build_ingredient_direction(tags, format_name),
        "price_point": build_price_point(format_name, signals, int(theme.get("mentions", 0))),
        "positioning": build_positioning(theme_label, format_name, top_pain_phrase),
        "capabilities": capabilities_for_format(format_name),
    }


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


def parse_refinement_fields(text: str) -> Dict[str, str]:
    """Parse key/value fields from flexible model output formats."""
    fields = {
        "product_name": "",
        "format": "",
        "target_consumer": "",
        "ingredients": "",
        "price_point": "",
        "positioning": "",
    }

    # 1) Try JSON object first.
    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            if isinstance(obj, dict):
                for key in fields:
                    value = clean_generated_field_value(key, str(obj.get(key, "")))
                    if value:
                        fields[key] = value
        except Exception:
            pass

    # 2) Flexible line parsing with common aliases.
    aliases = {
        "product_name": {"product_name", "name", "product", "concept_name"},
        "format": {"format", "product_format", "type"},
        "target_consumer": {"target_consumer", "consumer", "target", "target_profile", "audience"},
        "ingredients": {"ingredients", "formulation", "ingredient_direction", "formula"},
        "price_point": {"price_point", "price", "pricing", "mrp"},
        "positioning": {"positioning", "competitive_positioning", "value_proposition", "position"},
    }
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-*").strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key_norm = re.sub(r"[^a-z_ ]", "", key.lower()).replace(" ", "_").strip("_")
        value = value.strip()
        if not value:
            continue
        for canonical_key, alias_set in aliases.items():
            if key_norm in alias_set:
                fields[canonical_key] = clean_generated_field_value(canonical_key, value)
                break

    # 3) Fallback: if still mostly empty, use generated text as positioning.
    filled = sum(bool(v.strip()) for v in fields.values())
    if filled == 0:
        compact_text = " ".join(text.split())
        if compact_text:
            fields["positioning"] = compact_text[:220]

    return fields


def extract_json_array(text: str) -> Optional[List[Dict[str, object]]]:
    """Extract JSON array of objects from noisy model output."""
    cleaned = text.strip()
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).replace("```", "").strip()

    # Try direct parse first.
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, list):
            return [item for item in obj if isinstance(item, dict)]
    except Exception:
        pass

    # Try to extract the first array block.
    match = re.search(r"\[[\s\S]*\]", cleaned)
    if match:
        block = match.group(0).strip()
        # Normalize common invalid JSON patterns.
        block = block.replace("\n", " ").replace("\r", " ")
        block = re.sub(r",\s*([}\]])", r"\1", block)  # trailing commas
        # single quotes -> double quotes (best effort)
        block = re.sub(r"(?<!\\)'", '"', block)
        try:
            arr = json.loads(block)
            if isinstance(arr, list):
                return [item for item in arr if isinstance(item, dict)]
        except Exception:
            pass

    # Fallback: parse object blocks and wrap as list.
    object_matches = re.findall(r"\{[\s\S]*?\}", cleaned)
    parsed_objects: List[Dict[str, object]] = []
    for obj_text in object_matches:
        candidate = re.sub(r",\s*([}\]])", r"\1", obj_text)
        candidate = re.sub(r"(?<!\\)'", '"', candidate)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                parsed_objects.append(parsed)
        except Exception:
            continue
    if parsed_objects:
        return parsed_objects
    return None


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


def refine_concept_with_local_llm(
    concept: Dict[str, object],
    evidence: str,
    use_local_llm: bool,
    llm_model: str,
    local_files_only: bool = False,
) -> Tuple[Dict[str, object], bool, str]:
    """Optionally refine concept brief language using a local transformers model."""
    if not use_local_llm:
        return concept, False, "disabled"

    text2text = get_transformers_text2text_pipeline(
        model_id=llm_model,
        local_files_only=local_files_only,
    )
    pipeline_obj, load_error = text2text
    if pipeline_obj is None:
        return concept, False, f"model_load_failed: {load_error[:120]}"

    prompt = f"""
You are a product strategist. Rewrite this concept brief clearly for consumer products.
Keep facts consistent with evidence. Keep concise and practical.
Return ONLY these 4 lines:
product_name: ...
target_consumer: ...
ingredients: ...
positioning: ...

INPUT:
product_name: {concept.get("name")}
format: {concept.get("format")}
target_consumer: {concept.get("target_consumer")}
ingredients: {concept.get("ingredients")}
positioning: {concept.get("positioning")}
evidence: {evidence}
"""
    try:
        tokenizer = pipeline_obj["tokenizer"]
        model = pipeline_obj["model"]
        kind = pipeline_obj["kind"]

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if kind == "causal" and decoded.startswith(prompt.strip()):
            response_text = decoded[len(prompt.strip()) :].strip()
        else:
            response_text = decoded
    except Exception:
        return concept, False, "generation_failed"

    parsed = parse_refinement_fields(response_text)
    if not any(v.strip() for v in parsed.values()):
        return concept, False, "parse_failed"

    refined = concept.copy()
    before_snapshot = (
        str(refined.get("name", "")),
        str(refined.get("target_consumer", "")),
        str(refined.get("ingredients", "")),
        str(refined.get("positioning", "")),
    )
    refined["name"] = parsed.get("product_name", refined["name"])
    refined["target_consumer"] = parsed.get("target_consumer", refined["target_consumer"])
    refined["ingredients"] = parsed.get("ingredients", refined["ingredients"])
    refined["positioning"] = parsed.get("positioning", refined["positioning"])
    after_snapshot = (
        str(refined.get("name", "")),
        str(refined.get("target_consumer", "")),
        str(refined.get("ingredients", "")),
        str(refined.get("positioning", "")),
    )
    changed = before_snapshot != after_snapshot
    return refined, changed, "ok" if changed else "no_change"


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


def build_concept_evidence(theme: Dict[str, object], signals: Dict[str, float]) -> str:
    """Build short citation text from available consumer data signals."""
    quotes = theme.get("quotes", [])
    quote_parts = []
    for idx, quote in enumerate(quotes[:2], start=1):
        quote_parts.append(f'Q{idx}: "{quote}"')
    quote_text = " | ".join(quote_parts) if quote_parts else "No direct quotes available."

    return (
        f"Mentions in uploaded data: {theme.get('mentions', 0)}; "
        f"Review mentions: {int(signals['review_mentions'])}; "
        f"Forum mentions: {int(signals['forum_mentions'])}; "
        f"Search interest avg: {signals['search_interest_avg']}."
        f" {quote_text}"
    )


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
    selected_capabilities: set,
    use_local_llm: bool = False,
    llm_model: str = "google/flan-t5-base",
    local_files_only: bool = False,
    min_concepts: int = 5,
    max_concepts: int = 10,
) -> pd.DataFrame:
    """Generate concept briefs from pain themes using LLM only."""
    if not themes or not use_local_llm:
        df = pd.DataFrame()
        df.attrs["llm_attempted_count"] = 0
        df.attrs["llm_generated_count"] = 0
        df.attrs["llm_failure_reasons"] = {"llm_required": 1}
        return df

    signals = build_data_signal_summary(data_df)
    concepts: List[Dict[str, object]] = []
    llm_attempted_count = 0
    llm_generated_count = 0
    llm_failure_reasons: Counter = Counter()

    # Deterministic concept-construction path intentionally disabled per request.
    # The app now uses LLM-only concept brief generation.

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
# Top navigation
# -------------------------------
sections = ["Upload Data", "Insights", "Product Concepts"]
current_index = sections.index(st.session_state.active_tab)
selected_section = st.radio(
    "Sections",
    options=sections,
    index=current_index,
    horizontal=True,
)
if selected_section != st.session_state.active_tab:
    st.session_state.active_tab = selected_section


# -------------------------------
# Upload Data section
# -------------------------------
if st.session_state.active_tab == "Upload Data":
    st.subheader("Import Data")
    st.write("Choose one source to import data.")

    selected_source = st.radio(
        "Data source",
        options=["Reddit", "Marketplace (Amazon/Nykaa/Flipkart etc.)"],
        horizontal=True,
    )

    if selected_source == "Reddit":
        st.markdown("### Reddit")
        st.write(
            "Enter a subreddit name and fetch posts + comments from the last 3 months "
            "(or as many as Reddit API allows)."
        )

        default_client_id = st.secrets.get("REDDIT_CLIENT_ID", "")
        default_client_secret = st.secrets.get("REDDIT_CLIENT_SECRET", "")
        default_user_agent = st.secrets.get(
            "REDDIT_USER_AGENT", "ai-product-inventor-app/0.1 by streamlit-user"
        )

        subreddit_name = st.text_input("Subreddit name (without r/)", value="entrepreneur")
        max_posts = st.number_input(
            "Max posts to fetch",
            min_value=10,
            max_value=1000,
            value=200,
            step=10,
            help="API limits and subreddit activity affect final result count.",
        )
        max_comments_per_post = st.number_input(
            "Max comments per post",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
        )

        if default_client_id and default_client_secret:
            st.caption("Using Reddit credentials from Streamlit secrets.")
            client_id = default_client_id
            client_secret = default_client_secret
            user_agent = default_user_agent
        else:
            st.warning(
                "Reddit API credentials are not configured in secrets. "
                "Please enter them below."
            )
            client_id = st.text_input("Reddit Client ID")
            client_secret = st.text_input("Reddit Client Secret", type="password")
            user_agent = st.text_input(
                "Reddit User Agent", value="ai-product-inventor-app/0.1 by streamlit-user"
            )

        if st.button("Fetch Reddit Data", type="primary"):
            if not subreddit_name.strip():
                st.error("Please enter a subreddit name (example: entrepreneur).")
            elif not client_id.strip() or not client_secret.strip() or not user_agent.strip():
                st.error(
                    "Reddit credentials are required. Provide Client ID, Client Secret, and User Agent."
                )
            else:
                try:
                    with st.spinner("Fetching subreddit posts and comments..."):
                        reddit_client = build_reddit_client(
                            client_id=client_id.strip(),
                            client_secret=client_secret.strip(),
                            user_agent=user_agent.strip(),
                        )
                        reddit_df = fetch_reddit_data(
                            reddit_client=reddit_client,
                            subreddit_name=subreddit_name.strip(),
                            max_posts=int(max_posts),
                            max_comments_per_post=int(max_comments_per_post),
                        )

                    if reddit_df.empty:
                        st.warning(
                            "No recent Reddit data was found in the last 3 months, "
                            "or API access was limited."
                        )
                    else:
                        st.session_state.current_dataset = reddit_df.copy()
                        st.session_state.current_dataset_name = f"reddit_r_{subreddit_name.strip()}"
                        with st.expander(
                            f"Reddit dataset: r/{subreddit_name.strip()}",
                            expanded=True,
                        ):
                            render_dataset_summary_block(reddit_df)
                            st.download_button(
                                label="Download Reddit Data (XLSX)",
                                data=dataframe_to_xlsx_bytes(reddit_df, sheet_name="reddit_data"),
                                file_name=f"reddit_r_{subreddit_name.strip()}_last_3_months.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            )
                            st.success("Dataset ready for Insights.")

                except PrawcoreException as exc:
                    st.error(
                        "Could not fetch Reddit data. Check credentials, subreddit name, and internet access. "
                        f"Technical detail: {exc}"
                    )
                except Exception as exc:
                    st.error(
                        "Unexpected Reddit import error. "
                        "Please retry and verify your inputs. "
                        f"Technical detail: {exc}"
                    )

    if selected_source == "Marketplace (Amazon/Nykaa/Flipkart etc.)":
        st.markdown("### Marketplace (Amazon/Nykaa/Flipkart etc.)")
        st.write("Upload one XLSX file with exactly two columns: `Title` and `Body`.")
        st.download_button(
            label="Download Marketplace XLSX Template",
            data=build_marketplace_template_xlsx_bytes(),
            file_name="marketplace_reviews_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Use this template format: exactly two columns, Title and Body.",
        )

        uploaded_file = st.file_uploader(
            "Choose XLSX file",
            type=["xlsx"],
            accept_multiple_files=False,
            help="Required columns: Title and Body only.",
        )

        if not uploaded_file:
            st.info("No XLSX uploaded yet. Please upload a file with Title and Body columns.")
        else:
            with st.expander(f"File: {uploaded_file.name}", expanded=True):
                try:
                    df = read_xlsx_file(uploaded_file)
                    prepared_df = prepare_marketplace_dataframe(df)
                    st.session_state.current_dataset = prepared_df.copy()
                    st.session_state.current_dataset_name = uploaded_file.name
                    render_dataset_summary_block(prepared_df)
                    st.success("Dataset ready for Insights.")
                except pd.errors.EmptyDataError:
                    st.error("This XLSX file is empty or invalid. Please upload a valid file with headers.")
                except ValueError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(
                        "Could not read this XLSX file. Please upload a valid .xlsx file "
                        "using the provided template format. "
                        f"Technical detail: {exc}"
                    )

    next_disabled = st.session_state.current_dataset is None
    if st.button("Next -> Insights", disabled=next_disabled):
        st.session_state.active_tab = "Insights"
        st.rerun()


# -------------------------------
# Insights section
# -------------------------------
if st.session_state.active_tab == "Insights":
    st.subheader("Insights")

    if st.session_state.current_dataset is None:
        st.info("Upload or fetch data in Upload Data first, then click Next.")
    else:
        data_df = st.session_state.current_dataset.copy()
        data_name = st.session_state.current_dataset_name or "current dataset"

        st.caption(f"Source dataset: {data_name}")
        texts = get_analysis_texts(data_df)

        if not texts:
            st.warning("No usable text found for analysis. Ensure title/body have content.")
        else:
            st.markdown("### Top Recurring Keywords (Pain Signals)")
            pain_df = build_pain_point_table(texts, top_n=15)
            if pain_df.empty:
                st.info("Not enough complaint-like text to extract pain phrases yet.")
            else:
                st.dataframe(
                    pain_df[["pain_point", "count", "example_1", "example_2"]],
                    use_container_width=True,
                )

            st.markdown("### Sentiment Breakdown")
            sentiment_df = build_sentiment_table(texts)
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric(
                "% Negative",
                f"{sentiment_df.loc[sentiment_df['sentiment'] == 'Negative', 'percentage'].iloc[0]}%",
            )
            metric_col2.metric(
                "% Neutral",
                f"{sentiment_df.loc[sentiment_df['sentiment'] == 'Neutral', 'percentage'].iloc[0]}%",
            )
            metric_col3.metric(
                "% Positive",
                f"{sentiment_df.loc[sentiment_df['sentiment'] == 'Positive', 'percentage'].iloc[0]}%",
            )
            fig, ax = plt.subplots()
            ax.pie(
                sentiment_df["count"],
                labels=sentiment_df["sentiment"],
                autopct="%1.1f%%",
                startangle=90,
            )
            ax.axis("equal")
            st.pyplot(fig)

            st.markdown("### Top Pain Themes (Cluster Preview)")
            themes = build_theme_previews(texts, min_themes=3, max_themes=5)
            if not themes:
                st.info("Not enough data to build themes yet.")
            else:
                for idx, theme in enumerate(themes, start=1):
                    st.markdown(f"**Theme {idx}: {theme['label']}**")
                    st.write(f"Mentions: {theme['mentions']}")
                    quotes = theme.get("quotes", [])
                    if quotes:
                        st.write(f"Quote 1: \"{quotes[0]}\"")
                    if len(quotes) > 1:
                        st.write(f"Quote 2: \"{quotes[1]}\"")
                    st.divider()

    next_disabled = st.session_state.current_dataset is None
    if st.button("Next -> Product Concepts", disabled=next_disabled):
        st.session_state.active_tab = "Product Concepts"
        st.rerun()


# -------------------------------
# Placeholder sections
# -------------------------------
if st.session_state.active_tab == "Product Concepts":
    st.subheader("Product Concepts")
    if st.session_state.current_dataset is None:
        st.info("Upload/fetch data first, then go to Insights and Product Concepts.")
    else:
        concept_df_source = st.session_state.current_dataset.copy()
        concept_texts = get_analysis_texts(concept_df_source)
        concept_themes = build_theme_previews(concept_texts, min_themes=3, max_themes=5)
        concept_pain_df = build_pain_point_table(concept_texts, top_n=20)
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
        st.markdown("### LLM Settings")
        llm_col1, llm_col2 = st.columns(2)
        use_local_llm = True
        llm_model = llm_col1.text_input("Model ID", value="google/flan-t5-base")
        local_files_only = llm_col2.checkbox(
            "Local files only",
            value=False,
            help="When OFF, model can auto-download on first run.",
        )
        st.caption(
            "Product concepts are generated using the LLM only. First run may download model files."
        )

        if not concept_themes:
            st.info("No pain themes detected yet. Refine data quality or add more complaint-heavy data.")
        else:
            concepts_df = generate_product_concepts(
                themes=concept_themes,
                pain_df=concept_pain_df,
                data_df=concept_df_source,
                selected_capabilities=selected_capabilities,
                use_local_llm=use_local_llm,
                llm_model=llm_model,
                local_files_only=local_files_only,
                min_concepts=5,
                max_concepts=concept_count,
            )

            if concepts_df.empty:
                failures = concepts_df.attrs.get("llm_failure_reasons", {})
                failure_text = ", ".join([f"{k}: {v}" for k, v in failures.items()]) if failures else "unknown"
                st.warning(
                    "Could not generate concept briefs from current themes. "
                    f"LLM reasons: {failure_text}"
                )
            else:
                attempted = int(concepts_df.attrs.get("llm_attempted_count", 0))
                generated = int(concepts_df.attrs.get("llm_generated_count", 0))
                failures = concepts_df.attrs.get("llm_failure_reasons", {})
                if generated > 0:
                    st.success(
                        f"LLM generated {generated} concept brief(s) "
                        f"from {attempted} theme-level prompt(s)."
                    )
                if failures:
                    failure_text = ", ".join([f"{k}: {v}" for k, v in failures.items()])
                    st.info(f"LLM warnings: {failure_text}")

                st.markdown("### Full Product Concept Briefs")
                for idx, row in concepts_df.iterrows():
                    with st.expander(
                        f"{idx + 1}. {row['product_name']} ({row['format']})",
                        expanded=(idx < 3),
                    ):
                        st.write(f"**Product Name:** {row['product_name']}")
                        st.write(f"**Target Consumer Profile:** {row['target_consumer']}")
                        st.write(f"**Key Ingredients / Formulation Direction:** {row['ingredients']}")
                        st.write(f"**Suggested Price Point:** {row['price_point']}")
                        st.write(f"**Format:** {row['format']}")
                        st.write(f"**Competitive Positioning:** {row['positioning']}")
