"""
Task 2: Temporal & Entity Analysis
===================================
Assignment 1 — Disaster Study Module

Sub-tasks:
  2.1  Response Delta Calculation
  2.2  Entity Recognition (NER) — spaCy + curated keyword regex
  2.3  Sentiment Volatility       — VADER

Input : scraping/gdacs_earthquake_data.csv   (30-column dataset)
Output: task2/task2_results.json             (machine-readable)
        console output                        (human-readable report)
"""

import os
import re
import json
import math
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
CSV_PATH  = BASE_DIR / "task1" / "gdacs_earthquake_data.csv"
OUT_DIR   = Path(__file__).resolve().parent
OUT_JSON  = OUT_DIR / "task2_results.json"

# ── NLP Models ───────────────────────────────────────────────────────────────
print("Loading NLP models...")
nlp = spacy.load("en_core_web_sm")
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()
print("Models loaded.\n")

# ── Curated Entity / Pattern Dictionaries ────────────────────────────────────

# Humanitarian actors used for curated keyword matching (supplements spaCy's generic NER)
KNOWN_NGOS = {
    "Red Cross", "ICRC", "IFRC", "UNICEF", "UN", "United Nations",
    "WFP", "World Food Programme", "WHO", "World Health Organization",
    "OCHA", "UNDP", "UNFPA", "IOM", "IRC",
    "Médecins Sans Frontières", "MSF", "Doctors Without Borders",
    "Save the Children", "Oxfam", "CARE", "Islamic Relief",
    "Catholic Relief Services", "CRS", "World Vision",
    "Mercy Corps", "Direct Relief", "Team Rubicon",
    "International Rescue Committee",
}

KNOWN_AGENCIES = {
    "USAID", "DSWD", "DROMIC", "NDMO", "NDRF", "NDRRMC",
    "FEMA", "CBCP", "WB", "World Bank", "Asian Development Bank", "ADB",
    "European Commission", "EU", "ECHO",
    "Coast Guard", "Army", "Navy", "Air Force",
}

# Regex patterns for extracting quantitative signals from raw headline text
RE_DEATH = re.compile(
    r"(?:kill(?:ed|s)|dead|deaths?|fatalities|casualties|toll)"
    r"(?:\s+(?:rises?|climbs?|reaches?|stands?))?"
    r"(?:\s+to)?\s+(?:at\s+least\s+)?(\d[\d,]*)",
    re.IGNORECASE,
)
RE_MONEY = re.compile(
    r"\$\s?(\d[\d,\.]*)\s?(billion|million|m\b|bn\b|thousand|k\b)?",
    re.IGNORECASE,
)
RE_INJURED = re.compile(
    r"(?:injur(?:ed|ies)|wounded|hurt)\s+(?:at\s+least\s+)?(\d[\d,]*)",
    re.IGNORECASE,
)

# ── Utility ────────────────────────────────────────────────────────────────

def parse_peak_news_date(peak_str: str, event_dt: datetime) -> datetime | None:
    """
    Convert 'DD/MM' into a full datetime using the event year as base.
    Handles year rollover: if peak month < event month (e.g. event Dec, peak Jan),
    increment year by 1.
    """
    if not peak_str or peak_str == "N/A":
        return None
    try:
        day, month = map(int, peak_str.split("/"))
        year = event_dt.year
        # If peak month is earlier in the year than event month,
        # peak likely falls in the next calendar year.
        if month < event_dt.month:
            year += 1
        return datetime(year, month, day)
    except (ValueError, AttributeError):
        return None


def humanise_delta(days: float) -> str:
    d = int(round(days))
    if d == 0:
        return "same day"
    elif d == 1:
        return "1 day"
    elif d < 7:
        return f"{d} days"
    elif d < 14:
        return f"~1 week ({d} days)"
    else:
        weeks = d // 7
        remainder = d % 7
        return f"~{weeks} week{'s' if weeks > 1 else ''} ({d} days)"


def to_normalised_amount_usd(value: float, unit: str) -> float | None:
    unit = (unit or "").lower().strip()
    if unit in ("billion", "bn"):
        return value * 1_000_000_000
    if unit in ("million", "m"):
        return value * 1_000_000
    if unit in ("thousand", "k"):
        return value * 1_000
    return value  # assume raw dollars if no unit


def extract_monetary_mentions(text: str) -> list[dict]:
    results = []
    for m in RE_MONEY.finditer(text):
        raw_val = float(m.group(1).replace(",", ""))
        unit    = m.group(2) or ""
        norm    = to_normalised_amount_usd(raw_val, unit)
        results.append({
            "raw_text": m.group(0).strip(),
            "value_usd": norm,
        })
    return results


def extract_death_toll(text: str) -> list[int]:
    return [int(m.group(1).replace(",", "")) for m in RE_DEATH.finditer(text)]


def extract_injury_count(text: str) -> list[int]:
    return [int(m.group(1).replace(",", "")) for m in RE_INJURED.finditer(text)]


def split_headlines(headline_str: str) -> list[str]:
    """Split the pipe-delimited headline string into individual items."""
    if not headline_str or headline_str == "N/A":
        return []
    parts = [h.strip() for h in headline_str.split("|")]
    return [p for p in parts if len(p) > 10]  # filter artefacts


def vader_classify(compound: float) -> str:
    """Map VADER compound score to a descriptive label."""
    if compound >= 0.5:
        return "strongly positive (relief/recovery narrative)"
    if compound >= 0.1:
        return "mildly positive"
    if compound >= -0.1:
        return "neutral / factual"
    if compound >= -0.5:
        return "mildly negative (concern/alarm)"
    return "strongly negative (alarmist/crisis)"


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("TASK 2: TEMPORAL & ENTITY ANALYSIS")
print("=" * 65)

df = pd.read_csv(CSV_PATH, dtype=str)

# Normalise scraper-inserted 'N/A' strings to proper NaN so pandas operations work correctly
df.replace("N/A", np.nan, inplace=True)

print(f"Loaded {len(df)} events from {CSV_PATH.name}\n")

results = {}  # dict to accumulate all sub-task outputs; serialised to JSON at the end

# ═══════════════════════════════════════════════════════════════════════════════
# SUB-TASK 2.1 — RESPONSE DELTA CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

print("─" * 65)
print("2.1  RESPONSE DELTA CALCULATION")
print("     Delta_T = T_MediaPeak − T_SystemAlert")
print("─" * 65)

delta_records = []

for _, row in df.iterrows():
    label = row["label"]

    # Parse system alert datetime
    try:
        alert_dt = datetime.strptime(
            str(row["event_date_utc"]).strip(), "%d %b %Y %H:%M UTC"
        )
    except (ValueError, TypeError):
        alert_dt = None

    peak_dt = parse_peak_news_date(
        str(row.get("peak_news_day", "")).strip(), alert_dt
    ) if alert_dt else None

    if alert_dt and peak_dt:
        delta_days = (peak_dt - alert_dt).days
        delta_label = humanise_delta(delta_days)
    else:
        delta_days  = None
        delta_label = "unavailable"

    rec = {
        "event":           label,
        "country":         row["country"],
        "period":          row["period"],
        "alert_date":      str(row["event_date_utc"]),
        "peak_news_day":   str(row.get("peak_news_day", "N/A")),
        "delta_days":      delta_days,
        "delta_human":     delta_label,
        "total_articles":  row.get("total_articles", "N/A"),
        "peak_articles":   row.get("peak_news_count", "N/A"),
    }
    delta_records.append(rec)

    print(f"\n  Event  : {label}")
    print(f"  Alert  : {rec['alert_date']}")
    print(f"  Peak   : {rec['peak_news_day']} (day with most articles)")
    print(f"  ΔT     : {rec['delta_human']}")
    print(f"  Volume : {rec['peak_articles']} articles on peak day / {rec['total_articles']} total")

# --- Comparative insight ---
valid = [r for r in delta_records if r["delta_days"] is not None]
if len(valid) >= 2:
    ph = next((r for r in valid if r["country"] == "Philippines" and r["period"] == "Historical"), None)
    pr = next((r for r in valid if r["country"] == "Philippines" and r["period"] == "Recent"), None)
    ah = next((r for r in valid if r["country"] == "Afghanistan" and r["period"] == "Historical"), None)
    ar = next((r for r in valid if r["country"] == "Afghanistan" and r["period"] == "Recent"), None)

    print("\n  ── Comparative Insight ────────────────────────────────────")
    if ph and pr:
        faster = "faster" if pr["delta_days"] < ph["delta_days"] else "slower"
        diff   = abs((pr["delta_days"] or 0) - (ph["delta_days"] or 0))
        print(f"  Philippines: Recent event hit media peak {diff} day(s) {faster} than Historical.")
        print(f"    Historical ΔT = {ph['delta_human']}  |  Recent ΔT = {pr['delta_human']}")
    if ah and ar:
        faster = "faster" if ar["delta_days"] < ah["delta_days"] else "slower"
        diff   = abs((ar["delta_days"] or 0) - (ah["delta_days"] or 0))
        print(f"  Afghanistan: Recent event hit media peak {diff} day(s) {faster} than Historical.")
        print(f"    Historical ΔT = {ah['delta_human']}  |  Recent ΔT = {ar['delta_human']}")

results["2.1_response_delta"] = delta_records

# ═══════════════════════════════════════════════════════════════════════════════
# SUB-TASK 2.2 — ENTITY RECOGNITION (NER)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 65)
print("2.2  ENTITY RECOGNITION (NER)")
print("     Using spaCy en_core_web_sm + curated keyword matching")
print("─" * 65)

ner_records = []

for _, row in df.iterrows():
    label     = row["label"]
    headlines = split_headlines(str(row.get("news_headlines", "")))
    full_text = " ".join(headlines)

    if not full_text.strip():
        print(f"\n  {label}: no headlines available.")
        continue

    # Run spaCy NER on the concatenated headline text (cap at 100k chars to stay within token limits)
    doc = nlp(full_text[:100_000])

    # Separate entity counters per label
    spacy_orgs   = Counter()
    spacy_gpes   = Counter()
    spacy_people = Counter()
    spacy_money  = Counter()

    for ent in doc.ents:
        t = ent.text.strip()
        if ent.label_ == "ORG":
            spacy_orgs[t]   += 1
        elif ent.label_ == "GPE":
            spacy_gpes[t]   += 1
        elif ent.label_ == "PERSON":
            spacy_people[t] += 1
        elif ent.label_ == "MONEY":
            spacy_money[t]  += 1

    # Overlay curated dictionary matching on top of spaCy — catches well-known acronyms spaCy misses
    found_ngos    = {}
    found_agencies = {}
    lower_text    = full_text.lower()

    for ngo in KNOWN_NGOS:
        count = len(re.findall(r"\b" + re.escape(ngo.lower()) + r"\b", lower_text))
        if count > 0:
            found_ngos[ngo] = count

    for agency in KNOWN_AGENCIES:
        count = len(re.findall(r"\b" + re.escape(agency.lower()) + r"\b", lower_text))
        if count > 0:
            found_agencies[agency] = count

    # Extract numeric signals: death counts, injuries, monetary figures via regex
    death_counts   = extract_death_toll(full_text)
    injury_counts  = extract_injury_count(full_text)
    money_mentions = extract_monetary_mentions(full_text)

    # Collapse all monetary mentions to a single max figure (most newsworthy relief pledge)
    fund_values = [m["value_usd"] for m in money_mentions if m["value_usd"] is not None]
    max_fund    = max(fund_values) if fund_values else None

    # Build record
    rec = {
        "event": label,
        "ngos_found":    dict(sorted(found_ngos.items(),    key=lambda x: -x[1])),
        "agencies_found":dict(sorted(found_agencies.items(),key=lambda x: -x[1])),
        "govts_mentioned": dict(spacy_gpes.most_common(10)),
        "orgs_spacy":    dict(spacy_orgs.most_common(10)),
        "persons":       dict(spacy_people.most_common(5)),
        "death_toll_mentions": sorted(set(death_counts)),
        "injury_mentions":     sorted(set(injury_counts)),
        "max_stated_death_toll": max(death_counts) if death_counts else None,
        "max_stated_injuries":   max(injury_counts) if injury_counts else None,
        "fund_mentions":         [m["raw_text"] for m in money_mentions],
        "max_fund_usd":          max_fund,
        "headline_count": len(headlines),
    }
    ner_records.append(rec)

    # --- Pretty print ---
    print(f"\n  ── {label} ({len(headlines)} headlines) ──────────────────────────")

    if found_ngos:
        top_ngos = sorted(found_ngos.items(), key=lambda x: -x[1])[:5]
        print(f"  NGOs         : {', '.join(f'{k}({v})' for k, v in top_ngos)}")
    else:
        print("  NGOs         : none identified")

    if found_agencies:
        top_ag = sorted(found_agencies.items(), key=lambda x: -x[1])[:5]
        print(f"  Agencies     : {', '.join(f'{k}({v})' for k, v in top_ag)}")

    top_gpe = [k for k, _ in spacy_gpes.most_common(5)]
    if top_gpe:
        print(f"  Geographies  : {', '.join(top_gpe)}")

    top_org = [k for k, _ in spacy_orgs.most_common(5)]
    if top_org:
        print(f"  Orgs (spaCy) : {', '.join(top_org)}")

    if death_counts:
        print(f"  Death tolls  : {sorted(set(death_counts))}  (max stated: {max(death_counts):,})")
    else:
        print("  Death tolls  : no explicit figures found in headlines")

    if injury_counts:
        print(f"  Injuries     : {sorted(set(injury_counts))}  (max stated: {max(injury_counts):,})")

    if money_mentions:
        print(f"  Funds/Amounts: {', '.join(m['raw_text'] for m in money_mentions[:6])}")
        if max_fund:
            print(f"  Largest fund : ${max_fund/1_000_000:.1f}M USD")
    else:
        print("  Funds/Amounts: none identified in headlines")

results["2.2_entity_recognition"] = ner_records

# ═══════════════════════════════════════════════════════════════════════════════
# SUB-TASK 2.3 — SENTIMENT VOLATILITY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 65)
print("2.3  SENTIMENT VOLATILITY")
print("     Scoring each headline with VADER → compound ∈ [−1, +1]")
print("─" * 65)

sentiment_records = []

for _, row in df.iterrows():
    label     = row["label"]
    headlines = split_headlines(str(row.get("news_headlines", "")))

    if not headlines:
        print(f"\n  {label}: no headlines, skipping.")
        continue

    # Score each headline independently with VADER
    scores = [sia.polarity_scores(h)["compound"] for h in headlines]

    mean_score = float(np.mean(scores))
    std_score  = float(np.std(scores))   # higher std = more volatile / polarised coverage
    min_score  = float(np.min(scores))
    max_score  = float(np.max(scores))
    n_negative = sum(1 for s in scores if s < -0.1)
    n_positive = sum(1 for s in scores if s > 0.1)
    n_neutral  = len(scores) - n_negative - n_positive

    # Identify the single most extreme headlines for qualitative illustration
    sorted_by_score = sorted(zip(scores, headlines), key=lambda x: x[0])
    most_negative   = sorted_by_score[0]
    most_positive   = sorted_by_score[-1]

    tone_label = vader_classify(mean_score)

    # Alarmism ratio: share of headlines crossing the "strongly negative" threshold (compound < -0.5)
    strong_neg = sum(1 for s in scores if s < -0.5)
    alarmism_ratio = strong_neg / len(scores)

    rec = {
        "event":           label,
        "country":         row["country"],
        "period":          row["period"],
        "n_headlines":     len(headlines),
        "mean_compound":   round(mean_score, 4),
        "std_compound":    round(std_score,  4),
        "min_compound":    round(min_score,  4),
        "max_compound":    round(max_score,  4),
        "n_negative":      n_negative,
        "n_neutral":       n_neutral,
        "n_positive":      n_positive,
        "alarmism_ratio":  round(alarmism_ratio, 4),
        "tone_label":      tone_label,
        "most_negative_headline": most_negative[1],
        "most_positive_headline": most_positive[1],
    }
    sentiment_records.append(rec)

    print(f"\n  ── {label} ──────────────────────────────────────────────────")
    print(f"  Headlines    : {len(headlines)}")
    print(f"  Mean score   : {mean_score:+.4f}  ({tone_label})")
    print(f"  Std dev      : {std_score:.4f}  (higher = more volatile / mixed coverage)")
    print(f"  Range        : [{min_score:+.4f}, {max_score:+.4f}]")
    print(f"  Breakdown    : {n_negative} negative | {n_neutral} neutral | {n_positive} positive")
    print(f"  Alarmism %   : {alarmism_ratio*100:.1f}%  (headlines with compound < −0.5)")
    print(f"  Most negative: \"{most_negative[1][:90]}...\"" if len(most_negative[1]) > 90 else f"  Most negative: \"{most_negative[1]}\"")
    print(f"  Most positive: \"{most_positive[1][:90]}...\"" if len(most_positive[1]) > 90 else f"  Most positive: \"{most_positive[1]}\"")

results["2.3_sentiment"] = sentiment_records

# --- Cross-event comparisons ---
print("\n  ── Cross-Event Sentiment Comparison ──────────────────────────")

countries = df["country"].unique()
for country in countries:
    country_recs = [r for r in sentiment_records if r["country"] == country]
    if len(country_recs) < 2:
        continue
    hist = next((r for r in country_recs if r["period"] == "Historical"), None)
    rec  = next((r for r in country_recs if r["period"] == "Recent"),    None)
    if not hist or not rec:
        continue

    delta_mean = rec["mean_compound"] - hist["mean_compound"]
    delta_alarm = rec["alarmism_ratio"] - hist["alarmism_ratio"]
    delta_std   = rec["std_compound"] - hist["std_compound"]

    print(f"\n  {country}:")
    print(f"    Historical tone : {hist['mean_compound']:+.4f}  ({hist['tone_label']})")
    print(f"    Recent tone     : {rec['mean_compound']:+.4f}  ({rec['tone_label']})")
    print(f"    Δ tone          : {delta_mean:+.4f}  ({'more positive' if delta_mean > 0 else 'more negative'} in recent coverage)")
    print(f"    Δ alarmism      : {delta_alarm*100:+.1f}pp  ({'higher' if delta_alarm > 0 else 'lower'} % of strongly negative headlines)")
    print(f"    Δ volatility    : {delta_std:+.4f}  ({'more volatile' if delta_std > 0 else 'more consistent'} tone in recent)")

    if abs(delta_mean) < 0.05:
        obs = "Reporting tone has remained largely stable between the two events."
    elif delta_mean > 0:
        obs = "Coverage has shifted towards a more recovery/relief narrative in the recent event."
    else:
        obs = "Recent coverage is markedly more negative — potentially reflecting greater severity or desperation."

    if delta_alarm > 0.05:
        obs += " Alarmism has increased, suggesting more crisis-driven reporting."
    elif delta_alarm < -0.05:
        obs += " Alarmism has decreased — reporting may be more measured/analytical today."

    print(f"    Observation : {obs}")

    # Store comparison in results
    for r in results["2.3_sentiment"]:
        if r["country"] == country:
            if r["period"] == "Recent":
                r["delta_vs_historical_mean"] = round(delta_mean, 4)
                r["delta_vs_historical_alarmism"] = round(delta_alarm, 4)

# ═══════════════════════════════════════════════════════════════════════════════
# OVERALL SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("SUMMARY TABLE")
print("=" * 65)

header = f"{'Event':<30} {'ΔT':>7} {'Articles':>9} {'Sentiment':>10} {'Alarmism':>9}"
print(header)
print("-" * 65)

for dr in delta_records:
    # find matching sentiment
    sr = next((s for s in sentiment_records if s["event"] == dr["event"]), {})
    print(
        f"{dr['event']:<30} "
        f"{(str(dr['delta_days'])+' d') if dr['delta_days'] is not None else 'N/A':>7} "
        f"{str(dr['total_articles']):>9} "
        f"{sr.get('mean_compound', float('nan')):>+10.4f} "
        f"{sr.get('alarmism_ratio', float('nan'))*100:>8.1f}%"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 65)

# Custom JSON serialiser — pandas/numpy types aren't natively JSON-serialisable
def json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serialisable: {type(obj)}")

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=json_safe, ensure_ascii=False)

print(f"Results saved → {OUT_JSON}")
print("Done.")
