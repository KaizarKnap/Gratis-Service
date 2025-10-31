# streamlit_app.py
# -----------------
# Gratis weggegeven orders — Analyzer (verbeterde versie)
# - Hybride detectie: Regels + (optioneel) TF-IDF + LogisticRegression
# - (Optioneel) NL lemmatisering met spaCy voor robuustere matching
# - Uitleg per rij: waarom gedetecteerd ("reason"), model-score
# - Performance: caching, veilig voor Streamlit Cloud
# - Export: Excel met tabs per "Dienst facturatie" (indien aanwezig)

import io
import re
import datetime as dt
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

import spacy
from spacy.cli import download as spacy_download

# scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    spacy.load("nl_core_news_sm")
except OSError:
    spacy_download("nl_core_news_sm")
    
# =============== Page & header ===============
st.set_page_config(page_title="Gratis weggegeven orders - Analyzer", layout="wide")
st.title("🧾 Gratis weggegeven orders — Analyzer")
st.markdown(
    """
Deze app vindt en analyseert orders die **gratis** of **niet te factureren** zijn
(op basis van *notitie facturatie*), met filters op **periode** en **orderstatus**,
en export naar Excel met **tabs per Dienst facturatie**.

**Nieuw in deze versie**  
- Hybride detectie (regels **+** optioneel ML-model)  
- Optionele **lemmatisering** (spaCy) voor betere matching van vervoegingen  
- **Uitleg per rij** waarom iets als gratis is aangemerkt  
- Sneller en stabieler voor Streamlit Cloud
"""
)

# =============== Helper: spaCy (optioneel) ===============
@st.cache_resource(show_spinner=False)
def _load_spacy():
    """Probeer NL spaCy model te laden; zo niet, gebruik None."""
    try:
        import spacy
        return spacy.load("nl_core_news_sm")
    except Exception:
        return None

nlp = _load_spacy()
if nlp is None:
    st.info("ℹ️ spaCy NL-model niet gevonden. De app werkt, maar zonder lemmatisering. "
            "Installeer evt.: `python -m spacy download nl_core_news_sm`")

# =============== Regex patronen (uitbreidbaar) ===============
INCLUDE_PATTERNS_DEFAULT = [
    r"\bgratis\b(?!\s*(verzend|bezorg|transport|rijkosten|koerier|levering|leverings|aflever|zending|shipping))",
    r"\bkosteloos\b(?!\s*(verzend|bezorg|transport|rijkosten))",
    r"\bkostenloos\b(?!\s*(verzend|bezorg|transport|rijkosten))",
    r"\bzonder\s*kosten\b(?!\s*(verzend|bezorg|transport|rijkosten))",
    r"\b0\s*(euro|€)\b",
    r"\bpro\s*bono\b",
    r"\bfree\s*(of\s*charge)?\b(?!\s*(shipping))",
    r"\bkostenvrij\b(?!\s*(verzend|bezorg|transport|rijkosten))",
    r"\bvergoeding\s*=\s*0\b",
    r"\bniet\s*factureren\b",
    r"\bniet\s*in\s*rekening\s*brengen\b",
    r"\bgeheel\s*gratis\b|\bvolledig\s*gratis\b",
]
EXCLUDE_PATTERNS_DEFAULT = [
    r"\bgratis\s*(verzend|bezorg)kosten\b",
    r"\bgratis\s*verzending\b",
    r"\bfree\s*shipping\b",
    r"\btransport\s*(gratis|kosteloos|kostenloos)\b",
    r"\brijkosten\s*(gratis|kosteloos|kostenloos)\b",
    r"\bbezorgkosten\s*(gratis|kosteloos|kostenloos)\b",
]

@st.cache_data(show_spinner=False)
def _compile_patterns(include: List[str], exclude: List[str]) -> Tuple[List[re.Pattern], List[re.Pattern]]:
    inc = [re.compile(p, flags=re.IGNORECASE) for p in include]
    exc = [re.compile(p, flags=re.IGNORECASE) for p in exclude]
    return inc, exc

def guess_column(possible_names, columns):
    for name in possible_names:
        for c in columns:
            if c.lower() == name.lower():
                return c
    for name in possible_names:
        for c in columns:
            if name.lower() in c.lower():
                return c
    return None

def normalize_text_simple(s: str) -> str:
    if pd.isna(s):
        return ""
    return str(s).strip().lower()

def normalize_text_lemma(s: str) -> str:
    if pd.isna(s):
        return ""
    if nlp is None:
        return str(s).strip().lower()
    doc = nlp(str(s).lower())
    return " ".join([t.lemma_ for t in doc if not t.is_space])

# =============== Instellingen-sectie ===============
with st.container():
    st.markdown("### ⚙️ Instellingen")
    c1, c2 = st.columns([2, 2])
    with c1:
        uploaded = st.file_uploader("📂 Upload Excel-bestand", type=["xlsx", "xls"])
    with c2:
        status_filter = st.multiselect(
            "Orderstatus (alleen deze worden geanalyseerd)",
            options=["Confirmed", "Edited", "Cancelled", "WaitingOnConfirmation", "Draft", "Completed"],
            default=["Confirmed", "Edited", "Completed", "WaitingOnConfirmation", "Draft"],
        )

    st.divider()

    with st.expander("🔍 Detectie-instellingen (geavanceerd)", expanded=False):
        st.caption(
            "Voeg extra **inclusie**- en **exclusie**-regex toe en kies of je het ML-model wil gebruiken. "
            "Het eindresultaat combineert regels + model (gewichten instelbaar)."
        )
        colA, colB = st.columns(2)
        with colA:
            include_add = st.text_area("Extra inclusie-termen (regex, één per regel)", height=80, value="")
        with colB:
            exclude_add = st.text_area("Extra exclusie-termen (regex, één per regel)", height=80, value="")

        use_model = st.toggle("📘 Model gebruiken (TF-IDF + LogReg)", value=False)
        colC, colD = st.columns(2)
        with colC:
            use_lemmatize = st.toggle("🔤 Lemmatisering (spaCy) inschakelen", value=(nlp is not None))
        with colD:
            alpha_rules = st.slider("Weging regels (vs. model)", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                                    help="0.0 = alleen model, 1.0 = alleen regels")

        thresh = st.slider("Detectiedrempel (eindscore)", min_value=0.10, max_value=0.90, value=0.50, step=0.05)

# =============== Data inlezen ===============
if uploaded is None:
    st.info("⬆️ Upload eerst een Excel-bestand om te starten.")
    st.stop()

def smart_read_excel(file):
    """Detecteert automatisch header-rij; valt veilig terug als detectie faalt."""
    try:
        preview = pd.read_excel(file, header=None, nrows=20, engine="openpyxl")
        best_row, best_score = 0, -1
        for i, row in preview.iterrows():
            values = [str(x).strip() for x in row if pd.notna(x)]
            # Schatting voor 'header-achtig'
            text_ratio = sum(x.replace(" ", "").isalpha() for x in values) / max(len(values), 1)
            unique_ratio = len(set(values)) / max(len(values), 1)
            score = len(values) + text_ratio * 5 + unique_ratio * 5
            if score > best_score:
                best_score = score
                best_row = i
        df_ = pd.read_excel(file, header=best_row, engine="openpyxl")
        st.success(f"✅ Kolomkoppen automatisch gevonden op rij {best_row + 1}")
        st.caption(f"Gevonden kolommen: {', '.join(map(str, df_.columns[:8]))}{' ...' if len(df_.columns) > 8 else ''}")
        return df_
    except Exception as e:
        st.warning(f"Header-detectie mislukt ({e}). Ik lees met standaard header (eerste rij).")
        df_ = pd.read_excel(file, engine="openpyxl")
        return df_

df = smart_read_excel(uploaded)
orig_columns = df.columns.tolist()

# Kolommen raden + fallback in sidebar
col_orderstatus = guess_column(["orderstatus", "status"], df.columns) or st.sidebar.selectbox("Kies kolom: Orderstatus", df.columns)
col_date = guess_column(["uitvoerdatum", "orderdatum", "datum"], df.columns) or st.sidebar.selectbox("Kies kolom: Datum", df.columns)
col_note = guess_column(["notitie facturatie", "facturatie notitie", "notitie", "opmerking"], df.columns) or st.sidebar.selectbox("Kies kolom: Notitie facturatie", df.columns)
col_dienst = guess_column(["dienst facturatie", "dienst", "facturatie dienst"], df.columns) or st.sidebar.selectbox("Kies kolom: Dienst facturatie", df.columns)
col_created_by = guess_column(["aangemaakt door", "aangemaakt_door", "auteur", "gebruiker"], df.columns) or st.sidebar.selectbox("Kies kolom: Aangemaakt door", df.columns)
col_location = guess_column(["locatienummer", "locatie", "plaats"], df.columns) or st.sidebar.selectbox("Kies kolom: Locatie", df.columns)
col_amount = guess_column(["verkooptarief", "totaal", "bedrag", "prijs"], df.columns) or st.sidebar.selectbox("Kies kolom: Verkooptarief (totaal)", df.columns)

# Datumkolom check
if col_date not in df.columns:
    st.error("Ik kan de datumkolom niet vinden. Selecteer of hernoem de juiste kolom.")
    st.stop()

# Datum naar datetime
df[col_date] = pd.to_datetime(df[col_date], errors="coerce", dayfirst=True)

# Filter op status
if col_orderstatus in df.columns:
    df = df[df[col_orderstatus].astype(str).isin(status_filter)]
else:
    st.warning("Orderstatus-kolom niet gevonden; alle regels worden meegenomen.")

# =============== Periode-selectie ===============
st.subheader("📅 Periode selectie")

df["Datum_dt"] = pd.to_datetime(df[col_date], errors="coerce", dayfirst=True)
df["Datum_nl"] = df["Datum_dt"].dt.strftime("%d-%m-%Y")
df["Datum_kort"] = df["Datum_dt"].dt.strftime("%a %d %b %Y")

min_date = df["Datum_dt"].min().date() if pd.notna(df["Datum_dt"].min()) else dt.date(2000, 1, 1)
max_date = df["Datum_dt"].max().date() if pd.notna(df["Datum_dt"].max()) else dt.date.today()

today = dt.date.today()
default_start = dt.date(2025, 1, 1)
if default_start < min_date: default_start = min_date
default_end = min(max_date, today)

c1, c2 = st.columns(2)
with c1:
    start_date = st.date_input("Begindatum", value=default_start, min_value=min_date, max_value=max_date, format="DD-MM-YYYY")
with c2:
    end_date = st.date_input("Einddatum", value=default_end, min_value=min_date, max_value=max_date, format="DD-MM-YYYY")

if start_date > end_date:
    st.warning("De begindatum ligt na de einddatum – pas dit aan.")
    st.stop()

st.caption(f"Geselecteerde periode: {start_date.strftime('%d-%m-%Y')} t/m {end_date.strftime('%d-%m-%Y')}")
mask_period = (df["Datum_dt"] >= pd.to_datetime(start_date)) & (df["Datum_dt"] <= pd.to_datetime(end_date))
df = df[mask_period].copy()

# =============== Detectie: Regels + (optioneel) Model ===============
# Patronen samenstellen met gebruiker-input
INCLUDE_USER = [p.strip() for p in include_add.strip().splitlines() if p.strip()]
EXCLUDE_USER = [p.strip() for p in exclude_add.strip().splitlines() if p.strip()]
INCLUDE_ALL = INCLUDE_PATTERNS_DEFAULT + INCLUDE_USER
EXCLUDE_ALL = EXCLUDE_PATTERNS_DEFAULT + EXCLUDE_USER
INC_RX, EXC_RX = _compile_patterns(INCLUDE_ALL, EXCLUDE_ALL)

def rule_match(note: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Geeft terug: (rule_is_free, include_hit, exclude_hit)"""
    text = str(note or "")
    inc_hit = None
    exc_hit = None
    for p in INC_RX:
        if p.search(text):
            inc_hit = p.pattern
            break
    for p in EXC_RX:
        if p.search(text):
            exc_hit = p.pattern
            break
    return (inc_hit is not None and exc_hit is None), inc_hit, exc_hit

# === Labeling voor model (in-memory) ===
LABEL_KEY = "labels_memory_v2"
if LABEL_KEY not in st.session_state:
    st.session_state[LABEL_KEY] = {}  # {row_index: 0/1}

with st.expander("✍️ Optioneel: Voorbeelden labelen voor het model"):
    st.caption("Label enkele notities als **Gratis weggegeven** (ja/nee). Het model leert hiervan.")
    if col_note in df.columns and len(df) > 0:
        sample = df[[col_note]].dropna().sample(min(20, len(df)), random_state=42)
        for idx, row in sample.iterrows():
            note_text = str(row[col_note])[:400]
            c1, c2, c3 = st.columns([6, 1, 1])
            with c1:
                st.text_area(f"Notitie (rij {idx})", value=note_text, key=f"note_{idx}", height=80, disabled=True)
            with c2:
                if st.button("✅ Ja", key=f"yes_{idx}"):
                    st.session_state[LABEL_KEY][idx] = 1
            with c3:
                if st.button("❌ Nee", key=f"no_{idx}"):
                    st.session_state[LABEL_KEY][idx] = 0
        st.write(
            f"Gelabeld: {sum(v==1 for v in st.session_state[LABEL_KEY].values())} gratis / "
            f"{sum(v==0 for v in st.session_state[LABEL_KEY].values())} niet-gratis"
        )
    else:
        st.info("Geen notitie-kolom of geen data beschikbaar voor labeling.")

# === Model trainen (optioneel) ===
model_pipe: Optional[Pipeline] = None
if use_model and col_note in df.columns:
    try:
        labeled_idx = list(st.session_state[LABEL_KEY].keys())
        if len(labeled_idx) >= 4:  # minimum voor zinvolle fit
            y = np.array([st.session_state[LABEL_KEY][i] for i in labeled_idx])
            texts = df.loc[labeled_idx, col_note].fillna("").astype(str).tolist()
            # Normalisatie t.b.v. robuustheid (optioneel lemmatize)
            if use_lemmatize:
                texts = [normalize_text_lemma(t) for t in texts]
            else:
                texts = [normalize_text_simple(t) for t in texts]
            model_pipe = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                ("clf", LogisticRegression(max_iter=1000))
            ])
            model_pipe.fit(texts, y)
        else:
            st.info("Voor het ML-model zijn ≥4 gelabelde voorbeelden nodig.")
    except Exception as e:
        st.warning(f"Model trainen mislukt: {e}")
        model_pipe = None

# === Einddetectie ===
if col_note not in df.columns:
    st.error("Kolom met notities (bijv. 'Notitie facturatie') ontbreekt. Kies/controleer deze in de instellingen.")
    st.stop()

notes_series = df[col_note].fillna("").astype(str)

# Rules
rule_flags = []
rule_inc_hits = []
rule_exc_hits = []
for txt in notes_series:
    flag, inc_hit, exc_hit = rule_match(txt)
    rule_flags.append(flag)
    rule_inc_hits.append(inc_hit)
    rule_exc_hits.append(exc_hit)
rule_flags = pd.Series(rule_flags, index=df.index).astype(bool)

# Model-scores
model_scores = pd.Series(0.0, index=df.index)
if model_pipe is not None:
    try:
        if use_lemmatize:
            norm_notes = notes_series.map(normalize_text_lemma)
        else:
            norm_notes = notes_series.map(normalize_text_simple)
        model_scores = pd.Series(model_pipe.predict_proba(norm_notes.tolist())[:, 1], index=df.index)
    except Exception as e:
        st.warning(f"Voorspellen met model mislukt: {e}")

# Combineer
rule_scores = rule_flags.astype(float)
final_scores = alpha_rules * rule_scores + (1 - alpha_rules) * model_scores
is_free = final_scores >= thresh

# Bewaar uitleg/reden
df["__rule__"] = rule_flags
df["__rule_inc__"] = rule_inc_hits
df["__rule_exc__"] = rule_exc_hits
df["__model_score__"] = model_scores.round(3)
df["__final_score__"] = final_scores.round(3)
df["__is_free__"] = is_free

# =============== KPI's & Overzichten ===============
free_df = df[df["__is_free__"]].copy()

st.subheader("📈 Overzicht")
cA, cB, cC, cD = st.columns(4)
total_free = 0.0
if col_amount in free_df.columns and len(free_df) > 0:
    total_free = (
        free_df[col_amount]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
        .replace("", "0")
        .astype(float)
        .sum()
    )
with cA:
    st.metric("Aantal gratis orders (regels)", len(free_df))
with cB:
    st.metric("Periode", f"{start_date.strftime('%d-%m-%Y')} → {end_date.strftime('%d-%m-%Y')}")
with cC:
    st.metric("Unieke locaties", free_df[col_location].nunique() if col_location in free_df.columns else 0)
with cD:
    st.metric("Totaal weggegeven (€)", f"{total_free:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

# Grafiek per dag
st.subheader("📅 Weggegeven per dag (aantal)")
if len(free_df) == 0:
    st.info("Geen gratis orders gevonden in de geselecteerde periode/filters.")
else:
    daily = (
        free_df.set_index(col_date)
        .sort_index()
        .groupby(pd.Grouper(freq="D"))
        .size()
        .rename("aantal")
        .to_frame()
    )
    st.line_chart(daily)

# Wie heeft weggegeven
st.subheader("👤 Wie heeft orders weggegeven (aantal)?")
if col_created_by in free_df.columns:
    who = free_df.groupby(col_created_by).size().sort_values(ascending=False).rename("aantal").to_frame()
    st.dataframe(who)
else:
    st.info("Kolom 'Aangemaakt door' niet gevonden. Kies/controleer deze in de instellingen.")

# Locaties
st.subheader("📍 Locaties met gratis weggegeven orders (totaal)")
if col_location in free_df.columns:
    loc = free_df.groupby(col_location).size().sort_values(ascending=False).rename("aantal").to_frame()
    st.dataframe(loc)
else:
    st.info("Kolom 'Locatie' niet gevonden. Kies/controleer deze in de instellingen.")

# =============== Detail & Uitleg ===============
st.subheader("📑 Detailtabel (inclusief uitleg)")
st.caption("Gefilterde regels die als **gratis** zijn gedetecteerd. Kolommen `__*__` geven de uitleg en scores.")
show_cols = orig_columns  # originele volgorde
explain_cols = ["__is_free__", "__final_score__", "__rule__", "__rule_inc__", "__rule_exc__", "__model_score__"]
preview_cols = [c for c in show_cols] + [c for c in explain_cols if c not in show_cols]
preview = free_df.reindex(columns=preview_cols, fill_value=np.nan).head(1000)
st.dataframe(preview)

# =============== Export ===============
st.divider()
st.subheader("⬇️ Exporteren naar Excel (tabs per Dienst facturatie)")

def make_excel_bytes(df_in: pd.DataFrame, dienst_col: Optional[str], sheet_max_name: int = 31) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        if dienst_col and dienst_col in df_in.columns:
            for dienst, part in df_in.groupby(dienst_col):
                sheet = str(dienst) if pd.notna(dienst) and str(dienst).strip() else "Onbekend"
                sheet = sheet[:sheet_max_name]
                part = part.reindex(columns=preview_cols, fill_value=np.nan)
                part.to_excel(writer, index=False, sheet_name=sheet)
        else:
            df_in.reindex(columns=preview_cols, fill_value=np.nan).to_excel(writer, index=False, sheet_name="Data")
    return buf.getvalue()

excel_bytes = make_excel_bytes(free_df, col_dienst if col_dienst in free_df.columns else None)

st.download_button(
    "Exporteren naar Excel",
    data=excel_bytes,
    file_name=f"gratis_orders_{start_date.strftime('%d-%m-%Y')}_{end_date.strftime('%d-%m-%Y')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    help="Exporteert gefilterde regels, per tab gesplitst op Dienst facturatie (indien kolom aanwezig).",
)

# =============== Transparantie ===============
with st.expander("🔎 Hoe detecteer ik 'gratis'?"):
    st.markdown(
        """
**Regel-gebaseerd:** zoekt naar termen als *gratis, kosteloos, zonder kosten, 0 euro, niet factureren*,
maar **sluit expliciet** termen uit die enkel over **verzend/transport/bezorg**-kosten gaan
(bijv. *gratis verzendkosten*, *free shipping*).

**Lerend (optioneel):** label in het paneel enkele voorbeelden; het model (TF-IDF + LogisticRegression)
geeft een waarschijnlijkheid terug. De eindscore is:  
**Eindscore = α × (regelscore) + (1−α) × (modelscore)**, en wordt vergeleken met de **drempel**.
"""
    )
