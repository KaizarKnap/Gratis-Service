
import io
import re
import datetime as dt
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Gratis weggegeven orders - Analyzer", layout="wide")

st.title("🧾 Gratis weggegeven orders — Analyzer")

st.markdown(
    """
Deze app helpt je bij het vinden en analyseren van orders die **gratis** zijn weggegeven
(op basis van *notitie facturatie*), met filters op **periode** en **orderstatus**,
en exporteert naar Excel met **tabs per Dienst facturatie**.
"""
)

# =============== Helpers ===============
def guess_column(possible_names, columns):
    cols_lower = {c.lower(): c for c in columns}
    for name in possible_names:
        for c in columns:
            if c.lower() == name.lower():
                return c
    # fuzzy contains
    for name in possible_names:
        for c in columns:
            if name.lower() in c.lower():
                return c
    return None

def normalize_text(s):
    if pd.isna(s):
        return ""
    return str(s).strip().lower()

INCLUDE_PATTERNS_DEFAULT = [
    r"gratis(?!\s*(verzend|bezorg|transport|rijkosten|koerier|levering|leverings|aflever|zending|shipping))",
    r"kosteloos(?!\s*(verzend|bezorg|transport|rijkosten))",
    r"kostenloos(?!\s*(verzend|bezorg|transport|rijkosten))",
    r"zonder\s*kosten(?!\s*(verzend|bezorg|transport|rijkosten))",
    r"0\s*(euro|€)",
    r"pro\s*bono",
    r"free\s*(of\s*charge)?(?!\s*(shipping))",
    r"kostenvrij(?!\s*(verzend|bezorg|transport|rijkosten))",
    r"vergoeding\s*=\s*0",
    r"niet\s*factureren",
    r"niet\s*in\s*rekening\s*brengen",
    r"geheel\s*gratis|volledig\s*gratis",
]

EXCLUDE_PATTERNS_DEFAULT = [
    r"gratis\s*(verzend|bezorg)kosten",
    r"gratis\s*verzending",
    r"free\s*shipping",
    r"transport\s*(gratis|kosteloos|kostenloos)",
    r"rijkosten\s*(gratis|kosteloos|kostenloos)",
    r"bezorgkosten\s*(gratis|kosteloos|kostenloos)",
]

def compile_patterns(patterns):
    return [re.compile(p, flags=re.IGNORECASE) for p in patterns]

INCLUDE_RX = compile_patterns(INCLUDE_PATTERNS_DEFAULT)
EXCLUDE_RX = compile_patterns(EXCLUDE_PATTERNS_DEFAULT)

def rule_based_is_free(note: str) -> bool:
    text = str(note or "")
    # If any include matches and no exclude matches, it's free
    inc = any(p.search(text) for p in INCLUDE_RX)
    exc = any(p.search(text) for p in EXCLUDE_RX)
    return bool(inc and not exc)

# =============== Sidebar ===============
st.sidebar.header("⚙️ Instellingen")

uploaded = st.sidebar.file_uploader("Upload Excel-bestand", type=["xlsx", "xls"])

status_filter = st.sidebar.multiselect(
    "Orderstatus (alleen deze worden geanalyseerd)",
    options=["Confirmed", "Edited", "Cancelled", "WaitingOnConfirmation", "Draft", "Completed"],
    default=["Confirmed", "Edited"],
)

st.sidebar.divider()

st.sidebar.subheader("Detectie-instellingen (leren)")
st.sidebar.caption(
    "Je kunt extra **inclusie**- en **exclusie**-woorden/regex toevoegen. "
    "Ook kun je voorbeelden labelen om een eenvoudig model te trainen."
)

include_add = st.sidebar.text_area("Extra inclusie-termen (regex, één per regel)", height=80, value="")
exclude_add = st.sidebar.text_area("Extra exclusie-termen (regex, één per regel)", height=80, value="")

use_model = st.sidebar.toggle("Model leren van gelabelde voorbeelden (TF‑IDF + LogReg)", value=False)

# =============== Main ===============
if uploaded is None:
    st.info("⬆️ Upload eerst een Excel-bestand om te starten.")
    st.stop()

# Read Excel
try:
    xls = pd.ExcelFile(uploaded)
    df = pd.read_excel(xls, sheet_name=0)
except Exception as e:
    st.error(f"Kon Excel niet lezen: {e}")
    st.stop()

orig_columns = df.columns.tolist()

# Guess critical columns
col_orderstatus = guess_column(["orderstatus", "status"], df.columns) or st.sidebar.selectbox("Kies kolom: Orderstatus", df.columns)
col_date = guess_column(["uitvoerdatum", "orderdatum", "datum"], df.columns) or st.sidebar.selectbox("Kies kolom: Datum", df.columns)
col_note = guess_column(["notitie facturatie", "facturatie notitie", "notitie", "opmerking"], df.columns) or st.sidebar.selectbox("Kies kolom: Notitie facturatie", df.columns)
col_dienst = guess_column(["dienst facturatie", "dienst", "facturatie dienst"], df.columns) or st.sidebar.selectbox("Kies kolom: Dienst facturatie", df.columns)
col_created_by = guess_column(["aangemaakt door", "aangemaakt_door", "auteur", "gebruiker"], df.columns) or st.sidebar.selectbox("Kies kolom: Aangemaakt door", df.columns)
col_location = guess_column(["locatienummer", "locatie", "plaats"], df.columns) or st.sidebar.selectbox("Kies kolom: Locatie", df.columns)
col_amount = guess_column(["verkooptarief", "totaal", "bedrag", "prijs"], df.columns) or st.sidebar.selectbox("Kies kolom: Verkooptarief (totaal)", df.columns)

# Cast date
if col_date not in df.columns:
    st.error("Ik kan de datumkolom niet vinden. Selecteer of hernoem de juiste kolom.")
    st.stop()

df[col_date] = pd.to_datetime(df[col_date], errors="coerce")

# Filter by status
if col_orderstatus in df.columns:
    df = df[df[col_orderstatus].astype(str).isin(status_filter)]
else:
    st.warning("Orderstatus-kolom niet gevonden; alle regels worden meegenomen.")

# Period selector
min_date = pd.to_datetime(df[col_date]).min()
max_date = pd.to_datetime(df[col_date]).max()
if pd.isna(min_date) or pd.isna(max_date):
    min_date = dt.date(2000,1,1)
    max_date = dt.date.today()

period = st.slider(
    "Periode",
    min_value=min_date.date(),
    max_value=max_date.date(),
    value=(max(min_date.date(), max_date.date() - dt.timedelta(days=30)), max_date.date()),
)
start_date, end_date = [pd.to_datetime(d) for d in period]
mask_period = (df[col_date] >= start_date) & (df[col_date] <= end_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
df = df[mask_period].copy()

# Build detection patterns (extend with user inputs)
if include_add.strip():
    INCLUDE_USER = [p.strip() for p in include_add.strip().splitlines() if p.strip()]
else:
    INCLUDE_USER = []
if exclude_add.strip():
    EXCLUDE_USER = [p.strip() for p in exclude_add.strip().splitlines() if p.strip()]
else:
    EXCLUDE_USER = []

INCLUDE_RX_ALL = compile_patterns(INCLUDE_PATTERNS_DEFAULT + INCLUDE_USER)
EXCLUDE_RX_ALL = compile_patterns(EXCLUDE_PATTERNS_DEFAULT + EXCLUDE_USER)

def rule_engine_series(notes: pd.Series) -> pd.Series:
    def _is_free(t):
        text = str(t or "")
        inc = any(p.search(text) for p in INCLUDE_RX_ALL)
        exc = any(p.search(text) for p in EXCLUDE_RX_ALL)
        return bool(inc and not exc)
    return notes.fillna("").map(_is_free)

# Optional: simple active learning
LABEL_KEY = "labels_memory"

if LABEL_KEY not in st.session_state:
    st.session_state[LABEL_KEY] = {}  # {row_index: label}

with st.expander("✍️ Optioneel: Voorbeelden labelen voor het model"):
    st.caption("Label enkele notities als **Gratis weggegeven** (ja/nee). Het model leert hiervan.")
    sample = df[[col_note]].dropna().sample(min(20, len(df)), random_state=42) if len(df) > 0 else pd.DataFrame(columns=[col_note])
    for idx, row in sample.iterrows():
        note_text = str(row[col_note])[:400]
        current = st.session_state[LABEL_KEY].get(idx)
        col1, col2, col3 = st.columns([6,1,1])
        with col1:
            st.text_area(f"Notitie (rij {idx})", value=note_text, key=f"note_{idx}", height=80, disabled=True)
        with col2:
            if st.button("✅ Ja", key=f"yes_{idx}"):
                st.session_state[LABEL_KEY][idx] = 1
        with col3:
            if st.button("❌ Nee", key=f"no_{idx}"):
                st.session_state[LABEL_KEY][idx] = 0
    st.write(f"Gelabeld: {sum(v==1 for v in st.session_state[LABEL_KEY].values())} gratis / {sum(v==0 for v in st.session_state[LABEL_KEY].values())} niet-gratis")

# Compute rule-based predictions
rules_free = rule_engine_series(df[col_note] if col_note in df.columns else pd.Series([], dtype=str))

# Train model if requested and we have labels
model_pred = pd.Series(False, index=df.index)
if use_model and len(st.session_state[LABEL_KEY]) >= 4:
    labeled_idx = list(st.session_state[LABEL_KEY].keys())
    y = np.array([st.session_state[LABEL_KEY][i] for i in labeled_idx])
    X_text = df.loc[labeled_idx, col_note].fillna("").astype(str)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    try:
        pipe.fit(X_text, y)
        model_scores = pipe.predict_proba(df[col_note].fillna("").astype(str))[:,1]
        # Combine rules + model (conservative: require either rules True OR high model prob)
        model_pred = pd.Series(model_scores >= 0.6, index=df.index)
    except Exception as e:
        st.warning(f"Model trainen mislukt: {e}")

# Final decision: rules OR model (if enabled)
is_free = rules_free | model_pred

df["__is_free__"] = is_free

free_df = df[df["__is_free__"]].copy()

# ====== KPI's ======
st.subheader("📈 Overzicht")
colA, colB, colC, colD = st.columns(4)
total_free = float(free_df[col_amount].fillna(0).replace({",": "."}, regex=True).astype(str).str.replace(",", ".").str.replace(" ", "").astype(float)) if col_amount in free_df.columns else 0.0
with colA:
    st.metric("Aantal gratis orders (regels)", len(free_df))
with colB:
    st.metric("Periode", f"{start_date.date()} → {end_date.date()}")
with colC:
    st.metric("Unieke locaties", free_df[col_location].nunique() if col_location in free_df.columns else 0)
with colD:
    st.metric("Totaal weggegeven (€)", f"{total_free:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

# ====== Grafiek per dag ======
st.subheader("📅 Weggegeven per dag (aantal)")
if len(free_df) == 0:
    st.info("Geen gratis orders gevonden in de geselecteerde periode/filters.")
else:
    daily = free_df.set_index(col_date).sort_index().groupby(pd.Grouper(freq="D")).size().rename("aantal").to_frame()
    st.line_chart(daily)

# ====== Tabel: wie heeft weggegeven ======
st.subheader("👤 Wie heeft orders weggegeven (aantal)?")
if col_created_by in free_df.columns:
    who = free_df.groupby(col_created_by).size().sort_values(ascending=False).rename("aantal").to_frame()
    st.dataframe(who)
else:
    st.info("Kolom 'Aangemaakt door' niet gevonden. Kies/controleer deze in de instellingen.")

# ====== Locatie overview (over totale lijst, niet per dienst) ======
st.subheader("📍 Locaties met gratis weggegeven orders (totaal)")
if col_location in free_df.columns:
    loc = free_df.groupby(col_location).size().sort_values(ascending=False).rename("aantal").to_frame()
    st.dataframe(loc)
else:
    st.info("Kolom 'Locatie' niet gevonden. Kies/controleer deze in de instellingen.")

# ====== Detailtabel & export per Dienst facturatie ======
st.subheader("📑 Detailtabel")
st.caption("Gefilterde regels die als **gratis** zijn gedetecteerd.")

show_cols = orig_columns  # zelfde opmaak/volgorde als input
preview = free_df.reindex(columns=show_cols, fill_value=np.nan).head(1000)
st.dataframe(preview)

st.divider()
st.subheader("⬇️ Exporteren naar Excel (tabs per Dienst facturatie)")

def make_excel_bytes(df_in: pd.DataFrame, dienst_col: str, sheet_max_name: int = 31) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        if dienst_col in df_in.columns:
            for dienst, part in df_in.groupby(dienst_col):
                sheet = str(dienst) if pd.notna(dienst) and str(dienst).strip() else "Onbekend"
                sheet = sheet[:sheet_max_name]
                part = part.reindex(columns=show_cols, fill_value=np.nan)
                part.to_excel(writer, index=False, sheet_name=sheet)
        else:
            # fallback: alles op 1 tab
            df_in.reindex(columns=show_cols, fill_value=np.nan).to_excel(writer, index=False, sheet_name="Data")
    return buf.getvalue()

excel_bytes = make_excel_bytes(free_df, col_dienst if col_dienst in free_df.columns else None)

st.download_button(
    "Exporteren naar Excel",
    data=excel_bytes,
    file_name=f"gratis_orders_{start_date.date()}_{end_date.date()}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    help="Exporteert gefilterde regels, met aparte tabs per Dienst facturatie (indien kolom aanwezig).",
)

# ====== Transparantie over detectie ======
with st.expander("🔎 Hoe detecteer ik 'gratis'?"):
    st.markdown(
        """
        **Regel-gebaseerd:** zoekt naar termen als *gratis, kosteloos, zonder kosten, 0 euro, niet factureren*,
        maar **sluit expliciet** termen uit die enkel over **verzending/transport/bezorg-**kosten gaan
        (bv. *gratis verzendkosten*, *free shipping*).
        
        **Lerend (optioneel):** label in het paneel enkele voorbeelden; het model (TF‑IDF + LogisticRegression)
        verhoogt/vermindert de kans dat een notitie als *gratis* wordt aangemerkt. De uiteindelijke selectie is:
        **Regels OR (model ≥ 0.60)**.
        """
    )
