import io, time, re, json
import streamlit as st
from pypdf import PdfReader
from docx import Document
from rapidfuzz import fuzz
import pandas as pd

# ---------- UI ----------
st.set_page_config(page_title="Analyse de CV", layout="wide")
st.title("Analyse de CV (Notebook → App)")

MAX_MB  = int(st.secrets.get("limits", {}).get("MAX_FILE_MB", 5))
MAX_PGS = int(st.secrets.get("limits", {}).get("MAX_PAGES", 8))

# ---------- Cellule 4 : extraction ----------
def read_pdf(file_bytes: bytes, max_pages=MAX_PGS) -> str:
    r = PdfReader(io.BytesIO(file_bytes))
    pages = r.pages[:max_pages]
    return "\n".join([p.extract_text() or "" for p in pages])

def read_docx(file_bytes: bytes) -> str:
    with open("/tmp/tmp.docx","wb") as f: f.write(file_bytes)
    doc = Document("/tmp/tmp.docx")
    return "\n".join(p.text for p in doc.paragraphs)

def clean_text(t: str) -> str:
    t = t.replace("\x00"," ").strip()
    t = re.sub(r"[ \t]+"," ", t)
    return t

# ---------- Cellule 5 : features heuristiques ----------
KW = ["python","nlp","pandas","docker","streamlit","aws","azure","sql","ml","deeplearning"]
def keyword_features(text: str) -> dict:
    kw_score = sum(fuzz.partial_ratio(k, text) for k in KW) / (100*len(KW))
    years = re.findall(r"(\d+)\s*(ans|years?)", text.lower())
    exp_years = max([int(n) for (n,_) in years], default=0)
    return {"kw_score": kw_score, "exp_years": exp_years}

# ---------- Cellule 6 : similarité ----------
USE_EMB = st.toggle("Activer embeddings (nécessite torch, peut être lent)", value=False)

# TF-IDF (chemin léger)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
@st.cache_resource
def get_vectorizer():
    return TfidfVectorizer(ngram_range=(1,2), max_features=20000)

def tfidf_similarity(a: str, b: str) -> float:
    vec = get_vectorizer()
    X = vec.fit_transform([a, b])
    return float(cosine_similarity(X[0], X[1])[0][0])

# Embeddings (chemin complet)
if USE_EMB:
    from sentence_transformers import SentenceTransformer, util
    @st.cache_resource
    def load_model():
        return SentenceTransformer("all-MiniLM-L6-v2")
    def emb_similarity(a: str, b: str) -> float:
        m = load_model()
        return float(util.cos_sim(m.encode(a), m.encode(b)))

# ---------- Cellule 7 : résumé LLM (optionnel) ----------
def summarize_llm(text: str, lang="fr") -> str:
    key = st.secrets.get("llm", {}).get("OPENAI_API_KEY")
    if not key:
        return "Clé LLM absente (ajoute-la dans Secrets)."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        prompt = f"Résume ce CV en 5 puces ({lang}) et liste 5 forces:\n{text[:6000]}"
        out = client.chat.completions.create(model="gpt-4o-mini",
                                             messages=[{"role":"user","content":prompt}],
                                             temperature=0.3)
        return out.choices[0].message.content
    except Exception as e:
        return f"Résumé indisponible: {e}"

# ---------- Cellule 8 : combinaison ----------
def combine_scores(semantic: float, kw_score: float, feats: dict) -> float:
    # Pondération simple ; ajuste selon ton notebook
    bonus = min(feats.get("exp_years",0), 10) / 100.0   # +0.00..0.10
    return round(0.8*semantic + 0.2*kw_score + bonus, 3)

# ---------- UI : inputs (Cellules 3/10) ----------
left, right = st.columns([1,1])
with left:
    files = st.file_uploader("PDF/DOCX (max 5 Mo chacun)", type=["pdf","docx"], accept_multiple_files=True)
with right:
    job_desc = st.text_area("Description du poste", "Data Scientist Python, NLP, Cloud", height=160)
    want_summary = st.checkbox("Générer un résumé LLM (optionnel)")

# ---------- Traitement (Cellules 3–9) ----------
rows = []
if files and job_desc:
    for f in files:
        if f.size > MAX_MB*1024*1024:
            st.error(f"{f.name} : fichier trop volumineux (> {MAX_MB} Mo).")
            continue
        raw = f.read()
        text = read_pdf(raw) if f.type.endswith("pdf") else read_docx(raw)
        text = clean_text(text)
        feats = keyword_features(text)
        sem = emb_similarity(text, job_desc) if USE_EMB else tfidf_similarity(text, job_desc)
        final = combine_scores(sem, feats["kw_score"], feats)
        summary = summarize_llm(text) if want_summary else ""
        rows.append({
            "fichier": f.name, "score_final": final, "sim": round(sem,3),
            "kw_score": round(feats["kw_score"],3), "exp_years": feats["exp_years"],
            "resume": summary
        })

# ---------- Affichage & export (Cellule 9) ----------
if rows:
    df = pd.DataFrame(rows).sort_values("score_final", ascending=False)
    st.subheader("Résultats")
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger CSV", csv, file_name="resultats_cv.csv", mime="text/csv")
