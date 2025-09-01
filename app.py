import io, time
import streamlit as st
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz

st.set_page_config(page_title="Analyse de CV", layout="wide")
st.title("Analyse de CV")

MAX_MB = 5
KW = ["python", "nlp", "pandas", "docker", "streamlit"]
MODEL_NAME = "all-MiniLM-L6-v2"

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

def read_pdf(file_bytes: bytes) -> str:
    r = PdfReader(io.BytesIO(file_bytes))
    return "\n".join([p.extract_text() or "" for p in r.pages])

def read_docx(file_bytes: bytes) -> str:
    with open("/tmp/tmp.docx","wb") as f: f.write(file_bytes)
    doc = Document("/tmp/tmp.docx")
    return "\n".join(p.text for p in doc.paragraphs)

# TODO: colle ici tes fonctions de nettoyage/scoring issues de tes 10 cellules

col1, col2 = st.columns([1,1])
with col1:
    file = st.file_uploader("PDF/DOCX (max 5 Mo)", type=["pdf","docx"])
with col2:
    job_desc = st.text_area("Description du poste", "Data Scientist Python, NLP, Cloud", height=120)

if file:
    if file.size > MAX_MB*1024*1024:
        st.error("Fichier trop volumineux.")
        st.stop()

    raw = file.read()
    text = read_pdf(raw) if file.type.endswith("pdf") else read_docx(raw)
    st.subheader("Texte extrait (aperçu)")
    st.text_area("Contenu", text[:5000], height=220)

    if st.button("Analyser / Scorer"):
        t0 = time.time()
        model = load_model()
        s_sem = float(util.cos_sim(model.encode(text), model.encode(job_desc)))
        kw_score = sum(fuzz.partial_ratio(k, text) for k in KW) / (100*len(KW))
        final = 0.8*s_sem + 0.2*kw_score
        st.metric("Score d’adéquation", f"{final:.3f}", delta=f"{time.time()-t0:.2f}s")
