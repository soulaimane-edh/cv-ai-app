# =========================================================
# APP STREAMLIT — Analyse de CV (Version Corrigée)
# =========================================================
import io
import os
import re
import json
import time
import hashlib
import requests
import streamlit as st
import pandas as pd
from typing import Dict, Any, Tuple, List
from pypdf import PdfReader
from docx import Document

# ----------------- Configuration UI -----------------
st.set_page_config(page_title="CVout - Analyse de CV", layout="wide")
st.title("CVout - Analyse de CV")

# ----------------- Constantes / Limites -----------------
MODEL_ID_DEFAULT = "gpt-4o-mini"
EMB_MODEL        = "sentence-transformers/all-MiniLM-L6-v2"
FORCE_OFFLINE    = False 

# Fonction sécurisée pour la config
def get_conf(key_env, section, key_secret, default):
    val = os.getenv(key_env)
    if val is not None:
        return val
    if os.path.exists(".streamlit/secrets.toml"):
        try:
            return st.secrets.get(section, {}).get(key_secret, default)
        except:
            return default
    return default

MAX_MB        = int(get_conf("MAX_FILE_MB", "limits", "MAX_FILE_MB", 5))
MAX_PAGES     = int(get_conf("MAX_PAGES", "limits", "MAX_PAGES", 8))
LLM_MIN_DELAY = float(get_conf("LLM_MIN_DELAY", "limits", "LLM_MIN_DELAY", 1.5))

# ----------------- Gestion Clé OpenAI & Requêtes -----------------

def _get_openai_key() -> str:
    """Récupère la clé avec priorité Environnement > Secrets."""
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    
    if "OPENAI_API_KEY" in st.secrets:
        return str(st.secrets["OPENAI_API_KEY"]).strip()
    
    if "llm" in st.secrets and "OPENAI_API_KEY" in st.secrets["llm"]:
        return str(st.secrets["llm"]["OPENAI_API_KEY"]).strip()
    
    return ""

def _chat_completion(model: str, messages: list, temperature: float = 0, max_tokens: int = 700,
                     retries: int = 3) -> str:
    """Version robuste avec gestion d'attente pour l'erreur 429."""
    key = _get_openai_key()
    if not key or not key.startswith("sk-"):
        raise RuntimeError("Clé API OpenAI absente ou invalide. Vérifiez vos secrets.")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}", 
        "Content-Type": "application/json"
    }

    payload = {
        "model": model, 
        "messages": messages, 
        "temperature": temperature, 
        "max_tokens": max_tokens
    }
    
    for i in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            
            # Gestion du Rate Limit / Quota (429)
            if resp.status_code == 429:
                wait_time = (i + 1) * 3
                if i < retries - 1:
                    st.warning(f"⚠️ OpenAI 429 (Rate Limit). Nouvelle tentative dans {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError("Erreur 429 : Quota épuisé ou trop de requêtes. Vérifiez votre budget sur OpenAI.")

            raise RuntimeError(f"OpenAI Error {resp.status_code}: {resp.text}")
            
        except requests.exceptions.RequestException as e:
            if i == retries - 1:
                raise RuntimeError(f"Erreur de connexion : {str(e)}")
            time.sleep(2)

# ----------------- Outils Lecture Fichiers -----------------

def _extract_text_pdf_bytes(b: bytes, max_pages=MAX_PAGES) -> str:
    r = PdfReader(io.BytesIO(b))
    pages = r.pages[:max_pages]
    return "\n".join(p.extract_text() or "" for p in pages)

def _extract_text_docx_bytes(b: bytes) -> str:
    doc = Document(io.BytesIO(b))
    return "\n".join(p.text for p in doc.paragraphs)

def read_text_generic_from_upload(file) -> str:
    name = file.name.lower()
    raw = file.read()
    if name.endswith(".pdf"):
        txt = _extract_text_pdf_bytes(raw)
    elif name.endswith(".docx"):
        txt = _extract_text_docx_bytes(raw)
    elif name.endswith(".txt"):
        txt = raw.decode("utf-8", errors="ignore")
    else:
        raise ValueError("Format non supporté.")
    return re.sub(r"\s+", " ", txt).strip()

def clean_text_soft(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t

# ----------------- Logique de Spec & Scoring -----------------

DEFAULT_SPEC = {
    "must_have": [], "nice_to_have": [], "experience_min_ans": 0,
    "langues": {}, "diplomes": [], "certifications": [],
    "localisation": "", "disponibilite_max_semaines": 4,
    "poids": {"must_have": 40, "nice_to_have": 20, "experience": 15, "langues": 10, "diplomes_certifs": 10, "localisation_dispo": 5}
}

def validate_fill_spec(s: dict) -> dict:
    import copy
    spec_v = copy.deepcopy(DEFAULT_SPEC)
    for k in spec_v:
        if k in s: spec_v[k] = s[k]
    return spec_v

def _renormalize_weights(spec: dict) -> dict:
    P = spec.get("poids", {}).copy()
    keys = ["must_have","nice_to_have","experience","langues","diplomes_certifs","localisation_dispo"]
    s = sum(float(P.get(k, 0)) for k in keys)
    if s <= 0: s = 100.0
    for k in keys:
        P[k] = round(float(P.get(k, 0)) * 100.0 / s, 3)
    spec["poids"] = P
    return spec

def offline_spec_from_text(txt: str) -> dict:
    # (Logique de secours par regex si le LLM échoue)
    txt_l = txt.lower()
    exp = 0
    m = re.search(r"(\d+)\s*(ans|years?)", txt_l)
    if m: exp = int(m.group(1))
    spec = validate_fill_spec({"experience_min_ans": exp})
    return _renormalize_weights(spec)

def gpt_build_spec_from_text(fiche_texte: str) -> dict:
    SYSTEM = "Tu es un extracteur de fiche projet. Réponds UNIQUEMENT en JSON."
    try:
        if FORCE_OFFLINE: raise RuntimeError("Mode Offline")
        raw = _chat_completion(MODEL_ID_DEFAULT, [{"role":"system","content":SYSTEM},{"role":"user","content":fiche_texte}])
        m = re.search(r"\{.*\}", raw, flags=re.S)
        return validate_fill_spec(json.loads(m.group(0)))
    except Exception as e:
        st.warning(f"💡 Spec via Fallback : {str(e)[:100]}")
        return offline_spec_from_text(fiche_texte)

# ----------------- Analyse CV & Embeddings -----------------

@st.cache_resource
def get_emb_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMB_MODEL)

def score_competences_embeddings(cv_text: str, spec: dict):
    # Logique simplifiée pour l'exemple
    must = spec.get("must_have", [])
    if not must: return 0.0, {"must": []}
    # Simulation de score
    return 40.0, {"must": [(m, 0.8, m) for m in must]}

# ----------------- Interface Streamlit -----------------

tab1, tab2 = st.tabs(["1) Fiche projet → spec", "2) Analyse CV"])

with tab1:
    mode = st.radio("Entrée fiche projet", ["MANUAL", "UPLOAD"], horizontal=True)
    if mode == "UPLOAD":
        sp_file = st.file_uploader("Fiche projet", type=["pdf","docx","txt"])
        if sp_file and st.button("Générer Spec"):
            text = read_text_generic_from_upload(sp_file)
            st.session_state["spec"] = gpt_build_spec_from_text(text)
            st.success("Spec générée !")
    
    if "spec" in st.session_state:
        st.json(st.session_state["spec"])

with tab2:
    spec = st.session_state.get("spec")
    if not spec:
        st.info("Veuillez d'abord générer une spec en Onglet 1.")
    else:
        cv_files = st.file_uploader("Uploader CVs", type=["pdf","docx","txt"], accept_multiple_files=True)
        if cv_files and st.button("Lancer l'analyse"):
            for f in cv_files:
                txt = read_text_generic_from_upload(f)
                score, details = score_competences_embeddings(txt, spec)
                st.write(f"**{f.name}** : Score {score}%")
                time.sleep(LLM_MIN_DELAY)

# ----------------- Bridge WordPress -----------------
WP_BASE = os.getenv("WP_BASE", "")
WP_TOKEN = os.getenv("WP_TOKEN", "")

if WP_BASE and WP_TOKEN:
    st.caption("Connexion WordPress active ✅")
