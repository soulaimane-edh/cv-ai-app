# =========================================================
# APP STREAMLIT — Analyse de CV (Notebook → App)
# - LLM via API OpenAI (retries/backoff + fallback OFFLINE)
# - Construction de spec (LLM -> fallback regex)
# - Scoring "façon notebook" (cellule 6) mais borné par la spec:
#     * Similarité CV complet ↔ skill
#     * Seuils must/nice + moyenne, puis pondération par POIDS
# - Règles strictes (cellule 8) + renormalisation POIDS = 100
# - Option embeddings locale (all-MiniLM-L6-v2) activable
# =========================================================
import io
import os
import re
import json
import time
import math
import hashlib
import requests
import streamlit as st
import pandas as pd
from typing import Dict, Any, Tuple, List
from pypdf import PdfReader
from docx import Document
























# ----------------- UI de base -----------------
st.set_page_config(page_title="Analyse de CV (Notebook → App)", layout="wide")
st.title("Analyse de CV — par ED-dahmani Soulaimane")

# ----------------- Constantes / limites -----------------
MODEL_ID_DEFAULT = "gpt-4o-mini"              # ou "gpt-5" / "gpt-4o-mini" suivant ton compte
EMB_MODEL        = "sentence-transformers/all-MiniLM-L6-v2"

MAX_MB        = int(st.secrets.get("limits", {}).get("MAX_FILE_MB", 5))
MAX_PAGES     = int(st.secrets.get("limits", {}).get("MAX_PAGES", 8))
LLM_MIN_DELAY = float(st.secrets.get("limits", {}).get("LLM_MIN_DELAY", 1.2))

# ----------------- Sidebar : paramètres LLM -----------------















# ----------------- Clé OpenAI + appel HTTP (avec retries) -----------------
def _get_openai_key() -> str:
    key = (st.secrets.get("llm", {}) or {}).get("OPENAI_API_KEY")
    key = key or st.secrets.get("OPENAI_API_KEY")
    key = key or os.getenv("OPENAI_API_KEY")
    return (str(key).strip() if key else "")

def _chat_completion(model: str, messages: list, temperature: float = 0, max_tokens: int = 700,
                     retries: int = 4) -> str:
    """Appel /v1/chat/completions avec retries + respect éventuel de Retry-After."""
    key = _get_openai_key()
    if not key or not key.startswith("sk-"):
        raise RuntimeError("OPENAI_API_KEY absente/invalide (Settings → Secrets).")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    # Support optionnel org/projet si précisé dans Secrets/env
    org  = (st.secrets.get("llm", {}) or {}).get("OPENAI_ORG")     or os.getenv("OPENAI_ORG")
    proj = (st.secrets.get("llm", {}) or {}).get("OPENAI_PROJECT") or os.getenv("OPENAI_PROJECT")
    if org:  headers["OpenAI-Organization"] = org
    if proj: headers["OpenAI-Project"]      = proj

    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}

    delay = 2.0
    last_err = ""
    for attempt in range(retries):
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        if resp.status_code in (429, 500, 502, 503, 504):
            last_err = resp.text[:300]
            ra = resp.headers.get("retry-after")
            if ra:
                try:
                    delay = max(delay, float(ra))
                except Exception:
                    pass
            time.sleep(delay + 0.2 * attempt)
            delay = min(delay * 2, 20.0)
            continue

        raise RuntimeError(f"OpenAI HTTP {resp.status_code}: {resp.text[:300]}")

    raise RuntimeError(f"OpenAI indisponible après retries. Dernier message: {last_err}")

# ----------------- Outils lecture fichiers/texte -----------------
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
        raise ValueError("Format non supporté (PDF/DOCX/TXT).")
    return re.sub(r"\s+", " ", txt).strip()

def clean_text_soft(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t

# ----------------- Spec par défaut + validate + renormalisation -----------------
DEFAULT_SPEC = {
    "must_have": [], "nice_to_have": [],
    "experience_min_ans": 0,
    "langues": {}, "diplomes": [], "certifications": [],
    "localisation": "", "disponibilite_max_semaines": 4,
    "poids": {
        "must_have": 40, "nice_to_have": 20, "experience": 15,
        "langues": 10, "diplomes_certifs": 10, "localisation_dispo": 5
    }
}

def validate_fill_spec(s: dict) -> dict:
    import copy
    spec_v = copy.deepcopy(DEFAULT_SPEC)
    for k in spec_v:
        if k in s: spec_v[k] = s[k]
    if "poids" not in spec_v or not isinstance(spec_v["poids"], dict):
        spec_v["poids"] = dict(DEFAULT_SPEC["poids"])
    for k in DEFAULT_SPEC["poids"]:
        if k not in spec_v["poids"]:
            spec_v["poids"][k] = DEFAULT_SPEC["poids"][k]
    spec_v["must_have"] = list(spec_v.get("must_have", []))
    spec_v["nice_to_have"] = list(spec_v.get("nice_to_have", []))
    spec_v["langues"] = dict(spec_v.get("langues", {}))
    spec_v["diplomes"] = list(spec_v.get("diplomes", []))
    spec_v["certifications"] = list(spec_v.get("certifications", []))
    spec_v["localisation"] = str(spec_v.get("localisation", ""))
    try: spec_v["experience_min_ans"] = int(spec_v.get("experience_min_ans", 0))
    except: spec_v["experience_min_ans"] = 0
    try: spec_v["disponibilite_max_semaines"] = int(spec_v.get("disponibilite_max_semaines", 4))
    except: spec_v["disponibilite_max_semaines"] = 4
    return spec_v

def _renormalize_weights(spec: dict) -> dict:
    """S'assure que la somme des poids = 100 (évite des scores >100)."""
    P = spec.get("poids", {}).copy()
    keys = ["must_have","nice_to_have","experience","langues","diplomes_certifs","localisation_dispo"]
    s = sum(float(P.get(k, 0)) for k in keys)
    if s <= 0:
        P = {"must_have":40,"nice_to_have":20,"experience":15,"langues":10,"diplomes_certifs":10,"localisation_dispo":5}
        s = 100.0
    if abs(s - 100.0) > 1e-6:
        for k in keys:
            P[k] = round(float(P.get(k, 0)) * 100.0 / s, 3)
    spec["poids"] = P
    return spec

# ----------------- Fallback OFFLINE pour construire la spec -----------------
def offline_spec_from_text(txt: str) -> dict:
    """Spec minimale extraite depuis fiche projet (sans LLM)."""
    txt_l = txt.lower()

    def grab_section(header_keywords):
        pat = r"(?:" + "|".join([re.escape(k.lower()) for k in header_keywords]) + r")\s*[:\-]\s*(.+)"
        m = re.search(pat, txt_l)
        if not m:
            return []
        line = m.group(1)
        items = [re.sub(r"\s+", " ", w).strip(" .;-") for w in re.split(r"[,;/•|]", line)]
        return [it for it in items if it]

    must = grab_section(["must have", "obligatoire", "exigé", "requis", "required"])
    nice = grab_section(["nice to have", "souhaité", "optionnel", "plus"])

    KNOWN = [
        "python","sql","spark","airflow","aws","docker","kubernetes","power bi","pandas",
        "scikit-learn","tensorflow","pytorch","azure","gcp","tableau","excel","dbt"
    ]
    if not must and not nice:
        for k in KNOWN:
            if re.search(rf"(?<![a-z0-9_]){re.escape(k)}(?![a-z0-9_])", txt_l):
                must.append(k)
        must = list(dict.fromkeys(must))[:6]

    exp = 0
    m = re.search(r"(\d+)\s*(ans|years?)\s+(?:d['e]|\s)*exp", txt_l)
    if m: exp = int(m.group(1))
    else:
        m = re.search(r"exp[\w\s:]*?(\d+)\s*(ans|years?)", txt_l)
        if m: exp = int(m.group(1))

    langues = {}
    for code in ["fr","en","de","es","ar","it"]:
        p = re.search(rf"\b{code}\b[^A-Z0-9]{{0,20}}(A1|A2|B1|B2|C1|C2)", txt, flags=re.I)
        if p: langues[code] = p.group(1).upper()

    diplomes = []
    if re.search(r"\b(licence|master|ing[eé]nieur|bachelor|bac\+)\b", txt_l):
        diplomes = ["Diplôme supérieur"]

    certifs = []
    if re.search(r"\b(certification|certifi[eé]|aws certified|azure|gcp)\b", txt_l):
        certifs = ["Certification détectée"]

    loc = ""
    m = re.search(r"localisation\s*[:\-]\s*([^\n,;]+)", txt, flags=re.I)
    if m: loc = m.group(1).strip()

    dispo = 4
    m = re.search(r"(\d+)\s*(semaines?)\s*(?:max|maximum|disponibilit[eé])", txt_l)
    if m: dispo = int(m.group(1))

    spec = validate_fill_spec({
        "must_have": must, "nice_to_have": nice,
        "experience_min_ans": exp,
        "langues": langues, "diplomes": diplomes, "certifications": certifs,
        "localisation": loc, "disponibilite_max_semaines": dispo,
        "poids": {"must_have":40,"nice_to_have":20,"experience":15,"langues":10,"diplomes_certifs":10,"localisation_dispo":5}
    })
    return _renormalize_weights(spec)

# ----------------- Cellule 3 : Spec via LLM (fallback si erreur) -----------------
SPEC_SYSTEM = """
Tu es SPEC_BUILDER. À partir d'une fiche projet en texte libre,
tu renvoies UNIQUEMENT un JSON valide qui respecte exactement ce schéma :
{
  "must_have": [string],
  "nice_to_have": [string],
  "experience_min_ans": number,
  "langues": {"fr|en|...": "A1|A2|B1|B2|C1|C2"},
  "diplomes": [string],
  "certifications": [string],
  "localisation": string,
  "disponibilite_max_semaines": number,
  "poids": {
    "must_have": number, "nice_to_have": number, "experience": number,
    "langues": number, "diplomes_certifs": number, "localisation_dispo": number
  }
}
Règles :
- Valeurs réalistes.
- Si info absente → valeurs par défaut raisonnables.
- Les poids doivent approx. sommer 100.
- Réponds en UNE SEULE structure JSON, sans texte autour.
"""

def gpt_build_spec_from_text(fiche_texte: str, model_id: str = None) -> dict:
    """Spec via LLM ; fallback regex si l'appel échoue ou FORCE_OFFLINE."""
    model_id = model_id or st.session_state.get("MODEL_ID", MODEL_ID_DEFAULT)
    msgs = [{"role":"system","content":SPEC_SYSTEM},
            {"role":"user","content":fiche_texte}]
    try:
        if st.session_state.get("FORCE_OFFLINE"):
            raise RuntimeError("FORCE_OFFLINE activé")
        txt = _chat_completion(model_id, msgs, temperature=0, max_tokens=700).strip()
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if not m:
            raise ValueError("JSON non trouvé")
        raw = m.group(0)
        try:
            spec = validate_fill_spec(json.loads(raw))
        except Exception:
            cleaned = re.sub(r",\s*}", "}", raw)
            cleaned = re.sub(r",\s*]", "]", cleaned)
            spec = validate_fill_spec(json.loads(cleaned))
    except Exception as e:
        st.warning("💡 Spec sans LLM (fallback) — " + str(e)[:180])
        spec = offline_spec_from_text(fiche_texte)
    return _renormalize_weights(spec)

# ----------------- Cellule 4 : Lecture CV -----------------
def read_cv_text_from_upload(file) -> str:
    name = file.name.lower()
    raw  = file.read()
    if name.endswith(".pdf"):  txt = _extract_text_pdf_bytes(raw)
    elif name.endswith(".docx"): txt = _extract_text_docx_bytes(raw)
    elif name.endswith(".txt"):  txt = raw.decode("utf-8", errors="ignore")
    else: raise ValueError("Format non supporté (PDF/DOCX/TXT).")
    return clean_text_soft(txt)

# ----------------- Cellule 5 : Extraction safe (LLM + fallback) -----------------
def offline_extract_from_text(cv_text: str) -> dict:
    data = {
        "experience_ans": {"value": None, "evidence": []},
        "disponibilite_semaines": {"value": None, "evidence": []},
        "langues": [],
        "diplomes_obtenus": [],
        "diplomes_en_cours": [],
        "certifications": [],
        "localisation": {"value": "", "evidence": []},
    }
    m = re.search(r"(\d+)\s*(ans|year|years)", cv_text, flags=re.I)
    if m: data["experience_ans"]["value"] = float(m.group(1))
    m = re.search(r"(\d+)\s*(semaines?|weeks?)", cv_text, flags=re.I)
    if m: data["disponibilite_semaines"]["value"] = float(m.group(1))

    langs = []
    for code in ["fr","en","de","es","ar","it"]:
        p = re.search(rf"\b{code}\b[^A-Z0-9]*\(?\s*(A1|A2|B1|B2|C1|C2)\s*\)?", cv_text, flags=re.I)
        if p: langs.append({"code": code, "niveau": p.group(1).upper(), "evidence": []})
    data["langues"] = langs

    m = re.search(r"Localisation\s*[:\-]\s*([^\n,;]+)", cv_text, flags=re.I)
    if m: data["localisation"]["value"] = m.group(1).strip()

    if re.search(r"\b(licence|master|engineer|ing[eé]nieur|bachelor|bac\+)\b", cv_text, flags=re.I):
        data["diplomes_obtenus"].append({"label": "Diplôme détecté", "year": None, "evidence": []})
    if re.search(r"\b(certification|certifi[eé])\b", cv_text, flags=re.I):
        data["certifications"].append({"label": "Certification détectée", "evidence": []})
    return data

def gpt_extract_profile_safe(cv_text: str, model_id: str = None) -> dict:
    """Extraction via LLM ; fallback offline si clé absente/erreur/forced."""
    if st.session_state.get("FORCE_OFFLINE") or not _get_openai_key():
        return offline_extract_from_text(cv_text)

    SYSTEM = """
    Tu es un extracteur STRICT. Renvoie UN SEUL JSON avec ce schéma EXACT :
    {
      "experience_ans": {"value": number|null, "evidence": [{"text": string, "start": number, "end": number}]},
      "disponibilite_semaines": {"value": number|null, "evidence": [{"text": string, "start": number, "end": number}]},
      "langues": [{"code":"fr|en|...","niveau":"A1|A2|B1|B2|C1|C2","evidence":[{"text": string, "start": number, "end": number}]}],
      "diplomes_obtenus": [{"label": string, "year": number|null, "evidence":[{"text": string, "start": number, "end": number}]}],
      "diplomes_en_cours": [{"label": string, "evidence":[{"text": string, "start": number, "end": number}]}],
      "certifications": [{"label": string, "evidence":[{"text": string, "start": number, "end": number}]}],
      "localisation": {"value": string|null, "evidence":[{"text": string, "start": number, "end": number}]}
    }
    - N'INVENTE RIEN.
    """
    model_id = model_id or st.session_state.get("MODEL_ID", MODEL_ID_DEFAULT)
    msgs = [{"role":"system","content":SYSTEM},
            {"role":"user","content":cv_text[:120000]}]

    try:
        raw = _chat_completion(model_id, msgs, temperature=0, max_tokens=900).strip()
        m = re.search(r"\{.*\}", raw, flags=re.S)
        js = m.group(0) if m else "{}"
        try:
            data = json.loads(js)
        except Exception:
            js = re.sub(r",\s*}", "}", js); js = re.sub(r",\s*]", "]", js)
            data = json.loads(js or "{}")
    except Exception as e:
        st.warning(f"LLM indisponible ({str(e)[:160]}). Passage en extraction locale.")
        data = offline_extract_from_text(cv_text)

    data.setdefault("experience_ans", {"value": None, "evidence": []})
    data.setdefault("disponibilite_semaines", {"value": None, "evidence": []})
    data.setdefault("localisation", {"value": "", "evidence": []})
    data.setdefault("langues", [])
    data.setdefault("diplomes_obtenus", [])
    data.setdefault("diplomes_en_cours", [])
    data.setdefault("certifications", [])
    return data

def enforce_evidence(extraction: dict, cv_text: str) -> dict:
    def _ev_for(val: str):
        if not val: return []
        i = cv_text.lower().find(str(val).lower())
        if i < 0: return []
        return [{"text": cv_text[i:i+len(str(val))], "start": i, "end": i+len(str(val))}]
    for k in ("experience_ans","disponibilite_semaines","localisation"):
        slot = extraction.get(k, {})
        val = slot.get("value") if isinstance(slot, dict) else None
        if val and not slot.get("evidence"):
            slot["evidence"] = _ev_for(val); extraction[k] = slot
    return extraction

def fill_with_regex_if_missing(extraction: dict, cv_text: str) -> dict:
    if not extraction.get("experience_ans", {}).get("value"):
        m = re.search(r"(\d+)\s*(ans|year|years)", cv_text, flags=re.I)
        if m: extraction["experience_ans"] = {"value": float(m.group(1)), "evidence": []}
    if not extraction.get("disponibilite_semaines", {}).get("value"):
        m = re.search(r"(\d+)\s*(semaines?|weeks?)", cv_text, flags=re.I)
        if m: extraction["disponibilite_semaines"] = {"value": float(m.group(1)), "evidence": []}
    return extraction

# ----------------- Cellule 6 : Embeddings + scoring "notebook-like" -----------------
USE_EMB = st.toggle("Activer embeddings (S-BERT) — nécessite torch (plus lent)", value=True)

@st.cache_resource
def get_emb_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMB_MODEL)

def score_competences_embeddings(cv_text: str, spec: Dict[str, Any],
                                 thr_must: float = 0.28, thr_nice: float = 0.25
                                 ) -> Tuple[float, Dict[str, List[Tuple[str, float, str]]]]:
    """
    Logique alignée à la cellule 6 du notebook :
    - Similarité entre le CV (texte entier) et chaque skill.
    - On calcule la moyenne des similarités par groupe (must/nice),
      on applique des seuils de couverture, puis on pondère par POIDS de la spec.
    - Fallback sans embeddings = simple keyword strict (pas de fuzzy permissif).
    """
    from statistics import mean

    P = spec.get("poids", {})
    w_must = float(P.get("must_have", 40.0))
    w_nice = float(P.get("nice_to_have", 20.0))

    must = [s.strip() for s in spec.get("must_have", []) if s.strip()]
    nice = [s.strip() for s in spec.get("nice_to_have", []) if s.strip()]
    proofs = {"must": [], "nice": []}

    sims_must, sims_nice = [], []

    if USE_EMB:
        try:
            from sentence_transformers import util
            model = get_emb_model()
            cv_vec = model.encode(cv_text, normalize_embeddings=True)

            def sim(skill: str) -> float:
                v = model.encode(skill, normalize_embeddings=True)
                return float(util.cos_sim(cv_vec, v).item())

            for sk in must:
                s = sim(sk); proofs["must"].append((sk, round(s,3), sk if s >= thr_must else ""))
                sims_must.append(max(0.0, s))
            for sk in nice:
                s = sim(sk); proofs["nice"].append((sk, round(s,3), sk if s >= thr_nice else ""))
                sims_nice.append(max(0.0, s))
        except Exception:
            # en cas d'erreur d'embeddings, on retombe sur le strict keyword
            LOW = cv_text.lower()
            def hit(skill: str) -> float:
                return 1.0 if re.search(rf"(?<![a-z0-9_]){re.escape(skill.lower())}(?![a-z0-9_])", LOW) else 0.0
            for sk in must:
                s = hit(sk); proofs["must"].append((sk, s, sk if s >= 1.0 else "")); sims_must.append(s)
            for sk in nice:
                s = hit(sk); proofs["nice"].append((sk, s, sk if s >= 1.0 else "")); sims_nice.append(s)
    else:
        # Fallback (OFFLINE) : keyword strict
        LOW = cv_text.lower()
        def hit(skill: str) -> float:
            return 1.0 if re.search(rf"(?<![a-z0-9_]){re.escape(skill.lower())}(?![a-z0-9_])", LOW) else 0.0
        for sk in must:
            s = hit(sk); proofs["must"].append((sk, s, sk if s >= 1.0 else "")); sims_must.append(s)
        for sk in nice:
            s = hit(sk); proofs["nice"].append((sk, s, sk if s >= 1.0 else "")); sims_nice.append(s)

    pts = 0.0
    if must:
        avg_must   = mean(sims_must)
        cover_must = mean([1.0 if s >= thr_must else 0.0 for s in sims_must])
        pts += w_must * min(1.0, max(0.0, avg_must)) * cover_must

    if nice:
        avg_nice   = mean(sims_nice)
        cover_nice = mean([1.0 if s >= thr_nice else 0.0 for s in sims_nice])
        pts += w_nice * min(1.0, max(0.0, avg_nice)) * cover_nice

    return round(pts, 2), proofs

# ----------------- Cellule 8 : Règles strictes -----------------
ORDER_CEFR = {"A1":1,"A2":2,"B1":3,"B2":4,"C1":5,"C2":6}
def _num(slot, default=None):
    if isinstance(slot, dict): slot = slot.get("value", None)
    try: return float(slot)
    except (TypeError, ValueError): return default
def _str(slot):
    if isinstance(slot, dict): return slot.get("value", "")
    return slot or ""

def score_autres_criteres(ex: Dict[str, Any], spec: Dict[str, Any]) -> Tuple[float, str, Dict[str, Any]]:
    """
    Règles strictes :
    - Expérience : points uniquement si un minimum est exigé (>0)
    - Langues : points uniquement si des langues sont exigées ET niveau >= requis
    - Diplômes/Certifs : points uniquement s'il y en a au moins un
    - Localisation/Dispo : points uniquement si la spec impose la contrainte
    """
    P = spec.get("poids", {})
    w_exp  = float(P.get("experience", 15))
    w_lang = float(P.get("langues", 10))
    w_dc   = float(P.get("diplomes_certifs", 10))
    w_locd = float(P.get("localisation_dispo", 5))

    pts = 0.0

    # Expérience
    need = spec.get("experience_min_ans", None)
    yrs  = _num(ex.get("experience_ans"))
    if isinstance(need, (int, float)) and need > 0 and isinstance(yrs, (int, float)):
        pts += w_exp * min(1.0, yrs / max(1, int(need)))

    # Langues
    lang_req = spec.get("langues", {}) or {}
    if lang_req:
        per_lang = w_lang / len(lang_req)
        for code, req in lang_req.items():
            have = next((d.get("niveau") for d in ex.get("langues", [])
                         if (d.get("code") or "").lower() == code.lower()), None)
            if have and ORDER_CEFR.get((have or "").upper(), 0) >= ORDER_CEFR.get((req or "").upper(), 0):
                pts += per_lang

    # Diplômes/Certifs (au moins un)
    if ex.get("diplomes_obtenus") or ex.get("certifications"):
        pts += w_dc

    # Localisation
    want_loc = (spec.get("localisation") or "").strip()
    if want_loc:
        ex_loc = _str(ex.get("localisation")).lower()
        ok = any(seg.strip().lower() in ex_loc for seg in want_loc.split("|") if seg.strip())
        if ok:
            pts += w_locd / 2

    # Disponibilité
    dmax = spec.get("disponibilite_max_semaines", None)
    dval = _num(ex.get("disponibilite_semaines"))
    if isinstance(dmax, (int, float)) and isinstance(dval, (int, float)) and dval <= dmax:
        pts += w_locd / 2

    com = "Règles: exp(min>0), langues requises, diplômes/certifs≥1, localisation/dispo si imposées."
    return round(pts, 2), com, {"ok": True}

# ----------------- Commentaire déterministe (texte fixe) -----------------
def build_commentaire_deterministe(score_final, evidences, extraction, spec):
    def _fmt_langues(lang_list):
        if not lang_list: return "non mentionnées"
        parts=[]
        for it in lang_list:
            code=(it.get("code") or "").upper()
            lvl=(it.get("niveau") or "").upper()
            if code and lvl: parts.append(f"{code} ({lvl})")
            elif code: parts.append(code)
        return ", ".join(parts) if parts else "non mentionnées"

    target_exp = spec.get("experience_min_ans", None)
    target_txt = f"{target_exp} an(s)" if isinstance(target_exp,(int,float)) else "non précisé"
    exp_val = _num(extraction.get("experience_ans"))
    exp_txt = "non mentionnée" if exp_val is None else f"{exp_val:.0f} an(s)"
    dispo_val = _num(extraction.get("disponibilite_semaines"))
    dispo_txt = f"{dispo_val} semaine(s)" if isinstance(dispo_val,(int,float)) else "non mentionnée"
    loc_val = _str(extraction.get("localisation"))
    loc_txt = loc_val if loc_val.strip() else "non mentionnée"
    dipl_txt = ", ".join([d.get("label","") for d in extraction.get("diplomes_obtenus", [])]) or "non mentionnés"
    certs_txt= ", ".join([c.get("label","") for c in extraction.get("certifications", [])]) or "non mentionnées"

    def decision_band(score):
        return ("Très bon match – à convoquer" if score>=85 else
                "Bon match – à prioriser" if score>=70 else
                "Moyen – à examiner" if score>=55 else
                "Faible – non prioritaire")

    rows = []
    for kind, thr in (("must",0.55),("nice",0.50)):
        top = sorted(evidences.get(kind,[]), key=lambda x: x[1], reverse=True)
        for sk,sim,phr in top[:4]:
            if sim >= thr:
                rows.append(f"• {sk} (sim={sim:.3f})" + (f" — « {phr[:120]}… »" if phr else ""))

    com = []
    com.append(f"Score final : {score_final:.2f} %.")
    com.append("Compétences :"); com.extend(rows if rows else ["• Aucune preuve forte."])
    com.append(f"Expérience : {exp_txt} (objectif : {target_txt}).")
    com.append(f"Langues : {_fmt_langues(extraction.get('langues', []))}.")
    com.append(f"Diplômes : {dipl_txt}. Certifs : {certs_txt}.")
    com.append(f"Localisation : {loc_txt}. Disponibilité : {dispo_txt}.")
    com.append(f"Recommandation : {decision_band(score_final)}.")
    return "\n".join(com)

# ----------------- Cache extraction (évite appels LLM récurrents) -----------------
def _hash_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()

@st.cache_data(ttl=3600, show_spinner=False)
def _extract_cached(h: str, text: str, model_id: str):
    return gpt_extract_profile_safe(text, model_id=model_id)

# ----------------- UI onglets -----------------
tab1, tab2, tab3 = st.tabs(["1) Fiche projet → spec", "2) Analyse CV", "3) Démo (cellule 10)"])

with tab1:
    _key_dbg = _get_openai_key(); _mask = (_key_dbg[:3]+"…"+_key_dbg[-4:]) if _key_dbg else "—"
    st.caption("🔐 Clé OpenAI  : " + ("oui ("+_mask+")" if _key_dbg else "non"))

    mode = st.radio("Mode d'entrée fiche projet", ["UPLOAD_DOC", "UPLOAD_JSON", "MANUAL"], horizontal=True)
    sp_file = None
    if mode in ("UPLOAD_DOC","UPLOAD_JSON"):
        sp_file = st.file_uploader("Fiche projet (PDF/DOCX/TXT ou JSON)", type=["pdf","docx","txt","json"])

    if st.button("Construire la spec (cellule 3)"):
        if mode == "MANUAL":
            spec = validate_fill_spec(DEFAULT_SPEC)
        elif not sp_file:
            st.error("Charge un fichier."); st.stop()
        elif sp_file.name.lower().endswith(".json"):
            raw = json.loads(sp_file.read().decode("utf-8", errors="ignore"))
            spec = validate_fill_spec(raw)
        else:
            text = read_text_generic_from_upload(sp_file)
            spec = gpt_build_spec_from_text(text, model_id=st.session_state["MODEL_ID"])

        spec = _renormalize_weights(spec)
        st.session_state["spec"] = spec
        st.success("✅ Spec chargée.")
        st.json(spec)
        st.download_button("Télécharger spec_active.json",
                           json.dumps(spec, ensure_ascii=False, indent=2).encode("utf-8"),
                           "spec_active.json", "application/json")

with tab2:
    spec = st.session_state.get("spec")
    if not spec:
        st.info("➡️ Onglet 1 : génère/charge la spec avant d'analyser des CV.")
    else:
        files = st.file_uploader("CV (PDF/DOCX/TXT) — multiples", type=["pdf","docx","txt"], accept_multiple_files=True)
        want_llm_comment = st.checkbox("Générer un commentaire RH (LLM)")
        if files and st.button("Analyser (cellules 4→9)"):
            rows=[]
            for idx, f in enumerate(files, start=1):
                if f.size > MAX_MB * 1024 * 1024:
                    st.error(f"{f.name} : > {MAX_MB} Mo"); continue

                t0 = time.time()
                cv_text = read_cv_text_from_upload(f)

                # Extraction (cache + fallback)
                cv_hash = _hash_text(cv_text)
                extraction = _extract_cached(cv_hash, cv_text, st.session_state["MODEL_ID"])
                extraction = enforce_evidence(extraction, cv_text)
                extraction = fill_with_regex_if_missing(extraction, cv_text)

                # Scoring
                pts_mn, evidences = score_competences_embeddings(cv_text, spec)
                pts_autres, com_autres, details_autres = score_autres_criteres(extraction, spec)
                score_final = round(min(100.0, pts_mn + pts_autres), 2)

                # Commentaire LLM (optionnel)
                comment_rh = ""
                if want_llm_comment and not st.session_state.get("FORCE_OFFLINE") and _get_openai_key():
                    try:
                        comment_rh = _chat_completion(
                            st.session_state["MODEL_ID"],
                            [{"role":"user","content":f"""Rôle: assistant RH. Commentaire factuel (5–7 lignes).
- Score final: {score_final} %
- Preuves must (top): {sorted(evidences.get('must',[]), key=lambda x: x[1], reverse=True)[:3]}
- Preuves nice (top): {sorted(evidences.get('nice',[]), key=lambda x: x[1], reverse=True)[:3]}
- Détails règles: {details_autres}
Contraintes: style FR pro, phrases courtes, terminer par une recommandation."""}],
                            temperature=0.2, max_tokens=220
                        ).strip()
                    except Exception as e:
                        st.warning(f"LLM indisponible pour le commentaire ({str(e)[:120]}).")

                rows.append({
                    "fichier": f.name,
                    "score_final": score_final,
                    "points_embeddings": pts_mn,
                    "points_regles": pts_autres,
                    "exp_years": _num(extraction.get("experience_ans")),
                    "commentaire": comment_rh
                })

                with st.expander(f"Détails — {f.name}"):
                    st.markdown("**Extraction (JSON)**"); st.json(extraction)
                    st.markdown("**Preuves sémantiques**"); st.write(evidences)
                    st.markdown("**Commentaire déterministe**")
                    st.text(build_commentaire_deterministe(score_final, evidences, extraction, spec))
                    if comment_rh: st.markdown("**Commentaire RH (LLM)**"); st.write(comment_rh)
                    st.caption(f"latence: {time.time()-t0:.2f}s — mode: {'embeddings' if USE_EMB else 'keyword-only'}")

                if idx < len(files):
                    time.sleep(LLM_MIN_DELAY)

            if rows:
                df = pd.DataFrame(rows).sort_values("score_final", ascending=False)
                st.subheader("Résultats")
                st.dataframe(df, use_container_width=True)
                st.download_button("Télécharger CSV", df.to_csv(index=False).encode("utf-8"),
                                   "resultats_cv.csv", "text/csv")








with tab3:
    st.write("**Cellule 10 : Test rapide**")
    spec_demo = _renormalize_weights(validate_fill_spec({
        "must_have": ["python","sql","power bi"],
        "nice_to_have": ["airflow","docker"],
        "experience_min_ans": 1,
        "langues": {"fr":"B2","en":"B1"},
        "poids": {"must_have":50,"nice_to_have":30,"experience":10,"langues":5,"diplomes_certifs":3,"localisation_dispo":2}
    }))
    cv_demo = """Mohamed Soulaimane — Data Analyst
Compétences: Python, SQL, Airflow, Docker, Power BI
Expérience: 2 ans en analytics et BI
Langues: FR (C1), EN (B2)
Localisation: Rabat-Salé
Disponibilité: 2 semaines
Certifications: Google Data Analytics
Diplômes: Licence Informatique (2021)
"""
    if st.button("Lancer le test"):
        ext = gpt_extract_profile_safe(cv_demo, model_id=st.session_state["MODEL_ID"])
        ext = enforce_evidence(ext, cv_demo)
        ext = fill_with_regex_if_missing(ext, cv_demo)
        pts_mn, evid = score_competences_embeddings(cv_demo, spec_demo)
        pts_autres, _, _ = score_autres_criteres(ext, spec_demo)
        score = round(min(100.0, pts_mn + pts_autres), 2)
        st.metric("SCORE_FINAL (demo)", f"{score} %")

















# === BRIDGE WORDPRESS — À COLLER À LA FIN DE app.py ===
import os, requests, streamlit as st

# 0) Si tu n'as PAS déjà une variable `result`, on en fabrique une minimale pour tester
if "result" not in locals():
    prediction = locals().get("prediction") or locals().get("label") or "OK"
    try:
        score = float(locals().get("score") or 0.87)
    except Exception:
        score = 0.87

    # Récupère quelques params de la page WP (optionnel)
    qp = getattr(st, "query_params", None)
    qp = dict(qp) if qp is not None else st.experimental_get_query_params()
    def _get(q, k, d):
        v = q.get(k, d)
        return (v[0] if isinstance(v, list) and v else v) or d
    lang = _get(qp, "lang", "fr")
    theme = _get(qp, "theme", "light")

    result = {"label": str(prediction), "score": score, "lang": lang, "theme": theme}
    st.json(result)  # affichage pour vérifier

# 1) Secrets / variables d'environnement (ne fait PAS crasher si absents)
WP_BASE  = os.getenv("WP_BASE")  or st.secrets.get("WP_BASE", "")
WP_TOKEN = os.getenv("WP_TOKEN") or st.secrets.get("WP_TOKEN", "")

# 2) Envoi vers WordPress uniquement si tout est configuré
if WP_BASE and WP_TOKEN:
    try:
        resp = requests.post(
            f"{WP_BASE}/wp-json/streamlit-bridge/v1/result",
            json=result,
            headers={"X-Streamlit-Token": WP_TOKEN},
            timeout=10
        )
        resp.raise_for_status()
        st.caption("Résultat envoyé à WordPress ✅")
    except Exception as e:
        st.caption(f"Envoi WordPress impossible : {e}")
else:
    st.caption("WP_BASE / WP_TOKEN non configurés — aucun envoi (l'app continue).")




