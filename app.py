# =========================================================
# APP STREAMLIT = Notebook (10 cellules) → Application
# Scoring identique au Colab (cellules 6 & 8)
# - Embeddings phrase-level + aliases + boost + sigmoïde
# - Poids spec utilisés partout (somme = 100)
# - Fallback sans embeddings (RapidFuzz)
# - Appel OpenAI par HTTP (requests) pour la construction de spec/extractions
# =========================================================
import io
import os
import re
import json
import time
import math
import requests
import streamlit as st
import pandas as pd
from typing import Dict, Any, Tuple, List
from pypdf import PdfReader
from docx import Document

# ----------------- Config UI -----------------
st.set_page_config(page_title="Analyse de CV (Notebook → App)", layout="wide")
st.title("Analyse de CV — reprise fidèle du notebook")

# ----------------- Constantes -----------------
MODEL_ID  = "gpt-4o-mini"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_SEED = 42

MAX_MB  = int(st.secrets.get("limits", {}).get("MAX_FILE_MB", 5))
MAX_PGS = int(st.secrets.get("limits", {}).get("MAX_PAGES", 8))

# ----------------- OpenAI (clé + HTTP) -----------------
def _get_openai_key() -> str:
    key = (st.secrets.get("llm", {}) or {}).get("OPENAI_API_KEY")
    key = key or st.secrets.get("OPENAI_API_KEY")
    key = key or os.getenv("OPENAI_API_KEY")
    return (str(key).strip() if key else "")

def _chat_completion(model: str, messages: list, temperature: float = 0, max_tokens: int = 700) -> str:
    key = _get_openai_key()
    if not key or not key.startswith("sk-"):
        raise RuntimeError("OPENAI_API_KEY absente ou invalide (Settings → Secrets).")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenAI HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]

# ----------------- Lecture fichiers -----------------
def _extract_text_pdf_bytes(b: bytes, max_pages=MAX_PGS) -> str:
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

# ----------------- Poids spec (somme = 100) -----------------
DEFAULT_SPEC = {
    "must_have": [], "nice_to_have": [],
    "experience_min_ans": 0,
    "langues": {}, "diplomes": [], "certifications": [],
    "localisation": "", "disponibilite_max_semaines": 4,
    "poids": {
        "must_have": 40, "nice_to_have": 15, "experience": 15,
        "langues": 10, "diplomes_certifs": 10, "localisation_dispo": 10
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

# -------- Cellule 3 : Spec via GPT (identique) --------
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
- Valeurs réalistes (ex: expérience 0..20).
- Si info absente → mets une valeur par défaut raisonnable.
- Les poids doivent approx. sommer 100 (±5).
- Réponds en UNE SEULE structure JSON, sans texte autour.
"""

def gpt_build_spec_from_text(fiche_texte: str, model_id: str = None) -> dict:
    model_id = model_id or MODEL_ID
    msgs = [{"role":"system","content":SPEC_SYSTEM},
            {"role":"user","content":fiche_texte}]
    txt = _chat_completion(model_id, msgs, temperature=0, max_tokens=700).strip()
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if not m: raise ValueError("JSON non trouvé dans la réponse du modèle.")
    raw = m.group(0)
    try: return json.loads(raw)
    except Exception:
        cleaned = re.sub(r",\s*}", "}", raw)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        return json.loads(cleaned)

# ----------------- Cellule 4 : Lecture CV -----------------
def read_cv_text_from_upload(file) -> str:
    name = file.name.lower()
    raw  = file.read()
    if name.endswith(".pdf"):  txt = _extract_text_pdf_bytes(raw)
    elif name.endswith(".docx"): txt = _extract_text_docx_bytes(raw)
    elif name.endswith(".txt"):  txt = raw.decode("utf-8", errors="ignore")
    else: raise ValueError("Format non supporté (PDF/DOCX/TXT).")
    return clean_text_soft(txt)

# ----------------- Cellule 5 : Extraction safe (LLM) -----------------
def gpt_extract_profile_safe(cv_text: str, model_id: str = MODEL_ID) -> dict:
    # Si pas de clé → squelette vide, puis regex
    if not _get_openai_key():
        return {
            "experience_ans": {"value": None, "evidence": []},
            "disponibilite_semaines": {"value": None, "evidence": []},
            "langues": [], "diplomes_obtenus": [], "diplomes_en_cours": [],
            "certifications": [], "localisation": {"value": "", "evidence": []}
        }
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
    - N'INVENTE RIEN. Si l'info n'est pas explicite -> null/"" ou vide.
    """
    msgs = [{"role":"system","content":SYSTEM},
            {"role":"user","content":cv_text[:120000]}]
    raw = _chat_completion(model_id, msgs, temperature=0, max_tokens=900).strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    js = m.group(0) if m else "{}"
    try: data = json.loads(js)
    except Exception:
        js = re.sub(r",\s*}", "}", js); js = re.sub(r",\s*]", "]", js)
        data = json.loads(js or "{}")
    # normalisation légère
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

# ----------------- Cellule 6 (du notebook) -----------------
USE_EMB = st.toggle("Activer embeddings (S-BERT) — nécessite torch", value=False)

@st.cache_resource
def get_emb_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMB_MODEL)

# Aliases (extrait du notebook)
SKILL_ALIASES = {
    "python": ["python"],
    "sql": ["sql","t-sql","postgresql","mysql","athena","redshift"],
    "spark": ["spark","pyspark","apache spark"],
    "airflow": ["airflow","apache airflow","dag","dags"],
    "aws": ["aws","amazon web services","s3","emr","glue","athena","redshift"],
    "docker": ["docker","container","conteneurisation"],
    "kubernetes": ["kubernetes","k8s"],
}

def expand_terms(skill: str) -> List[str]:
    return SKILL_ALIASES.get(skill.lower().strip(), [skill])

def split_sentences(text: str) -> List[str]:
    # split simple : lignes + ponctuation forte
    raw = re.split(r"[\n\.!\?;]+", text)
    # nettoyage et filtre phrases non vides
    sents = [re.sub(r"\s+", " ", s).strip() for s in raw]
    return [s for s in sents if s]

def keyword_hit(aliases: List[str], text: str) -> bool:
    t = text.lower()
    for a in aliases:
        a = a.lower().strip()
        if re.search(rf"(?<![a-z0-9_]){re.escape(a)}(?![a-z0-9_])", t):
            return True
    return False

BOOST_ADD = 0.15     # +0.15 si mot-clé trouvé
BOOST_CAP = 0.85     # plafond 0.85

def best_sim(term: str, sentences: List[str], sent_embs, emb_model) -> Tuple[float, str]:
    """Meilleure similarité phrase pour 'term' + phrase associée.
       - avec embeddings si dispo
       - fallback RapidFuzz sinon
    """
    if emb_model is not None and sent_embs is not None:
        from sentence_transformers import util
        v = emb_model.encode(term, normalize_embeddings=True)
        sims = util.cos_sim(sent_embs, v).squeeze(1).tolist()
        idx = max(range(len(sims)), key=lambda i: sims[i]) if sims else 0
        return float(sims[idx]) if sims else 0.0, (sentences[idx] if sims else "")
    else:
        # fallback : RapidFuzz
        from rapidfuzz import fuzz
        best = (0.0, "")
        for s in sentences:
            sc = fuzz.partial_ratio(term.lower(), s.lower()) / 100.0
            if sc > best[0]: best = (sc, s)
        return best

def best_sim_skill(skill: str, sentences: List[str], full_text: str, sent_embs, emb_model) -> Tuple[float, str]:
    aliases = expand_terms(skill)
    best = (0.0, "")
    for term in aliases:
        sim, phr = best_sim(term, sentences, sent_embs, emb_model)
        if sim > best[0]: best = (sim, phr)
    if keyword_hit(aliases, full_text):
        boosted = min(BOOST_CAP, best[0] + BOOST_ADD)
        best = (boosted, best[1])
    return best

def map_points(sim: float, max_pts: float, t0=0.35, t1=0.85) -> float:
    if sim <= t0: return 0.0
    if sim >= t1: return max_pts
    x = (sim - t0) / (t1 - t0)
    return max_pts * (1 / (1 + math.exp(-10 * (x - 0.5))))

def score_competences_embeddings(cv_text: str, spec: Dict[str, Any],
                                 seuil_must=0.60, seuil_nice=0.50
                                 ) -> Tuple[float, Dict[str, List[Tuple[str, float, str]]]]:
    P = spec["poids"]
    sents = split_sentences(cv_text)
    emb_model = get_emb_model() if USE_EMB else None
    sent_embs = None
    if emb_model is not None:
        sent_embs = emb_model.encode(sents, normalize_embeddings=True)

    total = 0.0
    proofs = {"must": [], "nice": []}

    # MUST
    must = [s for s in spec.get("must_have", []) if s.strip()]
    if must:
        part = P["must_have"] / len(must)
        for skill in must:
            sim, phr = best_sim_skill(skill, sents, cv_text, sent_embs, emb_model)
            if sim >= seuil_must:
                total += map_points(sim, part)
            proofs["must"].append((skill, round(sim, 3), phr))

    # NICE
    nice = [s for s in spec.get("nice_to_have", []) if s.strip()]
    if nice:
        part = P["nice_to_have"] / len(nice)
        for skill in nice:
            sim, phr = best_sim_skill(skill, sents, cv_text, sent_embs, emb_model)
            if sim >= seuil_nice:
                total += map_points(sim, part)
            proofs["nice"].append((skill, round(sim, 3), phr))

    return round(total, 2), proofs

# ----------------- Cellule 8 : Règles (identique au notebook) -----------------
ORDER_CEFR = {"A1":1,"A2":2,"B1":3,"B2":4,"C1":5,"C2":6}

def _num(slot, default=None):
    if isinstance(slot, dict): slot = slot.get("value", None)
    try: return float(slot)
    except (TypeError, ValueError): return default

def _str(slot):
    if isinstance(slot, dict): return slot.get("value", "")
    return slot or ""

def score_autres_criteres(ex: Dict[str, Any], spec: Dict[str, Any]) -> Tuple[float, str, Dict[str, Any]]:
    if not ex or "error" in ex:
        return 0.0, f"Extraction invalide ({ex.get('error','?')})", {"ok": False}

    P = spec["poids"]
    pts = 0.0

    # Expérience
    need = int(spec.get("experience_min_ans", 0) or 0)
    yrs  = _num(ex.get("experience_ans"), 0) or 0
    pts += P["experience"] * (min(1.0, yrs / max(1, need)) if need else 1.0)

    # Langues (niveau ≥ requis)
    lang_req = spec.get("langues", {}) or {}
    if lang_req:
        per_lang = P["langues"] / len(lang_req)
        for code, req in lang_req.items():
            have = next((d.get("niveau") for d in ex.get("langues", []) if (d.get("code") or "").lower()==code.lower()), None)
            if have and ORDER_CEFR.get((have or "").upper(),0) >= ORDER_CEFR.get((req or "").upper(),0):
                pts += per_lang

    # Diplômes/Certif : au moins un
    dipl_ok = bool(ex.get("diplomes_obtenus") or ex.get("diplomes") or ex.get("certifications"))
    if dipl_ok: pts += P["diplomes_certifs"]

    # Localisation (1/2 du poids)
    ex_loc = _str(ex.get("localisation")).lower()
    wanted = (spec.get("localisation") or "").lower()
    loc_ok = True
    if "remote" not in wanted and wanted.strip():
        loc_ok = any(city.strip().lower() in ex_loc for city in wanted.split("|") if city.strip())
    if loc_ok: pts += P["localisation_dispo"] / 2

    # Disponibilité (1/2 du poids)
    dmax = spec.get("disponibilite_max_semaines", 4)
    dval = _num(ex.get("disponibilite_semaines"), 10**6)
    if isinstance(dval, (int, float)) and dval <= dmax:
        pts += P["localisation_dispo"] / 2

    com = "Règles: exp/langues/diplômes+certifs/loc/dispo pondérées par la spec."
    return round(pts, 2), com, {"ok": True}

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

    def _best(evid, kind="must", k=4, thr=0.55):
        rows=[(sk,float(sim),phr) for (sk,sim,phr) in evid.get(kind,[])]
        rows.sort(key=lambda r:r[1], reverse=True)
        rows=[r for r in rows if r[1] >= thr]
        return rows[:k]

    bullets = []
    for (sk,sim,phr) in _best(evidences,"must",4,0.55)+_best(evidences,"nice",4,0.50):
        bullets.append(f"• {sk} (sim={sim:.3f}) — « {phr[:120]}… »" if phr else f"• {sk} (sim={sim:.3f})")

    com = []
    com.append(f"Score final : {score_final:.2f} %.")
    com.append("Compétences :"); com.extend(bullets if bullets else ["• Aucune preuve forte."])
    com.append(f"Expérience : {exp_txt} (objectif : {target_txt}).")
    com.append(f"Langues : {_fmt_langues(extraction.get('langues', []))}.")
    com.append(f"Diplômes : {dipl_txt}. Certifs : {certs_txt}.")
    com.append(f"Localisation : {loc_txt}. Disponibilité : {dispo_txt}.")
    com.append(f"Recommandation : {decision_band(score_final)}.")
    return "\n".join(com)

# ----------------- Cellule 9 : Commentaire RH (LLM, option) -----------------
def gpt_commentaire(score: float, evidences, details_autres, model_id: str = MODEL_ID) -> str:
    if not _get_openai_key(): return "(LLM inactif : ajouter OPENAI_API_KEY)"
    def pick(ev, limit=3, thr=0.55):
        cands = [f"{skill} (sim={sim:.3f}) → « {phr[:120]}... »"
                 for (skill, sim, phr) in sorted(ev, key=lambda x: x[1], reverse=True)
                 if sim >= thr][:limit]
        return cands
    top_must = pick(evidences.get("must",[]))
    top_nice = pick(evidences.get("nice",[]))
    prompt = f"""Rôle: assistant RH. Commentaire factuel concis (5–7 lignes).
- Score final: {score} %
- Must: {top_must}
- Nice: {top_nice}
- Détails règles: {details_autres}"""
    msgs = [{"role":"user","content":prompt}]
    return _chat_completion(model_id, msgs, temperature=0.2, max_tokens=220).strip()

# ----------------- UI -----------------
tab1, tab2, tab3 = st.tabs(["1) Fiche projet → spec", "2) Analyse CV", "3) Démo (cellule 10)"])

with tab1:
    _key_dbg = _get_openai_key(); _mask = (_key_dbg[:3]+"…"+_key_dbg[-4:]) if _key_dbg else "—"
    st.caption("🔐 Clé OpenAI : " + ("oui ("+_mask+")" if _key_dbg else "non"))

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
            if not _get_openai_key():
                st.error("Ajoute OPENAI_API_KEY dans Secrets ou utilise MANUAL/UPLOAD_JSON.")
                st.stop()
            text = read_text_generic_from_upload(sp_file)
            spec = validate_fill_spec(gpt_build_spec_from_text(text))

        st.session_state["spec"] = spec
        st.success("✅ Spec chargée.")
        st.json(spec)
        st.download_button("Télécharger spec_active.json",
                           json.dumps(spec, ensure_ascii=False, indent=2).encode("utf-8"),
                           "spec_active.json", "application/json")

with tab2:
    spec = st.session_state.get("spec")
    if not spec:
        st.info("➡️ Onglet 1 : génère/charge la spec.")
    else:
        files = st.file_uploader("CV (PDF/DOCX/TXT) — multiples", type=["pdf","docx","txt"], accept_multiple_files=True)
        want_llm_comment = st.checkbox("Générer un commentaire RH (LLM)")
        if files and st.button("Analyser (cellules 4→9)"):
            rows=[]
            for f in files:
                if f.size > MAX_MB * 1024 * 1024:
                    st.error(f"{f.name} : > {MAX_MB} Mo"); continue
                t0=time.time()
                cv_text = read_cv_text_from_upload(f)
                extraction = gpt_extract_profile_safe(cv_text, model_id=MODEL_ID)
                extraction = enforce_evidence(extraction, cv_text)
                extraction = fill_with_regex_if_missing(extraction, cv_text)

                pts_mn, evidences = score_competences_embeddings(cv_text, spec)
                pts_autres, com_autres, details_autres = score_autres_criteres(extraction, spec)
                score_final = round(min(100.0, pts_mn + pts_autres), 2)

                comment_rh = gpt_commentaire(score_final, evidences, details_autres, MODEL_ID) if want_llm_comment else ""
                rows.append({
                    "fichier": f.name, "score_final": score_final,
                    "points_embeddings": pts_mn, "points_regles": pts_autres,
                    "exp_years": _num(extraction.get("experience_ans")),
                    "commentaire": comment_rh
                })

                with st.expander(f"Détails — {f.name}"):
                    st.markdown("**Extraction (JSON)**"); st.json(extraction)
                    st.markdown("**Preuves sémantiques**"); st.write(evidences)
                    st.markdown("**Commentaire déterministe**")
                    st.text(build_commentaire_deterministe(score_final, evidences, extraction, spec))
                    if comment_rh: st.markdown("**Commentaire RH (LLM)**"); st.write(comment_rh)
                    st.caption(f"latence: {time.time()-t0:.2f}s — mode: {'embeddings' if USE_EMB else 'fallback (RapidFuzz)'}")

            if rows:
                df = pd.DataFrame(rows).sort_values("score_final", ascending=False)
                st.subheader("Résultats")
                st.dataframe(df, use_container_width=True)
                st.download_button("Télécharger CSV", df.to_csv(index=False).encode("utf-8"),
                                   "resultats_cv.csv", "text/csv")

with tab3:
    st.write("**Cellule 10 : Test rapide**")
    spec_demo = validate_fill_spec({
        "must_have": ["python","sql","power bi"],
        "nice_to_have": ["airflow","docker"],
        "experience_min_ans": 1,
        "langues": {"fr":"B2","en":"B1"},
        "poids": {"must_have":50,"nice_to_have":30,"experience":10,"langues":5,"diplomes_certifs":3,"localisation_dispo":2}
    })
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
        ext = gpt_extract_profile_safe(cv_demo, model_id=MODEL_ID)
        ext = enforce_evidence(ext, cv_demo)
        ext = fill_with_regex_if_missing(ext, cv_demo)
        pts_mn, evid = score_competences_embeddings(cv_demo, spec_demo)
        pts_autres, _, _ = score_autres_criteres(ext, spec_demo)
        score = round(min(100.0, pts_mn + pts_autres), 2)
        st.metric("SCORE_FINAL (demo)", f"{score} %")
