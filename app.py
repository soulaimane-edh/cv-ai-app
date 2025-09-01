# =========================================================
# APP STREAMLIT = REPRISE DU NOTEBOOK (10 CELLULES)
# =========================================================
import io
import os
import re
import json
import time
import streamlit as st
import pandas as pd
from pypdf import PdfReader
from docx import Document
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Config UI ----------
st.set_page_config(page_title="Analyse de CV (Notebook → App)", layout="wide")
st.title("Analyse de CV — reprise du notebook (10 cellules)")

# ---------- Cellule 1 : Setup & variables ----------
MODEL_ID = "gpt-4o-mini"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_SEED = 42
MAX_MB = int(st.secrets.get("limits", {}).get("MAX_FILE_MB", 5))
MAX_PGS = int(st.secrets.get("limits", {}).get("MAX_PAGES", 8))

# ---------- Helpers OpenAI (clé/clients) ----------
import os
import streamlit as st

def _get_openai_key() -> str:
    """Récupère la clé depuis Secrets ([llm] ou top-level) ou l'ENV. Toujours une str propre."""
    key = (st.secrets.get("llm", {}) or {}).get("OPENAI_API_KEY")
    key = key or st.secrets.get("OPENAI_API_KEY")
    key = key or os.getenv("OPENAI_API_KEY")
    if key is None:
        return ""
    if isinstance(key, bytes):
        key = key.decode("utf-8", "ignore")
    key = str(key).strip()
    return key

def _get_openai_client():
    """Valide la clé, la met dans l'ENV, puis crée le client sans arguments."""
    key = _get_openai_key()
    if not key or not isinstance(key, str) or not key.startswith("sk-"):
        st.error("OPENAI_API_KEY absente ou invalide. Ajoute-la dans Settings → Secrets (avec des guillemets).")
        raise RuntimeError("Invalid or missing OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = key  # le SDK la lira automatiquement
    from openai import OpenAI
    return OpenAI()  # <-- IMPORTANT: aucun argument ici


# ---------- Outils communs (lecture fichiers/texte) ----------
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

# ---------- Cellule 3 : FICHE PROJET -> JSON spec ----------
DEFAULT_SPEC = {
    "must_have": [],
    "nice_to_have": [],
    "experience_min_ans": 0,
    "langues": {},
    "diplomes": [],
    "certifications": [],
    "localisation": "",
    "disponibilite_max_semaines": 4,
    "poids": {
        "must_have": 40,
        "nice_to_have": 15,
        "experience": 15,
        "langues": 10,
        "diplomes_certifs": 10,
        "localisation_dispo": 10,
    },
}

def validate_fill_spec(s: dict) -> dict:
    import copy
    spec_v = copy.deepcopy(DEFAULT_SPEC)
    for k in spec_v:
        if k in s:
            spec_v[k] = s[k]
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
    try:
        spec_v["experience_min_ans"] = int(spec_v.get("experience_min_ans", 0))
    except Exception:
        spec_v["experience_min_ans"] = 0
    try:
        spec_v["disponibilite_max_semaines"] = int(spec_v.get("disponibilite_max_semaines", 4))
    except Exception:
        spec_v["disponibilite_max_semaines"] = 4
    return spec_v

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
    client = _get_openai_client()
    model_id = model_id or MODEL_ID

    msgs = [
        {"role": "system", "content": SPEC_SYSTEM},
        {"role": "user", "content": fiche_texte},
    ]

    r = client.chat.completions.create(
        model=model_id, messages=msgs, temperature=0, max_tokens=700
    )
    txt = r.choices[0].message.content.strip()
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if not m:
        raise ValueError("JSON non trouvé dans la réponse du modèle.")
    raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception:
        cleaned = re.sub(r",\s*}", "}", raw)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        return json.loads(cleaned)

# ---------- Cellule 4 : LECTURE CV (via upload) ----------
def read_cv_text_from_upload(file) -> str:
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
    return clean_text_soft(txt)

# ---------- Cellule 5 : Extraction safe (profil) ----------
def gpt_extract_profile_safe(cv_text: str, model_id: str = MODEL_ID) -> dict:
    # si pas de clé, renvoyer un squelette vide (fallback regex ensuite)
    if not _get_openai_key():
        return {
            "experience_ans": {"value": None, "evidence": []},
            "disponibilite_semaines": {"value": None, "evidence": []},
            "langues": [],
            "diplomes_obtenus": [],
            "diplomes_en_cours": [],
            "certifications": [],
            "localisation": {"value": "", "evidence": []},
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
    RÈGLES :
    - N'INVENTE RIEN. Si l'info n'est pas explicitement écrite -> value null/"" ou liste vide.
    - Les 'evidence.text' doivent être des sous-chaînes EXACTES du CV, et 'start/end' leurs index caractères.
    - Diplômes : mets dans 'diplomes_obtenus' SEULEMENT les diplômes déjà obtenus (mention 'diplômé/obtenu/graduated' ou année).
      Sinon -> 'diplomes_en_cours'. Ne déduis pas Master/Ingénieur/Bac+5 à partir d'un cycle en cours.
    """
    client = _get_openai_client()
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": cv_text[:120000]}]
    r = client.chat.completions.create(model=model_id, messages=msgs, temperature=0, max_tokens=900)
    raw = r.choices[0].message.content.strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    js = m.group(0) if m else "{}"
    try:
        data = json.loads(js)
    except Exception:
        js = re.sub(r",\s*}", "}", js)
        js = re.sub(r",\s*]", "]", js)
        data = json.loads(js or "{}")

    # normalisation minimale
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
        if not val:
            return []
        i = cv_text.lower().find(str(val).lower())
        if i < 0:
            return []
        return [{"text": cv_text[i : i + len(str(val))], "start": i, "end": i + len(str(val))}]
    for k in ("experience_ans", "disponibilite_semaines", "localisation"):
        slot = extraction.get(k, {})
        val = slot.get("value") if isinstance(slot, dict) else None
        if val and not slot.get("evidence"):
            slot["evidence"] = _ev_for(val)
            extraction[k] = slot
    return extraction

def fill_with_regex_if_missing(extraction: dict, cv_text: str) -> dict:
    if not extraction.get("experience_ans", {}).get("value"):
        m = re.search(r"(\d+)\s*(ans|year|years)", cv_text, flags=re.I)
        if m:
            v = float(m.group(1))
            extraction["experience_ans"] = {"value": v, "evidence": []}
    if not extraction.get("disponibilite_semaines", {}).get("value"):
        m = re.search(r"(\d+)\s*(semaines?|weeks?)", cv_text, flags=re.I)
        if m:
            v = float(m.group(1))
            extraction["disponibilite_semaines"] = {"value": v, "evidence": []}
    return extraction

# ---------- Cellule 6 : Embeddings + scoring compétences ----------
USE_EMB = st.toggle("Activer embeddings (S-BERT) — nécessite torch", value=False)

@st.cache_resource
def get_emb_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMB_MODEL)

def score_competences_embeddings(cv_text: str, spec: dict, emb_model=None):
    must = [s.strip() for s in spec.get("must_have", []) if s.strip()]
    nice = [s.strip() for s in spec.get("nice_to_have", []) if s.strip()]
    evid = {"must": [], "nice": []}

    # Embeddings si emb_model fourni, sinon fallback mots-clés
    if emb_model is not None:
        from sentence_transformers import util
        cv_vec = emb_model.encode(cv_text, normalize_embeddings=True)

        def sim(skill: str) -> float:
            v = emb_model.encode(skill, normalize_embeddings=True)
            return float(util.cos_sim(cv_vec, v).item())

        ms = 0.0
        for sk in must:
            s = sim(sk)
            evid["must"].append((sk, s, sk if s >= 0.25 else ""))
            ms += s
        ns = 0.0
        for sk in nice:
            s = sim(sk)
            evid["nice"].append((sk, s, sk if s >= 0.22 else ""))
            ns += s

        pts = 0.0
        if must:
            pts += min(1.0, ms / len(must)) * 60
        if nice:
            pts += min(1.0, ns / len(nice)) * 40
        return round(pts, 2), evid

    # Fallback : présence des mots (léger)
    low = cv_text.lower()
    mh = sum(1 for sk in must if sk.lower() in low)
    nh = sum(1 for sk in nice if sk.lower() in low)
    for sk in must:
        evid["must"].append((sk, 1.0 if sk.lower() in low else 0.0, sk if sk.lower() in low else ""))
    for sk in nice:
        evid["nice"].append((sk, 1.0 if sk.lower() in low else 0.0, sk if sk.lower() in low else ""))
    pts = 0.0
    if must:
        pts += (mh / len(must)) * 60
    if nice:
        pts += (nh / len(nice)) * 40
    return round(pts, 2), evid

# ---------- Cellule 8 : Règles & commentaire ----------
def score_autres_criteres(extract: dict, spec: dict):
    p = spec.get("poids", {})
    w_exp = p.get("experience", 10)
    w_lang = p.get("langues", 5)
    w_dc = p.get("diplomes_certifs", 3)
    w_loc = p.get("localisation_dispo", 2)
    score = 0.0
    details = {}

    need = spec.get("experience_min_ans", None)
    yrs = extract.get("experience_ans", {}).get("value")
    if isinstance(need, (int, float)) and isinstance(yrs, (int, float)):
        part = min(1.0, yrs / max(1, need)) * w_exp
        score += part
        details["experience"] = round(part, 2)

    have = {l.get("code"): (l.get("niveau") or "").upper() for l in extract.get("langues", [])}
    need_langs = spec.get("langues", {})
    if isinstance(need_langs, dict) and need_langs:
        ok = 0
        tot = 0
        for code in need_langs.keys():
            tot += 1
            if code.lower() in (have or {}):
                ok += 1
        if tot:
            part = (ok / tot) * w_lang
            score += part
            details["langues"] = round(part, 2)

    d = len(extract.get("diplomes_obtenus", []) or [])
    c = len(extract.get("certifications", []) or [])
    if d + c > 0:
        score += w_dc
        details["diplomes_certifs"] = w_dc

    dmax = spec.get("disponibilite_max_semaines", None)
    dval = extract.get("disponibilite_semaines", {}).get("value")
    if isinstance(dmax, (int, float)) and isinstance(dval, (int, float)) and dval <= dmax:
        score += w_loc
        details["localisation_dispo"] = w_loc

    com = "Règles: exp/langues/diplômes/certifs/dispo agrégés."
    return round(score, 2), com, details

def build_commentaire_deterministe(score_final, evidences, extraction, spec):
    def _fmt_langues(lang_list):
        if not lang_list:
            return "non mentionnées"
        parts = []
        for it in lang_list:
            code = (it.get("code") or "").upper()
            lvl = (it.get("niveau") or "").upper()
            parts.append(f"{code} ({lvl})" if code and lvl else code)
        return ", ".join([p for p in parts if p]) or "non mentionnées"

    target_exp = spec.get("experience_min_ans", None)
    target_txt = f"{target_exp} an(s)" if isinstance(target_exp, (int, float)) else "non précisé"
    exp_val = extraction.get("experience_ans", {}).get("value")
    exp_txt = "non mentionnée" if exp_val is None else f"{exp_val:.0f} an(s)"
    dispo_val = extraction.get("disponibilite_semaines", {}).get("value")
    dispo_txt = f"{dispo_val} semaine(s)" if isinstance(dispo_val, (int, float)) else "non mentionnée"
    loc_val = extraction.get("localisation", {}).get("value") or ""
    loc_txt = loc_val if loc_val.strip() else "non mentionnée"
    dipl_txt = ", ".join([d.get("label", "") for d in extraction.get("diplomes_obtenus", [])]) or "non mentionnés"
    certs_txt = ", ".join([c.get("label", "") for c in extraction.get("certifications", [])]) or "non mentionnées"

    def decision_band(score):
        return (
            "Très bon match – à convoquer"
            if score >= 85
            else "Bon match – à prioriser"
            if score >= 70
            else "Moyen – à examiner"
            if score >= 55
            else "Faible – non prioritaire"
        )

    def _best(evid, kind="must", k=4, thr=0.55):
        rows = [(sk, float(sim), phr) for (sk, sim, phr) in evid.get(kind, [])]
        rows.sort(key=lambda r: r[1], reverse=True)
        rows = [r for r in rows if r[1] >= thr]
        return rows[:k]

    bullets = []
    for (sk, sim, phr) in _best(evidences, "must", 4, 0.55) + _best(evidences, "nice", 4, 0.50):
        bullets.append(f"• {sk} (sim={sim:.3f}) — « {phr[:120]}… »" if phr else f"• {sk} (sim={sim:.3f})")

    com = []
    com.append(f"Le candidat obtient un score final de {score_final:.2f} %.") 
    com.append("Compétences évaluées :")
    com.extend(bullets if bullets else ["• Aucune preuve forte détectée."])
    com.append(f"Expérience : {exp_txt} (objectif fiche : {target_txt}).")
    com.append(f"Langues : {_fmt_langues(extraction.get('langues', []))}.")
    com.append(f"Diplômes : {dipl_txt}. Certifications : {certs_txt}.")
    com.append(f"Localisation : {loc_txt}. Disponibilité : {dispo_txt}.")
    com.append(f"Recommandation automatique : {decision_band(score_final)}.")
    return "\n".join(com)

# ---------- Cellule 9 : Commentaire RH (LLM, option) ----------
def gpt_commentaire(score: float, evidences, details_autres, model_id: str = MODEL_ID) -> str:
    if not _get_openai_key():
        return "(LLM inactif : ajouter OPENAI_API_KEY dans Secrets)"
    def pick(ev, limit=3, thr=0.55):
        cands = [
            f"{skill} (sim={sim:.3f}) → « {phr[:120]}... »"
            for (skill, sim, phr) in sorted(ev, key=lambda x: x[1], reverse=True)
            if sim >= thr
        ][:limit]
        return cands
    top_must = pick(evidences.get("must", []))
    top_nice = pick(evidences.get("nice", []))
    prompt = f"""Rôle: assistant RH. Commentaire factuel et concis (5–7 lignes).
Données:
- Score final: {score} %
- Preuves must (top): {top_must}
- Preuves nice (top): {top_nice}
- Détails règles: {details_autres}
Contraintes: Style pro, FR, phrases courtes. Conclure par une recommandation."""
    client = _get_openai_client()
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=220,
    )
    return resp.choices[0].message.content.strip()

# ---------- UI (onglets) ----------
tab1, tab2, tab3 = st.tabs(["1) Fiche projet → spec", "2) Analyse CV", "3) Démo (cellule 10)"])

with tab1:
        _key_dbg = _get_openai_key()
    _mask = (_key_dbg[:3] + "…" + _key_dbg[-4:]) if _key_dbg else "—"
    st.caption("🔐 Clé OpenAI détectée : " + ("oui (" + _mask + ")" if _key_dbg else "non"))

    st.caption("🔐 Clé OpenAI chargée : " + ("oui" if _get_openai_key() else "non"))
    mode = st.radio("Mode d'entrée fiche projet", ["UPLOAD_DOC", "UPLOAD_JSON", "MANUAL"], horizontal=True)
    sp_file = None
    if mode in ("UPLOAD_DOC", "UPLOAD_JSON"):
        sp_file = st.file_uploader("Fiche projet (PDF/DOCX/TXT ou JSON)", type=["pdf", "docx", "txt", "json"])

    if st.button("Construire la spec (cellule 3)"):
        if mode == "MANUAL":
            spec = validate_fill_spec({
                "must_have": ["Python", "SQL", "Spark", "Airflow"],
                "nice_to_have": ["AWS", "Docker", "Kubernetes"],
                "experience_min_ans": 3,
                "langues": {"fr": "C1", "en": "B2"},
                "diplomes": [],
                "certifications": [],
                "localisation": "Casablanca",
                "disponibilite_max_semaines": 4,
                "poids": {"must_have": 40, "nice_to_have": 15, "experience": 15, "langues": 10, "diplomes_certifs": 10, "localisation_dispo": 10},
            })
        elif not sp_file:
            st.error("Charge un fichier."); st.stop()
        elif sp_file.name.lower().endswith(".json"):
            raw = json.loads(sp_file.read().decode("utf-8", errors="ignore"))
            spec = validate_fill_spec(raw)
        else:
            if not _get_openai_key():
                st.error("Pour construire la spec depuis PDF/DOCX/TXT, ajoute OPENAI_API_KEY dans Secrets ou utilise MANUAL/UPLOAD_JSON.")
                st.stop()
            text = read_text_generic_from_upload(sp_file)
            spec = validate_fill_spec(gpt_build_spec_from_text(text))

        st.session_state["spec"] = spec
        st.success("✅ Spec chargée en mémoire (session_state['spec']).")
        st.json(spec)
        st.download_button(
            "Télécharger spec_active.json",
            json.dumps(spec, ensure_ascii=False, indent=2).encode("utf-8"),
            "spec_active.json",
            "application/json",
        )

with tab2:
    spec = st.session_state.get("spec")
    if not spec:
        st.info("➡️ Va d'abord dans l'onglet 1 pour générer/charger la fiche projet (spec).")
    else:
        files = st.file_uploader("CV (PDF/DOCX/TXT) — multiples autorisés", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        want_llm_comment = st.checkbox("Générer un commentaire RH (LLM)")
        if files and st.button("Analyser (cellules 4→9)"):
            rows = []
            for f in files:
                if f.size > MAX_MB * 1024 * 1024:
                    st.error(f"{f.name} : trop volumineux (> {MAX_MB} Mo)")
                    continue
                t0 = time.time()
                cv_text = read_cv_text_from_upload(f)
                # cellule 5
                extraction = gpt_extract_profile_safe(cv_text, model_id=MODEL_ID)
                extraction = enforce_evidence(extraction, cv_text)
                extraction = fill_with_regex_if_missing(extraction, cv_text)
                # cellule 6
                emb_model = get_emb_model() if USE_EMB else None
                pts_mn, evidences = score_competences_embeddings(cv_text, spec, emb_model)
                # cellule 8
                pts_autres, com_autres, details_autres = score_autres_criteres(extraction, spec)
                score_final = round(pts_mn + pts_autres, 2)
                # cellule 9 (option)
                comment_rh = gpt_commentaire(score_final, evidences, details_autres, MODEL_ID) if want_llm_comment else ""
                rows.append({
                    "fichier": f.name,
                    "score_final": score_final,
                    "points_embeddings": pts_mn,
                    "points_regles": pts_autres,
                    "exp_years": extraction.get("experience_ans", {}).get("value"),
                    "commentaire": comment_rh,
                })
                with st.expander(f"Détails — {f.name}"):
                    st.markdown("**Extraction (JSON)**"); st.json(extraction)
                    st.markdown("**Preuves sémantiques (must/nice)**"); st.write(evidences)
                    st.markdown("**Commentaire déterministe**"); st.text(build_commentaire_deterministe(score_final, evidences, extraction, spec))
                    if comment_rh: st.markdown("**Commentaire RH (LLM)**"); st.write(comment_rh)
                    st.caption(f"latence: {time.time()-t0:.2f}s — méthode: {'embeddings' if USE_EMB else 'fallback/keywords'}")

            if rows:
                df = pd.DataFrame(rows).sort_values("score_final", ascending=False)
                st.subheader("Résultats")
                st.dataframe(df, use_container_width=True)
                st.download_button("Télécharger CSV", df.to_csv(index=False).encode("utf-8"), "resultats_cv.csv", "text/csv")

with tab3:
    st.write("**Cellule 10 : Test rapide sans upload**")
    spec_demo = {
        "must_have": ["python", "sql", "power bi"],
        "nice_to_have": ["airflow", "docker"],
        "experience_min_ans": 1,
        "langues": {"fr": "B2", "en": "B1"},
        "diplomes": [],
        "certifications": [],
        "disponibilite_max_semaines": 4,
        "poids": {"must_have": 50, "nice_to_have": 30, "experience": 10, "langues": 5, "diplomes_certifs": 3, "localisation_dispo": 2},
    }
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
        pts_mn, evid = score_competences_embeddings(cv_demo, spec_demo, get_emb_model() if USE_EMB else None)
        pts_autres, com_autres, det_autres = score_autres_criteres(ext, spec_demo)
        score = round(pts_mn + pts_autres, 2)
        st.metric("SCORE_FINAL (demo)", f"{score} %")
        st.text(build_commentaire_deterministe(score, evid, ext, spec_demo))
