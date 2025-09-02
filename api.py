# api.py
import os, io, re, json, base64, time, hashlib
from typing import List, Dict, Any, Tuple, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests

# =========================
# Configuration / constantes
# =========================
MODEL_ID_DEFAULT = os.getenv("OPENAI_MODEL_ID", "gpt-5-mini")
EMB_MODEL        = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
USE_EMB          = os.getenv("USE_EMB", "0") in ("1", "true", "True", "yes", "YES")

MAX_PAGES  = int(os.getenv("MAX_PAGES", "8"))  # pages PDF max pour extraction
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "5"))

# =========================
# Aides I/O & texte
# =========================
def clean_text_soft(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t

def _extract_text_pdf_bytes(b: bytes, max_pages: int = MAX_PAGES) -> str:
    from pypdf import PdfReader
    r = PdfReader(io.BytesIO(b))
    pages = r.pages[:max_pages]
    return "\n".join((p.extract_text() or "") for p in pages)

def _extract_text_docx_bytes(b: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(b))
    return "\n".join(p.text for p in doc.paragraphs)

def read_text_from_named_bytes(name: str, raw: bytes) -> str:
    n = (name or "").lower()
    if n.endswith(".pdf"):
        txt = _extract_text_pdf_bytes(raw)
    elif n.endswith(".docx"):
        txt = _extract_text_docx_bytes(raw)
    elif n.endswith(".txt"):
        txt = raw.decode("utf-8", errors="ignore")
    else:
        # Si JSON
        if n.endswith(".json"):
            try:
                obj = json.loads(raw.decode("utf-8", errors="ignore"))
                return json.dumps(obj, ensure_ascii=False)
            except Exception:
                pass
        raise ValueError("Format non supporté (PDF/DOCX/TXT/JSON).")
    return clean_text_soft(txt)

# =========================
# OpenAI (optionnel) + appels robustes
# =========================
def _get_openai_key() -> str:
    k = os.getenv("OPENAI_API_KEY", "")
    return (k.strip() if k else "")

def _chat_completion(model: str, messages: list, temperature: float = 0, max_tokens: int = 700,
                     retries: int = 4) -> str:
    key = _get_openai_key()
    if not key or not key.startswith("sk-"):
        raise RuntimeError("OPENAI_API_KEY absente/invalide (env).")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    org  = os.getenv("OPENAI_ORG", "")
    proj = os.getenv("OPENAI_PROJECT", "")
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

# =========================
# SPEC : défaut / validate / normaliser
# =========================
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

# =========================
# Construction de SPEC depuis fiche projet (LLM + fallback offline)
# =========================
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

def offline_spec_from_text(txt: str) -> dict:
    txt_l = (txt or "").lower()

    def grab_section(header_keywords):
        pat = r"(?:" + "|".join([re.escape(k.lower()) for k in header_keywords]) + r")\s*[:\-]\s*(.+)"
        m = re.search(pat, txt_l)
        if not m:
            return []
        line = m.group(1)
        items = [re.sub(r"\s+", " ", w).strip(" .;-") for w in re.split(r"[,;/•|]", line)]
        return [it for it in items if it]

    must = grab_section(["must have","obligatoire","exigé","requis","required"])
    nice = grab_section(["nice to have","souhaité","optionnel","plus"])

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
        p = re.search(rf"\b{code}\b[^A-Z0-9]{{0,20}}(A1|A2|B1|B2|C1|C2)", txt or "", flags=re.I)
        if p: langues[code] = p.group(1).upper()

    diplomes = []
    if re.search(r"\b(licence|master|ing[eé]nieur|bachelor|bac\+)\b", txt_l):
        diplomes = ["Diplôme supérieur"]

    certifs = []
    if re.search(r"\b(certification|certifi[eé]|aws certified|azure|gcp)\b", txt_l):
        certifs = ["Certification détectée"]

    loc = ""
    m = re.search(r"localisation\s*[:\-]\s*([^\n,;]+)", txt or "", flags=re.I)
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

def gpt_build_spec_from_text(fiche_texte: str, model_id: Optional[str] = None) -> dict:
    """Spec via LLM ; fallback regex si erreur ou si pas de clé."""
    model_id = model_id or MODEL_ID_DEFAULT
    msgs = [{"role":"system","content":SPEC_SYSTEM},
            {"role":"user","content":fiche_texte}]
    try:
        if not _get_openai_key():
            raise RuntimeError("No OPENAI_API_KEY")
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
    except Exception:
        spec = offline_spec_from_text(fiche_texte)
    return _renormalize_weights(spec)

# =========================
# Extraction profil depuis CV (LLM + offline)
# =========================
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

def gpt_extract_profile_safe(cv_text: str, model_id: Optional[str] = None) -> dict:
    """Extraction via LLM ; fallback offline si clé absente/erreur."""
    if not _get_openai_key():
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
    model_id = model_id or MODEL_ID_DEFAULT
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
    except Exception:
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

# =========================
# Scoring compétences (embeddings optionnels) + règles strictes
# =========================
def score_competences_embeddings(cv_text: str, spec: Dict[str, Any],
                                 thr_must: float = 0.28, thr_nice: float = 0.25
                                 ) -> Tuple[float, Dict[str, List[Tuple[str, float, str]]]]:
    """
    - Similarité CV entier ↔ skill (embeddings si dispo, sinon keywords stricts)
    - Moyenne par groupe + couverture (>= seuils), pondérée par POIDS de la spec
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
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer(EMB_MODEL)
            cv_vec = model.encode(cv_text, normalize_embeddings=True)

            def sim(skill: str) -> float:
                v = model.encode(skill, normalize_embeddings=True)
                return float(util.cos_sim(cv_vec, v).item())

            for sk in must:
                s = sim(sk); proofs["must"].append((sk, round(s,3), sk if s >= thr_must else "")); sims_must.append(max(0.0, s))
            for sk in nice:
                s = sim(sk); proofs["nice"].append((sk, round(s,3), sk if s >= thr_nice else "")); sims_nice.append(max(0.0, s))
        except Exception:
            # fallback: keyword strict
            LOW = cv_text.lower()
            def hit(skill: str) -> float:
                return 1.0 if re.search(rf"(?<![a-z0-9_]){re.escape(skill.lower())}(?![a-z0-9_])", LOW) else 0.0
            for sk in must:
                s = hit(sk); proofs["must"].append((sk, s, sk if s >= 1.0 else "")); sims_must.append(s)
            for sk in nice:
                s = hit(sk); proofs["nice"].append((sk, s, sk if s >= 1.0 else "")); sims_nice.append(s)
    else:
        # keyword strict
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
        avg_nice   = (sum(sims_nice)/len(sims_nice)) if sims_nice else 0.0
        cover_nice = (sum(1.0 if s >= thr_nice else 0.0 for s in sims_nice) / len(sims_nice)) if sims_nice else 0.0
        pts += w_nice * min(1.0, max(0.0, avg_nice)) * cover_nice

    return round(pts, 2), proofs

ORDER_CEFR = {"A1":1,"A2":2,"B1":3,"B2":4,"C1":5,"C2":6}
def _num(slot, default=None):
    if isinstance(slot, dict): slot = slot.get("value", None)
    try: return float(slot)
    except (TypeError, ValueError): return default
def _str(slot):
    if isinstance(slot, dict): return slot.get("value", "")
    return slot or ""

def score_autres_criteres(ex: Dict[str, Any], spec: Dict[str, Any]) -> Tuple[float, str, Dict[str, Any]]:
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

# =========================
# Commentaire déterministe
# =========================
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

# =========================
# Modèles d'entrée API
# =========================
class FileB64(BaseModel):
    name: str
    content_b64: str  # base64 du fichier

class InAnalyzeCV(BaseModel):
    # Tu peux fournir soit:
    # 1) spec_text (texte brut de la fiche projet)
    # 2) spec_file (fichier PDF/DOCX/TXT/JSON encodé en base64)
    spec_text: Optional[str] = ""
    spec_file: Optional[FileB64] = None
    cvs: List[FileB64] = []
    lang: str = "fr"
    want_llm_comment: bool = False  # si tu veux le commentaire RH via LLM (si clé dispo)

# =========================
# FastAPI app
# =========================
app = FastAPI(title="CV Analyzer API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/healthz")
def health():
    return {"ok": True, "use_embeddings": USE_EMB}

def _build_spec_from_inputs(spec_text: Optional[str], spec_file: Optional[FileB64]) -> dict:
    # 1) Si fichier JSON → spec directe
    if spec_file:
        raw = base64.b64decode(spec_file.content_b64)
        if spec_file.name.lower().endswith(".json"):
            try:
                obj = json.loads(raw.decode("utf-8", errors="ignore"))
                return _renormalize_weights(validate_fill_spec(obj))
            except Exception:
                pass
        # 2) Sinon, extraire le texte et construire via LLM + fallback
        text = read_text_from_named_bytes(spec_file.name, raw)
        return gpt_build_spec_from_text(text, model_id=MODEL_ID_DEFAULT)

    # 3) Pas de fichier → utiliser spec_text
    stxt = (spec_text or "").strip()
    if not stxt:
        # spec vide par défaut
        return _renormalize_weights(validate_fill_spec(DEFAULT_SPEC))
    return gpt_build_spec_from_text(stxt, model_id=MODEL_ID_DEFAULT)

@app.post("/analyze_cv")
def analyze_cv(inp: InAnalyzeCV):
    # 1) Construire SPEC à partir de la fiche projet (fichier ou texte)
    spec = _build_spec_from_inputs(inp.spec_text, inp.spec_file)

    results = []
    for item in inp.cvs:
        raw = base64.b64decode(item.content_b64)
        # sécurité taille
        if len(raw) > MAX_FILE_MB * 1024 * 1024:
            results.append({"file": item.name, "error": f"Fichier > {MAX_FILE_MB} Mo"})
            continue

        # 2) Lire le CV (texte)
        cv_text = read_text_from_named_bytes(item.name, raw)

        # 3) Extraction profil (LLM si clé, sinon offline)
        extraction = gpt_extract_profile_safe(cv_text, model_id=MODEL_ID_DEFAULT)
        extraction = enforce_evidence(extraction, cv_text)
        extraction = fill_with_regex_if_missing(extraction, cv_text)

        # 4) Scoring
        pts_mn, evidences = score_competences_embeddings(cv_text, spec)
        pts_autres, _, details_autres = score_autres_criteres(extraction, spec)
        score_final = round(min(100.0, pts_mn + pts_autres), 2)

        # 5) Commentaires
        commentaire_det = build_commentaire_deterministe(score_final, evidences, extraction, spec)
        commentaire_llm = ""
        if inp.want_llm_comment and _get_openai_key():
            try:
                commentaire_llm = _chat_completion(
                    MODEL_ID_DEFAULT,
                    [{"role":"user","content":f"""Rôle: assistant RH. Commentaire factuel (5–7 lignes).
- Score final: {score_final} %
- Preuves must (top): {sorted(evidences.get('must',[]), key=lambda x: x[1], reverse=True)[:3]}
- Preuves nice (top): {sorted(evidences.get('nice',[]), key=lambda x: x[1], reverse=True)[:3]}
- Détails règles: {details_autres}
Contraintes: style FR pro, phrases courtes, terminer par une recommandation."""}],
                    temperature=0.2, max_tokens=220
                ).strip()
            except Exception:
                commentaire_llm = ""

        results.append({
            "file": item.name,
            "score_final": score_final,
            "points_embeddings": pts_mn,
            "points_regles": pts_autres,
            "exp_years": _num(extraction.get("experience_ans")),
            "evidences": evidences,
            "extraction": extraction,
            "commentaire_deterministe": commentaire_det,
            "commentaire_rh": commentaire_llm
        })

    return {
        "spec": spec,
        "use_embeddings": USE_EMB,
        "results": results,
        "lang": inp.lang
    }

# Petit endpoint simple si besoin
class InSimple(BaseModel):
    text: str = ""
    lang: str = "fr"

@app.post("/analyze")
def analyze_simple(inp: InSimple):
    spec = gpt_build_spec_from_text(inp.text or "", model_id=MODEL_ID_DEFAULT)
    ex = offline_extract_from_text(inp.text or "")
    pts_kw, _ = score_competences_embeddings(inp.text or "", spec)
    pts_autres, _, _ = score_autres_criteres(ex, spec)
    score_final = min(100.0, round(pts_kw + pts_autres, 2))
    return {"label": "OK", "score": score_final, "spec": spec, "lang": inp.lang}
