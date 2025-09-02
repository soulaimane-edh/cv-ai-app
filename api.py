# api.py
import base64, io, os, re, json
from typing import List, Dict, Any, Tuple
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ---------- (1) Petits outils texte/fichiers ----------
def clean_text_soft(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t

def _extract_text_pdf_bytes(b: bytes) -> str:
    from pypdf import PdfReader
    r = PdfReader(io.BytesIO(b))
    return "\n".join((p.extract_text() or "") for p in r.pages[:8])

def _extract_text_docx_bytes(b: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(b))
    return "\n".join(p.text for p in doc.paragraphs)

def read_text_from_named_bytes(name: str, raw: bytes) -> str:
    name = name.lower()
    if name.endswith(".pdf"):  txt = _extract_text_pdf_bytes(raw)
    elif name.endswith(".docx"): txt = _extract_text_docx_bytes(raw)
    elif name.endswith(".txt"):  txt = raw.decode("utf-8", errors="ignore")
    else: raise ValueError("Format non supporté (PDF/DOCX/TXT).")
    return clean_text_soft(txt)

# ---------- (2) Spec minimale (mêmes règles que ton app) ----------
DEFAULT_SPEC = {
    "must_have": [], "nice_to_have": [],
    "experience_min_ans": 0,
    "langues": {}, "diplomes": [], "certifications": [],
    "localisation": "", "disponibilite_max_semaines": 4,
    "poids": {"must_have":40,"nice_to_have":20,"experience":15,"langues":10,"diplomes_certifs":10,"localisation_dispo":5}
}

def validate_fill_spec(s: dict) -> dict:
    import copy
    spec = copy.deepcopy(DEFAULT_SPEC)
    for k in spec: 
        if k in s: spec[k] = s[k]
    P0 = DEFAULT_SPEC["poids"]; P = spec.get("poids") or {}
    spec["poids"] = {k: P.get(k, P0[k]) for k in P0}
    spec["must_have"] = list(spec.get("must_have", []))
    spec["nice_to_have"] = list(spec.get("nice_to_have", []))
    spec["langues"] = dict(spec.get("langues", {}))
    spec["diplomes"] = list(spec.get("diplomes", []))
    spec["certifications"] = list(spec.get("certifications", []))
    try: spec["experience_min_ans"] = int(spec.get("experience_min_ans", 0))
    except: spec["experience_min_ans"] = 0
    try: spec["disponibilite_max_semaines"] = int(spec.get("disponibilite_max_semaines", 4))
    except: spec["disponibilite_max_semaines"] = 4
    return spec

def _renormalize_weights(spec: dict) -> dict:
    keys = ["must_have","nice_to_have","experience","langues","diplomes_certifs","localisation_dispo"]
    s = sum(float(spec["poids"].get(k,0)) for k in keys)
    if s <= 0: s = 100.0
    if abs(s-100.0) > 1e-6:
        for k in keys:
            spec["poids"][k] = round(float(spec["poids"].get(k,0))*100.0/s, 3)
    return spec

def offline_spec_from_text(txt: str) -> dict:
    txt_l = txt.lower()
    def grab(keys):
        pat = r"(?:" + "|".join([re.escape(k.lower()) for k in keys]) + r")\s*[:\-]\s*(.+)"
        m = re.search(pat, txt_l)
        if not m: return []
        line = m.group(1)
        return [re.sub(r"\s+"," ",w).strip(" .;-") for w in re.split(r"[,;/•|]", line) if w.strip()]
    must = grab(["must have","obligatoire","exigé","requis","required"])
    nice = grab(["nice to have","souhaité","optionnel","plus"])
    if not must and not nice:
        for k in ["python","sql","spark","airflow","aws","docker","power bi","pandas","scikit-learn","pytorch","azure","gcp","excel","dbt"]:
            if re.search(rf"(?<![a-z0-9_]){re.escape(k)}(?![a-z0-9_])", txt_l): must.append(k)
        must = list(dict.fromkeys(must))[:6]
    exp = 0
    m = re.search(r"(\d+)\s*(ans|years?)\s+(?:d['e]|\s)*exp", txt_l) or re.search(r"exp[\w\s:]*?(\d+)\s*(ans|years?)", txt_l)
    if m: exp = int(m.group(1))
    langues = {}
    for code in ["fr","en","de","es","ar","it"]:
        p = re.search(rf"\b{code}\b[^A-Z0-9]{{0,20}}(A1|A2|B1|B2|C1|C2)", txt, flags=re.I)
        if p: langues[code] = p.group(1).upper()
    diplomes = ["Diplôme supérieur"] if re.search(r"\b(licence|master|ing[eé]nieur|bachelor|bac\+)\b", txt_l) else []
    certifs  = ["Certification détectée"] if re.search(r"\b(certification|certifi[eé]|aws certified|azure|gcp)\b", txt_l) else []
    loc = (re.search(r"localisation\s*[:\-]\s*([^\n,;]+)", txt, flags=re.I) or [None,""])[1].strip() if re.search(r"localisation", txt, flags=re.I) else ""
    dispo = int((re.search(r"(\d+)\s*(semaines?)\s*(?:max|maximum|disponibilit[eé])", txt_l) or [None,4])[1]) if re.search(r"(\d+)\s*(semaines?)\s*(?:max|maximum|disponibilit[eé])", txt_l) else 4
    return _renormalize_weights(validate_fill_spec({
        "must_have": must, "nice_to_have": nice, "experience_min_ans": exp, "langues": langues,
        "diplomes": diplomes, "certifications": certifs, "localisation": loc,
        "disponibilite_max_semaines": dispo, "poids": DEFAULT_SPEC["poids"]
    }))

# ---------- (3) Scoring keyword strict (sans Streamlit) ----------
def score_keywords(cv_text: str, spec: Dict[str,Any]) -> Tuple[float, Dict[str, Any]]:
    LOW = cv_text.lower()
    def hit(skill: str) -> float:
        return 1.0 if re.search(rf"(?<![a-z0-9_]){re.escape(skill.lower())}(?![a-z0-9_])", LOW) else 0.0
    must = [s for s in spec.get("must_have",[]) if s.strip()]
    nice = [s for s in spec.get("nice_to_have",[]) if s.strip()]
    sims_must = [hit(s) for s in must]
    sims_nice = [hit(s) for s in nice]
    from statistics import mean
    P = spec.get("poids", {})
    w_must = float(P.get("must_have",40)); w_nice = float(P.get("nice_to_have",20))
    pts = 0.0
    if must:
        avg   = mean(sims_must)
        cover = mean([1.0 if s>=1.0 else 0.0 for s in sims_must])
        pts += w_must * min(1.0, avg) * cover
    if nice:
        avg   = mean(sims_nice)
        cover = mean([1.0 if s>=1.0 else 0.0 for s in sims_nice])
        pts += w_nice * min(1.0, avg) * cover
    return round(pts,2), {"must": list(zip(must, sims_must)), "nice": list(zip(nice, sims_nice))}

def score_autres(extraction: Dict[str,Any], spec: Dict[str,Any]) -> float:
    ORDER = {"A1":1,"A2":2,"B1":3,"B2":4,"C1":5,"C2":6}
    def _num(slot):
        if isinstance(slot, dict): slot = slot.get("value")
        try: return float(slot)
        except: return None
    def _str(slot):
        if isinstance(slot, dict): return slot.get("value","")
        return slot or ""
    P = spec.get("poids", {})
    w_exp=float(P.get("experience",15)); w_lang=float(P.get("langues",10)); w_dc=float(P.get("diplomes_certifs",10)); w_locd=float(P.get("localisation_dispo",5))
    pts=0.0
    need = spec.get("experience_min_ans",0); yrs=_num(extraction.get("experience_ans"))
    if need>0 and isinstance(yrs,(int,float)): pts += w_exp * min(1.0, yrs/max(1,int(need)))
    lang_req = spec.get("langues",{}) or {}
    if lang_req:
        per = w_lang / len(lang_req)
        for code, req in lang_req.items():
            have = next((d.get("niveau") for d in extraction.get("langues",[]) if (d.get("code") or "").lower()==code.lower()), None)
            if have and ORDER.get(have.upper(),0) >= ORDER.get((req or "").upper(),0): pts += per
    if extraction.get("diplomes_obtenus") or extraction.get("certifications"): pts += w_dc
    want_loc = (spec.get("localisation") or "").strip()
    if want_loc:
        ex_loc=_str(extraction.get("localisation")).lower()
        ok=any(seg.strip().lower() in ex_loc for seg in want_loc.split("|") if seg.strip())
        if ok: pts += w_locd/2
    dmax = spec.get("disponibilite_max_semaines"); dval=_num(extraction.get("disponibilite_semaines"))
    if isinstance(dmax,(int,float)) and isinstance(dval,(int,float)) and dval<=dmax: pts += w_locd/2
    return round(pts,2)

def offline_extract(cv_text: str) -> dict:
    data = {
        "experience_ans": {"value": None, "evidence": []},
        "disponibilite_semaines": {"value": None, "evidence": []},
        "langues": [], "diplomes_obtenus": [], "diplomes_en_cours": [],
        "certifications": [], "localisation": {"value": "", "evidence": []},
    }
    m = re.search(r"(\d+)\s*(ans|year|years)", cv_text, flags=re.I)
    if m: data["experience_ans"]["value"] = float(m.group(1))
    m = re.search(r"(\d+)\s*(semaines?|weeks?)", cv_text, flags=re.I)
    if m: data["disponibilite_semaines"]["value"] = float(m.group(1))
    langs=[]
    for code in ["fr","en","de","es","ar","it"]:
        p = re.search(rf"\b{code}\b[^A-Z0-9]*\(?\s*(A1|A2|B1|B2|C1|C2)\s*\)?", cv_text, flags=re.I)
        if p: langs.append({"code":code,"niveau":p.group(1).upper(),"evidence":[]})
    data["langues"]=langs
    if re.search(r"\b(licence|master|ing[eé]nieur|bachelor|bac\+)\b", cv_text, flags=re.I):
        data["diplomes_obtenus"].append({"label":"Diplôme détecté","year":None,"evidence":[]})
    if re.search(r"\b(certification|certifi[eé])\b", cv_text, flags=re.I):
        data["certifications"].append({"label":"Certification détectée","evidence":[]})
    return data

# ---------- (4) FastAPI ----------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class InSimple(BaseModel):
    text: str = ""
    lang: str = "fr"

class CVItem(BaseModel):
    name: str
    content_b64: str   # fichier encodé en base64 (PDF/DOCX/TXT)

class InAnalyze(BaseModel):
    spec_text: str = ""     # texte de la fiche projet
    cvs: List[CVItem] = []  # liste de CVs en base64
    lang: str = "fr"

@app.get("/healthz")
def health(): return {"ok": True}

@app.post("/analyze")
def analyze_simple(inp: InSimple):
    spec = offline_spec_from_text(inp.text or "")
    # démo : on score sur le même texte (tu peux changer pour y mettre un CV par défaut)
    ex = offline_extract(inp.text or "")
    pts_kw, _ = score_keywords(inp.text or "", spec)
    pts_autres = score_autres(ex, spec)
    score_final = min(100.0, round(pts_kw + pts_autres, 2))
    return {"label": "OK", "score": score_final, "spec": spec, "lang": inp.lang}

@app.post("/analyze_cv")
def analyze_cv(inp: InAnalyze):
    spec = offline_spec_from_text(inp.spec_text or "")
    out = []
    for item in inp.cvs:
        raw = base64.b64decode(item.content_b64)
        cv_text = read_text_from_named_bytes(item.name, raw)
        ex = offline_extract(cv_text)
        pts_kw, _ = score_keywords(cv_text, spec)
        pts_autres = score_autres(ex, spec)
        score_final = min(100.0, round(pts_kw + pts_autres, 2))
        out.append({"file": item.name, "score_final": score_final})
    return {"spec": spec, "results": out}
