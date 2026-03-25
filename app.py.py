#!/usr/bin/env python3
"""
ai_firewall_modern_ui_final.py

Single-file AI Firewall demo — updated:
- Minimal signatures + ML detection
- Signature matches -> blocked.log
- ML quarantined/blocked -> quarantine.log / blocked.log
- Allowed requests are now logged to allowed.log (so tester's benign commands appear)
- UI: Tester has two "normal command" buttons that produce allowed payloads
- AI explanation text for each decision
- Adaptive block threshold based on recent blocked volume
Run:
    python ai_firewall_modern_ui_final.py
Open:
    http://127.0.0.1:8080/login  (admin / adminpass)
"""

import os
import re
import json
import math
import urllib.parse
from datetime import datetime
from collections import Counter
from functools import wraps

from flask import (
    Flask, request, jsonify, make_response, send_file,
    redirect, url_for, session, render_template_string
)

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump, load

# -------------------------
# Paths & config
# -------------------------
BASE = os.path.dirname(__file__) or "."
MODEL_PATH = os.path.join(BASE, "model_minimal.joblib")
BLOCKED_LOG = os.path.join(BASE, "blocked.log")
QUARANTINE_LOG = os.path.join(BASE, "quarantine.log")
ALLOWED_LOG = os.path.join(BASE, "allowed.log")
os.makedirs(BASE, exist_ok=True)

BLOCK_THRESHOLD = 0.72
QUARANTINE_THRESHOLD = 0.45

ADMIN_USER = "admin"
ADMIN_PASS = "adminpass"

app = Flask(__name__)
app.secret_key = "change_this_secret_to_a_strong_value_for_sessions"

# -------------------------
# Signatures (focused)
# -------------------------
SIGNATURES = [
    (r"(?:')\s*or\s+(?:')?1(?:')?\s*=\s*(?:')?1", "SQLi"),
    (r"\bor\b\s+1\s*=\s*1\b", "SQLi"),
    (r"\bunion\b\s*(?:all\s*)?\bselect\b", "SQLi"),
    (r"(?i)(?:')\s*;\s*drop\s+table", "SQLi"),
    (r"\bselect\b[^;]{1,300}\bfrom\b", "SQLi"),
    (r"\binsert\s+into\b", "SQLi"),
    (r"<script[^>]*>", "XSS"),
    (r"on\w+\s*=", "XSS"),
    (r"javascript:", "XSS"),
    (r"\.\./\.\.", "Path-Traversal"),
    (r"/etc/passwd", "LFI"),
    (r"cmd\.exe", "Shell"),
    (r"/bin/bash", "Shell"),
    (r"rm\s+-rf", "Shell"),
    (r"\beval\s*\(", "RCE"),
    (r"\bsystem\s*\(", "RCE"),
    (r"base64_decode\s*\(", "RCE"),
    (r"\(\)\s*{\s*:\s*;\s*}\s*;", "Shellshock"),
]
COMPILED_SIGS = [(re.compile(p, re.IGNORECASE), t) for p, t in SIGNATURES]

# -------------------------
# Feature extraction & ML
# -------------------------
def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    l = len(s)
    return -sum((c / l) * math.log2(c / l) for c in counts.values())

def extract_features(text: str):
    txt = text or ""
    l = len(txt)
    words = re.findall(r"\w+", txt)
    wc = len(words)
    avgw = (sum(len(w) for w in words) / wc) if wc else 0.0
    digits = sum(ch.isdigit() for ch in txt)
    uppers = sum(ch.isupper() for ch in txt)
    non_alnum = sum(not ch.isalnum() and not ch.isspace() for ch in txt)
    entropy = shannon_entropy(txt)
    base64_like_flag = 1 if re.search(r"[A-Za-z0-9+/]{40,}={0,2}", txt) else 0
    suspicious_tokens = sum(
        1
        for t in ["SELECT", "DROP", "UNION", "<script", "cmd", "eval", "base64"]
        if re.search(re.escape(t), txt, re.IGNORECASE)
    )
    digit_ratio = digits / l if l else 0.0
    non_alnum_ratio = non_alnum / l if l else 0.0
    upper_ratio = uppers / l if l else 0.0
    return np.array(
        [
            l,
            wc,
            avgw,
            digit_ratio,
            non_alnum_ratio,
            upper_ratio,
            entropy,
            base64_like_flag,
            suspicious_tokens,
        ],
        dtype=float,
    )

FEATURE_NAMES = [
    "len",
    "word_count",
    "avg_word_len",
    "digit_ratio",
    "non_alnum_ratio",
    "upper_ratio",
    "entropy",
    "base64_like",
    "suspicious_tokens",
]

# -------------------------
# Training
# -------------------------
def make_synthetic_dataset(n_benign=500, n_malicious=500, seed=42):
    rng = np.random.default_rng(seed)
    benign = [
        "username=alice&action=view",
        "GET /index.html HTTP/1.1",
        "search=python programming",
        "page=2&sort=asc",
    ]
    malicious = [
        "username=admin' OR '1'='1' --",
        "'; DROP TABLE users; --",
        "<script>alert('xss')</script>",
        "cmd.exe /c whoami; rm -rf /",
        "() { :; }; /bin/bash -c 'echo pwned'",
        "SELECT password FROM users WHERE id=1",
        "/etc/passwd",
    ]
    X, y = [], []
    for _ in range(n_benign):
        s = rng.choice(benign)
        if rng.random() < 0.2:
            s += " token_" + str(rng.integers(1000, 9999))
        X.append(extract_features(s))
        y.append(0)
    for _ in range(n_malicious):
        t = rng.choice(malicious)
        if rng.random() < 0.35:
            noise = os.urandom(rng.integers(30, 80)).hex()
            payload = f"{t} {noise}"
        else:
            payload = t + " " + "!@#$%^&*()_+"
        X.append(extract_features(payload))
        y.append(1)
    return np.vstack(X), np.array(y)

def train_and_save_model():
    print("[*] Training ML model...")
    X, y = make_synthetic_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.18, random_state=1, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("[*] Training complete. Test classification report:")
    print(classification_report(y_test, preds, digits=3))
    dump(clf, MODEL_PATH)
    print("[*] Model saved:", MODEL_PATH)
    return clf

def load_or_train():
    if os.path.exists(MODEL_PATH):
        try:
            clf = load(MODEL_PATH)
            print("[*] Loaded model from", MODEL_PATH)
            return clf
        except Exception as e:
            print("[!] Loading model failed, retraining:", e)
    return train_and_save_model()

MODEL = load_or_train()

# -------------------------
# AI helper functions (NEW)
# -------------------------
def classify_attack_type_by_text(text: str) -> str:
    t = (text or "").lower()
    if "<script" in t or "onerror" in t:
        return "XSS"
    if "select" in t and "from" in t:
        return "SQLi"
    if "/etc/passwd" in t or "../" in t:
        return "LFI"
    if "cmd.exe" in t or "/bin/bash" in t or "rm -rf" in t:
        return "Shell"
    if "eval(" in t or "system(" in t or "base64_decode" in t:
        return "RCE"
    if shannon_entropy(t) > 4.5:
        return "Obfuscated/Encoded"
    return "Unknown"

def ai_explain_decision(text: str, prob: float, attack_type: str) -> str:
    """
    Explain in simple English why the request was blocked / quarantined / allowed.
    """
    txt = text or ""
    ent = shannon_entropy(txt)

    # Base sentence from probability
    if prob > 0.9:
        base = "Very high confidence malicious request."
    elif prob > 0.7:
        base = "Strong signs of malicious behavior."
    elif prob > 0.5:
        base = "Suspicious pattern detected."
    else:
        base = "Request appears normal."

    # Extra hints
    extras = []
    low = txt.lower()
    if ent > 4.5:
        extras.append("payload appears highly encoded/obfuscated")
    if any(k in low for k in ["select", "union", "drop"]):
        extras.append("SQL keywords present")
    if any(k in low for k in ["<script", "onerror", "alert("]):
        extras.append("XSS-like script content present")

    if extras:
        base += " Indicators: " + ", ".join(extras) + "."

    if attack_type and attack_type != "Unknown":
        base += " Classified as: %s." % attack_type

    return base

def adaptive_block_threshold() -> float:
    """
    Adaptive threshold:
    - Default is BLOCK_THRESHOLD
    - If many blocks in blocked.log, slightly lower threshold (be more strict)
    Very simple heuristic, but enough to auto-tune.
    """
    try:
        if os.path.exists(BLOCKED_LOG):
            with open(BLOCKED_LOG, "r", encoding="utf-8") as f:
                blocked = sum(1 for _ in f)
            if blocked > 200:
                return max(0.55, BLOCK_THRESHOLD - 0.10)
            if blocked > 50:
                return BLOCK_THRESHOLD - 0.05
    except Exception:
        pass
    return BLOCK_THRESHOLD

# -------------------------
# Decision & Logging
# -------------------------
def check_signatures_on_texts(texts):
    for text in texts:
        for regex, atype in COMPILED_SIGS:
            if regex.search(text):
                return True, regex.pattern, atype
    return False, None, None

def append_log_file(path, record):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print("[!] Failed to write log:", e)

def decide(request_text: str, raw_body: str = ""):
    ts = datetime.utcnow().isoformat() + "Z"
    decoded_body = ""
    try:
        decoded_body = urllib.parse.unquote_plus(raw_body or "")
    except Exception:
        decoded_body = raw_body or ""
    full_text = raw_body or request_text or ""
    texts_to_check = [request_text or "", decoded_body or ""]

    # 1) Signature engine
    sig_hit, sig_pattern, sig_type = check_signatures_on_texts(texts_to_check)
    if sig_hit:
        ai_reason = "Matched known signature-based attack pattern. Blocking for safety."
        rec = {
            "timestamp": ts,
            "mode": "signature",
            "action": "BLOCK",
            "attack_type": sig_type,
            "signature": sig_pattern,
            "ml_score": 1.0,
            "ai_explanation": ai_reason,
            "context": full_text[:1000],
        }
        append_log_file(BLOCKED_LOG, rec)
        return {
            "status": "blocked",
            "reason": f"signature:{sig_pattern}",
            "attack_type": sig_type,
            "ml_score": 1.0,
            "ai_explanation": ai_reason,
        }

    # 2) ML + AI decision
    feats = extract_features(full_text[:2000]).reshape(1, -1)
    prob = float(MODEL.predict_proba(feats)[0, 1]) if hasattr(
        MODEL, "predict_proba"
    ) else float(MODEL.predict(feats)[0])

    attack_type = classify_attack_type_by_text(full_text)
    dynamic_block_threshold = adaptive_block_threshold()
    ai_reason = ai_explain_decision(full_text, prob, attack_type)

    if prob >= dynamic_block_threshold:
        rec = {
            "timestamp": ts,
            "mode": "ml",
            "action": "BLOCK",
            "attack_type": attack_type,
            "ml_score": prob,
            "ai_explanation": ai_reason,
            "context": full_text[:1000],
        }
        append_log_file(BLOCKED_LOG, rec)
        return {
            "status": "blocked",
            "reason": f"ml_score:{prob:.3f}",
            "attack_type": attack_type,
            "ml_score": prob,
            "ai_explanation": ai_reason,
        }

    if prob >= QUARANTINE_THRESHOLD:
        rec = {
            "timestamp": ts,
            "mode": "ml",
            "action": "QUARANTINE",
            "attack_type": attack_type,
            "ml_score": prob,
            "ai_explanation": ai_reason,
            "context": full_text[:1000],
        }
        append_log_file(QUARANTINE_LOG, rec)
        return {
            "status": "quarantined",
            "reason": f"ml_score:{prob:.3f}",
            "attack_type": attack_type,
            "ml_score": prob,
            "ai_explanation": ai_reason,
            "message": "Request quarantined",
        }

    # Allowed: record to allowed.log
    rec = {
        "timestamp": ts,
        "mode": "allowed",
        "action": "ALLOW",
        "attack_type": "None",
        "ml_score": prob,
        "ai_explanation": "Request considered normal by ML model.",
        "context": full_text[:1000],
    }
    append_log_file(ALLOWED_LOG, rec)
    return {
        "status": "allowed",
        "reason": f"ml_score:{prob:.3f}",
        "attack_type": "None",
        "ml_score": prob,
        "ai_explanation": "Request considered normal by ML model.",
    }

# -------------------------
# Endpoints
# -------------------------
@app.route("/", methods=["GET","POST"])
def root():
    parts = []
    parts.append(f"{request.method} {request.path} {request.environ.get('SERVER_PROTOCOL')}")
    for k, v in request.headers.items():
        parts.append(f"{k}: {v}")
    if request.query_string:
        parts.append("QS: " + request.query_string.decode(errors="ignore"))
    raw_body = request.get_data(as_text=True, cache=True) or ""
    parts.append("BODY: " + raw_body)
    text = "\n".join(parts)
    decision = decide(text, raw_body=raw_body)
    code = 403 if decision.get("status") == "blocked" else 200
    return make_response(jsonify(decision), code)

@app.route("/_health")
def health():
    return jsonify({"status":"ok","model":"rf_demo","features": FEATURE_NAMES})

@app.route("/metrics")
def metrics():
    blocked = quarantined = allowed = 0
    try:
        if os.path.exists(BLOCKED_LOG):
            with open(BLOCKED_LOG, "r", encoding="utf-8") as f:
                blocked = sum(1 for _ in f)
        if os.path.exists(QUARANTINE_LOG):
            with open(QUARANTINE_LOG, "r", encoding="utf-8") as f:
                quarantined = sum(1 for _ in f)
        if os.path.exists(ALLOWED_LOG):
            with open(ALLOWED_LOG, "r", encoding="utf-8") as f:
                allowed = sum(1 for _ in f)
    except Exception:
        pass
    return jsonify({"blocked": blocked, "quarantined": quarantined, "allowed": allowed})

# Admin helpers
def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if session.get("logged_in"):
            return f(*args, **kwargs)
        return redirect(url_for("login", next=request.path))
    return wrapped

@app.route("/logs/blocked")
@login_required
def download_blocked():
    if not os.path.exists(BLOCKED_LOG):
        return "", 200
    return send_file(BLOCKED_LOG, as_attachment=False)

@app.route("/logs/quarantine")
@login_required
def download_quarantine():
    if not os.path.exists(QUARANTINE_LOG):
        return "", 200
    return send_file(QUARANTINE_LOG, as_attachment=False)

@app.route("/logs/allowed")
@login_required
def download_allowed():
    if not os.path.exists(ALLOWED_LOG):
        return "", 200
    return send_file(ALLOWED_LOG, as_attachment=False)

@app.route("/clear_logs", methods=["POST"])
@login_required
def clear_logs():
    open(BLOCKED_LOG, "w", encoding="utf-8").close()
    open(QUARANTINE_LOG, "w", encoding="utf-8").close()
    open(ALLOWED_LOG, "w", encoding="utf-8").close()
    return "ok"

@app.route("/retrain")
@login_required
def retrain():
    global MODEL
    MODEL = train_and_save_model()
    return "retrained"

@app.route("/upload_logs", methods=["POST"])
@login_required
def upload_logs():
    if "file" not in request.files:
        return "no file", 400
    f = request.files["file"]
    target = request.form.get("target", "blocked")
    dest = BLOCKED_LOG if target == "blocked" else (QUARANTINE_LOG if target == "quarantine" else ALLOWED_LOG)
    content = f.read().decode(errors="ignore")
    appended = 0
    with open(dest, "a", encoding="utf-8") as out:
        for ln in content.splitlines():
            if not ln.strip():
                continue
            try:
                json.loads(ln)
                out.write(ln + "\n")
            except:
                out.write(json.dumps({"timestamp": datetime.utcnow().isoformat()+"Z", "imported": ln}) + "\n")
            appended += 1
    return jsonify({"appended": appended})

# -------------------------
# UI (single-page interactive)
# -------------------------
LOGIN_HTML = r"""
<!doctype html><html><head><meta charset="utf-8"><title>AI Firewall — Login</title>
<style>body{margin:0;height:100vh;display:flex;align-items:center;justify-content:center;background:#07080a;color:#e6eef8;font-family:Inter,Arial}.box{width:380px;padding:22px;border-radius:12px;background:linear-gradient(180deg,#071226,#041022);box-shadow:0 12px 40px rgba(0,0,0,0.6)}input{width:100%;padding:10px;margin:8px 0;border-radius:8px;border:1px solid rgba(255,255,255,0.04);background:transparent;color:inherit}button{width:100%;padding:10px;border-radius:8px;border:none;background:linear-gradient(90deg,#7c3aed,#06b6d4);color:white;font-weight:700}.small{font-size:12px;color:#94a3b8;margin-top:8px}.err{color:#fb7185;margin-bottom:6px}</style></head><body>
<div class="box"><h2>AI Firewall — Admin Login</h2>{% if error %}<div class="err">{{ error }}</div>{% endif %}<form method="post"><input name="username" placeholder="username" required autofocus><input name="password" placeholder="password" type="password" required><input type="hidden" name="next" value="{{ next }}"><button type="submit">Sign in</button></form><div class="small">Demo: admin/adminpass</div></div></body></html>
"""

UI_HTML = r"""
<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AI Firewall — Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#07080a;--muted:#9aa7b6;--accent1:#7c3aed;--accent2:#06b6d4}
*{box-sizing:border-box;font-family:Inter,Arial}body{margin:0;background:var(--bg);color:#e6eef8}.app{display:flex;min-height:100vh}.sidebar{width:220px;padding:20px;background:linear-gradient(180deg,#050608,#081018)}.logo{width:56px;height:56px;border-radius:10px;background:linear-gradient(135deg,var(--accent1),var(--accent2));display:flex;align-items:center;justify-content:center;font-weight:800;color:white;cursor:pointer}.brand{margin-top:12px;font-weight:700}.nav{margin-top:18px}.nav a{display:block;color:#cbd5e1;padding:8px;border-radius:8px;text-decoration:none;margin-bottom:6px}.main{flex:1;padding:20px;overflow:auto}.card{background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding:16px;border-radius:12px}.controls{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}textarea{width:100%;min-height:140px;padding:12px;border-radius:10px;border:1px solid rgba(255,255,255,0.03);background:transparent;color:inherit;resize:vertical}.btn{padding:10px 12px;border-radius:10px;border:none;cursor:pointer}.btn-primary{background:linear-gradient(90deg,var(--accent1),var(--accent2));color:white;font-weight:700}.response{min-height:120px;padding:12px;border-radius:10px;background:linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.005));white-space:pre-wrap}.log-item{padding:10px;border-radius:8px;margin-bottom:8px;background:linear-gradient(180deg,rgba(255,255,255,0.01),rgba(255,255,255,0.005));display:flex;gap:10px;align-items:flex-start}.badge{padding:6px 8px;border-radius:8px;font-weight:700}.badge.sig{background:#ef4444;color:white}.badge.ml{background:#f59e0b;color:white}.badge.all{background:#10b981;color:white}
.small{color:var(--muted)}
</style>
<script>
function show(id){
  const ids = ["home","tester","logs","settings"];
  ids.forEach(x=>{
    const el = document.getElementById(x);
    if(!el) return;
    if(x===id){ el.style.display="block"; }
    else{ el.style.display="none"; }
  });
}
</script>
</head><body onload="show('home')">
<div class="app">
  <aside class="sidebar">
    <div class="logo" onclick="show('home')">AF</div>
    <div class="brand">AI Firewall</div>
    <nav class="nav">
      <a href="#home" onclick="show('home')">Home</a>
      <a href="#tester" onclick="show('tester')">Tester</a>
      <a href="#logs" onclick="show('logs')">Logs</a>
      <a href="#settings" onclick="show('settings')">Settings</a>
      <a href="/logout" style="color:#fb7185">Logout</a>
    </nav>
  </aside>
  <main class="main">
    <div class="card" id="home" style="display:none">
      <h2>Overview</h2>
      <div class="small">
        Blocked: <span id="m_blocked">0</span> —
        Quarantined: <span id="m_quar">0</span> —
        Allowed: <span id="m_allowed">0</span>
      </div>
    </div>

    <div class="card" id="tester" style="margin-top:12px;display:none">
      <h3>Live Tester</h3>
      <div class="small">Send sample or custom payloads</div>
      <textarea id="payload">username=alice&action=view</textarea>
      <div class="controls">
        <button class="btn btn-primary" onclick="sendPayload()">Send Payload</button>
        <button class="btn" onclick="setPayload(`username=admin' OR '1'='1' --`)">SQLi sample</button>
        <button class="btn" onclick="setPayload('<script>alert(1)</script>')">XSS sample</button>
        <button class="btn" onclick="setPayload(generateHighEntropy())">High entropy (ML)</button>
        <!-- NEW: two normal benign commands -->
        <button class="btn" onclick="setPayload('action=list_files&path=/home/user')">Normal: List Files</button>
        <button class="btn" onclick="setPayload('action=get_time')">Normal: Get Time</button>
        <button class="btn" onclick="retrain()">Retrain</button>
      </div>
      <div style="margin-top:12px">
        <div class="small">Response</div>
        <div id="resp" class="response">No response yet</div>
      </div>
    </div>

    <div class="card" id="logs" style="margin-top:12px;display:none">
      <h3>Activity Logs</h3>
      <div class="small">Blocked / Quarantined / Allowed</div>
      <div style="margin-top:8px">
        <a href="/logs/blocked">Download blocked.log</a> |
        <a href="/logs/quarantine">Download quarantine.log</a> |
        <a href="/logs/allowed">Download allowed.log</a>
      </div>
      <div id="events" style="margin-top:12px"></div>
    </div>

    <div class="card" id="settings" style="margin-top:12px;display:none">
      <h3>Settings</h3>
      <div class="controls">
        <button class="btn" onclick="retrain()">Retrain Model</button>
        <button class="btn" onclick="clearLogs()">Clear Logs</button>
      </div>
    </div>

  </main>
</div>

<script>
function setPayload(v){
  const p=document.getElementById('payload');
  if(p) p.value=v;
}
function generateHighEntropy(){
  let s='';
  for(let i=0;i<500;i++){
    s+=Math.random().toString(36).substring(2);
  }
  return s;
}
async function sendPayload(){
  const payload = document.getElementById('payload').value || '';
  const resp = document.getElementById('resp');
  resp.textContent = 'Sending...';
  try{
    const r = await fetch('/', { method:'POST', headers:{'Content-Type':'text/plain'}, body: payload });
    const txt = await r.text();
    resp.textContent = 'HTTP ' + r.status + '\n\n' + txt;
    await refreshAll();
    await loadEvents();
  }catch(e){
    resp.textContent = 'Error: ' + e;
  }
}
async function retrain(){
  const resp = document.getElementById('resp');
  resp.textContent='Retraining...';
  try{
    const r=await fetch('/retrain');
    const t=await r.text();
    resp.textContent=t;
    await refreshAll();
  }catch(e){
    resp.textContent='Retrain failed';
  }
}
async function clearLogs(){
  if(!confirm('Clear logs?')) return;
  await fetch('/clear_logs', { method:'POST' });
  await refreshAll();
  await loadEvents();
}

async function refreshAll(){
  try{
    const r = await fetch('/metrics');
    if(r.ok){
      const j=await r.json();
      document.getElementById('m_blocked').textContent=j.blocked;
      document.getElementById('m_quar').textContent=j.quarantined;
      document.getElementById('m_allowed').textContent=j.allowed;
    }
  }catch(e){}
}
function safeParse(line){
  try{return JSON.parse(line);}catch(e){return null;}
}
function escapeHtml(s){
  if(!s) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function renderLogItem(obj){
  if(!obj) return '';
  const ts = obj.timestamp || '';
  const mode = obj.mode || '';
  const act = obj.action || '';
  const type = obj.attack_type || (obj.attack || 'Unknown');
  const sig = obj.signature ? 'Rule: ' + obj.signature : '';
  const ml = obj.ml_score !== undefined ? 'ML: ' + Number(obj.ml_score).toFixed(3) : '';
  const ctx = obj.context ? escapeHtml(obj.context).slice(0,300) : '';
  const badge = mode === 'signature'
    ? '<span class="badge sig">Signature</span>'
    : (mode === 'ml'
        ? '<span class="badge ml">ML</span>'
        : '<span class="badge all">Allowed</span>');
  const details = [sig, ml].filter(Boolean).join(' · ');
  return `<div class="log-item"><div>${badge}</div><div><div style="font-weight:700">${escapeHtml(type)} · ${escapeHtml(act)}</div><div class="small">${escapeHtml(ts)}${details? ' · '+escapeHtml(details):''}</div><div class="small" style="margin-top:6px">${ctx}</div></div></div>`;
}

async function loadEvents(){
  const container = document.getElementById('events');
  if(!container) return;
  container.innerHTML='Loading...';
  let out='';
  // blocked
  try{
    const r = await fetch('/logs/blocked');
    if(r.status===200){
      const t=await r.text();
      const lines=t.trim()?t.trim().split('\n').reverse():[];
      if(lines.length){
        out += '<div><strong>Blocked</strong></div>';
        for(let i=0;i<Math.min(lines.length,50);i++){
          const obj=safeParse(lines[i]);
          out += obj ? renderLogItem(obj) : '<div class="log-item">'+escapeHtml(lines[i])+'</div>';
        }
      }
    }
  }catch(e){}
  // quarantine
  try{
    const r2 = await fetch('/logs/quarantine');
    if(r2.status===200){
      const t2=await r2.text();
      const lines2=t2.trim()?t2.trim().split('\n').reverse():[];
      if(lines2.length){
        out += '<div style="margin-top:12px"><strong>Quarantined</strong></div>';
        for(let i=0;i<Math.min(lines2.length,50);i++){
          const obj=safeParse(lines2[i]);
          out += obj ? renderLogItem(obj) : '<div class="log-item">'+escapeHtml(lines2[i])+'</div>';
        }
      }
    }
  }catch(e){}
  // allowed
  try{
    const r3 = await fetch('/logs/allowed');
    if(r3.status===200){
      const t3=await r3.text();
      const lines3=t3.trim()?t3.trim().split('\n').reverse():[];
      if(lines3.length){
        out += '<div style="margin-top:12px"><strong>Allowed</strong></div>';
        for(let i=0;i<Math.min(lines3.length,50);i++){
          const obj=safeParse(lines3[i]);
          out += obj ? renderLogItem(obj) : '<div class="log-item">'+escapeHtml(lines3[i])+'</div>';
        }
      }
    }
  }catch(e){}
  container.innerHTML = out || 'No recent events.';
}

window.addEventListener('load', async function(){
  await refreshAll();
  await loadEvents();
});
</script>
</body></html>
"""

@app.route("/login", methods=["GET","POST"])
def login():
    next_url = request.args.get("next") or "/ui"
    if request.method == "POST":
        user = request.form.get("username","")
        pwd = request.form.get("password","")
        if user == ADMIN_USER and pwd == ADMIN_PASS:
            session["logged_in"] = True
            return redirect(next_url)
        return render_template_string(LOGIN_HTML, error="Invalid credentials", next=next_url)
    return render_template_string(LOGIN_HTML, error=None, next=next_url)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/ui")
@login_required
def ui():
    return render_template_string(UI_HTML)

# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    print("AI Firewall running at http://127.0.0.1:8080/login")
    print("File:", os.path.abspath(__file__))
    app.run(host="127.0.0.1", port=8080, debug=False, threaded=True)