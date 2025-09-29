import re, math, unicodedata
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ==== Load your trained model (no retraining needed) ====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("model_distilbert")
model = AutoModelForTokenClassification.from_pretrained("model_distilbert").to(DEVICE).eval()

# ==== Lightweight rules (same spirit as notebook) ====
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
TOK_RE   = re.compile(r"[A-Za-z0-9_\-]{16,}")      # long tokens
PREFIXES = ("ghp_", "AKIA", "xoxb-", "xoxa-", "slk-")
CTX      = {"password","passwd","pwd","secret","token","apikey","api_key"}

def entropy(s):
    if not s: return 0.0
    from collections import Counter
    c = Counter(s); n = len(s)
    return -sum((v/n)*math.log2(v/n) for v in c.values())

def normalize(s):                      # NFC & strip zero-width
    s = unicodedata.normalize("NFKC", s)
    return s.replace("\u200b","")

def detect_rules(text):
    t = normalize(text)
    spans = []
    for m in EMAIL_RE.finditer(t):
        spans.append(["EMAIL", m.start(), m.end()])
    for m in TOK_RE.finditer(t):
        tok_s = m.group(0)
        left  = t[max(0, m.start()-40):m.start()].lower()
        ok_prefix = tok_s.startswith(PREFIXES)
        ok_entropy = entropy(tok_s) > 3.5
        ok_context = any(k in left for k in CTX)
        if ok_prefix or ok_entropy or ok_context:
            # extend to the right until end of token chars/punct
            start, end = m.start(), m.end()
            while end < len(t) and t[end] in ":;,.\"'`()[]{}":
                end += 1
            spans.append(["SECRET", start, end])
    return spans

# ==== Model inference (BIO-ish) ====
def detect_model(text):
    enc = tok(text, return_offsets_mapping=True, truncation=True, max_length=256, return_tensors="pt")
    offs = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits[0].softmax(-1).cpu().numpy()
    pred = logits.argmax(-1).tolist()

    spans = []
    i = 0
    while i < len(pred):
        if pred[i] in (1, 2):                 # 1->EMAIL, 2->SECRET (matches your fine-tune)
            lbl = "EMAIL" if pred[i] == 1 else "SECRET"
            s, e = offs[i]
            j = i + 1
            while j < len(pred) and pred[j] == pred[i]:
                s = min(s, offs[j][0]); e = max(e, offs[j][1]); j += 1
            if e > s:
                spans.append([lbl, s, e])
            i = j
        else:
            i += 1
    return spans

# ==== Merge rules + model ====
def detect_hybrid(text):
    r = {tuple(x) for x in detect_rules(text)}
    m = {tuple(x) for x in detect_model(text)}
    spans = sorted(r.union(m), key=lambda z: (z[1], z[2]))
    return [list(s) for s in spans]

# ==== Views ====
def strip_text(text, spans):
    if not spans: return text
    chars = list(text)
    for lbl, s, e in sorted(spans, key=lambda x: x[1], reverse=True):
        chars[s:e] = ""
    safe = "".join(chars)
    # clean up spaces/punct if you like
    safe = re.sub(r"\s{2,}", " ", safe).strip()
    return safe

TRAFFIC_CSS = """
.safe-box    { background:#1e4620; color:#a3f7b5; padding:10px; border-radius:8px; border:1px solid #28a745; font-weight:600; }
.warning-box { background:#2b2b2b; color:#ffcc00; padding:10px; border-radius:8px; border:1px solid #ffcc00; font-weight:600; }
.critical-box{ background:#460000; color:#ff7a7a; padding:10px; border-radius:8px; border:1px solid #ff1a1a; font-weight:600; }
"""

def ui_predict(text, mode):
    if not text or not text.strip():
        return "<em>Type or paste some text above…</em>"

    spans = detect_hybrid(text)

    # guardrail to ignore bogus spans that cover too much
    max_cover = int(0.7 * len(text))
    spans = [s for s in spans if s[2] > s[1] and (s[2]-s[1]) <= max_cover]

    # counts
    counts = {}
    for lbl,_,_ in spans:
        counts[lbl] = counts.get(lbl, 0) + 1

    # traffic lights (multiple boxes)
    boxes = []
    if not spans:
        boxes.append('<div class="safe-box">✅ No sensitive information detected. Safe to share.</div>')
    if counts.get("EMAIL"):
        n = counts["EMAIL"]
        boxes.append(f'<div class="warning-box">⚠️ Warning: Sensitive information detected.<br><b>Detected:</b> {n} EMAIL{"s" if n>1 else ""}<br>Handle with caution before sharing.</div>')
    if counts.get("SECRET"):
        n = counts["SECRET"]
        boxes.append(f'<div class="critical-box">🔴 Critical: Highly confidential information detected.<br><b>Detected:</b> {n} SECRET{"s" if n>1 else ""}<br>Please do <u>not</u> share this in public or insecure channels.</div>')

    # views
    unsafe_list = "\n".join(f"[{lbl}:{text[s:e]}]" for lbl,s,e in spans) or "–"
    safe = strip_text(text, spans)

    if mode == "Unsafe":
        body = f"<h4>Original</h4><pre>{text}</pre><h4>Unsafe / Redacted</h4><pre>{unsafe_list}</pre>"
    elif mode == "Safe":
        body = f"<h4>Safe</h4><pre>{safe}</pre>"
    else:
        body = f"<h4>Original</h4><pre>{text}</pre><h4>Unsafe / Redacted</h4><pre>{unsafe_list}</pre><h4>Safe</h4><pre>{safe}</pre>"

    return "".join(boxes) + body

with gr.Blocks(theme="gradio/soft", css=TRAFFIC_CSS) as demo:
    gr.Markdown("# 🔒 Sensitive Info Detector")
    inp  = gr.Textbox(label="Input text", lines=8, placeholder="Paste text here…")
    mode = gr.Radio(["Both","Unsafe","Safe"], value="Both", label="View")
    out  = gr.HTML(label="Output")
    btn  = gr.Button("Detect", variant="primary")
    btn.click(ui_predict, inputs=[inp, mode], outputs=out)

if __name__ == "__main__":
    demo.launch(share=True)
