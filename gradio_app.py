import gradio as gr
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODEL_ID = "Petitkitten/sensitive-info-detector-distilbert"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
mdl = AutoModelForTokenClassification.from_pretrained(MODEL_ID)
nlp = pipeline("token-classification", model=mdl, tokenizer=tok,
               aggregation_strategy="simple")  # still helpful

# ---------- utilities ----------
def _merge_spans(results):
    """
    Merge overlapping/adjacent spans with the same label so we get a single
    [EMAIL:john@x.com] instead of many fragments.
    """
    spans = [(r["entity_group"], int(r["start"]), int(r["end"])) for r in results]
    spans.sort(key=lambda x: (x[1], x[2]))  # by start, then end

    merged = []
    for lbl, s, e in spans:
        if not merged:
            merged.append([lbl, s, e])
            continue
        mlbl, ms, me = merged[-1]
        # merge if same label and overlapping or touching (<= 1 char gap)
        if lbl == mlbl and s <= me + 1:
            merged[-1][2] = max(me, e)
        else:
            merged.append([lbl, s, e])
    # back to tuples
    return [(lbl, s, e) for lbl, s, e in merged]

def _unsafe_list(text, spans):
    return "\n".join(f"[{lbl}:{text[s:e]}]" for lbl, s, e in spans) or "-"

def _redact_by_indices(text, spans):
    """
    Build the redacted text by slicing with the original indices (don’t use .replace).
    """
    out = []
    last = 0
    for lbl, s, e in spans:
        out.append(text[last:s])
        out.append(f"[REDACTED:{lbl}]")
        last = e
    out.append(text[last:])
    return "".join(out)

TRAFFIC_CSS = """
.safe-box {
  background:#1e4620; color:#a3f7b5; padding:10px; border-radius:8px;
  border:1px solid #28a745; font-weight:600;
}
.warning-box {
  background:#2b2b2b; color:#ffcc00; padding:10px; border-radius:8px;
  border:1px solid #ffcc00; font-weight:600;
}
.critical-box {
  background:#460000; color:#ff7a7a; padding:10px; border-radius:8px;
  border:1px solid #ff1a1a; font-weight:600;
}
"""

def detect_sensitive(text, mode):
    if not text or not text.strip():
        return "<em>Type or paste some text above…</em>"

    raw = nlp(text)
    spans = _merge_spans(raw)  # <<< IMPORTANT

    # counts per label (after merge)
    counts = {}
    for lbl, _, _ in spans:
        counts[lbl] = counts.get(lbl, 0) + 1

    # traffic-light boxes
    boxes = []
    if not spans:
        boxes.append("""
        <div class="safe-box">
          ✅ No sensitive information detected. Safe to share.
        </div>
        """)
    if counts.get("EMAIL"):
        boxes.append(f"""
        <div class="warning-box">
          ⚠️ Warning: Sensitive information detected.<br>
          <b>Detected:</b> {counts['EMAIL']} EMAIL{'s' if counts['EMAIL']>1 else ''}<br>
          Handle with caution before sharing.
        </div>
        """)
    if counts.get("SECRET"):
        boxes.append(f"""
        <div class="critical-box">
          🔴 Critical: Highly confidential information detected.<br>
          <b>Detected:</b> {counts['SECRET']} SECRET{'s' if counts['SECRET']>1 else ''}<br>
          Please do <u>not</u> share this in public or insecure channels.
        </div>
        """)

    original = text
    unsafe   = _unsafe_list(text, spans)
    safe     = _redact_by_indices(text, spans)

    if mode == "Unsafe":
        body = f"<h4>Original</h4><pre>{original}</pre><h4>Unsafe / Redacted</h4><pre>{unsafe}</pre>"
    elif mode == "Safe":
        body = f"<h4>Safe</h4><pre>{safe}</pre>"
    else:
        body = (f"<h4>Original</h4><pre>{original}</pre>"
                f"<h4>Unsafe / Redacted</h4><pre>{unsafe}</pre>"
                f"<h4>Safe</h4><pre>{safe}</pre>")

    return "".join(boxes) + body

# ---------- Gradio UI ----------
with gr.Blocks(theme="gradio/soft", css=TRAFFIC_CSS) as demo:
    gr.Markdown("# 🔒 Sensitive Info Detector")
    inp  = gr.Textbox(label="Input text", lines=7, placeholder="Paste text here…")
    view = gr.Radio(["Both","Unsafe","Safe"], value="Both", label="View")
    out  = gr.HTML(label="Output")
    btn  = gr.Button("Detect", variant="primary")

    gr.Examples(
        examples=[
            "The meeting is at 10am tomorrow. Nothing sensitive here.",
            "Contact me at maria.lopez@gmail.com about the draft.",
            "Login password is Summer2024! and the API key is ghp_abc123DEF456ghi789.",
            "Please email me at david.smith@gmail.com. Also, the system password is Winter2025! and the deployment key is ghp_XYZ123456789abcd. Do not share this outside the team.",
        ],
        inputs=inp
    )

    btn.click(detect_sensitive, inputs=[inp, view], outputs=out)

demo.launch(share=True, server_name="0.0.0.0")
