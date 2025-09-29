def entities_view(text, spans):
    """Return ONLY the detected entities as [LABEL:VALUE] lines."""
    spans = _expand_and_merge(text, spans)
    if not spans: return "—"
    return "\n".join(f"[{lbl}:{text[s:e]}]" for lbl, s, e in spans)

def strip_text(text, spans):
    """Remove detected spans from the text (safe to share)."""
    spans = _expand_and_merge(text, spans)
    chars = list(text)
    for _, s, e in sorted(spans, key=lambda x: x[1], reverse=True):
        del chars[s:e]
    safe = "".join(chars)
    # tidy spacing
    safe = re.sub(r"\s{2,}", " ", safe)
    safe = re.sub(r"\s+([,.;:!?])", r"\1", safe)
    return safe.strip()

EXAMPLES = [
    "My email is MaithaHabib@hotmailcom",
    "Email me at alice99@corp.local and use password Tr0ub4dor&3",
    "Set API_KEY=ghp_ABC123456789xyz when deploying",
    "This is a safe line, nothing secret here",
]


def warning_box(spans):
    """
    Return a styled HTML notice:
      • Yellow warning box if any spans
      • Green success box if none
    """
    if not spans:
        return """
<div style="margin:8px 0;padding:12px;border-radius:10px;border:1px solid #b6e3b6;background:#eef9ee;color:#0f5132">
  ✅ No sensitive information detected.
</div>"""

    # counts per label for a pro touch
    counts = {}
    for lbl, _, _ in spans:
        counts[lbl] = counts.get(lbl, 0) + 1
    items = "".join(f"<li>{lbl}: <b>{n}</b></li>" for lbl, n in sorted(counts.items()))

    return f"""
<div style="margin:8px 0;padding:12px;border-radius:10px;border:1px solid #ffecb5;background:#fff8e1;color:#664d03">
  <b>⚠️ Sensitive information detected</b>
  <ul style="margin:8px 0 0 18px">{items}</ul>
  <div style="margin-top:6px">Please avoid sharing this information in public or insecure channels.</div>
</div>"""


import gradio as gr
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# 1. Load model + tokenizer from Hugging Face
MODEL_ID = "Petitkitten/sensitive-info-detector-distilbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)

# 2. Create pipeline
nlp = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# --- Web UI only (keep your pipeline unchanged) ---
!pip -q install gradio==4.44.0
import gradio as gr

# ---------- traffic-light styles ----------
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

def ui_predict(text, mode):
    if not text or not text.strip():
        return "<em>Type or paste some text above…</em>"

    # 1) detect
    spans = detect_hybrid(text)

    # 2) guardrail
    max_cover = int(0.7 * len(text))
    spans = [s for s in spans if s[2] > s[1] and (s[2]-s[1]) <= max_cover]

    # 3) counts per type
    counts = {}
    for lbl,_,_ in spans:
        counts[lbl] = counts.get(lbl, 0) + 1

    # 4) build multiple traffic-light boxes
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

    # 5) output view
    original = text
    unsafe_list = "\n".join(f"[{lbl}:{text[s:e]}]" for lbl,s,e in spans) or "-"
    safe = strip_text(text, spans)

    if mode == "Unsafe":
        html = f"<h4>Original</h4><pre>{original}</pre><h4>Unsafe / Redacted</h4><pre>{unsafe_list}</pre>"
    elif mode == "Safe":
        html = f"<h4>Safe</h4><pre>{safe}</pre>"
    else:
        html = f"<h4>Original</h4><pre>{original}</pre><h4>Unsafe / Redacted</h4><pre>{unsafe_list}</pre><h4>Safe</h4><pre>{safe}</pre>"

    return "".join(boxes) + html




with gr.Blocks(theme="gradio/soft", css=TRAFFIC_CSS) as demo:
    gr.Markdown("# 🔒 Sensitive Info Detector")
    inp  = gr.Textbox(label="Input text", lines=7, placeholder="Paste text here…")
    view = gr.Radio(["Both","Unsafe","Safe"], value="Both", label="View")
    out  = gr.HTML(label="Output")
    btn  = gr.Button("Detect", variant="primary")

    # examples (optional)
    gr.Examples(
        examples=[
            "The meeting is at 10am tomorrow. Nothing sensitive here.",
            "Contact me at maria.lopez@gmail.com about the draft.",
            "Login password is Summer2024! and the API key is ghp_abc123DEF456ghi789.",
            "Please email me at david.smith@gmail.com. Also, the system password is Winter2025! and the deployment key is ghp_XYZ123456789abcd. Do not share this outside the team.",
        ],
        inputs=inp
    )

    btn.click(ui_predict, inputs=[inp, view], outputs=out)

demo.launch(share=True, server_name="0.0.0.0")
