import gradio as gr
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# 1. Load model + tokenizer from Hugging Face
MODEL_ID = "Petitkitten/sensitive-info-detector-distilbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)

# 2. Create pipeline
nlp = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# 3. Custom CSS for traffic-light style
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

# 4. Detector function
def detect_sensitive(text, mode):
    if not text or not text.strip():
        return "<em>Type or paste some text above…</em>"

    results = nlp(text)

    # Track counts
    counts = {}
    unsafe_list = []
    for r in results:
        label = r["entity_group"]
        counts[label] = counts.get(label, 0) + 1
        unsafe_list.append(f"[{label}:{text[r['start']:r['end']]}]")

    # Safe version
    safe = text
    for r in results:
        safe = safe.replace(text[r["start"]:r["end"]], "[REDACTED]")

    # Build traffic-light boxes
    boxes = []
    if not results:
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

    # Output view
    original = text
    unsafe = "\n".join(unsafe_list) or "-"
    
    if mode == "Unsafe":
        html = f"<h4>Original</h4><pre>{original}</pre><h4>Unsafe / Redacted</h4><pre>{unsafe}</pre>"
    elif mode == "Safe":
        html = f"<h4>Safe</h4><pre>{safe}</pre>"
    else:
        html = (
            f"<h4>Original</h4><pre>{original}</pre>"
            f"<h4>Unsafe / Redacted</h4><pre>{unsafe}</pre>"
            f"<h4>Safe</h4><pre>{safe}</pre>"
        )

    return "".join(boxes) + html

# 5. Gradio UI
with gr.Blocks(theme="gradio/soft", css=TRAFFIC_CSS) as demo:
    gr.Markdown("# 🔒 Sensitive Info Detector")
    inp  = gr.Textbox(label="Input text", lines=7, placeholder="Paste text here…")
    view = gr.Radio(["Both","Unsafe","Safe"], value="Both", label="View")
    out  = gr.HTML(label="Output")
    btn  = gr.Button("Detect", variant="primary")

    # Examples
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
