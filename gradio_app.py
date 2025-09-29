import gradio as gr
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# 1. Load your model + tokenizer directly from Hugging Face
MODEL_ID = "Petitkitten/sensitive-info-detector-distilbert"  # your repo ID
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)

# 2. Create pipeline
nlp = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# 3. Detector function
def detect_sensitive(text):
    results = nlp(text)
    if not results:
        return "✅ No sensitive info detected.", text

    unsafe = text
    for r in results:
        unsafe = unsafe.replace(text[r["start"]:r["end"]], f"[{r['entity_group']}:{text[r['start']:r['end']]}]")

    safe = text
    for r in results:
        safe = safe.replace(text[r["start"]:r["end"]], "[REDACTED]")

    return results, f"Unsafe: {unsafe}\nSafe: {safe}"

# 4. Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🔒 Sensitive Info Detector")
    inp = gr.Textbox(lines=6, label="Paste text here...")
    out1 = gr.JSON(label="Detected Entities")
    out2 = gr.Textbox(label="Outputs")
    btn = gr.Button("Detect")
    btn.click(detect_sensitive, inputs=inp, outputs=[out1, out2])

demo.launch()
