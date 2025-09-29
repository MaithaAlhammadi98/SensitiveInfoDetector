# SensitiveInfoDetector


# Sensitive Info Detector (EMAIL + SECRET)

Web demo that loads a fine-tuned DistilBERT (`model_distilbert/`) and detects emails + secrets (tokens/passwords), combining regex rules with the model. No training is run at launch.

## Quick start (Colab)
```bash
!git clone https://github.com/<you>/SensitiveInfoDetector.git
%cd SensitiveInfoDetector
!pip install -r requirements.txt
!python gradio_app.py
