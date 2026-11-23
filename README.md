# FractureAI

**FractureAI – An AI-powered bone fracture detection and assessment system using Deep Learning, Computer Vision, Explainable AI, and Retrieval-Augmented Generation (RAG) to support fracture screening and provide intelligent healthcare assistance.**

## Installation and Setup

1. Install Python 3.10 or 3.11 (recommended for TensorFlow compatibility).
2. Open this folder in VS Code.
3. Open the VS Code terminal in this folder.
4. Create and activate a virtual environment.

**Windows:**
```bash
py -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

5. Install packages:

```bash
pip install -r requirements.txt
```

6. Ensure `FractureAI_Final.keras` is inside the `model` folder.
7. The optional `model/FractureAI_threshold.json` file can contain:

```json
{"threshold": 0.43699625}
```

8. Start the website:

```bash
python app.py
```

9. Open http://127.0.0.1:5000 in your browser.

## Important Notes

- This is a development starter, not a production medical system.
- Change `FLASK_SECRET_KEY` before deployment.
- Do not upload real patient data to a public deployment until privacy, consent, access control, encryption, retention and applicable legal requirements have been reviewed.
- The prediction is experimental and must not be used as a diagnosis.
- The appointment form records a request only; it does not contact a clinic.
- The chat page may require further LLM/RAG integration depending on the current implementation.
- Ensure the model's input preprocessing matches the exact preprocessing used in the Colab training notebook (RGB, 224x224, float32 [0,255]).