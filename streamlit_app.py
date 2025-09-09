import os, re, json, time
import numpy as np
import pandas as pd
import streamlit as st
import torch
from openai import OpenAI
from sklearn.metrics import f1_score, accuracy_score, classification_report
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import HfApi
import openAI_prompt

# ---------- CONFIG ----------
VALUES = [
    "Fairness","Autonomy","Quality of Life","Safety","Life",
    "Honesty","Innovation","Responsibility","Sustainability","Economic Growth and Preservation"
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128

# Read secrets/env only (no text input for keys)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
HF_TOKEN = st.secrets.get("HUGGINGFACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))  # or "gpt-4o"
client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="BERT vs GPT Comparator", layout="wide")
st.title("Compare GPT vs Fine-tuned BERT (Values Classification)")

with st.sidebar:
    st.header("Models & Auth")
    bert_path = st.text_input("BERT path/repo", "mzq34/bert-values-classifier")
    temperature = st.slider("GPT temperature", 0.0, 1.5, 0.0, 0.05)
    max_output_tokens = 128

    # Status badges
    st.caption(f"🔑 OpenAI key: {'✅ found' if bool(OPENAI_API_KEY) else '❌ missing'}")
    st.caption(f"🤗 HF token: {'✅ found' if bool(HF_TOKEN) else '⚠️ optional/blank'}")
    st.caption(f"🧠 OpenAI model: `{OPENAI_MODEL}`")

# ---------- LOAD BERT ----------
@st.cache_resource(show_spinner=True)
def load_bert(path_or_repo: str, token: str | None):
    kw = {}
    if token:
        kw["token"] = token
    tok = AutoTokenizer.from_pretrained(path_or_repo, **kw)
    model = AutoModelForSequenceClassification.from_pretrained(path_or_repo, **kw).to(DEVICE).eval()
    # normalize mappings
    id2label = model.config.id2label
    if isinstance(id2label, dict):
        id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    return tok, model, id2label, label2id

bert_tok, bert_model, id2label, label2id = load_bert(bert_path, HF_TOKEN or "")

def predict_bert(texts: List[str]):
    enc = bert_tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        logits = bert_model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_ids = probs.argmax(axis=1)
    labels = [id2label[int(i)] for i in pred_ids]
    confs = [float(probs[i, pred_ids[i]]) for i in range(len(texts))]
    return labels, confs, probs

# ---------- GPT CALL ----------
def normalize_label(text: str) -> str:
    t = text.strip()
    # exact or prefix match
    for v in VALUES:
        if t.lower() == v.lower() or t.lower().startswith(v.lower()):
            return v
    # contains match
    tl = t.lower()
    for v in VALUES:
        if v.lower() in tl:
            return v
    # last resort: simple word scan
    words = re.findall(r"[A-Za-z][A-Za-z ]+", t)
    for w in words:
        w = w.strip()
        if w in VALUES:
            return w
    return "Honesty"

def gpt_label(sentence: str, *, model: str, temperature: float, max_tokens: int) -> str:
    
    user_prompt = f"Sentence: {sentence}\nLabel:"

    system_prompt = openAI_prompt.prompt_gpt(user_prompt)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        return normalize_label(text)
    except Exception as e:
        st.warning(f"OpenAI call failed: {e}")
        return "Honesty"

def batch_gpt(texts, model, temperature, max_tokens) -> List[str]:
    return [gpt_label(t, model=model, temperature=temperature, max_tokens=max_tokens) for t in texts]


# ---------- UI: Single sentence ----------
st.subheader("Single sentence")
col1, col2 = st.columns(2)

with col1:
    sent = st.text_area("Sentence", height=120, placeholder="e.g., We should redistribute the wealth.")
    if st.button("Compare"):
        if not OPENAI_API_KEY:
            st.error("No OpenAI API key found in secrets. Add OPENAI_API_KEY in Streamlit -> Settings -> Secrets.")
            st.stop()
        elif sent.strip():
            b_label, b_conf, _ = predict_bert([sent])
            g_label = batch_gpt([sent], OPENAI_MODEL, temperature, 128)[0]
            st.markdown(f"**BERT** → `{b_label[0]}` (conf {b_conf[0]:.2f})")
            st.markdown(f"**GPT**  → `{g_label}`")
            st.markdown(f"**Agreement:** {'Yes!' if b_label[0]==g_label else 'No :('}")

with col2:
    st.info("GPT is prompted to return exactly one of your 10 labels. Keep temperature at 0 for determinism.")




