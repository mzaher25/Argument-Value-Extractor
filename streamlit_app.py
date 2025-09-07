import os, re, json, time
import numpy as np
import pandas as pd
import streamlit as st
import torch
import openAI_prompt
import retrieval
from sklearn.metrics import f1_score, accuracy_score, classification_report
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import HfApi

# ---------- CONFIG ----------
VALUES = [
    "Fairness","Autonomy","Quality of Life","Safety","Life",
    "Honesty","Innovation","Responsibility","Sustainability","Economic Growth and Preservation"
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128

st.set_page_config(page_title="BERT vs GPT Comparator", layout="wide")
st.title("Compare GPT-4o vs Fine-tuned BERT (Values Classification)")

with st.sidebar:
    st.header("Models & Auth")
    bert_path = st.text_input(
        "BERT path/repo", "mzq34/bert-values-classifier"
    )
    hf_token = st.text_input("HF token (if needed)", type="password", value=os.getenv("HUGGINGFACE_HUB_TOKEN",""))

    # OpenAI setup
    default_model = os.getenv("OPENAI_MODEL", "gpt-4o")  # you can put the exact name you have access to
    openai_model = st.text(f"OpenAI model: {default_model}")
    openai_api_key = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY",""))
    temperature = st.slider("GPT temperature", 0.0, 1.5, 0.0, 0.05)
    max_output_tokens = st.slider("GPT max tokens", 8, 128, 32, 4)

# ---------- LOAD BERT ----------
@st.cache_resource(show_spinner=True)
def load_bert(path_or_repo: str, token: str):
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

bert_tok, bert_model, id2label, label2id = load_bert(bert_path, hf_token)

def predict_bert(texts: List[str]):
    enc = bert_tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        logits = bert_model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_ids = probs.argmax(axis=1)
    labels = [id2label[int(i)] for i in pred_ids]
    confs = [float(probs[i, pred_ids[i]]) for i in range(len(texts))]
    return labels, confs, probs

# ---------- GPT CALL (Responses API first, fallback to ChatCompletions) ----------

def normalize_label(text: str) -> str:
    t = text.strip()
    for v in VALUES:
        if t.lower().startswith(v.lower()) or v.lower() in t.lower():
            return v
    # fallback heuristic
    words = re.findall(r"[A-Za-z][A-Za-z ]+", t)
    for w in words:
        w = w.strip()
        if w in VALUES:
            return w
    return "Honesty"

def gpt_label(sentence: str, model: str, api_key: str, temperature: float, max_tokens: int) -> str:
    import openai
    openai.api_key = api_key

    user_prompt = f"Sentence: {sentence}\n"

    # Fallback: Chat Completions
    chat = openai.chat.completions.create(
    model = "chatgpt-4o-latest",
    messages= [{"role": "user", "content":openAI_prompt.prompt_gpt(sentence)}]
)

    text = chat.choices[0].message.content or ""

    return normalize_label(text)

def batch_gpt(texts: List[str], model: str, api_key: str, temperature: float, max_tokens: int) -> List[str]:
    out = []
    for t in texts:
        try:
            out.append(gpt_label(t, model, api_key, temperature, max_tokens))
        except Exception:
            out.append("Honesty")
            time.sleep(0.02)
    return out

# ---------- UI: Single sentence ----------
st.subheader("Single sentence")
col1, col2 = st.columns(2)

with col1:
    sent = st.text_area("Sentence", height=120, placeholder="e.g., We should redistribute the wealth.")
    if st.button("Compare"):
        if not openai_api_key:
            st.error("Please provide OPENAI_API_KEY in the sidebar.")
        elif sent.strip():
            b_label, b_conf, _ = predict_bert([sent])
            g_label = batch_gpt([sent], openai_model, openai_api_key, temperature, max_output_tokens)[0]
            st.markdown(f"**BERT** → `{b_label[0]}` (conf {b_conf[0]:.2f})")
            st.markdown(f"**GPT**  → `{g_label}`")
            st.markdown(f"**Agreement:** {'Yes!' if b_label[0]==g_label else 'No :('}")

with col2:
    st.info("GPT is prompted to return exactly one of your 10 labels. Keep temperature at 0 for determinism.")


