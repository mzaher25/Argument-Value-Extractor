#File that builds the streamlit app, gets user input and passes it through both models

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


VALUES = [
    "Fairness","Autonomy","Quality of Life","Safety","Life",
    "Honesty","Innovation","Responsibility","Sustainability","Economic Growth and Preservation"
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128

# Read secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o"))  

client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="Argument Value Extractor", layout="wide")
st.title("Argument Value Extractor: Fine-Tuned BERT vs GPT+RAG Comparator")

with st.sidebar:
    st.header("Models & Auth")
    bert_path = "mzq34/bert-values-classifier"
    temperature = st.slider("GPT temperature", 0.0, 1.5, 0.0, 0.05)
    max_output_tokens = 128

    st.caption(f"OpenAI model: `{OPENAI_MODEL}`")
    st.caption(f"BERT path: `{bert_path}`")

# BERT
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

bert_tok, bert_model, id2label, label2id = load_bert(bert_path, "")

def predict_bert(texts: List[str]):
    enc = bert_tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        logits = bert_model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_ids = probs.argmax(axis=1)
    labels = [id2label[int(i)] for i in pred_ids]
    confs = [float(probs[i, pred_ids[i]]) for i in range(len(texts))]
    return labels, confs, probs

# GPT call
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

    #builds prompt in openAI_prompt file
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


# UI
st.subheader("About:")
st.markdown("Input an argumentative sentence to see which value each model outputs.  The BERT model is a fine-tuned base BERT, utilizing a created dataset of 300 annotated sentences. For the GPT model, I first created an ontology of values, which includes the definition and way they are measured, along with keywords. The input is first passed into a word embedding model, then using cosine similarity + the ontology the top 3 values are filtered to ground the prediction. This is passed to GPT-4o along with the user sentence. For more info, like fine-tuning, model specifics, and to see the ontology, check out the repo linked above!")
st.subheader("Values:")
st.markdown("Each model classifies a sentence into one of the following: Fairness, Autonomy, Quality of Life, Safety, Life, Honesty, Innovation, Responsibility, Sustainability, Economic Growth and Preservation")
st.subheader("Suggested Sentences:")
st.markdown(":green[Agreements:]")
st.markdown("We must protect the young.")
st.markdown("You shouldn’t lie.")
st.markdown("We need to lower crime rates in our city.")
st.markdown(":red[Disagreements:]")
st.markdown("We should prevent murders.")
st.markdown("We should explore new energy sources that are renewable.")
st.markdown("We should really expand our efforts to develop new medicines.")


col1, col2 = st.columns(2)

with col1:
    sent = st.text_area("Sentence", height=120, placeholder="e.g., We should prevent crime.")
    if st.button("Compare"):
        if not OPENAI_API_KEY:
            st.error("GPT CALL ERROR")
            st.stop()
        elif sent.strip():
            b_label, b_conf, _ = predict_bert([sent])
            g_label = batch_gpt([sent], OPENAI_MODEL, temperature, 128)[0]
            st.markdown(f"**BERT** → `{b_label[0]}` (conf {b_conf[0]:.2f})")
            st.markdown(f"**GPT**  → `{g_label}`")
            st.markdown(f"**Agreement:** {'Yes!' if b_label[0]==g_label else 'No :('}")

with col2:
    st.info("GPT is prompted to return one of 10 labels. Keep temperature at 0 for determinism.")




