#This file loads in the ontology
#It compares given sentences and generates a list of 3 top values
import yaml, json 

from sentence_transformers import SentenceTransformer
import numpy as np


model = SentenceTransformer("intfloat/e5-base-v2")

def enc_passages(texts):
    X = model.encode([f"passage: {t}" for t in texts],
                     normalize_embeddings=True, convert_to_numpy=True)
    return X  

def enc_query(text):
    q = model.encode([f"query: {text}"],
                     normalize_embeddings=True, convert_to_numpy=True)
    return q  

# build value entries (name + def + measures + keywords)
def value_text(v):
    parts = [f"{v['name']}. {v['definition']}"]
    if v.get("measures"):  parts.append("Measures: " + ", ".join(v["measures"]))
    if v.get("keywords"):  parts.append("Keywords: " + ", ".join(v["keywords"]))
    return " ".join(parts)

def run_retrival(argument):
    with open( "/Users/maryzaher/val/val_ontology.yaml", "r") as f:
      ontology = yaml.safe_load(f)
    
    values = ontology["values"] 
    value_texts = [value_text(v) for v in values]
    V = enc_passages(value_texts)              

    #embed the argument 
    q = enc_query(argument)                    

    # cosine similarity (dot qoduct since normalized)
    sims = (V @ q.T).ravel()                   
    topk_idx = sims.argsort()[::-1][:3]

    #candidates list + similarity score for each in top k
    candidates = [{
        "id": values[i]["id"],
        "name": values[i]["name"],
        "similarity": float(sims[i]),
        "definition": values[i]["definition"],
        "measures_topk": sorted(values[i]["measures"] or [])[:3]
    } for i in topk_idx]

    return candidates 
