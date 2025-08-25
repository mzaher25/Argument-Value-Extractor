import yaml, json 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def embed():
  pass 
  
def retrieve(argument, path = "ontology.yaml", top_k = 3):
  with open(path, "r") as f:
    ontology = yaml.safe_load(f)

  
