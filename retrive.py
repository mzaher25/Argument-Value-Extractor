import yaml, json 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def embed(entries,model = TfidfVectorizer):
  return model().fit(entries)
  
def retrieve(argument, path = "val_ontology.yaml", top_k = 3):
  #gets values from ontology 
  with open(path, "r") as f:
    ontology = yaml.safe_load(f)

  #gets them into list format 
  vals_list = []
  for val in ontology["values"]:
    vals_list.append(f"{val["id"]}. {val["definition"]}. Measures: {(", ").join(val["measures"]})

  embedded = embed(vals_list)

  #get a 1d list of the cos similarity between args and vals 
  similarity = cosine_similarity(embedded.transform(arguments), embedded.transform(entries)).ravel() #flatten here bc shape is 1,N

  #get most similar values 
  return similarity.argsort()[::-1][:top_k]

if name == "main":
  argument = "death is bad!"
  retrieve(argument)
                    
          


  
