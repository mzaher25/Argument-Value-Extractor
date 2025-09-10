#File that creates the dataloader to be used for fine-tuning
#Then it creates the functions used to fine-tune BERT 

import torch
import numpy 
import pandas
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
import datasets
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


df = pandas.read_csv("/Users/maryzaher/val/100_sent.csv - Sheet1 (5).csv").dropna(subset=["sentence","primary_value"])
print(df)
from sklearn.model_selection import train_test_split

#generates train, test, validation dfs from random seed
train_df, tmp_df = train_test_split(df, test_size=0.3, stratify=df["primary_value"], random_state=7)
val_df,  test_df = train_test_split(tmp_df, test_size=0.5, stratify=tmp_df["primary_value"], random_state=7)


#load in tokenizer (wordpiece)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


val2id = {"Fairness":0,"Autonomy":1, "Quality of Life": 2, "Safety":3, "Life":4, "Honesty":5, "Innovation":6, "Responsibility":7, "Sustainability":8, "Economic Growth and Preservation":9}
id2val = {0:"Fairness",1:"Autonomy", 2:"Quality of Life",3: "Safety", 4:"Life", 5:"Honesty", 6:"Innovation", 7:"Responsibility", 8:"Sustainability", 9:"Economic Growth and Preservation"}

class ValueDataset(Dataset):
    def __init__(self,data,max_len=256):
        self.sents = df["sentence"].tolist()
        self.labels = [val2id[v] for v in df["primary_value"].tolist()]
        self.df = data.reset_index(drop=True)
        self.max_len = max_len
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        enc = tokenizer(self.sents[idx], truncation=True, padding="max_length", max_length=256, return_tensors="pt")
        item = {k: v.squeeze(0) for k,v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

#creates the training and testing loaders
train_loader = DataLoader(ValueDataset(train_df), batch_size=16, shuffle=True)
test_loader  = DataLoader(ValueDataset(test_df),  batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=10, id2label=val2id, label2id=id2val).to(device)
    
#training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

def run_epoch(model, ds, optim, train = True):
    model.train()
    total_loss = 0.0
    
    for batch in ds:
        batch = {k: v.to(device) for k,v in batch.items()}
        out = model(**batch)
        loss = out.loss
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        total_loss += loss.item()
    return total_loss/len(train_loader)

def train(model, optim, num_epochs):
    for i in range(num_epochs):
        bl = run_epoch(model, train_loader,optim)
        print("Epoch:", i, "Loss:",bl)


