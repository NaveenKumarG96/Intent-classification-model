from fastapi import FastAPI, Request, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForSequenceClassification, AdamW
import torch.nn as nn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import sys,os

full_dir = os.path.join(os.path.dirname(__file__), 'training')
sys.path.append(full_dir)

app = FastAPI() 

tokenizer = AutoTokenizer.from_pretrained("roberta-base")


classifier = torch.load('lstm_classifier.pth').eval()


class QueryRequest(BaseModel):
    query: str

def preprocess_data(data, tokenizer):
    input_text = data
    

    tokenized_data = tokenizer(
        input_text,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = tokenized_data["input_ids"]
    attention_mask = tokenized_data["attention_mask"]

    return input_ids, attention_mask


def classify_text(text):
    # Tokenize the text
    input_ids, attention_mask = preprocess_data(text,tokenizer)
    
    with torch.no_grad():
        logits = classifier(input_ids, attention_mask)
    print(logits)

    predicted_intent_index = torch.argmax(logits, dim=1).item()


    intent_labels = {0: "Churn", 1: "Escalation", 2: "Churn and Escalation" , 3:'No Intent Found'}


    predicted_intent = intent_labels[predicted_intent_index]

    return predicted_intent 



@app.get("/")
async def get_index():
    return FileResponse("form.html")

# Route to do intent classifier post call
@app.post("/intent")
async def classify(query_data: QueryRequest):
    user_query = query_data.query
    predicted_intent = classify_text(user_query) 

    return {"intent": predicted_intent}

