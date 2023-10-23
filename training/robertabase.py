import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, classification_report
import torch.nn as nn
from roberta_lstm import LSTMClassifier

train_data_path = '../data/train.json'

test_data_path = '../data/test.json'

def load_data(data_path):
    with open(data_path, 'r') as json_file:
        data = json.load(json_file)
    return data

train_data = load_data(train_data_path)

test_data = load_data(test_data_path)

label_mapping = {"Churn": 0, "Escalation": 1,  'Churn and Escalation':2,"No Intent Found": 3}

# Tokenize and encode data
def preprocess_data(data, tokenizer, label_mapping):
    input_texts = [example["text"] for example in data]
    labels = [label_mapping[example["intent"]] for example in data]

    tokenized_data = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = tokenized_data["input_ids"]
    attention_mask = tokenized_data["attention_mask"]

    return input_ids, attention_mask, labels

# Load the RoBERTa tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
num_labels = len(label_mapping)
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)

# Preprocess training and test data
train_input_ids, train_attention_mask, train_labels = preprocess_data(train_data, tokenizer, label_mapping)
test_input_ids, test_attention_mask, test_labels = preprocess_data(test_data, tokenizer, label_mapping)

# Create PyTorch datasets
class IntentDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }
    

train_dataset = IntentDataset(train_input_ids, train_attention_mask, train_labels)
test_dataset = IntentDataset(test_input_ids, test_attention_mask, test_labels)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize and train the RoBERTa model

optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()


    
# Initialize the LSTM classifier
hidden_size = 64
num_layers = 1
lstm_classifier = LSTMClassifier(model = model, hidden_size=hidden_size, num_labels = 4)




num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_classifier.to(device)
lstm_classifier.train()

for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        # roberta = model.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # hidden_states = roberta.last_hidden_state
        # print(hidden_states.shape)
        optimizer.zero_grad()
        logits = lstm_classifier(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

model_int8 = torch.quantization.quantize_dynamic(
    lstm_classifier,
    {torch.nn.Linear},  
    dtype=torch.qint8)

torch.save(model_int8,'../lstm_classifier.pth') 

# Evaluation on the validation set
lstm_classifier.eval()
val_preds = []
val_true = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        logits = lstm_classifier(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(logits, dim=1)
        val_preds.extend(preds.cpu().numpy())
        val_true.extend(labels.cpu().numpy())

# Convert predictions and ground truth to their original labels
pred_labels = [label for label, index in label_mapping.items() for pred in val_preds if index == pred]
true_labels = [label for label, index in label_mapping.items() for true in val_true if index == true]

# Print classification report
print("test data")
print(accuracy_score(true_labels, pred_labels))
print(classification_report(true_labels, pred_labels))





