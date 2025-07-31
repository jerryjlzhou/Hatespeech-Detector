import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification
from src.util.bert_data import BertData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = BertData(csv_file="data/processed/train.csv", tokenizer="sentence-transformers/all-MiniLM-L12-v2")
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
model = AutoModelForSequenceClassification.from_pretrained("sentence-transformers/all-MiniLM-L12-v2", num_labels=3)
model.to(device)
opt = AdamW(model.parameters(), lr=5e-5)
crit = torch.nn.CrossEntropyLoss()

print("Using device:", device)

model.train()
for batch in train_loader:
	input_ids = batch['input_ids'].to(device)
	attention_mask = batch['attention_mask'].to(device)
	labels = batch['labels'].to(device)

	outputs = model(input_ids=input_ids, attention_mask=attention_mask)
	logits = outputs.logits
	loss = crit(logits, labels)

	opt.zero_grad()
	loss.backward()
	opt.step()

	print(f"Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "./src/bert/saved/bert_finetuned.pt")
