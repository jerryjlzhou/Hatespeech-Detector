import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification
from src.util.bert_data import BertData
from sklearn.metrics import confusion_matrix, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = BertData(csv_file="data/processed/train.csv", tokenizer="bert-base-uncased")
val_dataset = BertData(csv_file="data/processed/val.csv", tokenizer="bert-base-uncased")
test_dataset = BertData(csv_file="data/processed/test.csv", tokenizer="bert-base-uncased")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
model.to(device)

opt = AdamW(model.parameters(), lr=1e-5)
crit = torch.nn.CrossEntropyLoss()

num_epochs = 50
checkpoint_path = "./src/bert/saved/checkpoint.pt"
best_model_path = "./src/bert/saved/best_model.pt"

patience = 3
best_val_loss = float('inf')
epochs_no_improve = 0

start_epoch = 0
if os.path.exists(checkpoint_path):
    print(f"Resuming from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
    print(f"Resuming from epoch {start_epoch+1}")

print("Using device:", device)

for epoch in range(start_epoch, num_epochs):
    model.train()
    running_train_loss = 0
    for batch_idx, batch in enumerate(train_loader, 1):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = crit(logits, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

        running_train_loss += loss.item()

        if batch_idx % 10 == 0 or batch_idx == len(train_loader):
            avg_loss = running_train_loss / batch_idx
            print(f"Epoch {epoch+1} Batch {batch_idx}/{len(train_loader)} - Train loss: {avg_loss:.4f}")

    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = crit(logits, labels)
            val_loss += loss.item()

            preds = logits.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = val_correct / val_total
    print(f"Epoch {epoch+1} - Validation loss: {avg_val_loss:.4f}, accuracy: {val_accuracy:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved: {best_model_path}")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s)")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'val_loss': avg_val_loss,
        'val_accuracy': val_accuracy,
        'best_val_loss': best_val_loss,
        'epochs_no_improve': epochs_no_improve
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    if epochs_no_improve >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

model.load_state_dict(torch.load(best_model_path))
model.eval()

test_loss = 0
test_correct = 0
test_total = 0
all_labels = []
all_preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = crit(logits, labels)
        test_loss += loss.item()

        preds = logits.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

avg_test_loss = test_loss / len(test_loader)
test_accuracy = test_correct / test_total

print(f"Test loss: {avg_test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")

cm = confusion_matrix(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')

print("Confusion Matrix:")
print(cm)
print(f"F1 Score (weighted): {f1:.4f}")
