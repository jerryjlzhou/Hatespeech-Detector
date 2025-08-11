import pandas as pd
import ast
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F


# -----------------------------
# Config
# -----------------------------
BATCH_SIZE = 1024
EPOCHS = 100
LR = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using: {DEVICE}")

# -----------------------------
# Load data
# -----------------------------
def load_data(path):
    df = pd.read_csv(path)
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    targets = [ast.literal_eval(t) if isinstance(t, str) else [] for t in df["targets"]]
    return texts, labels, targets

train_texts, train_labels, train_targets = load_data("../../data/processed/train.csv")
val_texts, val_labels, val_targets = load_data("../../data/processed/val.csv")

# Label encoders
label_encoder = LabelEncoder()
y_train_label = label_encoder.fit_transform(train_labels)
y_val_label = label_encoder.transform(val_labels)

mlb = MultiLabelBinarizer()
y_train_targets = mlb.fit_transform(train_targets)
y_val_targets = mlb.transform(val_targets)

num_single_classes = len(label_encoder.classes_)
num_multi_labels = len(mlb.classes_)

# -----------------------------
# Embed texts (Frozen MiniLM)
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L12-v2")

X_train = embedder.encode(train_texts, convert_to_tensor=True, show_progress_bar=True)
X_val = embedder.encode(val_texts, convert_to_tensor=True, show_progress_bar=True)

# -----------------------------
# Dataset & Dataloader
# -----------------------------
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, y_label, y_multi):
        self.embeddings = embeddings
        self.y_label = y_label
        self.y_multi = y_multi

    def __len__(self):
        return len(self.y_label)

    def __getitem__(self, idx):
        return {
            "x": self.embeddings[idx],
            "label": torch.tensor(self.y_label[idx], dtype=torch.long),
            "multi": torch.tensor(self.y_multi[idx], dtype=torch.float32)
        }

train_dataset = EmbeddingDataset(X_train, y_train_label, y_train_targets)
val_dataset = EmbeddingDataset(X_val, y_val_label, y_val_targets)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# -----------------------------
# Model (Classifier on frozen embeddings)
# -----------------------------
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, num_multilabels):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.out_label = nn.Linear(prev_dim, num_classes)
        self.out_multi = nn.Linear(prev_dim, num_multilabels)

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.out_label(x), self.out_multi(x)


model = Classifier(
    input_dim=X_train.shape[1],
    hidden_dims=[512, 1024, 2048, 1024, 512],
    num_classes=num_single_classes,
    num_multilabels=num_multi_labels
).to(DEVICE)

# -----------------------------
# Training Setup
# -----------------------------
criterion_single = nn.CrossEntropyLoss()
criterion_multi = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# Training Loop with Early Stopping
# -----------------------------
best_f1 = 0
patience = 10
counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        x = batch["x"].to(DEVICE)
        y_label = batch["label"].to(DEVICE)
        y_multi = batch["multi"].to(DEVICE)

        optimizer.zero_grad()
        out_label, out_multi = model(x)

        loss1 = criterion_single(out_label, y_label)
        loss2 = criterion_multi(out_multi, y_multi)
        loss = loss1 + loss2

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Training Loss: {total_loss:.4f}")

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()
    all_preds_label = []
    all_true_label = []
    all_preds_multi = []
    all_true_multi = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(DEVICE)
            y_label = batch["label"]
            y_multi = batch["multi"]

            out_label, out_multi = model(x)

            preds_label = torch.argmax(out_label, dim=1).cpu()
            preds_multi = (torch.sigmoid(out_multi) > 0.5).cpu()

            all_preds_label.extend(preds_label.tolist())
            all_true_label.extend(y_label.tolist())
            all_preds_multi.extend(preds_multi.tolist())
            all_true_multi.extend(y_multi.tolist())

    acc = accuracy_score(all_true_label, all_preds_label)
    f1 = f1_score(all_true_multi, all_preds_multi, average='micro')

    print(f"Validation Label Accuracy: {acc:.4f} | Multi-label F1: {f1:.4f}")

    # -----------------------------
    # Early Stopping Check
    # -----------------------------
    if f1 > best_f1:
        best_f1 = f1
        counter = 0
        torch.save(model.state_dict(), "best_model.pt")
        print("✓ Model improved — saving!")
    else:
        counter += 1
        print(f"✗ No improvement. Patience: {counter}/{patience}")
        if counter >= patience:
            print("🛑 Early stopping triggered.")
            break

import torch.nn.functional as F

# Load best model
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
print("✅ Model loaded and ready!")

while True:
    text = input("\nEnter a sentence (or type 'exit' to quit): ").strip()
    if text.lower() == "exit":
        break

    # Encode text using frozen MiniLM
    embedding = embedder.encode([text], convert_to_tensor=True).to(DEVICE)

    with torch.no_grad():
        out_label, out_multi = model(embedding)

        # Single-label (softmax)
        probs_label = F.softmax(out_label, dim=1)
        conf_label, pred_label_idx = torch.max(probs_label, dim=1)
        pred_label = label_encoder.inverse_transform([pred_label_idx.item()])[0]

        # Multi-label (sigmoid)
        probs_multi = torch.sigmoid(out_multi).cpu().numpy()[0]
        threshold = 0.5
        pred_multi_labels = [mlb.classes_[i] for i, p in enumerate(probs_multi) if p > threshold]

    print(f"\n📌 Prediction:")
    print(f"  ➤ Label: {pred_label} (confidence: {conf_label.item():.4f})")
    print(f"  ➤ Targets: {pred_multi_labels if pred_multi_labels else 'None'}")
