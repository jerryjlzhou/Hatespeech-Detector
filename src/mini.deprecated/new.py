import os
import ast
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sentence_transformers import SentenceTransformer
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# =============================================================================
#                            BASE PARAMETERS
# =============================================================================

BATCH_SIZE = 2048
EPOCHS = 500
LEARNING_RATE = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_WEIGHTS = torch.tensor([5, 2, 3], dtype=torch.float32).to(DEVICE)

CRITERION_LABEL = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
CRITERION_TARGETS = nn.BCEWithLogitsLoss()
PATIENCE = 10


print(DEVICE)
print(torch.cuda.is_available())

# Ensure Consistency Across Experimentations.
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
#                          DATASET & DATALOADER
# =============================================================================


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels, targets):
        self.embeddings = embeddings
        self.labels = labels
        self.targets = targets

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "embedding": self.embeddings[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "targets": torch.tensor(self.targets[idx], dtype=torch.float32)
        }


def load_data(path):
    df = pd.read_csv(path)
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # Safely parse the string representation of Python lists
    # in the 'targets' column into actual Python list objects.
    targets = [ast.literal_eval(t)
               if isinstance(t, str) else [] for t in df["targets"]]

    return texts, labels, targets


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def load_training_data(embedding_model="all-MiniLM-L12-v2", reduce_dimensionality=True):
    # # Paths for cached embeddings
    # save_dir = f"../../data/embeddings/{embedding_model}"
    # os.makedirs(save_dir, exist_ok=True)
    # train_path = os.path.join(save_dir, "train.pt")
    # val_path = os.path.join(save_dir, "val.pt")
    # test_path = os.path.join(save_dir, "test.pt")

    # cached = all(os.path.exists(p) for p in [train_path, val_path, test_path])

    # if cached:
    #     print("✅ Loaded cached embeddings.")
    #     X_train = torch.load(train_path)
    #     X_val = torch.load(val_path)
    #     X_test = torch.load(test_path)
    #     input_dim = X_train.shape[1]

    #     _, train_labels, train_targets = load_data("../../data/processed/train.csv")
    #     _, val_labels, val_targets = load_data("../../data/processed/val.csv")
    #     _, test_labels, test_targets = load_data("../../data/processed/test.csv")
    # else:
    print("🧠 Generating embeddings...")
    train_texts, train_labels, train_targets = load_data("../../data/processed/train.csv")
    val_texts, val_labels, val_targets = load_data("../../data/processed/val.csv")
    test_texts, test_labels, test_targets = load_data("../../data/processed/test.csv")

    embedder = SentenceTransformer(embedding_model)
    input_dim = embedder.get_sentence_embedding_dimension()

    X_train = embedder.encode(train_texts, convert_to_tensor=True, show_progress_bar=True)
    X_val = embedder.encode(val_texts, convert_to_tensor=True, show_progress_bar=True)
    X_test = embedder.encode(test_texts, convert_to_tensor=True, show_progress_bar=True)

    # torch.save(X_train, train_path)
    # torch.save(X_val, val_path)
    # torch.save(X_test, test_path)
    # print("💾 Saved embeddings to disk.")

    # Perform dimensionality reduction (PCA)
    # if reduce_dimensionality:
    #     print("🔽 Reducing dimensionality...")
    #     pca = PCA(n_components=2)
    #     X_train_reduced = pca.fit_transform(X_train.cpu())

    #     # Convert labels to numeric values
    #     le = LabelEncoder()
    #     train_labels_numeric = le.fit_transform(train_labels)

    #     # Plot the reduced embeddings (train set)
    #     plt.figure(figsize=(10, 8))
    #     scatter = plt.scatter(X_train_reduced[:, 0], X_train_reduced[:, 1], c=train_labels_numeric, cmap='viridis', s=20, alpha=0.7)

    #     # Create a colorbar with class labels as ticks
    #     cbar = plt.colorbar(scatter)
    #     cbar.set_ticks(np.arange(len(le.classes_)))
    #     cbar.set_ticklabels(le.classes_)

    #     plt.title("2D PCA of Train Embeddings")
    #     plt.xlabel("Principal Component 1")
    #     plt.ylabel("Principal Component 2")
    #     plt.show()

    # Label and target encoding
    le = LabelEncoder()
    y_train_label = le.fit_transform(train_labels)
    y_val_label = le.transform(val_labels)
    y_test_label = le.transform(test_labels)
    num_labels = len(le.classes_)

    mlb = MultiLabelBinarizer()
    y_train_targets = mlb.fit_transform(train_targets)
    y_val_targets = mlb.transform(val_targets)
    y_test_targets = mlb.transform(test_targets)
    num_targets = len(mlb.classes_)

    # Wrap in DataLoaders
    train_dataset = EmbeddingDataset(X_train, y_train_label, y_train_targets)
    val_dataset = EmbeddingDataset(X_val, y_val_label, y_val_targets)
    test_dataset = EmbeddingDataset(X_test, y_test_label, y_test_targets)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return input_dim, num_labels, num_targets, train_loader, val_loader, test_loader, le


# =============================================================================
#                            CLASSIFIER MODEL
# =============================================================================


class Classifier(nn.Module):
    def __init__(self, input_dim, num_labels, num_targets, hidden_dims=[], dropout=0.3):
        super().__init__()

        if hidden_dims is None or len(hidden_dims) == 0:
            hidden_dims = [input_dim]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)

        self.out_label = nn.Linear(prev_dim, num_labels)
        self.out_targets = nn.Linear(prev_dim, num_targets)

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.out_label(x), self.out_targets(x)


# =============================================================================
#                             TRAINING LOOP
# =============================================================================


def train_one_epoch(model, train_loader, optimizer):
    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    scaler = GradScaler(device="cuda")

    for batch in train_loader:
        x_embedding = batch["embedding"].to(DEVICE)
        y_label = batch["label"].to(DEVICE)
        y_targets = batch["targets"].to(DEVICE)

        optimizer.zero_grad()

        with autocast(device_type='cuda'):
            out_label, out_targets = model(x_embedding)

            label_loss = CRITERION_LABEL(out_label, y_label)
            targets_loss = CRITERION_TARGETS(out_targets, y_targets)
            loss = 0.5 * label_loss + 1.5 * targets_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        preds = torch.sigmoid(out_targets).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y_targets.cpu().numpy())

        class_preds = torch.argmax(out_label, dim=1)
        total_correct += (class_preds == y_label).sum().item()
        total_samples += y_label.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples
    f1 = f1_score(all_targets, np.array(all_preds) > 0.3, average="micro")

    return avg_loss, accuracy, f1


def evaluate_model(model, val_loader):
    model.eval()

    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            x_embedding = batch["embedding"].to(DEVICE)
            y_label = batch["label"].to(DEVICE)
            y_targets = batch["targets"].to(DEVICE)

            out_label, out_targets = model(x_embedding)

            label_loss = CRITERION_LABEL(out_label, y_label)
            targets_loss = CRITERION_TARGETS(out_targets, y_targets)
            loss = 0.5 * label_loss + 1.5 * targets_loss

            total_loss += loss.item()

            preds = torch.argmax(out_label, dim=1)
            total_correct += (preds == y_label).sum().item()
            total_samples += y_label.size(0)

            all_preds.extend(torch.sigmoid(out_targets).cpu().numpy())
            all_targets.extend(y_targets.cpu().numpy())

    accuracy = total_correct / total_samples
    f1 = f1_score(all_targets, np.array(all_preds) > 0.3, average="micro")
    avg_loss = total_loss / len(val_loader)

    return accuracy, f1, avg_loss


def train_and_evaluate(model, train_loader, val_loader, output_path="model.pt"):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_f1 = 0
    patience_counter = 0

    train_losses = []
    train_accuracies = []
    train_f1s = []
    val_losses = []
    val_accuracies = []
    val_f1s = []

    for epoch in range(EPOCHS):
        print(f"⚙️  EPOCH {epoch+1}:")

        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, optimizer)
        val_acc, val_f1, val_loss = evaluate_model(model, val_loader)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_f1s.append(val_f1)
        train_f1s.append(train_f1)

        print(f"► Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"► Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            print("► Model Improved ✅")
            torch.save(model.state_dict(), output_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"► No Improvement {patience_counter}/{PATIENCE} ❌")
            if patience_counter >= PATIENCE:
                print("🛑 Early Stopping Triggered")
                break

    return train_losses, train_accuracies, train_f1s, val_losses, val_accuracies, val_f1s


def plot_confusion_matrix(model, data_loader, label_encoder, device=DEVICE, save_path="confusion_matrix.png"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            x = batch["embedding"].to(device)
            y = batch["label"].to(device)

            out_label, _ = model(x)
            preds = torch.argmax(out_label, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def main():
    set_seed(42)
    input_dim, num_labels, num_targets, train_loader, val_loader, test_loader, le = load_training_data()

    model = Classifier(input_dim, num_labels, num_targets).to(DEVICE)
    
    train_losses, train_accs, train_f1s, val_losses, val_accs, val_f1s = train_and_evaluate(model, train_loader, val_loader)

    # Final test set evaluation
    acc, f1, _ = evaluate_model(model, test_loader)
    print(f"✅ Test Set Accuracy: {acc:.4f} | F1: {f1:.4f}")

    # Plot confusion matrix on test set
    plot_confusion_matrix(model, test_loader, le)

    # Plot accuracy
    plt.figure()
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_plot.png")
    plt.show()

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_plot.png")
    plt.show()

    plt.figure()
    plt.plot(train_f1s, label="Train F1 Score")
    plt.plot(val_f1s, label="Val F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Train vs Validation F1 Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("train_val_f1_plot.png")
    plt.show()


if __name__ == '__main__':
    main()
