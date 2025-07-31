import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def bert(text):
    model = AutoModelForSequenceClassification.from_pretrained("sentence-transformers/all-MiniLM-L12-v2", num_labels=3)
    model.load_state_dict(torch.load("src/bert/saved/bert_finetuned.pt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predicted_class = logits.argmax(dim=1).item()

    print(f"Input: {text}")
    if predicted_class == 0:
        print("Normal")
    elif predicted_class == 1:
        print("Offensive")
    elif predicted_class == 2:
        print("Hatespeech")

bert(content)