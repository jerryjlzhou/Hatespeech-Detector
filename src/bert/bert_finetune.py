import pandas as pd
from transformers import AutoTokenizer

data = pd.read_csv("../../data/processed/train.csv")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
encoded_text = []
for text in data['text']:
	encoded = tokenizer(text)
	encoded_text.append(encoded)