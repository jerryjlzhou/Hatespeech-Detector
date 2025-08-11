import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd

class BertData(Dataset):
	def __init__ (self, csv_file, tokenizer):
		self.data = pd.read_csv(csv_file)
		self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
	
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		text = self.data.loc[idx, "text"]
		label = self.data.loc[idx, "label_id"]

		encoded = self.tokenizer(text, padding='max_length', return_tensors='pt')
		item = {key: val.squeeze(0) for key, val in encoded.items()}
		item['labels'] = torch.tensor(label)

		return item
