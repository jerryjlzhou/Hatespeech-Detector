<p align='center'>
<img src="https://cdn-icons-png.flaticon.com/512/3100/3100333.png" width="75" />
</p>
<h1 align='center'>COMP9444 Group Project</h1>
<h2 align='center'>
Multi-Class Hate Speech and Offensive Language Detection in Social Media (026)
</h2>

## Processed dataset
In the `data/processed` directory, there are three CSVs, `train.csv`, `val.csv`, `test.csv`. This data is split in an **80-10-10** partition. View the preprocessing script at `/scripts/preprocess_data.py`.

To label the data, the **majority label** was chosen based on the three annotations. If there is a three way tie (e.g normal, offensive, hatespeech), an **arbitrary label** is assigned to the post.

Each CSV contains the following columns:
* **post_id** (string): Unique ID
* **text** (string): Full post text
* **label** (string): {normal, offensive, hatespeech}
* **label_id** (int): mapped to {0, 1, 2} respectively
* **targets** (List[string]): list of targets mentioned (empty if none)

**Example format:**
![alt text](image-2.png)

### Loading the data
```python
import pandas as pd

# Paths
train_path = 'data/processed/train.csv'
val_path = 'data/processed/val.csv'
test_path = 'data/processed/test.csv'

# Read
df_train = pd.read_csv(train_path)
df_val = pd.read_csv(val_path)
df_test = pd.read_csv(test_path)
```


- [Project Report Doc](https://docs.google.com/document/d/1J-bV2ESFtu3zjjpIabKhCD_pXtCT_vyP9_YbnlqTB4A/edit?usp=sharing)

- [Presentation Slides](https://docs.google.com/presentation/d/16rYcF_tRftwAjX_Pcsc6GPDLl-_3RfIK7lNZzRy2ieQ/edit?usp=sharing)

## Options for implementation

### First Model (No-context)
- all-MiniLM-L12-v2 (Andrew)
- TF-IDF + SVM (Jerry + Leslie)

### Second Model (handles context edge cases)
- Bert (Nathan)

### Utils
- Split Dataset (Jerry)
- Dataset Extraction (Those outside of Kaggle etc.)

### Questions to be answered
- How do we plug all of these together?
