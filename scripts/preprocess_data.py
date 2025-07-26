import json
import numpy as np
import pandas as pd
from collections import Counter

# Paths
RAW_PATH = 'data/raw/dataset.json'
DIVISIONS_PATH = 'data/raw/post_id_divisions.json'
OUTPUT_DIR = 'data/processed/'

# Label mapping
LABEL_MAP = {
    'normal': 0,
    'offensive': 1,
    'hatespeech': 2
}

def majority_label(annotators):
    """
    Given a list of annotator dicts with 'label' keys,
    returns the majority label. In case of tie, picks one arbitrarily.
    """
    labels = [ann['label'] for ann in annotators]
    most_common, count = Counter(labels).most_common(1)[0]
    return most_common

def preprocess_entry(entry):
    """
    Convert a raw entry into a single record:
    - post_id
    - text (joined tokens)
    - label (majority)
    - label_id (mapped integer)
    - targets (list)
    """
    post_id = entry['post_id']
    tokens = entry.get('post_tokens', [])
    text = ' '.join(tokens)
    annots = entry.get('annotators', [])
    label = majority_label(annots)
    label_id = LABEL_MAP[label]
    # Collect unique targets
    targets = []
    for a in annots:
        for t in a.get('target', []):
            if t not in targets and t.lower() != 'none':
                targets.append(t)
    return {
        'post_id': post_id,
        'text': text,
        'label': label,
        'label_id': label_id,
        'targets': targets
    }

def load_data():
    with open(RAW_PATH, 'r') as f:
        raw = json.load(f)
    with open(DIVISIONS_PATH, 'r') as f:
        divs = json.load(f)
    return raw, divs

def build_datasets():
    raw, divs = load_data()
    # Process all entries
    all_records = [preprocess_entry(v) for v in raw.values()]
    df = pd.DataFrame(all_records)

    # Split
    train_ids = set(divs['train'])
    val_ids = set(divs['val'])
    test_ids = set(divs['test'])

    df_train = df[df['post_id'].isin(train_ids)].reset_index(drop=True)
    df_val = df[df['post_id'].isin(val_ids)].reset_index(drop=True)
    df_test = df[df['post_id'].isin(test_ids)].reset_index(drop=True)

    # Save
    df_train.to_csv(OUTPUT_DIR + 'train.csv', index=False)
    df_val.to_csv(OUTPUT_DIR + 'val.csv', index=False)
    df_test.to_csv(OUTPUT_DIR + 'test.csv', index=False)

    print(f"Saved: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

if __name__ == '__main__':
    build_datasets()
