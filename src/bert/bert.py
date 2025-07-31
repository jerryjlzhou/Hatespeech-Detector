from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import umap
import hdbscan
import re
import numpy as np

def classify_speech(doc: str) -> float:
    docs = preprocess_sentence(doc)
    embed = vectorise_sentence(docs)


	# Excludes short examples (may not be required due to the nature of the problema)
    # if len(embed) < 3:
    #     return cosine_similarity(
    #         np.mean(embed1, axis=0, keepdims=True),
    #     )[0][0]

    u_embed1 = dim_reduce(embed1)
    u_embed2 = dim_reduce(embed2)

    # Clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
    labels1 = clusterer.fit_predict(u_embed1)
    label_set1 = sorted(set(labels1) - {-1})
    labels2 = clusterer.fit_predict(u_embed2)
    label_set2 = sorted(set(labels2) - {-1})

    top1 = topic_representation(label_set1, labels1, docs1)
    top2 = topic_representation(label_set2, labels2, docs2)

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([top1, top2])
    return cosine_similarity(tfidf[0], tfidf[1])[0][0]


def preprocess_sentence(doc: str) -> list[str]:
    sentences = re.split(r'\.\s+', doc)
    processed = []
    for sentence in sentences:
        sentence = sentence.lower()
        words = [word for word in sentence.split() if word not in ENGLISH_STOP_WORDS]
        if words:
            processed.append(" ".join(words))
    return processed

def vectorise_sentence(docs: list[str]) -> any:
    model = SentenceTransformer('../models/teambuilder')
    embeddings = model.encode(docs, show_progress_bar=False)
    # nan_mask = ~np.isnan(embeddings).any(axis=1)
    # embeddings = embeddings[nan_mask]
    return embeddings

def dim_reduce(embed: any) -> any:
    neighbors = max(2, min(5, len(embed) - 1))
    components = min(5, len(embed) - 1)
    reducer1 = umap.UMAP(n_components=components, random_state=42, n_neighbors=neighbors)
    return reducer1.fit_transform(embed)

def topic_representation(label_set: any, labels: any, docs: list[str]) -> str:
    label_to_index = {label: idx for idx, label in enumerate(label_set)}
    clusters = [[] for _ in label_set]
    for i, label in enumerate(labels):
        if label != -1:
            clusters[label_to_index[label]].append(docs[i])

    cluster_texts = [" ".join(docs) for docs in clusters]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cluster_texts)
    terms = vectorizer.get_feature_names_out()
    top_terms_per_cluster = []

    for row in tfidf_matrix:
        row_array = row.toarray().flatten().argsort()
        sorted_desc = row_array[::-1]
        top_indices = sorted_desc[:10]
        
        top_terms = []
        for i in top_indices:
            top_terms.append(terms[i])
        
        top_terms_per_cluster.append(top_terms)
    return " ".join([" ".join(cluster) for cluster in top_terms_per_cluster])