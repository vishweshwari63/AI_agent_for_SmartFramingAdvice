# build_index.py
import os
import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

DATA_PATH = "data/farmer_advisor_dataset.csv"
MODEL_NAME = "all-MiniLM-L6-v2"  # small and effective; adjust if desired
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

def load_dataset(path):
    df = pd.read_csv(path)
    # Expected: a column 'question' or 'text' and 'answer' or 'advice'
    if 'question' in df.columns:
        texts = df['question'].astype(str).tolist()
    elif 'text' in df.columns:
        texts = df['text'].astype(str).tolist()
    else:
        # fallback: join all columns
        texts = df.astype(str).agg(' '.join, axis=1).tolist()
    # keep answers
    answers = df['answer'].astype(str).tolist() if 'answer' in df.columns else [""] * len(df)
    return texts, answers, df

def main():
    print("Loading dataset...")
    texts, answers, df = load_dataset(DATA_PATH)
    print(f"{len(texts)} rows loaded.")
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    batch_size = 64
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    # L2 normalize (helps cosine via inner product)
    faiss.normalize_L2(embeddings)
    d = embeddings.shape[1]
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(d)  # using inner product on normalized vectors == cosine similarity
    index.add(embeddings)
    # Save index and metadata
    faiss.write_index(index, os.path.join(OUT_DIR, "faiss_index.bin"))
    meta = {
        "texts": texts,
        "answers": answers,
        "df_columns": df.columns.tolist()
    }
    with open(os.path.join(OUT_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print("Index and metadata saved to", OUT_DIR)

if __name__ == "__main__":
    main()
