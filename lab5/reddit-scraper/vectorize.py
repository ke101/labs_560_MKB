import re
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from database import get_db


WHITESPACE_RE = re.compile(r"\s+")


def tokenize(text: str) -> List[str]:
    text = text or ""
    text = text.lower()
    text = WHITESPACE_RE.sub(" ", text).strip()
    tokens = [t for t in text.split(" ") if len(t) >= 2]
    return tokens


def load_cleaned_corpus(db, limit: Optional[int] = None) -> Tuple[List[str], List[List[str]]]:
    sql = """
        SELECT post_id, title_clean, body_clean, ocr_text
        FROM cleaned_posts
        ORDER BY created_dt DESC
    """
    if limit is not None:
        sql += " LIMIT %s"
        rows = db.execute_query(sql, (limit,), fetch=True)
    else:
        rows = db.execute_query(sql, fetch=True)

    post_ids: List[str] = []
    tokens_list: List[List[str]] = []

    for r in rows:
        post_id = r["post_id"]
        title = r.get("title_clean") or ""
        body = r.get("body_clean") or ""
        ocr = r.get("ocr_text") or ""

        combined = (title + " " + body + " " + ocr).strip()
        tokens = tokenize(combined)
        if not tokens:
            tokens = ["empty"]

        post_ids.append(post_id)
        tokens_list.append(tokens)

    return post_ids, tokens_list


def load_missing_vector_post_ids(db) -> List[str]:
    sql = """
        SELECT cp.post_id
        FROM cleaned_posts cp
        LEFT JOIN post_vectors pv ON cp.post_id = pv.post_id
        WHERE pv.post_id IS NULL
        ORDER BY cp.created_dt DESC
    """
    rows = db.execute_query(sql, fetch=True)
    return [r["post_id"] for r in rows]


def train_doc2vec(tagged_docs: List[TaggedDocument], vector_size: int = 100) -> Doc2Vec:
    model = Doc2Vec(
        vector_size=vector_size,
        window=8,
        min_count=2,
        workers=4,
        epochs=20,
        dm=1,
        negative=10,
        seed=42,
    )
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=len(tagged_docs), epochs=model.epochs)
    return model


def main(vector_size: int = 100, batch_size: int = 500) -> None:
    db = get_db()

    missing_ids = load_missing_vector_post_ids(db)
    if not missing_ids:
        print("No missing vectors. post_vectors is already complete.")
        return

    print(f"Missing vectors: {len(missing_ids)}")

    # Train on full corpus (stable embeddings), then write only missing vectors
    post_ids, tokens_list = load_cleaned_corpus(db)
    print(f"Loaded cleaned corpus for training: n_docs={len(post_ids)}")

    tagged_docs = [TaggedDocument(words=tokens_list[i], tags=[post_ids[i]]) for i in range(len(post_ids))]

    print("Training Doc2Vec model...")
    model = train_doc2vec(tagged_docs, vector_size=vector_size)
    print("Doc2Vec training done.")

    total = 0
    batch: List[Dict[str, Any]] = []

    for post_id in missing_ids:
        # If the post_id exists in training tags, we can use model.dv directly.
        # Otherwise, we infer a vector from tokens (fallback).
        if post_id in model.dv:
            vec = model.dv[post_id]
        else:
            idx = post_ids.index(post_id)
            vec = model.infer_vector(tokens_list[idx])

        vec = np.asarray(vec, dtype=np.float32)
        batch.append({"post_id": post_id, "vector": vec, "model_name": "doc2vec"})

        if len(batch) >= batch_size:
            db.insert_vectors_batch(batch)
            total += len(batch)
            print(f"Stored missing vectors: {total}/{len(missing_ids)}")
            batch = []

    if batch:
        db.insert_vectors_batch(batch)
        total += len(batch)
        print(f"Stored missing vectors: {total}/{len(missing_ids)}")

    print("Vectorization done.")


if __name__ == "__main__":
    main()
