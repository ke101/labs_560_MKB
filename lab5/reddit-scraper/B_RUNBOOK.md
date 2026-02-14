# Preprocessing and Vectorization

## Goal
Populate:
- cleaned_posts (cleaned text, keywords, topic, anonymized author, timestamps)
- post_vectors (doc2vec embeddings, 100-d float32)

## Prerequisites
- MySQL running locally
- reddit_db exists and raw_posts imported (5097 rows)

## Environment Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Create .env in lab5/reddit-scraper:
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_DATABASE=reddit_db

## Run Preprocessing
python preprocess.py

Verify:
mysql -u root -e "USE reddit_db; SELECT COUNT(*) AS cleaned_cnt FROM cleaned_posts;"
mysql -u root -e "USE reddit_db; SELECT COUNT(*) AS raw_cnt FROM raw_posts;"

## Run Vectorization (Doc2Vec, 100-d)
python vectorize.py

Verify:
mysql -u root -e "USE reddit_db; SELECT COUNT(*) AS vec_cnt FROM post_vectors;"
mysql -u root -e "USE reddit_db; SELECT vector_dim, COUNT(*) AS cnt FROM post_vectors GROUP BY vector_dim;"
mysql -u root -e "USE reddit_db; SELECT post_id, vector_dim, OCTET_LENGTH(vector) AS bytes_len, model_name FROM post_vectors LIMIT 3;"
