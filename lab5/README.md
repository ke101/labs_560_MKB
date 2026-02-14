# 560 LAB5 mkb

> **Brief Description:** A Python-based pipeline that scrapes data about web seurity from Reddit, processes the text, and groups them into thematic clusters using K-Means and TF-IDF.

### Table of Contents
* [Overview](#overview)
* [Features](#features)
* [Architecture & Workflow](#architecture--workflow)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Configuration](#configuration)
* [Usage](#usage)
  * [1. Running the Scraper](#1-running-the-scraper)
  * [2. Running the Clustering Model](#2-running-the-clustering-model)
* [Results & Visualization](#results--visualization)
* [Project Structure](#project-structure)

---

### Overview
* **Target Data:** Data scraping from Reddit
* **Clustering Objective:** Group Similar Messages
* **Core Technologies:** ``

### Features
* 🕷️ **Automated Scraping:** Handles pagination, dynamic content rendering, and rate limiting.
* 🧹 **Data Preprocessing:** Cleans HTML tags, handles missing values, and normalizes text/features.
* 🧠 **Unsupervised Learning:** Implements `<Kmeans>` to discover natural groupings within the scraped dataset.
* 📊 **Visualization:** Generates 2D/3D scatter plots of the clusters`.

### Architecture & Workflow

1. **Data Ingestion:**
### Prequistises
* Python 3.9+


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

