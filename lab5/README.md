# 560 LAB5 mkb

> **Brief Description:** A Python-based pipeline that scrapes data about web seurity from Reddit, processes the text, and groups them into thematic clusters using K-Means and TF-IDF.

### Overview
* **Target Data:** Data scraping from Reddit
* **Clustering Objective:** Group Similar Messages
* **Core Technologies:** Web scraping, Embedding, Kmeans

### Features
* 🕷️ **Automated Scraping:** Handles pagination, dynamic content rendering, and rate limiting.
* 🧹 **Data Preprocessing:** Cleans HTML tags, handles missing values, and normalizes text/features.
* 🧠 **Unsupervised Learning:** Implements `<Kmeans>` to discover natural groupings within the scraped dataset.
* 📊 **Visualization:** Generates 2D/3D scatter plots of the clusters`.

### Prequistises
* Python 3.9+
### Environment Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Create .env in lab5/reddit-scraper:
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_DATABASE=reddit_db

### Run Scraper
`python scrape.py [data_length] --method [method]`

### Run Preprocessing
`python preprocess.py`

Verify:
mysql -u root -e "USE reddit_db; SELECT COUNT(*) AS cleaned_cnt FROM cleaned_posts;"
mysql -u root -e "USE reddit_db; SELECT COUNT(*) AS raw_cnt FROM raw_posts;"

### Run Vectorization (Doc2Vec, 100-d)
`python vectorize.py`

Verify:
mysql -u root -e "USE reddit_db; SELECT COUNT(*) AS vec_cnt FROM post_vectors;"
mysql -u root -e "USE reddit_db; SELECT vector_dim, COUNT(*) AS cnt FROM post_vectors GROUP BY vector_dim;"
mysql -u root -e "USE reddit_db; SELECT post_id, vector_dim, OCTET_LENGTH(vector) AS bytes_len, model_name FROM post_vectors LIMIT 3;"

### Run Cluster (Kmean) 
`python cluster.py [mysql_username_password]`
`python keyword.py [mysql_username_password]`

### Whole Pipeline
`python main.py [interval_num] [mysql_config]`

### Contributions
[scrape.py](reddit_scraper/scrape.py): Mingtao Ding

[database.py](reddit_scraper/database.py): Mingtao Ding

[setup_db.sql](reddit_scraper/settup_db.sql) Mingtao Ding

[preprocess.py](reddit_scraper/preprocess.py): Yi-Hsien Lou

[vectorize.py](reddit_scraper/vectorize.py): Yi-Hsien Lou

[cluster.py](reddit_scraper/cluster.py): Ke Wu

[keyword.py](reddit_scraper/keyword.py): Ke Wu

[find_optimal_k.py](reddit_scraper/find_optimal_k.py): Ke Wu


### More details about the project in README.pdf









