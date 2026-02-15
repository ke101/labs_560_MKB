import pandas as pd
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
import sys

DB_USER = sys.argv[1]         
DB_PASSWORD = sys.argv[2]
DB_HOST = "127.0.0.1"
DB_PORT = "3306"
DB_NAME = "reddit_db"

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

sql_query = "select cr.post_id, cluster_id, title_clean from cluster_results cr join cleaned_posts cp on cr.post_id = cp.post_id "
df = pd.read_sql(sql_query, engine)
#print(df.head(5))
def preprocess_text(text, min_word_length=2):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) >= min_word_length]
    
    return words

cluster_keywords = {}
cluster_texts = {}

for cluster_id in df['cluster_id'].unique():
    cluster_data = df[df['cluster_id'] == cluster_id]
    all_words = []
    for text in cluster_data['title_clean']:
        all_words.extend(preprocess_text(text))
    word_counts = Counter(all_words)
    top_keywords = word_counts.most_common(10)
    cluster_keywords[cluster_id] = top_keywords
    cluster_texts[cluster_id] = ' '.join(all_words)


for k,v in cluster_keywords.items():
    k_lst = ""
    for i in v:
        k_lst = k_lst + i[0]
    cluster_keywords[k] = k_lst

df_o = pd.DataFrame(list(cluster_keywords.items()), columns=["cluster_id", "keyword"])
df_o = df_o.sort_values(by="cluster_id")
df_o.to_sql(
    name="cluster_keywords",   
    con=engine,
    if_exists="append",       
    index=False
)