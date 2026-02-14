import pandas as pd
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import sys
import ast

DB_USER = sys.argv[1]         
DB_PASSWORD = sys.argv[2]
DB_HOST = "127.0.0.1"
DB_PORT = "3306"
DB_NAME = "reddit_db"

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

def decode(vec_bytes):
    if isinstance(vec_bytes, str):
        vec_bytes = ast.literal_eval(vec_bytes)
    return np.frombuffer(vec_bytes, dtype=np.float32)

df = pd.read_sql("SELECT * FROM post_vectors", engine)

df['d_vector'] = df['vector'].apply(decode)
X = np.vstack(df['d_vector'].values)

kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
df['cluster_id'] = kmeans.fit_predict(X)
df = df[["post_id", "cluster_id"]]
df.to_sql(
    name="cluster_results",   
    con=engine,
    if_exists="append",       
    index=False
)