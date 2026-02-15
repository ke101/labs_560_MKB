import pandas as pd
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


DB_USER = "kwu"          
DB_PASSWORD = "X135y,79092506"
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
#k = 5 
k_range = range(5,20)
iner = []
s_scores = []
for k in k_range:
    # n_init='auto' suppresses a common scikit-learn warning
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X)
    iner.append(kmeans.inertia_)
    score = silhouette_score(X, labels)
    s_scores.append(score)
plt.figure(figsize=(8, 5))
plt.plot(k_range, iner, marker='o', linestyle='-', color='b')
plt.title('The Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.xticks(k_range)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("kmean.png")
plt.figure(figsize=(8, 5))
plt.plot(k_range, s_scores, marker='o', linestyle='-', color='green')
plt.title('Silhouette Score for Optimal $k$')
plt.xlabel('Number of Clusters ($k$)')
plt.ylabel('Silhouette Score')
plt.xticks(k_range)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("s_kmean.png")