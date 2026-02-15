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

#Create visualizations
num_clusters = len(cluster_keywords)
fig, axes = plt.subplots(2, (num_clusters + 1) // 2, figsize=(15, 10))
fig.suptitle('Keyword Visualization by Cluster', fontsize=16, fontweight='bold')

axes = axes.flatten() if num_clusters > 1 else [axes]

colors = plt.cm.Set3(np.linspace(0, 1, num_clusters))

for idx, cluster_id in enumerate(sorted(cluster_keywords.keys())):
    ax = axes[idx]
    
    keywords = cluster_keywords[cluster_id][:10]
    words = [k[0] for k in keywords]
    counts = [k[1] for k in keywords]
    
    # Create bar chart display top keywords
    bars = ax.barh(words, counts, color=colors[idx])
    ax.set_xlabel('Frequency', fontweight='bold')
    ax.set_title(f'Cluster {cluster_id} (n={len(df[df["cluster_id"] == cluster_id])})', 
                 fontweight='bold', fontsize=12)
    ax.invert_yaxis()
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{int(width)}', ha='left', va='center', fontsize=9)

for idx in range(num_clusters, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('cluster_keywords.png', dpi=300, bbox_inches='tight')


# Create word clouds
fig2, axes2 = plt.subplots(1, num_clusters, figsize=(5*num_clusters, 5))
fig2.suptitle('Word Clouds by Cluster', fontsize=16, fontweight='bold')

if num_clusters == 1:
    axes2 = [axes2]

for idx, cluster_id in enumerate(sorted(cluster_keywords.keys())):
    if cluster_texts[cluster_id]:
        wordcloud = WordCloud(width=400, height=400, 
                             background_color='white',
                             colormap='Set3',
                             relative_scaling=0.5,
                             min_font_size=10).generate(cluster_texts[cluster_id])
        
        axes2[idx].imshow(wordcloud, interpolation='bilinear')
        axes2[idx].set_title(f'Cluster {cluster_id}', fontweight='bold', fontsize=12)
        axes2[idx].axis('off')

plt.tight_layout()
plt.savefig('cluster_wordclouds.png', dpi=300, bbox_inches='tight')


plt.close('all')

