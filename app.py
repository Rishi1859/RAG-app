import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.sklearn
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# === Sample Document Corpus (You can replace this with your own dataset) ===
documents = [
    "AI is transforming the future of work and automation.",
    "Football and cricket are popular sports in India.",
    "The government is launching new tech startups.",
    "Messi and Ronaldo are famous footballers.",
    "Data science and machine learning are key fields in AI.",
    "India won the cricket world cup.",
    "Startups are boosting the economy.",
    "Deep learning is part of machine learning.",
    "Cricket matches are fun to watch.",
    "AI will revolutionize healthcare."
]

# === Preprocessing Function ===
stop_words = set(stopwords.words('english'))

def preprocess(doc):
    words = word_tokenize(doc.lower())
    words = [w for w in words if w.isalpha() and w not in stop_words]
    return ' '.join(words)

cleaned_docs = [preprocess(doc) for doc in documents]

# === K-Means Clustering ===
tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=1)
X_tfidf = tfidf_vectorizer.fit_transform(cleaned_docs)

k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_tfidf)

print("\n📊 K-Means Clustering Results:")
for i, label in enumerate(kmeans.labels_):
    print(f"Doc {i+1}: Cluster {label}")

# === Latent Dirichlet Allocation (LDA) ===
count_vectorizer = CountVectorizer(max_df=0.9, min_df=1)
X_counts = count_vectorizer.fit_transform(cleaned_docs)

lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(X_counts)

print("\n🧠 LDA Topics:")
words = count_vectorizer.get_feature_names_out()
for i, topic in enumerate(lda.components_):
    top_words = [words[i] for i in topic.argsort()[-5:]]
    print(f"Topic {i+1}: {', '.join(top_words)}")

# === Optional: Document-topic distribution ===
doc_topic_dist = lda.transform(X_counts)
print("\n📄 Document-Topic Distribution (LDA):")
print(np.round(doc_topic_dist, 2))

# === Interactive Visualization ===
print("\n🧭 Opening interactive LDA visualization...")
pyLDAvis.enable_notebook()
vis = pyLDAvis.sklearn.prepare(lda, X_counts, count_vectorizer)
pyLDAvis.save_html(vis, 'lda_visualization.html')
print("Visualization saved to 'lda_visualization.html'")
