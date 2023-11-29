import json
import numpy as np
import gensim.downloader as api
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_rand_score
import random


# Convert sentence to glove embedding
def sentence_to_embedding(sentence, model):
    words = sentence.split()
    sentence_embedding = np.zeros(model.vector_size)
    count = 0
    for word in words:
        if word in model:
            sentence_embedding += model[word]
            count += 1
    return sentence_embedding


# clustering
def spherical_kmeans(embeddings, num_clusters=5):
    normalized_embeddings = normalize(embeddings)
    kmeans = KMeans(n_clusters=num_clusters).fit(normalized_embeddings)
    return kmeans.labels_


# map categories to numerical labels
def map_categories_to_labels(categories):
    unique_categories = list(set(categories))
    category_to_label = {category: i for i, category in enumerate(unique_categories)}
    labels = [category_to_label[category] for category in categories]
    return labels


def main():
    json_file = 'News_Category_Dataset_v3.json'

    # could change size of embedding by modifying 200 to -> 100 (check gensim for supported models)
    model = api.load("glove-wiki-gigaword-200")

    # read in headlines
    headlines = []
    categories = []
    with open(json_file, 'r') as file:
        for line in file:
            article = json.loads(line)
            headline = article['headline']
            category = article['category']
            headlines.append(headline)
            categories.append(category)

    # convert headlines to embeddings
    embeddings = np.array([sentence_to_embedding(headline, model) for headline in headlines])

    # MODIFY NUM CLUSTERS
    clusters = spherical_kmeans(embeddings, num_clusters=42) 

    # map categories to numerical labels and evaluate clustering alg
    category_labels = map_categories_to_labels(categories)
    ari_score = adjusted_rand_score(category_labels, clusters)
    print(f"Adjusted Rand Index: {ari_score}")

    # group headlines by predicted cluster
    cluster_to_headlines = {i: [] for i in range(42)}
    for idx, cluster_label in enumerate(clusters):
        cluster_to_headlines[cluster_label].append(headlines[idx])

    # print examples
    for cluster, headlines in cluster_to_headlines.items():
        print(f"\nCluster {cluster}:")
        sample_headlines = random.sample(headlines, min(5, len(headlines)))
        for headline in sample_headlines:
            print(f" - {headline}")

if __name__ == "__main__":
    main()
