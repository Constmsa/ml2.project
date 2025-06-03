
import pandas as pd
from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
from pre_processing_functions import preprocess
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score


def som_cluster(df, som_size=3, iterations=5000, sigma=1.0, learning_rate=0.5):
    df = df.copy()
    # 1. Extract features
    X = df[['lifetime_spend_groceries', 'lifetime_spend_electronics',
        'typical_hour', 'lifetime_spend_vegetables',
        'lifetime_spend_nonalcohol_drinks', 'lifetime_spend_alcohol_drinks',
        'lifetime_spend_meat', 'lifetime_spend_fish', 'lifetime_spend_hygiene',
        'lifetime_spend_videogames', 'lifetime_spend_petfood',
        'lifetime_total_distinct_products']].values

    # 2. Initialize and train SOM
    som = MiniSom(x=som_size,
                  y=som_size, 
                  input_len=X.shape[1], 
                  sigma=sigma, 
                  learning_rate=learning_rate)
    
    som.random_weights_init(X)
    np.random.seed(42)
    som.train(X, num_iteration=iterations, verbose=True)

    # 3. Assign cluster to each data point
    df['som_cluster'] = ([som.winner(X[i]) for i in range(0, len(X))])
    return df

def kmeans_clustering(df, n_clusters = 9, random_state = 42):
    df = df.copy() 
    data= df[['lifetime_spend_groceries', 'lifetime_spend_electronics',
        'typical_hour', 'lifetime_spend_vegetables',
        'lifetime_spend_nonalcohol_drinks', 'lifetime_spend_alcohol_drinks',
        'lifetime_spend_meat', 'lifetime_spend_fish', 'lifetime_spend_hygiene',
        'lifetime_spend_videogames', 'lifetime_spend_petfood',
        'lifetime_total_distinct_products']]
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(data)

    # Add cluster column to the DataFrame
    data_with_clusters = data.copy()
    data_with_clusters['Kmeans_cluster'] = cluster_labels

    return data_with_clusters

def hierarchical_clustering(df, n_clusters= 9, linkage= 'ward') :
   
    df = df.copy()
    data= df[['lifetime_spend_groceries', 'lifetime_spend_electronics',
        'typical_hour', 'lifetime_spend_vegetables',
        'lifetime_spend_nonalcohol_drinks', 'lifetime_spend_alcohol_drinks',
        'lifetime_spend_meat', 'lifetime_spend_fish', 'lifetime_spend_hygiene',
        'lifetime_spend_videogames', 'lifetime_spend_petfood',
        'lifetime_total_distinct_products']]
    
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    cluster_labels = model.fit_predict(data)

    # Add the cluster column to the DataFrame
    data_with_clusters = data.copy()
    data_with_clusters['hierarchical_cluster'] = cluster_labels

    return data_with_clusters

def clustering(path):
    df = preprocess(path)
    df['som_cluster'] = som_cluster(df,som_size=3, iterations=5000, sigma=1.0, learning_rate=0.5)['som_cluster']
    df['Kmeans_cluster'] = kmeans_clustering(df, n_clusters=9, random_state=42)['Kmeans_cluster']
    df['hierarchical_cluster'] = hierarchical_clustering(df, n_clusters=9, linkage='ward')['hierarchical_cluster']

    return df

def plot_silhouette(df, feature_cols, cluster_col):
    X = df[feature_cols].values
    labels = df[cluster_col].values
    n_clusters = len(np.unique(labels))

    silhouette_vals = silhouette_samples(X, labels)
    silhouette_avg = silhouette_score(X, labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10

    for i in np.unique(labels):
        ith_cluster_vals = silhouette_vals[labels == i]
        ith_cluster_vals.sort()

        size_cluster_i = ith_cluster_vals.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_vals,
                         facecolor=color, edgecolor=color, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_xlabel("Coeficiente de Silhueta")
    ax.set_ylabel("Label do Cluster")
    ax.set_title(f"Silhouette Plot - {cluster_col}")
    plt.show()