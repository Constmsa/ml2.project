import umap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def umap_soms(df, features, target):
    #Fit UMAP
    umap_object = umap.UMAP(n_neighbors=5, random_state=2)
    umap_embedding = umap_object.fit_transform(df[features])

    #Add UMAP coordinates to df
    df['umap_1'] = umap_embedding[:, 0]
    df['umap_2'] = umap_embedding[:, 1]

    #Plot UMAP colored by clusters
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='umap_1', y='umap_2', data=df, hue=df[target].astype(str),
                    palette='tab20', s=60, alpha=0.8)
    plt.title("UMAP Clusters - SOM Clustering")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='SOM Cluster')
    plt.tight_layout()
    plt.show()