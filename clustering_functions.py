import umap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from minisom import MiniSom
import numpy as np

def som_cluster(df, features, som_size=3, iterations=5000, sigma=1.0, learning_rate=0.5):
   
    df = df.copy()
    
    # 1. Extract features
    X = df[features].values

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


