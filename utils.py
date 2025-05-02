import matplotlib.pyplot as plt
import pandas as pd
import math
from minisom import MiniSom
import numpy as np
from matplotlib.patches import Patch

def plot_neurons_distances(matrix, neurons, highlight_point, starting_point):
  '''
  Plots neurons and distances to scatters in a 2-Dimensional plot.

  Arguments:
  - X(np.array): Array of two-dimensions.
  - neurons(list): List of neurons to plot.
  - highlight_point(tuple): x-y pair we want to highlight
  - starting_point(tuple): point to calculate distances from

  Returns:
  - None, but a plot is shown.
  '''

  fig, ax = plt.subplots()

  plt.xlim(0, 10)
  plt.scatter(matrix[:,0], matrix[:,1])
  plt.scatter(highlight_point[0], highlight_point[1], s=200, c='blue')
  for neuron in neurons:
    plt.scatter(neuron[0], neuron[1], c='orange', s=400, edgecolor='black')

  for neuron in neurons:
    plt.arrow(starting_point[0], 
              starting_point[1], 
              neuron[0]-starting_point[0], neuron[1]-starting_point[1],
              head_width=0, head_length=0, length_includes_head=True, color='black')
    
    x_dist = (starting_point[0]-neuron[0])**2
    y_dist = (starting_point[1]-neuron[1])**2

    euclidean = math.sqrt(x_dist+y_dist)
    ax.annotate(str(euclidean), xy=(neuron[0], neuron[1]))
    
    
def build_lattice(len_x, len_y):
  '''
  Builds lattice of len_x times len_y.

  Arguments:
  - len_x(int): Number of columns.
  - len_y(int): Number of rows.

  Return:
  - None, but a plot is shown
  '''
  grid = np.zeros((len_x, len_y))

  # Create the figure and axes for the plot
  fig, ax = plt.subplots()

  # Plot the grid using imshow
  ax.imshow(grid, cmap='binary', vmin=0, vmax=0)

  # Add gridlines to the plot
  ax.grid(True, which='both', color='black', linewidth=1)

  # Set the ticks for the x and y axes
  ax.set_xticks(np.arange(len_x+1))
  ax.set_yticks(np.arange(len_y+1))

  # Remove the tick labels
  ax.set_xticklabels(list(range(0,len_x+1)))
  ax.set_yticklabels(list(range(0,len_y+1))[::-1])
 
 
def visualize_data_points_grid(data, scaled_data, som_model, color_variable, color_dict):
  '''
  Plots scatter data points on top of a grid that represents
  the self-organizing map. 

  Each data point can be color coded with a "target" variable and 
  we are plotting the distance map in the background.

  Arguments:
  - som_model(minisom.som): Trained self-organizing map.
  - color_variable(str): Name of the column to use in the plot.

  Returns:
  - None, but a plot is shown.
  '''

  # Subset target variable to color data points
  target = data[color_variable]

  fig, ax = plt.subplots()

  # Get weights for SOM winners
  w_x, w_y = zip(*[som_model.winner(d) for d in scaled_data])
  w_x = np.array(w_x)
  w_y = np.array(w_y)

  # Plot distance back on the background
  plt.pcolor(som_model.distance_map().T, cmap='bone_r', alpha=.2)
  plt.colorbar()

  # Iterate through every data points - add some random perturbation just
  # to avoid getting scatters on top of each other.
  for c in np.unique(target):
      idx_target = target==c
      plt.scatter(w_x[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8,
                  w_y[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8, 
                  s=50, c=color_dict[c], label=c)

  ax.legend(bbox_to_anchor=(1.2, 1.05))
  plt.grid()
  plt.show()
  
  
def classify(som, scaled_data, financial_data):
    """
    Classifies each sample in data in one of the classes definited
    using the method labels_map.
    Returns a list of the same length of data where the i-th element
    is the class assigned to data[i].
    Arguments:
    - som(minisom.som): Trained som.
    - scaled_data(pd.DataFrame): Data to obtain winner node and classify according
    to most common class of target variable.
    - financial_data(pd.DataFrame): Data with with Sector information.
    
    Returns:
    - result(list): List with classified sector based on most representative
    sector on the BMU
    """
    
    # Get map of Labels
    winmap = som.labels_map(scaled_data, financial_data.Sector)
    
    # Create default class with most common
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    
    # Check sector of the winner in the loop
    result = []
    for d in scaled_data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result
    
    
def plot_feature_influence(trained_som, data):
  '''
  Plots feature influence on trained som nodes.

  Arguments:
  - trained_som(minisom.som): Trained self-organizing map.
  - data(pd.DataFrame): pandas dataframe with data fed to
  the self-organizing map.
  
  Returns:
  - None, but a plot is shows
  '''
  
  # Get feature names
  feature_names = data.columns

   # Get weights from SOM
  W = trained_som.get_weights()
  
  # Initialize lattice figure
  plt.figure(figsize=(10, 10))
  
  # Iterate on matplotlib grid and feature weights
  # on nodes.
  
  for i, f in enumerate(feature_names):
      plt.subplot(3, 3, i+1)
      plt.title(f)
      plt.pcolor(W[:,:,i].T, cmap='coolwarm')
      plt.xticks(np.arange(15+1))
      plt.yticks(np.arange(15+1))
  plt.tight_layout()
  plt.show()
  
  
def plot_most_important_variable(trained_som, features):
  '''
  Plots most important variable for each unit in the SOM.

  Arguments:
  - trained_som(MiniSom object): Trained Self-Organizing-Map.
  - features(list): List of columns used in the SOM training.

  Returns:
  - None, but a plot is shown
  '''

  W = trained_som.get_weights()

  plt.figure(figsize=(8, 8))
  for i in np.arange(W.shape[0]):
      for j in np.arange(W.shape[1]):
          feature = np.argmax(W[i, j , :])
          plt.plot([i+.5], [j+.5], 'o', color='C'+str(feature),
                  marker='s', markersize=24)

  legend_elements = [Patch(facecolor='C'+str(i),
                          edgecolor='w',
                          label=f) for i, f in enumerate(features)]

  plt.legend(handles=legend_elements,
            loc='center left',
            bbox_to_anchor=(1, .95))
          
  plt.xlim([0, 15])
  plt.ylim([0, 15])
  plt.show()


def train_plot_som(X):
  '''
  Trains and plots SOM on a toy example X.

  Arguments:
  - X(np.array): Simple array to apply SOM on.

  Returns:
  - None, but a plot with the trained SOM is shown.
  '''

  fig, ax = plt.subplots()

  plt.scatter(X[:,0], X[:,1])

  som = MiniSom(
      2, 2, 2, sigma=1, 
      learning_rate=1, neighborhood_function='gaussian', random_seed=42)
  som.train(X, 100)

  nodes = som.get_weights().reshape(4, 2)
  plt.scatter(nodes[:,0], nodes[:,1], c='purple', s=400, edgecolor='black')
  plt.show()