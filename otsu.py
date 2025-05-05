import matplotlib.pyplot as plt
import numpy as np
import math
from torch import nn
import torch

def findGreatestThreshold(img):
  img_tensor = torch.tensor(img).float()
  # First we flatten the image to get the 1D matrix.
  flatten_image = nn.Flatten()
  image_flattened = flatten_image(img_tensor)

  # Then we get the axial values of the image.
  counts, binominal_values= np.histogram(image_flattened, bins=6)
  # Counts = The number of times that intensity was displayed
  # Binominal_values = The levels of intensity in our case its [0., 0.16666667, 0.33333334, 0.5, 0.66666669, 0.83333331, 1.]

  # We get the total weight for our calculations.
  total_weight = np.sum(counts)

  # We assign an empty list to store the threshold values.
  List = []

  # We iterate for all the threshold values to get the highest.
  for i in range(len(binominal_values)-1):
    current_weight = 0
    current_intensity = 0
    foreground_intensity = 0
    foreground_weight = 0

    # We iterate for the background calculations.
    for j in range(i+1):
      current_weight += counts[j]
      current_intensity += counts[j]*binominal_values[j]

    # We iterate for the foreground calculations.
    for k in range(len(binominal_values)-2, i,-1):
      foreground_intensity += counts[k]*binominal_values[k]

    foreground_weight = total_weight - current_weight

    # We calculate the mean intensities for every threshold.
    Ub = current_intensity / current_weight
    Uf = foreground_intensity / foreground_weight if foreground_weight != 0 else 0.0

    # We calculate the weights for every threshold.
    Wb = current_weight / total_weight
    Wf = 1 - Wb

    # We calculate the variance for every threshold.
    variance = math.sqrt((Wb*Wf)*((Ub-Uf)*(Ub-Uf)))

    # We append the threshold values to the empty list to apply max() and get the optimal threshold value.
    List.append(variance)

    best_variance = max(List)

    thresholded_im = np.zeros(img.shape)

    thresholded_im[img >= best_variance] = 1

    return thresholded_im
     
