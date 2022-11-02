#!/home/ckg/miniconda3/envs/pt/bin/python

import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt 


def get_2d_items(dm, lv, N=None):
    """
    dm : data_matrix (2D matrix of projected X data)
    lv : labels_vector (1d vector of labels)
    """
    x, y = dm[:N, 0].real, dm[:N, 1].real #always dims 0 & 1 for 2D arrays
    n_labels = lv[:N]
    return x, y, n_labels #for 2D visualization

def get_3d_items(dm, lv, N=None):
    x, y, z = dm[:N, 0].real, dm[:N, 1].real, dm[:N, 0].real #always dims 0, 1 & 2 for 3D arrays
    n_labels = lv[:N]
    return x, y, z, n_labels #for 3D visualization
