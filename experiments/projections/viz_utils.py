#!/home/ckg/miniconda3/envs/pt/bin/python

import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt


def get_2d_items(dm, lv, N=None):
    """
    Parameters
        dm : data_matrix (2D matrix of projected X data)
        lv : labels_vector (1d vector of labels)
    """
    x, y = dm[:N, 0].real, dm[:N, 1].real #always dims 0 & 1 for 2D arrays
    n_labels = lv[:N]
    return x, y, n_labels #for 2D visualization

def get_3d_items(dm, lv, N=None):
    x, y, z = dm[:N, 0].real, dm[:N, 1].real, dm[:N, 2].real #always dims 0, 1 & 2 for 3D arrays
    n_labels = lv[:N] #the labels for the corresponding N instances
    return x, y, z, n_labels #for 3D visualization


# Given a Model & a Index -- Plot Original Image & Reconstruction
def plot_img_recon(model, idx):
    x, y = dataset.__getitem__(idx)[0], dataset.__getitem__(idx)[1]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5, 5))

    # Plot Original
    ax[0].imshow(x.squeeze(0))
    ax[0].set_title(f"Original Instance - {y}")

    # Compute Reconstruction and Reshape
    x = x.reshape(1, 784)
    x = x.to(device) #i.e; device = "cuda" if torch.cuda.is_available() else "cpu"
    out = model(x)
    out = out.reshape(28, 28).to("cpu").detach().numpy() #get output, reshape to img matrix, move to CPU as a Numpy Array

    # Plot Reconstruction
    ax[1].imshow(out)
    ax[1].set_title(f"Reconstruction - {y}")
