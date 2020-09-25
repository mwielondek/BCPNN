import numpy as np
import matplotlib.pyplot as plt

def dispmnist(x, ax=None, shape=(28,28)):
    """Display MNIST figure"""
    if ax is None:
        ax = plt
    ax.imshow(x.reshape(shape), cmap='gray_r')

def dispcompare(*vect, title=None, ylabels=None):
    """Display MNIST vectors vertically stacked above each other"""
    n_samples, _ = vect[0].shape
    n_vect = len(vect)
    fig, axs= plt.subplots(n_vect, n_samples, figsize=(2*n_samples, 1.5*n_vect))
    if title is not None:
        fig.suptitle(title)
    [axi.set_axis_off() for axi in axs.ravel()]
    if axs.ndim == 1:
        axs = [axs]
    if ylabels is None:
        ylabels = [''] * n_vect
    for v,ax_row,label in zip(vect, axs, ylabels):
        for i in range(n_samples):
            dispmnist(v[i], ax_row[i])
        ax_row[0].set_ylabel(label, rotation=0, ha='right', va='center')
        ax_row[0].axis('on')
        ax_row[0].set_frame_on(False)
        ax_row[0].get_xaxis().set_ticks([])
        ax_row[0].get_yaxis().set_ticks([])
    return fig, axs

def addsalt(X, prob=0.1):
    """Add salt to pattern with `prob` probability by using the complement"""
    xr = X.flatten()
    for i,_ in enumerate(xr):
        if np.random.choice([True, False], p=[prob, 1-prob]):
            xr[i] = max(0.8, 1 - xr[i])
    return xr.reshape(X.shape)
