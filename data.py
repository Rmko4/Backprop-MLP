from sklearn.datasets import make_gaussian_quantiles as mgq
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_FOLDER = 'data'
DEFAULT_PATH = Path(DEFAULT_FOLDER)

def create_dataset(N, save_path: Path = DEFAULT_FOLDER):
    gq = mgq(
        mean=None, cov=0.7, n_samples=N, n_features=2,
        n_classes=2, shuffle=True, random_state=None)
    u, y = gq
    if save_path:
        np.save(save_path / 'u', u)
        np.save(save_path / 'y', y)
    return gq


def load_data(path: Path = DEFAULT_FOLDER):
    u = np.load(path / 'u.npy')
    y = np.load(path / 'y.npy')
    return u, y


def plot_data(path: Path = DEFAULT_FOLDER):
    u, y = load_data(path)
    fig, ax = plt.subplots()
    scatter = ax.scatter(u[:, 0], u[:, 1], marker="o",
                         c=y, s=25, edgecolor="k")
    legend = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Scatterplot of dataset | N={len(y)}')
    plt.show()
