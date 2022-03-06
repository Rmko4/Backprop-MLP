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


def load_data(path: Path = DEFAULT_PATH):
    u = np.load(path / 'u.npy')
    y = np.load(path / 'y.npy')
    return u, y


def plot_data(path: Path = DEFAULT_PATH, print_title=False):
    u, y = load_data(path)
    plt.figure(figsize=(4.4,3.4))
    scatter = plt.scatter(u[:, 0], u[:, 1], marker="o",
                         c=y, s=20, edgecolor="k")
    legend = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.gca().add_artist(legend)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    if print_title:
        plt.title(f'Scatterplot of dataset | N={len(y)}')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # %% Requires TeX
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 12,
        "font.serif": ["Computer Modern Roman"]})
    plot_data()