import numpy as np

from sklearn.datasets import make_moons, make_s_curve, make_swiss_roll
from experiments.utils.constants import RANDOM_SEED

N = 1500

def generate_moon(n=N):
    Xm, _ = make_moons(n_samples=(n, 0), noise=0, random_state=RANDOM_SEED)
    theta = np.arctan2(Xm[:, 1], Xm[:, 0])
    ym = (theta - theta.min()) / (theta.max() - theta.min())

    Xs, _ = make_s_curve(n_samples=n, noise=0, random_state=RANDOM_SEED)
    z = Xs[:, 2].copy()

    X = Xm.copy()
    y = ym.copy()
    X = np.c_[X, z]

    order = [0, 2, 1]
    X = X[:, order]

    return X, y

def generate_s_curve(n=N):
    Xs, ys = make_s_curve(n_samples=n, noise=0, random_state=RANDOM_SEED)

    X = Xs.copy()
    y = ys.copy()

    return X, y

def generate_swiss_roll(n=N):
    Xsr, ysr = make_swiss_roll(n_samples=n, noise=0, random_state=RANDOM_SEED)

    X = Xsr.copy()
    y = ysr.copy()

    return X, y
