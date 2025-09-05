import numpy as np


def seasonal(t):
    a1, a2, a3 = np.random.uniform(-5, 5, 3)  # amplitude
    b1, b2, b3 = np.random.uniform(0, 1, 3)  # phase
    series = (
        a1 * np.sin(2 * np.pi * (t / 7 + b1))
        + a2 * np.sin(2 * np.pi * (t / 30 + b2))
        + a3 * np.sin(2 * np.pi * (t / 365 + b3))
    )
    return series


def trend(t):
    a4, a5 = np.random.uniform(-1, 1, 2)
    series = a4 + a5 * t / 365
    return series


def noise(y):
    return np.random.normal(0, 0.25 * np.mean(np.abs(y)), len(y))


def simple_seasonal(t):
    a1 = np.random.uniform(-5, 5, 1)
    series = a1 * np.sin(2 * np.pi * t / 7)
    return series


def simple(L):
    t = np.arange(L)
    y = simple_seasonal(t)
    return y


def single(L):
    t = np.arange(L)
    y = np.sin(2 * np.pi * t / 7)
    return y


def diverse(L):
    t = np.arange(L)
    y = seasonal(t) + trend(t)
    return y


def noisy(L):
    t = np.arange(L)
    y = seasonal(t) + trend(t)
    y += noise(y)
    return y
