"""Solutions for Homework 2: Machine Learning for Regression."""

from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "engine_displacement",
    "horsepower",
    "vehicle_weight",
    "model_year",
]
TARGET = "fuel_efficiency_mpg"
DATA_PATH = Path(__file__).parents[2] / "data" / "car_fuel_efficiency_2026.csv"


def split_data(df, seed):
    """Make the 60/20/20 split used in the lectures."""
    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)
    return (
        df.iloc[idx[:n_train]],
        df.iloc[idx[n_train:n_train + n_val]],
        df.iloc[idx[n_train + n_val:]],
    )


def prepare_x(df, fill_value):
    return df[FEATURES].fillna(fill_value).to_numpy()


def train_linear_regression(x, y, r=0.0):
    x = np.column_stack([np.ones(x.shape[0]), x])
    xtx = x.T @ x + r * np.eye(x.shape[1])
    weights = np.linalg.inv(xtx) @ x.T @ y
    return weights[0], weights[1:]


def rmse(y, y_pred):
    return np.sqrt(np.mean((y - y_pred) ** 2))


def score(train, validation, fill_value, r=0.0):
    x_train = prepare_x(train, fill_value)
    y_train = train[TARGET].to_numpy()
    w0, w = train_linear_regression(x_train, y_train, r)
    x_val = prepare_x(validation, fill_value)
    return rmse(validation[TARGET].to_numpy(), w0 + x_val @ w)


def main():
    df = pd.read_csv(DATA_PATH)[FEATURES + [TARGET]]
    train, validation, _ = split_data(df, seed=42)

    print("Q1:", df.columns[df.isna().any()][0])
    print("Q2:", df["horsepower"].median())
    print("Q3 (zero, mean):", round(score(train, validation, 0), 3),
          round(score(train, validation, train["horsepower"].mean()), 3))

    regularization = [0, 0.01, 0.1, 1, 5, 10, 100]
    q4_scores = {r: score(train, validation, 0, r) for r in regularization}
    print("Q4:", {r: float(round(value, 4)) for r, value in q4_scores.items()})

    seed_scores = []
    for seed in range(10):
        seed_train, seed_validation, _ = split_data(df, seed)
        seed_scores.append(score(seed_train, seed_validation, 0))
    print("Q5:", round(np.std(seed_scores), 3))

    train, validation, test = split_data(df, seed=9)
    combined = pd.concat([train, validation])
    print("Q6:", round(score(combined, test, 0, r=0.001), 3))


if __name__ == "__main__":
    main()
