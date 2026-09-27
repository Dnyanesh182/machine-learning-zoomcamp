import numpy as np
import pandas as pd

df = pd.read_csv("../../data/car_fuel_efficiency_2026.csv")

print(pd.__version__)
print(len(df))
print(df["fuel_type"].nunique())
print(df.isna().any().sum())
print(df.loc[df["origin"] == "Asia", "fuel_efficiency_mpg"].max())

horsepower = df["horsepower"]
print(horsepower.median())
print(horsepower.fillna(horsepower.mode().iloc[0]).median())

X = df.loc[df["origin"] == "Asia", ["vehicle_weight", "model_year"]].head(7).to_numpy()
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = np.linalg.inv(X.T @ X) @ X.T @ y
print(w.sum())