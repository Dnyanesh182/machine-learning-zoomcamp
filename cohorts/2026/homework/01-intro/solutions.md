# Homework 1 — Solutions

Dataset: `cohorts/2026/data/car_fuel_efficiency_2026.csv`

| Question | Answer |
| --- | --- |
| Q1. Pandas version | `2.3.3` |
| Q2. Records count | `10000` |
| Q3. Fuel types | `3` |
| Q4. Columns with missing values | `2` |
| Q5. Maximum fuel efficiency for cars from Asia | `41.2` |
| Q6. Did the horsepower median change after filling missing values with the mode? | Yes, it decreased (`254.0` to `252.0`) |
| Q7. Sum of weights | `0.369` (calculated value: `0.3691969690`) |

## Reproducible calculation

```python
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
```
