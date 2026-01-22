from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

df.boxplot(column=["HouseAge", "MedInc"])
plt.savefig('../figs/boxplot.png', dpi=2000)

# Optional: Display the plot
plt.show()

# Quick check
print(df.head())
print(df.shape)

# you can save the boxplot...