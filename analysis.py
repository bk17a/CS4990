# analysis.py
import pandas as pd
from clustering import lloyds, dbscan, kmedoids, manhattan_dist
from sklearn.preprocessing import MinMaxScaler
import matplotlib as plt


df = pd.read_csv("movies.csv")

df["MetacriticRating"] = df["MetacriticRating"].str.replace("/100", "").astype(float)
df["RottenTomatoesRating"] = (
    df["RottenTomatoesRating"].str.replace("%", "").astype(float)
)
columns = [
    "imdbRating",
    "MetacriticRating",
    "RottenTomatoesRating",
]  # do numerical vals
data = df[columns].dropna()

# normalize columns
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)
data_list = normalized_data.tolist()  # Convert to a list of lists

# Apply K-Means
k = 10  # Adjust based on silhouette or other metric
centers = lloyds(data_list, k, list(range(len(columns))), n=100)
print("K-Means Cluster Centers:", centers)

# Apply DBSCAN


# Apply K-Medoids
