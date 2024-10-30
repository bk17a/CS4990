import pandas as pd
from clustering import lloyds, kmedoids, dbscan, manhattan_dist
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv("movies.csv")

# Preprocess ratings
df["MetacriticRating"] = df["MetacriticRating"].str.replace("/100", "").astype(float)
df["RottenTomatoesRating"] = (
    df["RottenTomatoesRating"].str.replace("%", "").astype(float)
)

# Select numerical columns for clustering
columns = [
    "imdbRating",
    "MetacriticRating",
    "RottenTomatoesRating",
]
data = df[columns].dropna()

# Normalize columns
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)
data_list = normalized_data.tolist()  # Convert to a list of lists

# Apply K-Means
k = 10  # Adjust based on silhouette or other metric
centers = lloyds(data_list, k, list(range(len(columns))), n=100)
print("K-Means Cluster Centers:")
for i, center in enumerate(centers):
    formatted_center = ", ".join(f"{value:.4f}" for value in center)
    print(f"Cluster {i + 1}: [{formatted_center}]")

# Apply K-Medoids to numerical data
k = 10  # Adjust the number of clusters based analysis
medoids = kmedoids(data_list, k, manhattan_dist, n=100, eps=0.001)
print("K-Medoids Medoids:")
for i, medoid in enumerate(medoids):
    formatted_medoid = ", ".join(f"{value:.4f}" for value in medoid)
    print(f"Medoid {i + 1}: [{formatted_medoid}]")

# To handle categorical data
CATEGORY_MAP = {
    "Action": 1,
    "Adventure": 2,
    "Drama": 3,
    "Comedy": 4,
    "Sci-Fi": 5,
    "Fantasy": 6,
    "Horror": 7,
    "Thriller": 8,
    "Romance": 9,
    "Biography": 10,
    "Crime": 11,
    "History": 12,
}

RATING_MAP = {
    "G": 1,
    "PG": 2,
    "PG-13": 3,
    "R": 4,
    "NC-17": 5,
    "N/A": 6,
    "UR": 7,
    "TV-G": 8,
    "TV-PG": 9,
    "TV-14": 10,
    "TV-MA": 11,
    "Approved": 12,
    "M": 13,
}


def compare_categories(a, b):
    def compare(cata, catb, mapping):
        if cata == catb:
            return 0
        return abs(mapping.get(cata, 0) - mapping.get(catb, 0))

    categories_to_compare = ["Genre", "Rated"]
    total_score = 0
    valid_comparisons = 0

    for col in categories_to_compare:
        if col in a and col in b:
            if col == "Genre":
                genres_a = [genre.strip() for genre in a[col].split(",")]
                genres_b = [genre.strip() for genre in b[col].split(",")]
                for genre_a in genres_a:
                    for genre_b in genres_b:
                        score = compare(genre_a, genre_b, CATEGORY_MAP)
                        total_score += score
                        valid_comparisons += 1
                        print(
                            f"Comparing {col}: {genre_a} vs {genre_b} -> Score: {score}"
                        )
            elif col == "Rated":
                score = compare(a[col], b[col], RATING_MAP)
                total_score += score
                valid_comparisons += 1
                print(f"Comparing {col}: {a[col]} vs {b[col]} -> Score: {score}")

    return total_score / valid_comparisons if valid_comparisons > 0 else 0.0


# Choosing two movies to compare
movie_a = df.iloc[0]
movie_b = df.iloc[14]

# Print the genres and ratings for debugging
print(f"Movie A - Genre: {movie_a['Genre']}, Rated: {movie_a['Rated']}")
print(f"Movie B - Genre: {movie_b['Genre']}, Rated: {movie_b['Rated']}")

# Calculate the category comparison score
comparison_score = compare_categories(movie_a, movie_b)
print(
    f"Comparison Score between '{movie_a['Title']}' and '{movie_b['Title']}': {comparison_score}"
)

# Apply K-Medoids with compare_categories
centroids = kmedoids(data_list, k, compare_categories, n=5)
print("Cluster Centers from K-Medoids:")
for i, center in enumerate(centroids):
    formatted_center = ", ".join(f"{value:.4f}" for value in center)
    print(f"Center {i + 1}: [{formatted_center}]")
