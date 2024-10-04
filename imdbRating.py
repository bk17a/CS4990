import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_imdb_ratings(filename):
    df = pd.read_csv(filename)

    # Clean and process imdbRating column
    df["imdbRating"] = pd.to_numeric(df["imdbRating"], errors="coerce")

    # Drop rows with NaN values in 'imdbRating'
    df = df.dropna(subset=["imdbRating"])

    # Calculate summary
    summary = {
        "Mean": df["imdbRating"].mean(),
        "Median": df["imdbRating"].median(),
        "Min": df["imdbRating"].min(),
        "Max": df["imdbRating"].max(),
        "Range": df["imdbRating"].max() - df["imdbRating"].min(),
        "Standard Deviation": df["imdbRating"].std(),
        "Q1": df["imdbRating"].quantile(0.25),
        "Q2 (Median)": df["imdbRating"].quantile(0.50),
        "Q3": df["imdbRating"].quantile(0.75),
    }

    # Display summary
    print("Summary for imdbRating:")
    for stat, value in summary.items():
        print(f"{stat}: {value:.2f}")

        # Visualization
    plt.figure(figsize=(14, 6))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(df["imdbRating"], bins=10, kde=True, color="skyblue")
    plt.title("Distribution of IMDb Ratings")
    plt.xlabel("IMDb Rating")
    plt.ylabel("Frequency")

    # Box Plot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df["imdbRating"], color="lightgreen")
    plt.title("Box Plot of IMDb Ratings")
    plt.xlabel("IMDb Rating")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_imdb_ratings("movies.csv")
