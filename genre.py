import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_genre(filename):
    df = pd.read_csv(filename)

    # Split genres and create a new dataframe with one genre per row
    # Here we use stack to convert columns to rows
    genres = df["Genre"].str.split(", ").explode()

    # Count the occurrences of each individual genre
    genre_counts = genres.value_counts()

    # Display the genre counts
    print("\nIndividual Genre Counts:")
    print(genre_counts)

    # Most common genre
    most_common_genre = genre_counts.idxmax()
    print(f"\nMost Common Genre: {most_common_genre} ({genre_counts.max()})")

    # Create visualizations
    plt.figure(figsize=(12, 6))

    # Bar plot of genre counts
    plt.subplot(1, 2, 1)
    sns.barplot(x=genre_counts.index, y=genre_counts.values, palette="viridis")
    plt.title("Counts of Individual Genres")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Genres")
    plt.ylabel("Count")

    # Pie chart for the most common genre
    plt.subplot(1, 2, 2)
    plt.pie(
        [genre_counts.max(), genre_counts.sum() - genre_counts.max()],
        labels=[most_common_genre, "Other Genres"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#ff9999", "#66b3ff"],
    )
    plt.title("Most Common Genre vs Other Genres")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_genre("movies.csv")
