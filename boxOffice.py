import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_box_office(filename):
    df = pd.read_csv(filename)

    # Process BoxOffice column using raw string
    df["BoxOffice"] = df["BoxOffice"].replace({r"\$": "", r",": ""}, regex=True)
    df["BoxOffice"] = pd.to_numeric(df["BoxOffice"], errors="coerce")

    # Drop rows with N/A values
    df = df.dropna(subset=["BoxOffice"])

    # Calculate summary
    summary = {
        "Mean": df["BoxOffice"].mean(),
        "Median": df["BoxOffice"].median(),
        "Min": df["BoxOffice"].min(),
        "Max": df["BoxOffice"].max(),
        "Range": df["BoxOffice"].max() - df["BoxOffice"].min(),
        "Standard Deviation": df["BoxOffice"].std(),
        "Q1": df["BoxOffice"].quantile(0.25),
        "Q2 (Median)": df["BoxOffice"].quantile(0.50),
        "Q3": df["BoxOffice"].quantile(0.75),
    }

    # Display summary
    print("Summary Statistics for BoxOffice:")
    for stat, value in summary.items():
        print(f"{stat}: ${value:,.2f}")

    # Visualization
    plt.figure(figsize=(14, 6))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(
        df["BoxOffice"] / 1e6, bins=range(0, 901, 150), kde=True, color="skyblue"
    )
    plt.title("Distribution of Box Office Revenue")
    plt.xlabel("Box Office Revenue (Millions)")
    plt.ylabel("Frequency")

    # Box Plot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df["BoxOffice"] / 1e6, color="lightgreen")
    plt.title("Box Plot of Box Office Revenue")
    plt.xlabel("Box Office Revenue (Millions)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_box_office("movies.csv")
