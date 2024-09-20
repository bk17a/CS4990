import requests
import pandas as pd
from config import API_KEY
from movie_titles import movie_titles


def get_movie_data(title):
    url = f"http://www.omdbapi.com/?apikey={API_KEY}&t={title}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error code: {response.status_code}")
        return None


def process_movie_data(data):
    rotten_tomatoes_rating = "N/A"
    metacritic_rating = "N/A"

    for rating in data.get("Ratings", []):
        if rating["Source"] == "Rotten Tomatoes":
            rotten_tomatoes_rating = rating["Value"]
        elif rating["Source"] == "Metacritic":
            metacritic_rating = rating["Value"]

    return {
        "Title": data.get("Title"),
        "Length": data.get("Runtime"),
        "Year": data.get("Year"),
        "Rated": data.get("Rated"),
        "Released": data.get("Released"),
        "Genre": data.get("Genre"),
        "RottenTomatoesRating": rotten_tomatoes_rating,
        "MetacriticRating": metacritic_rating,
        "imdbRating": data.get("imdbRating"),
        "imdbVotes": data.get("imdbVotes"),
        "imdbID": data.get("imdbID"),
        "BoxOffice": data.get("BoxOffice"),
        "Awards": data.get("Awards") if data.get("Awards") != "N/A" else "No Awards",
        "Director": data.get("Director"),
        "Writer": data.get("Writer"),
        "Actors": data.get("Actors"),
        "Plot": data.get("Plot"),
        "Language": data.get("Language"),
        "Country": data.get("Country"),
        "Type:": data.get("Type"),
    }


def get_dataset(filename, rows=None):
    movie_data = []

    for title in movie_titles:
        data = get_movie_data(title)
        if data:
            movie_data.append(process_movie_data(data))

        if rows and len(movie_data) >= rows:
            break

        df = pd.DataFrame(movie_data)
        df.to_csv(filename, index=False)


get_dataset("movies.csv", rows=None)

df = pd.read_csv("movies.csv")
print(df)
