import requests
import pandas as pd
from config import API_KEY


def get_movie_data(title):
    url = f"http://www.omdbapi.com/?apikey={API_KEY}&t={title}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error code: {response.status_code}")
        return None


def get_dataset(filename, rows=None):
    movie_titles = [
        "Avengers: Endgame",
        "Avengers: Infinity War",
        "The Avengers",
        "Avengers: Age of Ultron",
        "Iron Man",
        "Thor: Ragnarok",
        "Captain America: Civil War",
        "Guardians of the Galaxy Vol. 3",
        "Spider-Man: No Way Home",
        "Spider-Man: Far From Home",
        "Spider-Man: No Way Home",
        "The Amazing Spider-Man",
        "The Amazing Spider-Man 2",
        "Spider-Man: Homecoming",
    ]
    movie_data = []

    for title in movie_titles:
        data = get_movie_data(title)
        if data:
            movie_data.append(
                {
                    "Title": data.get("Title"),
                    "Length": data.get("Runtime"),
                    "Year": data.get("Year"),
                    "Rated": data.get("Rated"),
                    "Released": data.get("Released"),
                    "Genre": data.get("Genre"),
                    "imdbRating": data.get("imdbRating"),
                    "imdbID": data.get("imdbID"),
                    "BoxOffice": data.get("BoxOffice"),
                    "Awards": data.get("Awards"),
                }
            )

        if rows and len(movie_data) >= rows:
            break

        df = pd.DataFrame(movie_data)
        df.to_csv(filename, index=False)


get_dataset("movies.csv", rows=None)

df = pd.read_csv("movies.csv")
print(df)
