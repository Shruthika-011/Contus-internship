from flask import Flask, request, jsonify, render_template
import pandas as pd
import ast
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


movies = pd.read_csv("movies.csv")
movies = movies[['title','overview','genres','vote_average']]
movies.dropna(inplace=True)


def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)


movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['genres_clean'] = movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])

movies['tags'] = movies['overview'] + movies['genres_clean']
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x).lower())


cv = CountVectorizer(max_features=3000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vectors)


platform_map = {
    "Netflix": "https://www.netflix.com",
    "Amazon Prime": "https://www.primevideo.com",
    "Disney+ Hotstar": "https://www.hotstar.com",
    "Zee5": "https://www.zee5.com"
}


all_genres = sorted(list({g for sub in movies['genres'] for g in sub}))


def recommend_movies(genre, min_rating):
    filtered = movies[
        (movies['genres'].apply(lambda x: genre in x)) &
        (movies['vote_average'] >= min_rating)
    ]

    if filtered.empty:
        return []

    indices = filtered.index.tolist()

    scores = []
    for i in indices:
        scores.append((i, sum(similarity[i])))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:10]

    result = []
    for i in scores:
        movie = movies.iloc[i[0]]

        # assign random platforms
        platforms = random.sample(list(platform_map.keys()), k=2)

        result.append({
            "title": movie['title'],
            "rating": float(movie['vote_average']),
            "platforms": [
                {"name": p, "link": platform_map[p]} for p in platforms
            ],
            "poster": "https://via.placeholder.com/300x450"
        })

    return result


@app.route('/')
def home():
    return render_template('front.html')

@app.route('/genres')
def get_genres():
    return jsonify(all_genres)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    genre = data['genre']
    rating = float(data['rating'])

    return jsonify(recommend_movies(genre, rating))


if __name__ == "__main__":
    app.run(debug=True)