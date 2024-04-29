import pandas as pd
from flask import Flask, render_template, request
import pickle
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import random

app = Flask(__name__)

# Load the movie dataset
movies_df = pd.read_csv('movies.csv')

last_two_genre = ['', '']

# Load the pre-trained TF-IDF vectorizer and similarity matrix
with open('movies_list.pkl', 'rb') as f:
    movies_list = pickle.load(f)

with open('similarity.pkl', 'rb') as f:
    similarity = pickle.load(f)

def get_similar_movies(movie_title):
    if 'genre' not in movies_df.columns:
        return []
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['genre'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    idx = movies_df[movies_df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices].tolist()

def search_movies(keyword):
    if 'title' not in movies_df.columns:
        return []
    return movies_df[movies_df['title'].str.contains(keyword, case=False)]['title'].tolist()

def get_movie_genre(movie_title):
    movie = movies_df[movies_df['title'] == movie_title]
    if not movie.empty:
        return movie['genre'].values[0]
    else:
        return None

def get_recommendations_based_on_genre(genre, num_recommendations=5):
    global last_two_genre
    if genre:
        last_two_genre.pop(0)  # Remove the oldest genre
        last_two_genre.append(genre)  # Add the new genre to the list
        similar_movies = movies_df[movies_df['genre'].str.contains(genre, na=False)]
        num_movies = min(num_recommendations, len(similar_movies))
        return similar_movies.sample(num_movies)['title'].tolist()
    else:
        return []

def get_recommendations_from_last_two_genre(num_recommendations=5):
    global last_two_genre
    recommendations = []
    for genre in last_two_genre:
        recommendations.extend(get_recommendations_based_on_genre(genre, num_recommendations))
    # Shuffle the recommendations to provide a mixed list
    random.shuffle(recommendations)
    return recommendations[:num_recommendations]


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        keyword = request.form['search']
        search_results = search_movies(keyword)
        return render_template('index.html', search_results=search_results)
    else:
        return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        movie_title = request.form['movie_title']
        
        movie_genre = get_movie_genre(movie_title)
        
        recommended_movies = get_recommendations_based_on_genre(movie_genre)

        last_two_recommendations = get_recommendations_from_last_two_genre()
        movie_overview = {movie: movies_df[movies_df['title'] == movie]['overview'].values[0] for movie in recommended_movies}
        movie_genre = {movie: movies_df[movies_df['title'] == movie]['genre'].values[0] for movie in recommended_movies}
        
        return render_template('recommend.html', movie_title=movie_title, recommended_movies=recommended_movies, movie_overview=movie_overview, movie_genre=movie_genre, last_two_recommendations=last_two_recommendations)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
